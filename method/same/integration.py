"""
SAME 方法实现（集成到 CLIntegration 生命周期）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.base.peft_extension import register_peft_extension
from method.factory import CLMethodFactory

_PEFT_EXT_REGISTERED = False


def ensure_peft_extension_registered() -> None:
    """
    按需注入 SAME 的 PEFT 映射（幂等）。
    即使 PEFT 目录已经静态注册了 mapping，这里也不会有副作用。
    """
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.peft.tuners.same import SAMEConfig, SAMEModel

    register_peft_extension(
        peft_type="MOE_LORA_SAME",
        config_cls=SAMEConfig,
        tuner_model_cls=SAMEModel,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("same")
class SameIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)

        self.expert_num: int = int(getattr(config, "expert_num", 8))
        self.feature_dim: int = int(getattr(config, "clip_feature_dim", 768))
        self.cur_task: int = int(getattr(config, "cur_task", 0))
        self.temparature: float = float(getattr(config, "temparature", 2.0))
        self.temparature_2: float = float(getattr(config, "temparature_2", 1.5))
        self.threshold: float = float(getattr(config, "threshold", 0.5))
        self.remaining_prob: float = float(getattr(config, "remaining_prob", 0.85))
        self.other_total_prob: float = float(getattr(config, "other_total_prob", 0.15))
        ratio = getattr(config, "top2_ratio", [3.0, 2.0])
        s = float(ratio[0] + ratio[1]) if isinstance(ratio, (list, tuple)) and len(ratio) == 2 else 5.0
        self.normalized_ratio = [float(ratio[0]) / s, float(ratio[1]) / s] if s > 0 else [0.6, 0.4]

        # anchors（和 hide_llava 类似）
        self.image_anchors: Optional[torch.nn.ParameterList] = None
        self.text_anchors: Optional[torch.nn.ParameterList] = None
        self.image_boundary: Optional[torch.nn.ParameterList] = None
        self.text_boundary: Optional[torch.nn.ParameterList] = None

        self._last_routing: Optional[torch.Tensor] = None
        self._model_ref = None

    def initialize_model(self, model) -> None:
        self._model_ref = model
        device = next(model.parameters()).device

        # 冻结 backbone（SAME 只训练 lora_）
        for _, p in model.named_parameters():
            p.requires_grad = False

        # init anchors
        if self.image_anchors is None:
            self.image_anchors = torch.nn.ParameterList(
                [torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False) for _ in range(self.expert_num)]
            ).to(device)
        if self.text_anchors is None:
            self.text_anchors = torch.nn.ParameterList(
                [torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False) for _ in range(self.expert_num)]
            ).to(device)
        if self.image_boundary is None:
            self.image_boundary = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False) for _ in range(self.expert_num)]
            ).to(device)
        if self.text_boundary is None:
            self.text_boundary = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False) for _ in range(self.expert_num)]
            ).to(device)

        # attach to model for checkpoint save/load convenience
        model.image_anchors = self.image_anchors
        model.text_anchors = self.text_anchors
        model.image_boundary = self.image_boundary
        model.text_boundary = self.text_boundary

        self._setup_same_lora(model)

        # 再次保证 anchors 不被 PEFT 改成可训练
        for name, p in model.named_parameters():
            if any(x in name for x in ("image_anchors", "text_anchors", "image_boundary", "text_boundary")):
                p.requires_grad = False

    def _setup_same_lora(self, model) -> None:
        ensure_peft_extension_registered()
        from PEFT.peft import get_peft_model
        from PEFT.peft.tuners.same import SAMEConfig

        target_modules = self._find_target_modules(model)

        peft_config = SAMEConfig(
            target_modules=target_modules,
            r=int(getattr(self.config, "lora_r", 64)),
            lora_alpha=int(getattr(self.config, "lora_alpha", 128)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            expert_num=self.expert_num,
            cur_task=int(getattr(self.config, "cur_task", self.cur_task)),
            task_type="CAUSAL_LM",
        )

        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            peft_model = get_peft_model(_base_model, peft_config)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, peft_config)

    def _find_target_modules(self, model) -> List[str]:
        target_modules = set()
        _base_model = getattr(model, "_base_model", None) or model
        for name, module in _base_model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(x in name for x in ("q_proj", "k_proj", "v_proj", "o_proj")):
                target_modules.add(name.split(".")[-1])
        return list(target_modules)

    def _extract_clip_features(self, model, images, input_ids, clip_tokenizer, text_tower):
        device = images.device if images is not None else next(model.parameters()).device

        # image features
        image_feat = None
        if images is not None:
            vision_tower = getattr(model, "vision_tower", None)
            if vision_tower and getattr(vision_tower, "is_loaded", False):
                with torch.no_grad():
                    raw = vision_tower(images)
                    raw = raw[0] if isinstance(raw, tuple) else raw
                    image_feat = raw.mean(dim=1) if raw.dim() == 3 else raw

        # text features
        if input_ids is None:
            text_feat = torch.randn(1, self.feature_dim, device=device)
        else:
            main_tokenizer = getattr(model, "tokenizer", None)
            if main_tokenizer is None:
                _base_model = getattr(model, "_base_model", None)
                if _base_model is not None:
                    main_tokenizer = getattr(_base_model, "tokenizer", None)
            tok = main_tokenizer or clip_tokenizer
            input_pad = np.where(
                input_ids.cpu().detach().numpy() != -200,
                input_ids.cpu().detach().numpy(),
                tok.pad_token_id,
            )
            decoded = tok.batch_decode(input_pad, skip_special_tokens=True)
            decoded_hidden = ["\n".join(d.split("\n")[1:]) for d in decoded]
            decoded_clip = [d.split(" ASSISTANT")[0] for d in decoded_hidden]
            clip_inputs = clip_tokenizer(
                decoded_clip,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                text_feat = text_tower(clip_inputs)
                text_feat = text_feat[0] if isinstance(text_feat, tuple) else text_feat
        if text_feat.dim() == 1:
            text_feat = text_feat.unsqueeze(0)
        return image_feat, text_feat

    def _routing_from_sims(self, sims: torch.Tensor) -> torch.Tensor:
        sims = sims.to(dtype=torch.float32)
        n = sims.shape[0]
        routing = torch.zeros(n, device=sims.device, dtype=torch.float32)
        sorted_indices = torch.argsort(sims, descending=True)
        max_id = int(sorted_indices[0].item())
        top2_id = int(sorted_indices[1].item()) if n > 1 else max_id

        prob_top1 = torch.softmax(sims * self.temparature, dim=0)[max_id]
        routing[max_id] = prob_top1

        if n > 1:
            remain_idx = sorted_indices[1:]
            remain_sims = sims[remain_idx]
            remain_probs = torch.softmax(remain_sims * self.temparature_2, dim=0)
            routing[remain_idx] = (1.0 - prob_top1) * remain_probs

        if routing[max_id] < self.threshold and n > 2:
            other_mask = torch.ones(n, dtype=torch.bool, device=sims.device)
            other_mask[max_id] = False
            other_mask[top2_id] = False
            other_sims = sims[other_mask]
            other_probs = torch.softmax(other_sims * 1.5, dim=0) * self.other_total_prob

            routing = torch.zeros(n, dtype=torch.float32, device=sims.device)
            routing[max_id] = self.normalized_ratio[0] * self.remaining_prob
            routing[top2_id] = self.normalized_ratio[1] * self.remaining_prob
            routing[other_mask] = other_probs

        routing = routing / (routing.sum() + 1e-8)
        return routing

    def _propagate_routing(self, model, routing: torch.Tensor) -> None:
        routing = routing.detach()
        self._last_routing = routing
        for module in model.modules():
            if module.__class__.__name__ == "SAMELinear":
                target_len = int(getattr(module, "expert_num", routing.numel()))
                routed = routing
                if routing.numel() != target_len:
                    routed = torch.zeros(target_len, dtype=routing.dtype, device=routing.device)
                    copy_len = min(target_len, routing.numel())
                    routed[:copy_len] = routing[:copy_len]
                    routed = routed / (routed.sum() + 1e-8)
                module.router = routed.to(device=next(module.parameters()).device, dtype=torch.float32)

    def on_input_prep(self, model, args: tuple, kwargs: dict, context: CLContext) -> None:
        images = kwargs.get("images", None)
        input_ids = args[0] if args else None

        clip_tokenizer = getattr(model, "clip_tokenizer", None)
        text_tower = getattr(model, "text_tower", None)
        if clip_tokenizer is None or text_tower is None:
            _base_model = getattr(model, "_base_model", None)
            if _base_model is not None:
                clip_tokenizer = clip_tokenizer or getattr(_base_model, "clip_tokenizer", None)
                text_tower = text_tower or getattr(_base_model, "text_tower", None)
            if _base_model is not None and hasattr(_base_model, "base_model"):
                clip_tokenizer = clip_tokenizer or getattr(_base_model.base_model, "clip_tokenizer", None)
                text_tower = text_tower or getattr(_base_model.base_model, "text_tower", None)
        if clip_tokenizer is None or text_tower is None:
            return

        # 增量解码阶段复用上一步路由
        if input_ids is None or (hasattr(input_ids, "shape") and input_ids.shape[1] <= 1):
            if self._last_routing is not None:
                self._propagate_routing(model, self._last_routing)
            return

        image_feat, text_feat = self._extract_clip_features(model, images, input_ids, clip_tokenizer, text_tower)
        device = text_feat.device

        # training: update anchors with current task id if available
        if model.training and context.task_id is not None and 0 <= int(context.task_id) < self.expert_num:
            tid = int(context.task_id)
            bs = int(text_feat.shape[0])
            with torch.no_grad():
                if image_feat is not None:
                    old = self.image_anchors[tid].data.clone()
                    cnt = self.image_boundary[tid].data.clone()
                    new_cnt = cnt + bs
                    self.image_anchors[tid].data.copy_((old * cnt + image_feat.sum(dim=0)) / new_cnt)
                    self.image_boundary[tid].data.copy_(new_cnt)
                oldt = self.text_anchors[tid].data.clone()
                cntt = self.text_boundary[tid].data.clone()
                new_cntt = cntt + bs
                self.text_anchors[tid].data.copy_((oldt * cntt + text_feat.sum(dim=0)) / new_cntt)
                self.text_boundary[tid].data.copy_(new_cntt)

            routing = torch.zeros(self.expert_num, device=device, dtype=torch.float32)
            routing[tid] = 1.0
            self._propagate_routing(model, routing)
            return

        # inference: 文本+图像相似度融合（迁移自 same_llava_arch）
        text_sims = []
        for t in range(self.expert_num):
            ta = self.text_anchors[t].to(device)
            text_sims.append(F.cosine_similarity(text_feat.unsqueeze(1), ta.unsqueeze(0), dim=2).max())
        text_sims_t = torch.stack(text_sims).to(device=device, dtype=torch.float32)

        if image_feat is not None:
            image_sims = []
            for t in range(self.expert_num):
                ia = self.image_anchors[t].to(device)
                image_sims.append(F.cosine_similarity(image_feat.unsqueeze(1), ia.unsqueeze(0), dim=2).max())
            image_sims_t = torch.stack(image_sims).to(device=device, dtype=torch.float32)
            sims_t = 0.5 * image_sims_t + 0.5 * text_sims_t
        else:
            sims_t = text_sims_t

        routing = self._routing_from_sims(sims_t)
        self._propagate_routing(model, routing)

    def on_forward_start(self, model, context: CLContext) -> None:
        return

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_step_end(self, model, context: CLContext, loss: Optional[torch.Tensor] = None) -> None:
        return

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        # SAME 任务结束：固化每层 covariance 快照
        for module in model.modules():
            if hasattr(module, "save_task_covariance_snapshot") and hasattr(module, "active_adapter"):
                adapter = getattr(module, "active_adapter", None)
                # PEFT sometimes uses list[str] for active adapters
                if isinstance(adapter, (list, tuple)) and adapter:
                    adapter = adapter[0]
                if not isinstance(adapter, str) or not adapter:
                    continue
                module.save_task_covariance_snapshot(adapter)

    def get_inference_config(self) -> Dict:
        return {"expert_num": self.expert_num, "feature_dim": self.feature_dim}

    def save_extra_state(self, output_dir: str) -> bool:
        os.makedirs(output_dir, exist_ok=True)

        if self._model_ref is None:
            return False

        same_state = {}
        # 注意：CLModel 会把底层模型挂在 `_base_model` 下，
        # 直接用 named_buffers() 会产生带 `_base_model.` 前缀的 key。
        # 这里优先从真实可训练模型上取 buffers，避免保存出的 key 在加载时对不上。
        model_for_buffers = getattr(self._model_ref, "_base_model", None) or self._model_ref
        for name, buf in model_for_buffers.named_buffers():
            if any(k in name for k in ("cov_U_prev", "cov_S_prev", "importance", "cov_prev_valid")):
                same_state[name] = buf.detach().cpu().clone()
        if not same_state:
            return False

        # 与原 same_train.py 对齐的命名
        torch.save(same_state, os.path.join(output_dir, "same_state.bin"))
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        # 优先兼容 same_train.py 产物
        candidates = [
            os.path.join(load_dir, "same_state.bin"),
            os.path.join(load_dir, "same_state.pt"),
        ]
        p = next((x for x in candidates if os.path.exists(x)), None)
        if p is None:
            return False

        state = torch.load(p, map_location="cpu")
        if model is None or not isinstance(state, dict):
            return False

        # 兼容两种 key:
        # 1) 直接从底层模型保存：`...`
        # 2) 从 CLModel 保存：`_base_model....`
        if any(k.startswith("_base_model.") for k in state.keys()):
            state = {k[len("_base_model."):]: v for k, v in state.items()}

        target = getattr(model, "_base_model", None) or model

        # 关键：不要依赖 load_state_dict 的 key 精确匹配。
        # SAME 的 buffers 存在于注入后的各层（例如 *.cov_prev_valid_default），不同包装层级会改变前缀。
        # 这里按 `named_buffers()` 做后缀匹配逐个拷贝，保证 cov_* 真正写入。
        tracked_buffers = []
        copied = 0
        missing = 0

        # 为提升性能，先把 state keys 变成 list（避免多次 view/iterator）
        state_keys = list(state.keys())

        for buf_name, buf in target.named_buffers():
            if not any(k in buf_name for k in ("cov_U_prev", "cov_S_prev", "importance", "cov_prev_valid")):
                continue

            tracked_buffers.append(buf_name)

            if buf_name in state:
                buf.data.copy_(state[buf_name].to(dtype=buf.dtype, device=buf.device))
                copied += 1
                continue

            # 后缀匹配（优先最长 key 命中）
            cands = [k for k in state_keys if k.endswith(buf_name) or buf_name.endswith(k)]
            if not cands:
                missing += 1
                continue
            # 若 buf_name.endswth(k)，说明 state_key 是更短的“尾部路径”，同样可用
            best = max(cands, key=len)
            buf.data.copy_(state[best].to(dtype=buf.dtype, device=buf.device))
            copied += 1

        self._model_ref = model

        # 主动验证：如果 cov_prev_valid 仍然全 False，直接报错，避免训练到 hook 才炸
        cov_valid = []
        for buf_name, buf in target.named_buffers():
            if "cov_prev_valid_" in buf_name:
                try:
                    cov_valid.append(bool(buf.detach().cpu().item()))
                except Exception:
                    pass

        if cov_valid and sum(cov_valid) == 0:
            raise RuntimeError(
                f"SAME extra state loaded from {p} but cov_prev_valid remains all False. "
                f"copied={copied}, missing={missing}, tracked={len(tracked_buffers)}"
            )

        return copied > 0

