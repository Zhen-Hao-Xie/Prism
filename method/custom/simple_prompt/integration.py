"""
simple_prompt：每任务一组可学习 soft prompt + CLIP 文本/图像原型路由（训练更新、推理 argmax 选任务）。

加载 `simple_prompt_state.pt` 后会在 stdout 打印 anchors 摘要；文件内还包含全部任务的 ``task_prompts``（含已冻结历史）。
各任务 soft prompt 的范数校验见 ``logging.getLogger(__name__)`` 的 DEBUG。
推理阶段（``model.eval()``）会在 **stdout** 打印每条样本的 ``route_infer``（与 anchors 摘要一样会进入 ``run.py`` 的 infer 日志）；训练阶段不打印以免刷屏。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from method.base.context import CLContext
from config.backbone.llava import CLIP_FEATURE_DIM
from method.base.integration import CLIntegration
from method.base.peft_extension import register_peft_extension
from method.factory import CLMethodFactory

_PEFT_EXT_REGISTERED = False
_LOG = logging.getLogger(__name__)


def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.peft_model import PeftModelForCausalLMSimplePrompt
    from PEFT.tuners.custom.simple_prompt import SimplePromptConfig, SimplePromptModel

    register_peft_extension(
        peft_type="SIMPLE_PROMPT",
        config_cls=SimplePromptConfig,
        tuner_model_cls=SimplePromptModel,
        task_type="CAUSAL_LM_SIMPLE_PROMPT",
        task_peft_model_cls=PeftModelForCausalLMSimplePrompt,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("simple_prompt")
class Simple_promptIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.feature_dim: int = int(CLIP_FEATURE_DIM)
        self.num_prompt_tokens: int = int(getattr(config, "num_prompt_tokens", 8))
        self.cur_task: int = int(getattr(config, "cur_task", 0))

        self.image_anchors: Optional[torch.nn.ParameterList] = None
        self.text_anchors: Optional[torch.nn.ParameterList] = None
        self.image_boundary: Optional[torch.nn.ParameterList] = None
        self.text_boundary: Optional[torch.nn.ParameterList] = None
        self._model_ref: Optional[Any] = None

    def initialize_model(self, model) -> None:
        self._model_ref = model
        device = next(model.parameters()).device

        for _, p in model.named_parameters():
            p.requires_grad = False

        if self.image_anchors is None:
            self.image_anchors = torch.nn.ParameterList(
                [torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False) for _ in range(self.task_num)]
            ).to(device)
        if self.text_anchors is None:
            self.text_anchors = torch.nn.ParameterList(
                [torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False) for _ in range(self.task_num)]
            ).to(device)
        if self.image_boundary is None:
            self.image_boundary = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False) for _ in range(self.task_num)]
            ).to(device)
        if self.text_boundary is None:
            self.text_boundary = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False) for _ in range(self.task_num)]
            ).to(device)

        model.image_anchors = self.image_anchors
        model.text_anchors = self.text_anchors
        model.image_boundary = self.image_boundary
        model.text_boundary = self.text_boundary
        model.task_num = self.task_num

        self._setup_simple_prompt_peft(model)

        for name, p in model.named_parameters():
            if any(x in name for x in ("image_anchors", "text_anchors", "image_boundary", "text_boundary")):
                p.requires_grad = False

        self._sync_prompt_trainability(model)

        spm = self._find_simple_prompt_model(model)
        if spm is not None:
            n_train = sum(1 for p in spm.task_prompts if p.requires_grad)
            tid = int(getattr(self.config, "cur_task", self.cur_task))
            n_elems = sum(p.numel() for p in spm.task_prompts if p.requires_grad)
            _LOG.debug(
                "peft_ready tasks=%s prompt_len=%s trainable_prompt_slots=%s",
                spm.num_tasks,
                spm.num_prompt_tokens,
                n_train,
            )
            print(
                f"[simple_prompt] Training updates only task_prompts slot {tid}; "
                f"num_prompt_tokens={spm.num_prompt_tokens}; trainable elements in this slot={n_elems:,} "
                f"(should match optimizer prompt parameter count). "
                f"At inference, slot is chosen by text_anchor routing, independent of training cur_task.",
                flush=True,
            )

    def _find_simple_prompt_model(self, model) -> Optional[Any]:
        from PEFT.tuners.custom.simple_prompt import SimplePromptModel

        root = getattr(model, "_base_model", None) or model
        for m in root.modules():
            if isinstance(m, SimplePromptModel):
                return m
        return None

    def _debug_task_prompt_norms(self, spm: Optional[Any], tag: str) -> None:
        if spm is None:
            _LOG.debug("%s task_prompts: SimplePromptModel not found", tag)
            return
        norms = [float(p.detach().float().norm().item()) for p in spm.task_prompts]
        _LOG.debug("%s task_prompts: num_slots=%d L2_norms=%s", tag, len(norms), norms)

    def _setup_simple_prompt_peft(self, model) -> None:
        ensure_peft_extension_registered()
        from PEFT import get_peft_model
        from PEFT.tuners.custom.simple_prompt import SimplePromptConfig

        _base = getattr(model, "_base_model", None) or model

        sp_cfg = SimplePromptConfig(
            num_tasks=self.task_num,
            num_prompt_tokens=self.num_prompt_tokens,
            cur_task=int(getattr(self.config, "cur_task", self.cur_task)),
            inference_mode=False,
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )

        if getattr(model, "_base_model", None) is not None:
            peft_model = get_peft_model(_base, sp_cfg)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, sp_cfg)

        peft_wrapped = getattr(model, "_base_model", None)
        if peft_wrapped is not None and hasattr(peft_wrapped, "print_trainable_parameters"):
            peft_wrapped.print_trainable_parameters()

    def _sync_prompt_trainability(self, model) -> None:
        spm = self._find_simple_prompt_model(model)
        if spm is None:
            return
        if model.training:
            tid = int(getattr(self.config, "cur_task", self.cur_task))
            spm.set_trainable_prompts(tid)
        else:
            spm.set_trainable_prompts(None)

    def _clip_tokenizer_and_text_tower(self, model: Any) -> Tuple[Optional[Any], Optional[Any]]:
        clip_tokenizer = getattr(model, "clip_tokenizer", None)
        text_tower = getattr(model, "text_tower", None)
        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            clip_tokenizer = clip_tokenizer or getattr(_base_model, "clip_tokenizer", None)
            text_tower = text_tower or getattr(_base_model, "text_tower", None)
            if hasattr(_base_model, "base_model"):
                clip_tokenizer = clip_tokenizer or getattr(_base_model.base_model, "clip_tokenizer", None)
                text_tower = text_tower or getattr(_base_model.base_model, "text_tower", None)
        return clip_tokenizer, text_tower

    def _extract_clip_features(self, model, images, input_ids, clip_tokenizer, text_tower):
        device = images.device if images is not None else next(model.parameters()).device

        if images is not None:
            vision_tower = getattr(model, "vision_tower", None)
            if vision_tower and getattr(vision_tower, "is_loaded", False):
                with torch.no_grad():
                    raw = vision_tower(images)
                    raw = raw[0] if isinstance(raw, tuple) else raw
                    image_feat = raw.mean(dim=1) if raw.dim() == 3 else raw
            else:
                image_feat = None
        else:
            image_feat = None

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

    def _update_prototypes(self, image_feat: Optional[torch.Tensor], text_feat: torch.Tensor, task_id: Optional[int]):
        if task_id is None or int(task_id) >= self.task_num:
            return
        tid = int(task_id)
        bs = int(text_feat.shape[0])
        if bs == 0:
            return
        with torch.no_grad():
            if image_feat is not None:
                old = self.image_anchors[tid].data.clone()
                cnt = self.image_boundary[tid].data.clone()
                new_cnt = cnt + bs
                self.image_anchors[tid].data.copy_((old * cnt + image_feat.sum(dim=0)) / new_cnt)
                self.image_boundary[tid].data.copy_(new_cnt)
                self.image_anchors[tid].requires_grad = False
                self.image_boundary[tid].requires_grad = False

            oldt = self.text_anchors[tid].data.clone()
            cntt = self.text_boundary[tid].data.clone()
            new_cntt = cntt + bs
            self.text_anchors[tid].data.copy_((oldt * cntt + text_feat.sum(dim=0)) / new_cntt)
            self.text_boundary[tid].data.copy_(new_cntt)
            self.text_anchors[tid].requires_grad = False
            self.text_boundary[tid].requires_grad = False
        

    def _predict_task(self, text_feat: torch.Tensor, *, log_route: bool = False) -> int:
        text_sims = []
        for t in range(self.task_num):
            anchor = self.text_anchors[t].to(text_feat.device)
            sim = F.cosine_similarity(text_feat.unsqueeze(1), anchor.unsqueeze(0), dim=2).max().item()
            text_sims.append(sim)
        predicted = int(torch.argmax(torch.tensor(text_sims, device=text_feat.device)).item())
        sims_rounded = [round(s, 4) for s in text_sims]
        _LOG.debug("route_infer sims=%s -> task=%s", sims_rounded, predicted)
        if log_route:
            print(
                f"[simple_prompt] route_infer: pred_task={predicted} "
                f"text_anchor_max_cos={sims_rounded}",
                flush=True,
            )
        return predicted

    def _propagate_task_id(self, model, task_id: int) -> None:
        from PEFT.tuners.custom.simple_prompt import SimplePromptModel

        for m in model.modules():
            if isinstance(m, SimplePromptModel):
                m.predicted_task_id = int(task_id)
        model._last_predicted_task_id = int(task_id)


    def _print_loaded_anchors(self, path: str) -> None:
        """从 checkpoint 恢复 anchors 后打印摘要（便于核对是否加载成功）。"""
        print(f"\n[simple_prompt] Loaded anchors from file: {path}")
        if self.image_anchors is not None:
            print("  image_anchors (L2):")
            for i, p in enumerate(self.image_anchors):
                print(f"    task {i}: {float(p.detach().float().norm().item()):.4f}")
        if self.text_anchors is not None:
            print("  text_anchors (L2):")
            for i, p in enumerate(self.text_anchors):
                print(f"    task {i}: {float(p.detach().float().norm().item()):.4f}")
        if self.image_boundary is not None:
            print("  image_boundary (count):")
            for i, p in enumerate(self.image_boundary):
                print(f"    task {i}: {float(p.detach().item()):.2f}")
        if self.text_boundary is not None:
            print("  text_boundary (count):")
            for i, p in enumerate(self.text_boundary):
                print(f"    task {i}: {float(p.detach().item()):.2f}")
        _LOG.debug("anchors_loaded path=%s", path)

    def _text_only_routing(self, model, input_ids, clip_tokenizer, text_tower) -> None:
        if input_ids is None or input_ids.shape[1] <= 1:
            if hasattr(model, "_last_predicted_task_id") and model._last_predicted_task_id is not None:
                self._propagate_task_id(model, int(model._last_predicted_task_id))
                _LOG.debug("text_only reuse_last_task=%s", model._last_predicted_task_id)
            return
        device = next(model.parameters()).device
        input_pad = np.where(
            input_ids.cpu().detach().numpy() != -200,
            input_ids.cpu().detach().numpy(),
            clip_tokenizer.pad_token_id,
        )
        decoded_inputs = clip_tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        decoded_hidden = ["\n".join(d.split("\n")[1:]) for d in decoded_inputs]
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
        pred = self._predict_task(text_feat, log_route=not model.training)
        self._propagate_task_id(model, pred)

    def on_input_prep(self, model, args: tuple, kwargs: dict, context: CLContext) -> None:
        images = kwargs.get("images", None)
        input_ids = args[0] if args else None
        clip_tokenizer, text_tower = self._clip_tokenizer_and_text_tower(model)
        if clip_tokenizer is None or text_tower is None:
            return

        if images is None:
            _LOG.debug("on_input_prep text_only")
            self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
            return

        image_feat, text_feat = self._extract_clip_features(model, images, input_ids, clip_tokenizer, text_tower)

        if model.training:
            self._update_prototypes(image_feat, text_feat, context.task_id)
            tid = int(context.task_id) if context.task_id is not None else int(getattr(self.config, "cur_task", 0))
            self._propagate_task_id(model, tid)
            # 与 on_forward_start 双保险：避免某些包装下 train 状态与可训标记短暂不一致
            self._sync_prompt_trainability(model)
        else:
            pred = self._predict_task(text_feat, log_route=True)
            self._propagate_task_id(model, pred)
            _LOG.debug("on_input_prep eval route_task=%s", pred)

    def pre_generate_hook(self, model, input_ids, images, context: CLContext) -> bool:
        clip_tokenizer, text_tower = self._clip_tokenizer_and_text_tower(model)
        if clip_tokenizer is None or text_tower is None:
            return True
        if images is None:
            tid = context.task_id if context and context.task_id is not None else 0
            self._propagate_task_id(model, int(tid))
            _LOG.debug("pre_generate text_only task=%s", tid)
            return True
        image_feat, text_feat = self._extract_clip_features(model, images, input_ids, clip_tokenizer, text_tower)
        pred = self._predict_task(text_feat, log_route=not model.training)
        self._propagate_task_id(model, pred)
        print(f"pre_generate mm route_task={pred}")
        return True

    def on_forward_start(self, model, context: CLContext) -> None:
        spm = self._find_simple_prompt_model(model)
        if spm is not None:
            cfg = spm.peft_config[spm.active_adapter]
            cfg.cur_task = int(getattr(self.config, "cur_task", self.cur_task))
        self._sync_prompt_trainability(model)
        spm = self._find_simple_prompt_model(model)

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_step_end(self, model, context: CLContext, loss: Optional[torch.Tensor] = None) -> None:
        return

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        print(f"[simple_prompt] Finished training for task {task_id}")

    def get_inference_config(self) -> Dict:
        return {"task_num": self.task_num, "num_prompt_tokens": self.num_prompt_tokens}

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        os.makedirs(output_dir, exist_ok=True)
        state: Dict[str, Any] = {}
        if self.image_anchors is not None:
            state["image_anchors"] = [p.cpu().clone() for p in self.image_anchors]
        if self.text_anchors is not None:
            state["text_anchors"] = [p.cpu().clone() for p in self.text_anchors]
        if self.image_boundary is not None:
            state["image_boundary"] = [p.cpu().clone() for p in self.image_boundary]
        if self.text_boundary is not None:
            state["text_boundary"] = [p.cpu().clone() for p in self.text_boundary]
        state["task_num"] = self.task_num
        state["expert_num"] = self.task_num
        state["num_prompt_tokens"] = self.num_prompt_tokens

        root = model if model is not None else self._model_ref
        spm = self._find_simple_prompt_model(root) if root is not None else None
        if spm is not None:
            self._debug_task_prompt_norms(spm, "save_extra_state")
            state["task_prompts"] = [p.detach().cpu().clone() for p in spm.task_prompts]
            _LOG.debug(
                "save_extra_state task_prompts: saved %d slots each shape %s",
                len(state["task_prompts"]),
                tuple(state["task_prompts"][0].shape) if state["task_prompts"] else (),
            )
        else:
            _LOG.debug(
                "save_extra_state: SimplePromptModel not found (root=%s), task_prompts not in checkpoint",
                type(root).__name__ if root is not None else None,
            )

        if not state:
            return False
        torch.save(state, os.path.join(output_dir, "simple_prompt_state.pt"))
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        path = os.path.join(load_dir, "simple_prompt_state.pt")
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict):
            return False

        loaded_any = False
        anchor_loaded = False

        if "image_anchors" in state and self.image_anchors is not None:
            for i, p in enumerate(state["image_anchors"]):
                if i < len(self.image_anchors):
                    self.image_anchors[i].data.copy_(p)
                    loaded_any = True
                    anchor_loaded = True
        if "text_anchors" in state and self.text_anchors is not None:
            for i, p in enumerate(state["text_anchors"]):
                if i < len(self.text_anchors):
                    self.text_anchors[i].data.copy_(p)
                    loaded_any = True
                    anchor_loaded = True
        if "image_boundary" in state and self.image_boundary is not None:
            for i, p in enumerate(state["image_boundary"]):
                if i < len(self.image_boundary):
                    self.image_boundary[i].data.copy_(p)
                    loaded_any = True
                    anchor_loaded = True
        if "text_boundary" in state and self.text_boundary is not None:
            for i, p in enumerate(state["text_boundary"]):
                if i < len(self.text_boundary):
                    self.text_boundary[i].data.copy_(p)
                    loaded_any = True
                    anchor_loaded = True

        if anchor_loaded:
            self._print_loaded_anchors(path)

        if "task_prompts" in state:
            tp = state["task_prompts"]
            if isinstance(tp, (list, tuple)):
                norms_ckpt = [float(x.detach().float().norm().item()) for x in tp if isinstance(x, torch.Tensor)]
                _LOG.debug("load_extra_state: file task_prompts num=%d L2_norms=%s", len(norms_ckpt), norms_ckpt)

        if model is not None:
            if self.image_anchors is not None:
                object.__setattr__(model, "image_anchors", self.image_anchors)
            if self.text_anchors is not None:
                object.__setattr__(model, "text_anchors", self.text_anchors)
            if self.image_boundary is not None:
                object.__setattr__(model, "image_boundary", self.image_boundary)
            if self.text_boundary is not None:
                object.__setattr__(model, "text_boundary", self.text_boundary)

        if "task_prompts" in state and model is not None:
            spm = self._find_simple_prompt_model(model)
            if spm is None:
                _LOG.debug("load_extra_state: checkpoint has task_prompts but SimplePromptModel not found on model")
            else:
                blobs = state["task_prompts"]
                if not isinstance(blobs, (list, tuple)):
                    _LOG.warning("load_extra_state: task_prompts in checkpoint is not a list, skip")
                else:
                    n = min(len(blobs), len(spm.task_prompts))
                    for i in range(n):
                        t = blobs[i]
                        if not isinstance(t, torch.Tensor):
                            continue
                        if t.shape != spm.task_prompts[i].shape:
                            _LOG.warning(
                                "load_extra_state: task_prompts[%d] shape ckpt=%s model=%s skip",
                                i,
                                tuple(t.shape),
                                tuple(spm.task_prompts[i].shape),
                            )
                            continue
                        spm.task_prompts[i].data.copy_(
                            t.to(device=spm.task_prompts[i].device, dtype=spm.task_prompts[i].dtype)
                        )
                        loaded_any = True
                    self._debug_task_prompt_norms(spm, "load_extra_state(after task_prompts copy)")
        elif "task_prompts" in state and model is None:
            _LOG.debug("load_extra_state: checkpoint has task_prompts but model=None, skip prompt restore")

        _LOG.debug("load_extra_state ok=%s keys=%s", loaded_any, list(state.keys()))
        return loaded_any
