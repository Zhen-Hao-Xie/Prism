from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.backbone.llava import CLIP_FEATURE_DIM
from method.base.context import CLContext
from method.base.integration import CLIntegration

_PRIOR_VEC_KEY = "_prior_expert_vec"
_LEGACY_PRIOR_KEY = "_last_routing"


def merge_tensor_bundles(
    first: Dict[str, torch.Tensor],
    second: Dict[str, torch.Tensor],
    *,
    on_duplicate_key: Literal["raise", "prefer_first", "prefer_second"] = "raise",
) -> Dict[str, torch.Tensor]:
    dup = set(first) & set(second)
    if dup and on_duplicate_key == "raise":
        sample = sorted(dup)[:32]
        raise ValueError(
            f"merge_tensor_bundles: duplicate keys ({len(dup)} total), e.g. {sample!r}"
        )
    if on_duplicate_key == "prefer_first":
        out = dict(second)
        out.update(first)
        return out
    out = dict(first)
    out.update(second)
    return out


def read_merge_write_safetensors(
    safetensors_path: str,
    extra: Dict[str, torch.Tensor],
    *,
    on_duplicate_key: Literal["raise", "prefer_first", "prefer_second"] = "prefer_second",
) -> bool:
    if not extra:
        return False

    from safetensors.torch import load_file, save_file

    if not os.path.isfile(safetensors_path):
        # PEFT may only have adapter_model.bin on some runs; still persist SAME extras.
        save_file(extra, safetensors_path)
        return True

    base: Dict[str, torch.Tensor] = load_file(safetensors_path)
    merged = merge_tensor_bundles(base, extra, on_duplicate_key=on_duplicate_key)
    save_file(merged, safetensors_path)
    return True

class PromptIntegration(CLIntegration):

    def __init__(self, config: Any):
        super().__init__(config)
        self.num_prompt_tokens: int = int(getattr(config, "num_prompt_tokens", 8))
        self.virtual_tokens: Optional[int] = getattr(config, "virtual_tokens", None)
        if self.virtual_tokens is not None:
            self.virtual_tokens = int(self.virtual_tokens)

    def initialize_model(self, model: nn.Module) -> None:
        return

    def on_input_prep(
        self,
        model: nn.Module,
        args: tuple,
        kwargs: dict,
        context: CLContext,
    ) -> None:
        return

    def on_forward_start(self, model: nn.Module, context: CLContext) -> None:
        return

    def on_forward_end(self, model: nn.Module, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_task_end(self, model: nn.Module, context: CLContext, task_id: int) -> None:
        return

    def get_inference_config(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "num_prompt_tokens": self.num_prompt_tokens,
        }
        if self.virtual_tokens is not None:
            out["virtual_tokens"] = self.virtual_tokens
        return out

class RouterIntegration(CLIntegration):
    _MCITBOX_SAME_PREFIX = "mcitbox.same."

    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.feature_dim: int = int(CLIP_FEATURE_DIM)
        self.cur_task: int = int(getattr(config, "cur_task", 0))

        self.image_anchors: Optional[torch.nn.ParameterList] = None
        self.text_anchors: Optional[torch.nn.ParameterList] = None
        self.image_boundary: Optional[torch.nn.ParameterList] = None
        self.text_boundary: Optional[torch.nn.ParameterList] = None

        self._model_ref: Any = None
        self._prior_expert_vec: Optional[torch.Tensor] = None
        self.mixture_logit_scale: float = float(
            getattr(config, "mixture_logit_scale", getattr(config, "routing_softmax_scale", 24.0))
        )
        self.peft_expert_layer_name: str = str(
            getattr(
                config,
                "peft_expert_layer_name",
                getattr(config, "peft_routing_module_name", "SAMELinear"),
            )
        )

    def initialize_model(self, model: Any) -> None:
        self._model_ref = model
        self._ensure_prototypes_on_model(model)

    def merge_extra_into_adapter_safetensors(
        self,
        output_dir: str,
        extra: Dict[str, torch.Tensor],
        *,
        on_duplicate_key: Literal["raise", "prefer_first", "prefer_second"] = "prefer_second",
    ) -> bool:
        st_path = os.path.join(output_dir, "adapter_model.safetensors")
        return read_merge_write_safetensors(
            st_path, extra, on_duplicate_key=on_duplicate_key
        )

    def _similarities_to_mixture(self, sims: torch.Tensor) -> torch.Tensor:
        x = sims.to(dtype=torch.float32).reshape(-1)
        if x.numel() != self.task_num:
            raise ValueError(
                f"_similarities_to_mixture: expected {self.task_num} scores, got {x.numel()}"
            )
        logits = x * self.mixture_logit_scale
        return F.softmax(logits, dim=0)

    def _ensure_prototypes_on_model(self, model: Any) -> None:
        device = next(model.parameters()).device
        if self.image_anchors is None:
            self.image_anchors = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                    for _ in range(self.task_num)
                ]
            ).to(device)
        if self.text_anchors is None:
            self.text_anchors = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                    for _ in range(self.task_num)
                ]
            ).to(device)
        if self.image_boundary is None:
            self.image_boundary = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                    for _ in range(self.task_num)
                ]
            ).to(device)
        if self.text_boundary is None:
            self.text_boundary = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                    for _ in range(self.task_num)
                ]
            ).to(device)

        model.image_anchors = self.image_anchors
        model.text_anchors = self.text_anchors
        model.image_boundary = self.image_boundary
        model.text_boundary = self.text_boundary
        model.task_num = self.task_num

    def _extract_clip_features(
        self, model: Any, images: Any, input_ids: Any, clip_tokenizer: Any, text_tower: Any
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        device = images.device if images is not None else next(model.parameters()).device

        image_feat: Optional[torch.Tensor] = None
        if images is not None:
            vision_tower = getattr(model, "vision_tower", None)
            if vision_tower and getattr(vision_tower, "is_loaded", False):
                with torch.no_grad():
                    raw = vision_tower(images)
                    raw = raw[0] if isinstance(raw, tuple) else raw
                    image_feat = raw.mean(dim=1) if raw.dim() == 3 else raw

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

    def _compute_cosine_similarities(
        self, image_feat: Optional[torch.Tensor], text_feat: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        text_sims: list = []
        for t in range(self.task_num):
            ta = self.text_anchors[t].to(device)
            text_sims.append(F.cosine_similarity(text_feat.unsqueeze(1), ta.unsqueeze(0), dim=2).max())
        text_sims_t = torch.stack(text_sims).to(device=device, dtype=torch.float32)

        if image_feat is not None:
            image_sims: list = []
            for t in range(self.task_num):
                ia = self.image_anchors[t].to(device)
                image_sims.append(F.cosine_similarity(image_feat.unsqueeze(1), ia.unsqueeze(0), dim=2).max())
            image_sims_t = torch.stack(image_sims).to(device=device, dtype=torch.float32)
            return 0.2 * image_sims_t + 0.8 * text_sims_t
        return text_sims_t

    def _update_running_prototypes(
        self,
        image_feat: Optional[torch.Tensor],
        text_feat: torch.Tensor,
        task_id: int,
    ) -> None:
        bs = int(text_feat.shape[0])
        with torch.no_grad():
            if image_feat is not None:
                old = self.image_anchors[task_id].data.clone()
                cnt = self.image_boundary[task_id].data.clone()
                new_cnt = cnt + bs
                self.image_anchors[task_id].data.copy_((old * cnt + image_feat.sum(dim=0)) / new_cnt)
                self.image_boundary[task_id].data.copy_(new_cnt)
            oldt = self.text_anchors[task_id].data.clone()
            cntt = self.text_boundary[task_id].data.clone()
            new_cntt = cntt + bs
            self.text_anchors[task_id].data.copy_((oldt * cntt + text_feat.sum(dim=0)) / new_cntt)
            self.text_boundary[task_id].data.copy_(new_cntt)

    def _batch_prepare(
        self,
        model: Any,
        images: Any,
        input_ids: Any,
        context: CLContext,
    ) -> None:
        clip_tokenizer, text_tower = self._clip_tokenizer_and_text_tower(model)
        if clip_tokenizer is None or text_tower is None:
            return

        if input_ids is None or (hasattr(input_ids, "shape") and input_ids.shape[1] <= 1):
            if self._prior_expert_vec is not None:
                self._write_expert_mix_to_modules(model, self._prior_expert_vec)
            return

        image_feat, text_feat = self._extract_clip_features(model, images, input_ids, clip_tokenizer, text_tower)
        device = text_feat.device

        if (
            model.training
            and torch.is_grad_enabled()
            and context.task_id is not None
            and 0 <= int(context.task_id) < self.task_num
        ):
            tid = int(context.task_id)
            self._update_running_prototypes(image_feat, text_feat, tid)
            mix = torch.zeros(self.task_num, device=device, dtype=torch.float32)
            mix[tid] = 1.0
            self._write_expert_mix_to_modules(model, mix)
            self._prior_expert_vec = mix.detach()
            return

        sims_t = self._compute_cosine_similarities(image_feat, text_feat, device)
        mix = self._similarities_to_mixture(sims_t)
        self._write_expert_mix_to_modules(model, mix)
        self._prior_expert_vec = mix.detach()

    def on_input_prep(
        self, model: Any, args: tuple, kwargs: dict, context: CLContext
    ) -> None:
        images = kwargs.get("images", None)
        input_ids = args[0] if args else None
        self._batch_prepare(model, images, input_ids, context)

    def pre_generate_hook(
        self, model: Any, input_ids: Any, images: Any, context: CLContext
    ) -> bool:
        self._batch_prepare(model, images, input_ids, context)
        return True

    def _write_expert_mix_to_modules(self, model: Any, mix: torch.Tensor) -> None:
        mix = mix.detach()
        target = self.peft_expert_layer_name
        for module in model.modules():
            if module.__class__.__name__ != target:
                continue
            target_len = int(getattr(module, "expert_num", mix.numel()))
            vec = mix
            if mix.numel() != target_len:
                vec = torch.zeros(target_len, dtype=mix.dtype, device=mix.device)
                copy_len = min(target_len, mix.numel())
                vec[:copy_len] = mix[:copy_len]
                vec = vec / (vec.sum() + 1e-8)
            module.router = vec.to(device=next(module.parameters()).device, dtype=torch.float32)

    def sync_anchors_to_model(self, model: Any) -> None:
        if model is None:
            return
        if self.image_anchors is not None:
            object.__setattr__(model, "image_anchors", self.image_anchors)
        if self.text_anchors is not None:
            object.__setattr__(model, "text_anchors", self.text_anchors)
        if self.image_boundary is not None:
            object.__setattr__(model, "image_boundary", self.image_boundary)
        if self.text_boundary is not None:
            object.__setattr__(model, "text_boundary", self.text_boundary)

    def load_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.image_anchors is not None:
            out["image_anchors"] = [p.detach().cpu().clone() for p in self.image_anchors]
        if self.text_anchors is not None:
            out["text_anchors"] = [p.detach().cpu().clone() for p in self.text_anchors]
        if self.image_boundary is not None:
            out["image_boundary"] = [p.detach().cpu().clone() for p in self.image_boundary]
        if self.text_boundary is not None:
            out["text_boundary"] = [p.detach().cpu().clone() for p in self.text_boundary]
        if self._prior_expert_vec is not None:
            t = self._prior_expert_vec.detach().cpu().contiguous().clone()
            out[_PRIOR_VEC_KEY] = t
            out[_LEGACY_PRIOR_KEY] = t.clone()
        return out

    def restore_state(self, state: Dict[str, Any], model: Optional[Any] = None) -> bool:
        anchors_ok = False
        prior_ok = False
        if "image_anchors" in state and isinstance(state["image_anchors"], (list, tuple)) and self.image_anchors is not None:
            for i, p in enumerate(state["image_anchors"]):
                if i < len(self.image_anchors) and isinstance(p, torch.Tensor):
                    self.image_anchors[i].data.copy_(
                        p.to(device=self.image_anchors[i].device, dtype=self.image_anchors[i].dtype)
                    )
            anchors_ok = True
        if "text_anchors" in state and isinstance(state["text_anchors"], (list, tuple)) and self.text_anchors is not None:
            for i, p in enumerate(state["text_anchors"]):
                if i < len(self.text_anchors) and isinstance(p, torch.Tensor):
                    self.text_anchors[i].data.copy_(
                        p.to(device=self.text_anchors[i].device, dtype=self.text_anchors[i].dtype)
                    )
            anchors_ok = True
        if "image_boundary" in state and isinstance(state["image_boundary"], (list, tuple)) and self.image_boundary is not None:
            for i, p in enumerate(state["image_boundary"]):
                if i < len(self.image_boundary) and isinstance(p, torch.Tensor):
                    self.image_boundary[i].data.copy_(
                        p.to(device=self.image_boundary[i].device, dtype=self.image_boundary[i].dtype)
                    )
            anchors_ok = True
        if "text_boundary" in state and isinstance(state["text_boundary"], (list, tuple)) and self.text_boundary is not None:
            for i, p in enumerate(state["text_boundary"]):
                if i < len(self.text_boundary) and isinstance(p, torch.Tensor):
                    self.text_boundary[i].data.copy_(
                        p.to(device=self.text_boundary[i].device, dtype=self.text_boundary[i].dtype)
                    )
            anchors_ok = True

        pv = None
        if _PRIOR_VEC_KEY in state and isinstance(state[_PRIOR_VEC_KEY], torch.Tensor):
            pv = state[_PRIOR_VEC_KEY]
        elif _LEGACY_PRIOR_KEY in state and isinstance(state[_LEGACY_PRIOR_KEY], torch.Tensor):
            pv = state[_LEGACY_PRIOR_KEY]
        if pv is not None:
            self._prior_expert_vec = pv.clone()
            prior_ok = True

        if model is not None:
            self._model_ref = model
            self.sync_anchors_to_model(model)
        return anchors_ok or prior_ok

    def _same_state_to_tensor_bundle(self, same_state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pref = self._MCITBOX_SAME_PREFIX
        out: Dict[str, torch.Tensor] = {}
        list_keys = ("image_anchors", "text_anchors", "image_boundary", "text_boundary")
        for lk in list_keys:
            if lk not in same_state:
                continue
            seq = same_state[lk]
            if not isinstance(seq, (list, tuple)):
                continue
            for i, t in enumerate(seq):
                if isinstance(t, torch.Tensor):
                    out[f"{pref}{lk}.{i}"] = t.detach().cpu().contiguous().clone()
        for pk in ("_prior_expert_vec", "_last_routing"):
            if pk in same_state and isinstance(same_state[pk], torch.Tensor):
                out[f"{pref}{pk}"] = same_state[pk].detach().cpu().contiguous().clone()
        for k, v in same_state.items():
            if k in list_keys or k in ("_prior_expert_vec", "_last_routing"):
                continue
            if isinstance(v, torch.Tensor):
                safe = k.replace(".", "__DOT__")
                out[f"{pref}buf.{safe}"] = v.detach().cpu().contiguous().clone()
        return out

    def _tensor_bundle_to_same_state(self, flat: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        pref = self._MCITBOX_SAME_PREFIX
        sub = {k: v for k, v in flat.items() if k.startswith(pref)}
        if not sub:
            return {}
        state: Dict[str, Any] = {}
        list_keys = ("image_anchors", "text_anchors", "image_boundary", "text_boundary")
        for lk in list_keys:
            pref_list = f"{pref}{lk}."
            idx_tensors: Dict[int, torch.Tensor] = {}
            for k, v in sub.items():
                if not k.startswith(pref_list):
                    continue
                tail = k[len(pref_list) :]
                if tail.isdigit():
                    idx_tensors[int(tail)] = v
            if idx_tensors:
                mx = max(idx_tensors)
                lst = [idx_tensors[i] for i in range(mx + 1) if i in idx_tensors]
                if lst:
                    state[lk] = lst
        for pk in ("_prior_expert_vec", "_last_routing"):
            key = f"{pref}{pk}"
            if key in sub:
                state[pk] = sub[key]
        buf_prefix = f"{pref}buf."
        for k, v in sub.items():
            if not k.startswith(buf_prefix):
                continue
            orig = k[len(buf_prefix) :].replace("__DOT__", ".")
            state[orig] = v
        return state

    def print_carryover_restore_summary(
        self, path: str, state: Dict[str, Any], tag: str = "[Router]"
    ) -> None:
        pass

    def save_carryover_file(self, output_dir: str, filename: str = "carryover_state.bin") -> bool:
        os.makedirs(output_dir, exist_ok=True)
        state = self.load_state()
        if not state:
            return False
        torch.save(state, os.path.join(output_dir, filename))
        return True

    def load_carryover_file(
        self, load_dir: str, model: Optional[Any] = None, filename: str = "carryover_state.bin"
    ) -> bool:
        p = os.path.join(load_dir, filename)
        if not os.path.exists(p):
            return False
        state = torch.load(p, map_location="cpu")
        if not isinstance(state, dict):
            return False
        ok = self.restore_state(state, model=model)
        if ok:
            self.print_carryover_restore_summary(p, state)
        return ok


