from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from method.base.context import CLContext
from method.custom.specialized_integration import RouterIntegration
from method.base.peft_extension import register_peft_extension
from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
from method.factory import CLMethodFactory

_PEFT_EXT_REGISTERED = False

_SAME_BUFFER_MARKERS = (
    "cov_U_prev",
    "cov_S_prev",
    "importance",
    "cov_prev_valid",
)


def _canonical_same_buffer_key(name: str) -> str:
    """Align buffer names across CLModel / PeftModel / saved checkpoint key variants."""
    s = name
    if s.startswith("_base_model."):
        s = s[len("_base_model.") :]
    while "base_model.model.model." in s:
        s = s.replace("base_model.model.model.", "base_model.model.", 1)
    return s


def _build_same_tensor_index(state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    idx: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not isinstance(k, str) or not isinstance(v, torch.Tensor):
            continue
        if not any(m in k for m in _SAME_BUFFER_MARKERS):
            continue
        idx[_canonical_same_buffer_key(k)] = v
    return idx


def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.tuners.custom.same import SAMEConfig, SAMEModel

    register_peft_extension(
        peft_type="MOE_LORA_SAME",
        config_cls=SAMEConfig,
        tuner_model_cls=SAMEModel,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("same")
class SameIntegration(RouterIntegration):
    def initialize_model(self, model) -> None:
        super().initialize_model(model)
        for _, p in model.named_parameters():
            p.requires_grad = False

        self._setup_same_lora(model)
        self.sync_same_cur_task(model)

    def sync_same_cur_task(self, model: Any) -> None:
        """Keep ``SAMELinear.cur_task`` aligned with ``config.cur_task`` (hook gating)."""
        from PEFT.tuners.custom.same import SAMELinear

        ct = int(getattr(self.config, "cur_task", self.cur_task))
        self.cur_task = ct
        for m in model.modules():
            if isinstance(m, SAMELinear):
                m.cur_task = ct

    def _any_same_cov_prev_valid(self, model: Any) -> bool:
        from PEFT.tuners.custom.same import SAMELinear

        for m in model.modules():
            if not isinstance(m, SAMELinear):
                continue
            adapter = getattr(m, "active_adapter", None)
            if isinstance(adapter, (list, tuple)) and adapter:
                adapter = adapter[0]
            if not isinstance(adapter, str) or not adapter:
                continue
            buf = getattr(m, f"cov_prev_valid_{adapter}", None)
            if (
                buf is not None
                and isinstance(buf, torch.Tensor)
                and buf.numel() == 1
                and bool(buf.detach().cpu().item())
            ):
                return True
        return False

    def _reconcile_cov_prev_valid_buffers(self, model: Any) -> None:
        """If ``cov_S_prev`` has energy but ``cov_prev_valid`` is False, fix flag (checkpoint mismatch)."""
        from PEFT.tuners.custom.same import SAMELinear

        for m in model.modules():
            if not isinstance(m, SAMELinear):
                continue
            adapter = getattr(m, "active_adapter", None)
            if isinstance(adapter, (list, tuple)) and adapter:
                adapter = adapter[0]
            if not isinstance(adapter, str) or not adapter:
                continue
            buf = getattr(m, f"cov_prev_valid_{adapter}", None)
            s_prev = getattr(m, f"cov_S_prev_{adapter}", None)
            if buf is None or s_prev is None:
                continue
            if not isinstance(buf, torch.Tensor) or buf.numel() != 1:
                continue
            if bool(buf.detach().cpu().item()):
                continue
            energy = (s_prev.detach().float() ** 2).sum().item()
            if energy > 1e-12:
                setattr(
                    m,
                    f"cov_prev_valid_{adapter}",
                    torch.tensor(True, device=buf.device, dtype=buf.dtype),
                )

    def on_forward_start(self, model, context: CLContext) -> None:
        return

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        """After first training forward on task>=1, snapshot prev cov if checkpoint left all flags False."""
        ct = int(getattr(self.config, "cur_task", getattr(self, "cur_task", 0)))
        if (
            ct >= 1
            and getattr(model, "training", False)
            and not getattr(self, "_same_fwd_bootstrap_done", False)
        ):
            if not self._any_same_cov_prev_valid(model):
                self._snapshot_same_carry_buffers(model)
            self._same_fwd_bootstrap_done = True
        return outputs

    def _setup_same_lora(self, model) -> None:
        ensure_peft_extension_registered()
        from PEFT import get_peft_model
        from PEFT.tuners.custom.same import SAMEConfig

        target_modules = self._find_target_modules(model)

        peft_config = SAMEConfig(
            target_modules=target_modules,
            r=int(getattr(self.config, "lora_r", 64)),
            lora_alpha=int(getattr(self.config, "lora_alpha", 128)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            expert_num=self.task_num,
            cur_task=int(getattr(self.config, "cur_task", self.cur_task)),
            tau_score=float(getattr(self.config, "tau_score", 0.1)),
            curvature_mu=float(getattr(self.config, "curvature_mu", 0.9)),
            window_size=int(getattr(self.config, "window_size", 3)),
            max_components=int(getattr(self.config, "max_components", 64)),
            cumulative_energy_ratio=float(getattr(self.config, "cumulative_energy_ratio", 0.9)),
            task_type="CAUSAL_LM",
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )

        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            peft_model = get_peft_model(_base_model, peft_config)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, peft_config)

    def _find_target_modules(self, model) -> List[str]:
        return collect_peft_target_linear_suffixes(model, self.config)

    def on_step_end(self, model, context: CLContext, loss: Optional[torch.Tensor] = None) -> None:
        return

    def _snapshot_same_carry_buffers(self, model: Any) -> None:
        """Materialize ``cov_*_prev`` / ``cov_prev_valid`` from running cov buffers (call before saving)."""
        for module in model.modules():
            if hasattr(module, "save_task_covariance_snapshot") and hasattr(module, "active_adapter"):
                adapter = getattr(module, "active_adapter", None)
                if isinstance(adapter, (list, tuple)) and adapter:
                    adapter = adapter[0]
                if not isinstance(adapter, str) or not adapter:
                    continue
                module.save_task_covariance_snapshot(adapter)

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        self._snapshot_same_carry_buffers(model)

    def get_inference_config(self) -> Dict:
        return {"task_num": self.task_num, "feature_dim": self.feature_dim}

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        os.makedirs(output_dir, exist_ok=True)
        same_state: Dict[str, Any] = dict(self.load_state())

        root = model if model is not None else self._model_ref
        if root is None:
            raise RuntimeError("SameIntegration.save_extra_state: both model and _model_ref are None")

        # Trainer may not invoke ``on_task_end`` before final ``save_model``; freeze carry buffers here.
        self._snapshot_same_carry_buffers(root)

        model_for_buffers = getattr(root, "_base_model", None) or root
        n_carry = 0
        for name, buf in model_for_buffers.named_buffers():
            if any(m in name for m in _SAME_BUFFER_MARKERS):
                same_state[name] = buf.detach().cpu().clone()
                n_carry += 1

        if n_carry == 0:
            raise RuntimeError(
                "SameIntegration.save_extra_state: no SAME carry-over buffers "
                "(cov_*/importance/cov_prev_valid) found on model; refusing empty save."
            )

        if not same_state:
            return False

        extra = self._same_state_to_tensor_bundle(same_state)
        self.merge_extra_into_adapter_safetensors(
            output_dir, extra, on_duplicate_key="prefer_second"
        )
        torch.save(same_state, os.path.join(output_dir, "same_state.bin"))
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        from safetensors.torch import load_file

        p: str
        state: Optional[Dict[str, Any]] = None
        st_path = os.path.join(load_dir, "adapter_model.safetensors")
        if os.path.isfile(st_path):
            flat = load_file(st_path)
            state = self._tensor_bundle_to_same_state(flat)
            if state:
                p = st_path

        if not state:
            p = os.path.join(load_dir, "same_state.bin")
            if not os.path.exists(p):
                return False
            state = torch.load(p, map_location="cpu")
            if not isinstance(state, dict):
                return False

        if any(k.startswith("_base_model.") for k in state.keys()):
            state = {k[len("_base_model."):]: v for k, v in state.items()}

        target = getattr(model, "_base_model", None) or model if model is not None else None

        copied = 0
        missing = 0
        tracked_buffers: List[str] = []
        state_keys = [k for k in state.keys() if isinstance(k, str)]
        tensor_index = _build_same_tensor_index(state)

        anchors_ok = self.restore_state(state, model=model)

        if target is not None:
            for buf_name, buf in target.named_buffers():
                if not any(m in buf_name for m in _SAME_BUFFER_MARKERS):
                    continue

                tracked_buffers.append(buf_name)

                if buf_name in state and isinstance(state[buf_name], torch.Tensor):
                    buf.data.copy_(state[buf_name].to(dtype=buf.dtype, device=buf.device))
                    copied += 1
                    continue

                ck = _canonical_same_buffer_key(buf_name)
                src = tensor_index.get(ck)
                if src is not None:
                    buf.data.copy_(src.to(dtype=buf.dtype, device=buf.device))
                    copied += 1
                    continue

                cands = [k for k in state_keys if k.endswith(buf_name) or buf_name.endswith(k)]
                if not cands:
                    missing += 1
                    continue
                best = max(cands, key=len)
                if isinstance(state[best], torch.Tensor):
                    buf.data.copy_(state[best].to(dtype=buf.dtype, device=buf.device))
                    copied += 1

        if model is not None:
            self.sync_same_cur_task(model)
            self._reconcile_cov_prev_valid_buffers(model)

        cov_valid: List[bool] = []
        if target is not None:
            for buf_name, buf in target.named_buffers():
                if "cov_prev_valid_" in buf_name:
                    try:
                        cov_valid.append(bool(buf.detach().cpu().item()))
                    except Exception:
                        pass

        if tracked_buffers and copied == 0:
            raise RuntimeError(
                f"SAME load_extra_state: checkpoint under {load_dir} has no matching buffer tensors "
                f"for {len(tracked_buffers)} tracked SAME buffers (key mismatch or empty save)."
            )

        if copied > 0 and cov_valid and sum(cov_valid) == 0:
            raise RuntimeError(
                f"SAME extra state loaded from {p} but cov_prev_valid remains all False. "
                f"copied={copied}, missing={missing}, tracked={len(tracked_buffers)}"
            )

        ok = copied > 0 or anchors_ok
        if ok:
            self.print_carryover_restore_summary(p, state, tag="[SAME]")
        return ok

