"""
SAME 方法实现（集成到 CLIntegration 生命周期）。
"""

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


def ensure_peft_extension_registered() -> None:
    """
    按需注入 SAME 的 PEFT 映射（幂等）。
    即使 PEFT 目录已经静态注册了 mapping，这里也不会有副作用。
    """
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
        """
        与 ``METHOD_CONFIG.peft_target_modules`` / ``--peft_target_modules`` 对齐：
        ``ffn`` / ``attn``（``attention``）/ ``linear``（``all``/``full``）等，见 ``PEFT.utils.peft_target_modules``。
        未配置时默认仅 attention（与原 SAME 行为一致）。
        """
        return collect_peft_target_linear_suffixes(model, self.config)

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
        return {"task_num": self.task_num, "feature_dim": self.feature_dim}

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        os.makedirs(output_dir, exist_ok=True)
        same_state: Dict[str, Any] = dict(self.load_state())

        if self._model_ref is not None:
            model_for_buffers = getattr(self._model_ref, "_base_model", None) or self._model_ref
            for name, buf in model_for_buffers.named_buffers():
                if any(k in name for k in ("cov_U_prev", "cov_S_prev", "importance", "cov_prev_valid")):
                    same_state[name] = buf.detach().cpu().clone()

        if not same_state:
            return False

        extra = self._same_state_to_tensor_bundle(same_state)
        self.merge_extra_into_adapter_safetensors(
            output_dir, extra, on_duplicate_key="prefer_second"
        )
        # 始终写 sidecar，避免仅依赖 safetensors 嵌入时因多卡竞态/旧 checkpoint 丢键而无法恢复
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

        anchors_ok = self.restore_state(state, model=model)

        if target is not None:
            for buf_name, buf in target.named_buffers():
                if not any(k in buf_name for k in ("cov_U_prev", "cov_S_prev", "importance", "cov_prev_valid")):
                    continue

                tracked_buffers.append(buf_name)

                if buf_name in state and isinstance(state[buf_name], torch.Tensor):
                    buf.data.copy_(state[buf_name].to(dtype=buf.dtype, device=buf.device))
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

        cov_valid: List[bool] = []
        if target is not None:
            for buf_name, buf in target.named_buffers():
                if "cov_prev_valid_" in buf_name:
                    try:
                        cov_valid.append(bool(buf.detach().cpu().item()))
                    except Exception:
                        pass

        if copied > 0 and cov_valid and sum(cov_valid) == 0:
            raise RuntimeError(
                f"SAME extra state loaded from {p} but cov_prev_valid remains all False. "
                f"copied={copied}, missing={missing}, tracked={len(tracked_buffers)}"
            )

        ok = copied > 0 or anchors_ok
        if ok:
            self.print_carryover_restore_summary(p, state, tag="[SAME]")
        return ok

