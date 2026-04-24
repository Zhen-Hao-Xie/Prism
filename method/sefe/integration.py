"""
SEFE: Answer Style Diversification & Regularized LoRA
This module provides the integration hooks for SEFE.
"""
from __future__ import annotations

import os
import torch
from typing import Any, Dict, List, Optional
import logging

from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.base.peft_extension import register_peft_extension
from method.base.peft_llm_targets import should_skip_module_for_peft_scan
from method.factory import CLMethodFactory

_PEFT_EXT_REGISTERED = False
_LOG = logging.getLogger(__name__)


def ensure_peft_extension_registered() -> None:
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    from PEFT.peft.tuners.sefe import SefeConfig, SefeModel
    from PEFT.peft.peft_model import PeftModelForCausalLM

    register_peft_extension(
        peft_type="SEFE",
        config_cls=SefeConfig,
        tuner_model_cls=SefeModel,
        task_type="CAUSAL_LM_SEFE",
        task_peft_model_cls=PeftModelForCausalLM,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("sefe")
class SefeIntegration(CLIntegration):
    """
    SEFE Method Integration (RegLoRA + ASD)
    ASD Data generation should be handled prior to training, usually in dataset logic.
    RegLoRA handles the regularized parameter preservation.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num = int(getattr(config, "task_num", 8))
        self.sefe_top_p = float(getattr(config, "sefe_top_p", 0.02))
        self.sefe_lambda_reg = float(
            getattr(config, "sefe_lambda_reg", 2500.0))
        self.cur_task = int(getattr(config, "cur_task", 0))
        self._model_ref = None

    def initialize_model(self, model) -> None:
        self._model_ref = model

        # Turn off gradients for the base model
        for _, p in model.named_parameters():
            p.requires_grad = False

        self._setup_sefe_peft(model)

        _LOG.info(
            f"[SEFE] Model initialized. Top P: {self.sefe_top_p}, Lambda Reg: {self.sefe_lambda_reg}")

    def _setup_sefe_peft(self, model) -> None:
        ensure_peft_extension_registered()
        from PEFT.peft import get_peft_model
        from PEFT.peft.tuners.sefe import SefeConfig

        target_modules = self._find_target_modules(model)

        sefe_cfg = SefeConfig(
            target_modules=target_modules,
            r=int(getattr(self.config, "lora_r", 64)),
            lora_alpha=int(getattr(self.config, "lora_alpha", 128)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            num_tasks=self.task_num,
            sefe_top_p=self.sefe_top_p,
            sefe_lambda_reg=self.sefe_lambda_reg,
            task_type="CAUSAL_LM",  # Using base PEFT type string mapped to PeftModelForCausalLM
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )

        _base = getattr(model, "_base_model", None) or model
        if getattr(model, "_base_model", None) is not None:
            peft_model = get_peft_model(_base, sefe_cfg)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, sefe_cfg)

        peft_wrapped = getattr(model, "_base_model", None)
        if peft_wrapped is not None and hasattr(peft_wrapped, "print_trainable_parameters"):
            peft_wrapped.print_trainable_parameters()

    def _find_target_modules(self, model) -> List[str]:
        target_modules = set()
        _base_model = getattr(model, "_base_model", None) or model
        for name, module in _base_model.named_modules():
            if should_skip_module_for_peft_scan(name, self.config):
                continue
            if isinstance(module, torch.nn.Linear) and any(x in name for x in ("q_proj", "k_proj", "v_proj", "o_proj")):
                target_modules.add(name.split(".")[-1])
        return list(target_modules)

    def on_input_prep(self, model, args: tuple, kwargs: dict, context: CLContext) -> None:
        pass

    def on_forward_start(self, model, context: CLContext) -> None:
        pass

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        if getattr(outputs, "loss", None) is not None:
            total_loss = self.compute_total_loss(outputs.loss, context)
            outputs.loss = total_loss
            if isinstance(outputs, tuple) and len(outputs) > 0:
                outputs = (total_loss,) + outputs[1:]
        return outputs

    def on_step_end(self, model, context: CLContext, loss: Optional[torch.Tensor] = None) -> None:
        pass

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        """
        Update mask based on current task delta without clearing LoRA.
        (Merging into base and clearing happens at the start of the next task)
        """
        from PEFT.peft.tuners.sefe import SefeLinear

        for name, module in model.named_modules():
            if isinstance(module, SefeLinear):
                adapter = getattr(module, "active_adapter", "default")
                if isinstance(adapter, (list, tuple)) and adapter:
                    adapter = adapter[0]
                if adapter:
                    module.update_mask(adapter)

        _LOG.info(
            f"[SEFE] Task {task_id} ended. RegLoRA masks updated.")

    def compute_total_loss(self, base_loss: torch.Tensor, context: CLContext) -> torch.Tensor:
        """
        Compute total loss with RegLoRA regularization.
        """
        return base_loss + context.get_total_auxiliary_loss()

    def get_inference_config(self) -> Dict:
        return {"task_num": self.task_num, "cur_task": getattr(self.config, "cur_task", 0)}

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        """
        Save the persistant masks.
        """
        os.makedirs(output_dir, exist_ok=True)
        sefe_state = {}

        model_for_buffers = getattr(
            self._model_ref, "_base_model", None) or self._model_ref
        for name, buf in model_for_buffers.named_buffers():
            if "weight_mask" in name:
                sefe_state[name] = buf.detach().cpu().clone()

        if not sefe_state:
            return False

        torch.save(sefe_state, os.path.join(output_dir, "sefe_state.bin"))
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        p = os.path.join(load_dir, "sefe_state.bin")
        if not os.path.exists(p):
            return False

        state = torch.load(p, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            return False

        if model is None:
            model = self._model_ref

        target = getattr(model, "_base_model",
                         None) or model if model is not None else None
        copied = 0
        state_keys = [k for k in state.keys() if isinstance(k, str)]

        if target is not None:
            for buf_name, buf in target.named_buffers():
                if "weight_mask" not in buf_name:
                    continue

                loaded_tensor = None
                if buf_name in state and isinstance(state[buf_name], torch.Tensor):
                    loaded_tensor = state[buf_name]
                else:
                    cands = [k for k in state_keys if k.endswith(
                        buf_name) or buf_name.endswith(k)]
                    if cands:
                        best = max(cands, key=len)
                        if isinstance(state[best], torch.Tensor):
                            loaded_tensor = state[best]

                if loaded_tensor is not None:
                    buf.data.copy_(loaded_tensor.to(
                        dtype=buf.dtype, device=buf.device))
                    copied += 1

        _LOG.info(
            f"[SEFE] Loaded accumulated sefe_state.")
