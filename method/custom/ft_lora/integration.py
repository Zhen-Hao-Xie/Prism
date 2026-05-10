"""
ft_lora：在 LLM 的 attention ∪ FFN（``attn_and_ffn``）注入标准 LoRA，逐任务顺序微调。

沿用 ``replay_lora`` 的 LoRA 注入方式（``PEFT.LoraConfig`` + ``get_peft_model``），但不做经验回放与额外状态；
前一任务的 checkpoint 仍可通过 ``previous_task_model_path`` / 推理 ``load_adapter`` 载入同一适配器继续训练。
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn

from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.factory import CLMethodFactory


@CLMethodFactory.register("ft_lora")
class Ft_loraIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)

    def initialize_model(self, model: nn.Module) -> None:
        for _, p in model.named_parameters():
            p.requires_grad = False
        self._setup_lora(model)
        r = int(getattr(self.config, "lora_r", 96))
        alpha = int(getattr(self.config, "lora_alpha", r * 2))
        print(
            f"[ft_lora] vanilla LoRA | target=attn+ffn | r={r} alpha={alpha} "
            f"| sequential fine-tuning (no replay)",
            flush=True,
        )

    def _find_target_modules(self, model: nn.Module) -> List[str]:
        return collect_peft_target_linear_suffixes(model, self.config)

    def _setup_lora(self, model: nn.Module) -> None:
        from PEFT import LoraConfig, get_peft_model

        target_modules = self._find_target_modules(model)
        r = int(getattr(self.config, "lora_r", 96))
        lora_config = LoraConfig(
            r=r,
            lora_alpha=int(getattr(self.config, "lora_alpha", r * 2)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )
        _base = getattr(model, "_base_model", None)
        if _base is not None:
            peft_model = get_peft_model(_base, lora_config)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, lora_config)
        peft_wrapped = getattr(model, "_base_model", None)
        if peft_wrapped is not None and hasattr(peft_wrapped, "print_trainable_parameters"):
            peft_wrapped.print_trainable_parameters()

    def on_input_prep(self, model: nn.Module, args: tuple, kwargs: dict, context: CLContext) -> None:
        return

    def on_forward_start(self, model: nn.Module, context: CLContext) -> None:
        return

    def on_forward_end(self, model: nn.Module, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_task_end(self, model: nn.Module, context: CLContext, task_id: int) -> None:
        return

    def get_inference_config(self) -> Dict[str, Any]:
        return {
            "lora_r": int(getattr(self.config, "lora_r", 96)),
            "peft_target_modules": getattr(self.config, "peft_target_modules", "attn_and_ffn"),
        }
