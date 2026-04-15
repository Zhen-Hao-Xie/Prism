# -*- encoding: utf-8 -*-
"""
Per-task soft prompt prefix for continual multimodal LM (LLaVA).

Training: only `cur_task` prompt parameters are trainable (set from integration).
Inference: `predicted_task_id` selects which prompt to prepend (set by CL integration).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ..utils import PeftType, PeftConfig


@dataclass
class SimplePromptConfig(PeftConfig):
    """PEFT config: one soft-prompt prefix per task."""

    num_tasks: int = field(default=8, metadata={"help": "Number of continual tasks / prompt slots"})
    num_prompt_tokens: int = field(default=8, metadata={"help": "Soft prompt length (virtual tokens)"})
    cur_task: int = field(default=0, metadata={"help": "Active task id during training (only this prompt trains)"})

    def __post_init__(self):
        self.peft_type = PeftType.SIMPLE_PROMPT
        if self.task_type is None:
            self.task_type = "CAUSAL_LM_SIMPLE_PROMPT"


class SimplePromptModel(nn.Module):
    """
    Wraps the HF causal LM: prepends task-specific soft prompts to `inputs_embeds`
    (or embeds `input_ids` then prepends).
    """

    def __init__(self, model: nn.Module, config: Dict[str, SimplePromptConfig], adapter_name: str):
        super().__init__()
        self.peft_config = config
        self.active_adapter = adapter_name
        cfg = self.peft_config[adapter_name]
        # Use the constructor argument `model` (not `self.model`) until `add_module` runs:
        # our __getattr__ delegates to the inner model; accessing `self.model` before it is
        # registered would recurse (__getattr__("model") -> getattr(self.model, ...) -> ...).
        inner_cfg = getattr(model, "config", None)
        if inner_cfg is None:
            hidden = 4096
        else:
            hidden = int(
                getattr(inner_cfg, "hidden_size", None)
                or getattr(inner_cfg, "d_model", 4096)
            )
        self.num_prompt_tokens = int(cfg.num_prompt_tokens)
        self.num_tasks = int(cfg.num_tasks)
        self.predicted_task_id: int = -1
        self.task_prompts = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.num_prompt_tokens, hidden) * 0.02)
                for _ in range(self.num_tasks)
            ]
        )
        self.add_module("model", model)
        self.config = getattr(model, "config", {"model_type": "custom"})

    def _active_task_id(self, training: bool) -> int:
        cfg = self.peft_config[self.active_adapter]
        if training:
            return int(getattr(cfg, "cur_task", 0))
        if self.predicted_task_id is not None and self.predicted_task_id >= 0:
            return int(self.predicted_task_id)
        return int(getattr(cfg, "cur_task", 0))

    def _prompt_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tid = self._active_task_id(self.training)
        tid = max(0, min(tid, self.num_tasks - 1))
        p = self.task_prompts[tid].to(device=device, dtype=dtype)
        return p.unsqueeze(0).expand(batch_size, -1, -1)

    def set_trainable_prompts(self, only_task_id: Optional[int]) -> None:
        """If only_task_id is None, freeze all prompt tensors; else train only that slot."""
        for i, p in enumerate(self.task_prompts):
            p.requires_grad = only_task_id is not None and i == int(only_task_id)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ):
        # Incremental decode: prompts already consumed in cache
        if past_key_values is not None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        if inputs_embeds is None and input_ids is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        if inputs_embeds is None:
            emb = self.model.get_input_embeddings()
            inputs_embeds = emb(input_ids)
            input_ids = None

        b, _, h = inputs_embeds.shape
        prompts = self._prompt_batch(b, inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)

        if attention_mask is not None:
            prefix = torch.ones(b, self.num_prompt_tokens, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix, attention_mask], dim=1)

        if labels is not None:
            ignore = torch.full(
                (b, self.num_prompt_tokens),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([ignore, labels], dim=1)

        if position_ids is not None:
            warnings.warn("[SimplePrompt] position_ids are ignored when prepending soft prompts.")
            position_ids = None

        return self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def __getattr__(self, name: str):
        # Subclass __getattr__ shadows nn.Module.__getattr__; restore submodule/parameter
        # lookup (e.g. task_prompts in _modules) before delegating to the wrapped LM.
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            pass
        try:
            inner = super().__getattribute__("_modules")["model"]
        except (AttributeError, KeyError):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from None
        return getattr(inner, name)
