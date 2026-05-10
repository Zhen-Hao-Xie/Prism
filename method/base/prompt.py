"""与 prompt tuning / soft prompt 类持续学习方法配套的 Integration 基类。

参照 ``RouterIntegration``（``method/base/router.py``）的组织方式：从 ``config`` 读取方法字段、
可选地将额外张量并入 ``adapter_model.safetensors``，并实现完整的 ``CLIntegration`` 生命周期。
与 ``RouterIntegration`` 不同，本类**不包含** CLIP 路由、专家混合向量、原型锚点等多任务路由逻辑，
适用于纯 PEFT Prompt / Prefix / P-Tuning 等「在前缀空间学习」的方法；具体 PEFT 注册应在子类的
``initialize_model`` 中完成。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn

from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.base.router import read_merge_write_safetensors


class PromptIntegration(CLIntegration):
    """Prompt 类方法的 Integration 基类（无 Router 侧混合与锚点）。"""

    def __init__(self, config: Any):
        super().__init__(config)
        # 常见 prompt 配置（子类或 ``config/methods/<name>.py`` 可按需提供）
        self.num_prompt_tokens: int = int(getattr(config, "num_prompt_tokens", 8))
        self.virtual_tokens: Optional[int] = getattr(config, "virtual_tokens", None)
        if self.virtual_tokens is not None:
            self.virtual_tokens = int(self.virtual_tokens)

    def initialize_model(self, model: nn.Module) -> None:
        """默认仅占位；子类在此注册 Prompt / Prefix PEFT 等。"""
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
        """导出与推理对齐的轻量配置（子类可扩展）。"""
        out: Dict[str, Any] = {
            "num_prompt_tokens": self.num_prompt_tokens,
        }
        if self.virtual_tokens is not None:
            out["virtual_tokens"] = self.virtual_tokens
        return out
