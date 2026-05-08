"""
Zeroshot：纯 LLaVA，不注入 PEFT，方法侧无任何路由/锚点/额外 loss。

仍使用 ``CLModel`` 包装以保持与现有训练/推理管线一致；所有生命周期钩子为空操作。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.factory import CLMethodFactory


@CLMethodFactory.register("zeroshot")
class ZeroshotIntegration(CLIntegration):
    def initialize_model(self, model) -> None:
        """不注入 LoRA/PEFT，不改冻结策略（由 ``ModelArguments`` / 训练脚本决定）。"""
        return

    def on_input_prep(self, model, args: tuple, kwargs: dict, context: CLContext) -> None:
        return

    def on_forward_start(self, model, context: CLContext) -> None:
        return

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_step_end(self, model, context: CLContext, loss: Optional[Any] = None) -> None:
        return

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        return

    def get_inference_config(self) -> Dict:
        return {}
