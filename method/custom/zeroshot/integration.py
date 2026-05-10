"""
Zeroshot：纯 LLaVA 基线，仅用于推理/评估。

不注入 PEFT，无路由/锚点/额外 loss；不参与持续学习训练（``run.py train --method zeroshot`` 会被拒绝）。
推理时仍可用 ``CLModel`` 包装以保持管线一致；所有生命周期钩子为空操作；无需 ``task_num`` / benchmark。
"""

from __future__ import annotations

from typing import Any, Dict

from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.factory import CLMethodFactory


@CLMethodFactory.register("zeroshot")
class ZeroshotIntegration(CLIntegration):
    def initialize_model(self, model) -> None:
        """不注入 LoRA/PEFT。"""
        return

    def on_input_prep(self, model, args: tuple, kwargs: dict, context: CLContext) -> None:
        return

    def on_forward_start(self, model, context: CLContext) -> None:
        return

    def on_forward_end(self, model, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_task_end(self, model, context: CLContext, task_id: int) -> None:
        return

    def get_inference_config(self) -> Dict:
        return {}
