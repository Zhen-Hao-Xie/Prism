# method/base/context.py
"""
持续学习上下文管理模块
在 Forward 过程中传递状态，避免污染模型实例变量
"""
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class CLContext:
    is_training: bool = True
    task_id: Optional[int] = None
    step_count: int = 0

    auxiliary_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    def clear(self):
        """清空 step 级缓存"""
        self.auxiliary_losses.clear()
    
    def add_auxiliary_loss(self, name: str, loss: torch.Tensor):
        self.auxiliary_losses[name] = loss
    
    def get_total_auxiliary_loss(self) -> torch.Tensor:
        if not self.auxiliary_losses:
            return torch.tensor(0.0)
        return sum(self.auxiliary_losses.values())