# method/base/integration.py
"""
CL 方法与模型集成的统一接口
每个方法 (SP, HiDe-LLaVA, RanPAC, SEFE) 都需要实现这个接口
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch.nn as nn
import torch

from .context import CLContext


class CLIntegration(ABC):
    """
    持续学习集成接口
    定义了方法生命周期中的所有拦截点
    """
    
    def __init__(self, config: Any):
        self.config = config
        
    @abstractmethod
    def initialize_model(self, model: nn.Module) -> None:
        """
        模型初始化时调用
        - 注册 Hook
        - 初始化 Router/Prototype/Gram 矩阵等
        - 配置 PEFT
        """
        pass
    
    @abstractmethod
    def on_input_prep(
        self, 
        model: nn.Module, 
        args: tuple, 
        kwargs: dict, 
        context: CLContext
    ) -> None:
        """
        输入准备阶段拦截 (prepare_inputs_labels_for_multimodal)
        - SP/HiDe-LLaVA: CLIP 路由，设置 PEFT 层的 task_id
        - RanPAC: 无需操作
        """
        pass
    
    @abstractmethod
    def on_forward_start(
        self, 
        model: nn.Module, 
        context: CLContext
    ) -> None:
        """
        Forward 开始前调用
        - 清空 Context 缓存
        - 重置状态
        """
        pass
    
    @abstractmethod
    def on_forward_end(
        self, 
        model: nn.Module, 
        outputs: Any, 
        context: CLContext
    ) -> Any:
        """
        Forward 结束后调用
        - SEFE: 注入正则 Loss
        - RanPAC: 注入辅助 Loss
        - SP: 清理状态
        返回修改后的 outputs
        """
        pass
    
    @abstractmethod
    def on_step_end(
        self, 
        model: nn.Module, 
        context: CLContext,
        loss: Optional[torch.Tensor] = None
    ) -> None:
        """
        训练步结束后调用 (Trainer Callback)
        - SP: 更新原型
        - RanPAC: 更新 Gram 矩阵
        - HiDe-LLaVA: 更新 Anchors
        """
        pass
    
    @abstractmethod
    def on_task_end(
        self, 
        model: nn.Module, 
        context: CLContext,
        task_id: int
    ) -> None:
        """
        任务训练结束后调用
        - SP: 冻结原型，更新混淆矩阵
        - RanPAC: 保存 Gram 矩阵
        """
        pass
    
    @abstractmethod
    def get_inference_config(self) -> Dict:
        """
        返回推理时需要的配置
        """
        pass
    def save_extra_state(self, output_dir: str, model=None) -> bool:
        """
        保存方法特定状态（原型、anchors、Gram 矩阵等）
        
        默认实现：不做任何操作
        子类根据需要重写
        
        Args:
            output_dir: 输出目录
            model: 可选，当前训练/保存时的 CL 包装模型（部分方法用于读取 PEFT 等权重）
            
        Returns:
            bool: 是否保存成功
        """
        # 默认实现：跳过
        return False
    
    def load_extra_state(self, load_dir: str) -> bool:
        """
        加载方法特定状态
        
        默认实现：不做任何操作
        子类根据需要重写
        
        Args:
            load_dir: 加载目录
            
        Returns:
            bool: 是否加载成功
        """
        # 默认实现：跳过
        return False
    
    def compute_total_loss(
        self, 
        base_loss: torch.Tensor, 
        context: CLContext
    ) -> torch.Tensor:
        """
        计算总 Loss = base_loss + auxiliary_losses + regularization
        默认实现，子类可重写
        """
        return base_loss + context.get_total_auxiliary_loss()