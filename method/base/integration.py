# method/base/integration.py
"""
CL 方法与模型集成的统一接口
每个方法都需要实现这个接口
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
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

    @property
    def peft_exclude_module_path_segments(self) -> Optional[List[str]]:
        """
        与 ``PEFT.utils.config.PeftConfig.exclude_module_path_segments`` 一致：
        ``None`` 使用 LLaVA 默认跳过集；``[]`` 关闭路径过滤；非空列表为自定义跳过分段名。
        在 ``config/methods/<method>.py`` 的 ``METHOD_CONFIG`` 中设置 ``exclude_module_path_segments``。
        """
        return getattr(self.config, "exclude_module_path_segments", None)

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
        - RanPAC: 注入辅助 Loss
        - SP: 清理状态
        返回修改后的 outputs
        """
        pass
    
    def on_step_end(
        self,
        model: nn.Module,
        context: CLContext,
        loss: Optional[torch.Tensor] = None,
    ) -> None:
        """
        训练步结束后调用（``CLTrainerCallback``，在 backward 与优化器更新之后）。
        - SP / RanPAC / HiDe 等：更新原型、Gram、Anchors…
        - 依赖 ``on_training_batch_end`` 的逻辑无需在此重复实现。
        """
        return
    
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
    
    def load_extra_state(self, load_dir: str, model=None) -> bool:
        """
        加载方法特定状态
        
        默认实现：不做任何操作
        子类根据需要重写
        
        Args:
            load_dir: 加载目录
            model: 可选，与 ``common/load_model.py`` 中调用约定一致（如 SAME/HiDe 从 PEFT 读 buffer）
            
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

    def prepare_training_data(self, data_args: Any, model_args: Any, training_args: Any = None) -> None:
        """
        在 ``make_supervised_data_module`` 之前调用，用于在不改训练脚本的前提下调整数据配置。

        典型用途：经验回放将历史样本写入侧车 JSON 并设置 ``data_args.memory_data_path``（与
        ``LazySupervisedDataset`` 已有逻辑对齐）；其它方法可忽略此钩子。

        Args:
            data_args: 训练数据参数（可原地修改）。
            model_args: 模型/方法参数。
            training_args: 训练参数（可选，用于 ``output_dir``、分布式 rank 等）。
        """
        return

    def on_training_batch_end(
        self,
        model: nn.Module,
        context: CLContext,
        batch: Dict[str, Any],
        *,
        loss: Optional[torch.Tensor] = None,
        trainer: Any = None,
    ) -> None:
        """
        由 ``LLaVATrainer.training_step`` 在每次 micro-batch 的 backward 之后调用（含梯度累积）。
        ``batch`` 与 ``compute_loss`` 收到的字典一致；若数据管线提供 ``cl_raw_example``（``list[dict]``），
        则为与 ``LazySupervisedDataset`` JSON 对齐的原始样本，可供经验回放等写入缓冲区。

        默认无操作；需在 backward 之后访问当前 ``batch`` 的方法应重写本方法。
        """
        return

    def should_store_training_example(
        self,
        model: nn.Module,
        context: CLContext,
        raw_example: Dict[str, Any],
        batch: Dict[str, Any],
        *,
        example_index: int,
        loss: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        是否将 ``raw_example`` 写入某类训练期缓冲区（如经验回放）。供 ``on_training_batch_end`` 内循环调用，
        便于子类按样本特征、``batch`` 张量、``loss`` 等自定义策略。默认 ``False``。
        """
        return False