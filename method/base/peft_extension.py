"""
PEFT 动态扩展注册工具。

目标：新增方法时，尽量在 method/<method_name>/ 目录内完成改动，
通过运行时注册把自定义 Config/Model 注入到 PEFT 映射中。
"""
from typing import Optional, Type


def register_peft_extension(
    peft_type: str,
    config_cls: Type,
    tuner_model_cls: Optional[Type] = None,
    task_type: Optional[str] = None,
    task_peft_model_cls: Optional[Type] = None,
) -> None:
    """
    将自定义 PEFT 组件注入 PEFT 内部映射表（幂等）。

    Args:
        peft_type: 自定义 peft_type 字符串，例如 "MOE_LORA_HiDe"
        config_cls: 自定义配置类（继承 PeftConfig）
        tuner_model_cls: 自定义 tuner model（用于 PeftModel 内部构造）
        task_type: 可选，任务类型键，例如 "CAUSAL_LM_HiDe"
        task_peft_model_cls: 可选，task_type 对应的 PeftModel 包装类
    """
    from PEFT import mapping as peft_mapping
    from PEFT import peft_model as peft_model_module

    peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls

    if tuner_model_cls is not None:
        peft_model_module.PEFT_TYPE_TO_MODEL_MAPPING[peft_type] = tuner_model_cls

    if task_type is not None and task_peft_model_cls is not None:
        peft_mapping.MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type] = task_peft_model_cls
