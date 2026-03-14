import torch
from typing import List
import sys

# HiDe 路径（请根据实际情况修改）
sys.path.append('/mnt/haiyangguo/mywork/CL-MLLM/LLaVA-HiDe')
from PEFT.peft import HiDeMOELoraConfig, TaskType, get_peft_model


def find_all_peft_target(model) -> List[str]:
    """找到所有需要注入 LoRA 的 Linear 层（跳过多模态组件）"""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'text_tower']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def create_lora_config(training_args, model_args, model):
    """创建 LoRA 配置"""
    kwargs = {
        "task_embedding_dim": model_args.task_embedding_dim,
        "expert_num": model_args.expert_num,
        "cur_task": model_args.cur_task,
    }
    lora_config = HiDeMOELoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_all_peft_target(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type=TaskType.CAUSAL_LM_HiDe,
        **kwargs
    )
    return lora_config


def apply_lora(model, lora_config, training_args):
    """应用 LoRA 到模型"""
    if training_args.bits == 16:
        if training_args.bf16:
            model = model.to(torch.bfloat16)
        if training_args.fp16:
            model = model.to(torch.float16)
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    return model