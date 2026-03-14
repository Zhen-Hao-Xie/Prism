import os
import torch
from PEFT.peft import WEIGHTS_NAME, set_peft_model_state_dict

def load_from_checkpoint(
    model, 
    checkpoint_path, 
    merge_lora=False,
    for_incremental_training=False
):
    """
    统一的 checkpoint 加载函数
    
    Args:
        model: 要加载权重的模型
        checkpoint_path: checkpoint 路径
        merge_lora: 是否合并 LoRA 权重（推理时常用）
        for_incremental_training: 是否为增量训练加载（影响键名处理方式）
    
    Returns:
        model: 加载完权重的模型
    """
    print(f'Loading checkpoint from {checkpoint_path}...')
    
    # 1. 加载 non_lora 权重
    non_lora_path = os.path.join(checkpoint_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        non_lora_weights = torch.load(non_lora_path, map_location='cpu')
        
        # 处理键名（根据加载模式选择不同的处理方式）
        if for_incremental_training:
            # 增量训练模式：保留 base_model. 前缀结构
            non_lora_weights = {
                (k[11:] if k.startswith('base_model.') else k): v 
                for k, v in non_lora_weights.items()
            }
            if any(k.startswith('model.model.') for k in non_lora_weights):
                non_lora_weights = {
                    (k[6:] if k.startswith('model.') else k): v 
                    for k, v in non_lora_weights.items()
                }
            model.base_model.model.load_state_dict(non_lora_weights, strict=False)
        else:
            # 推理模式：直接加载
            non_lora_weights = {
                (k[11:] if k.startswith('base_model.') else k): v 
                for k, v in non_lora_weights.items()
            }
            model.load_state_dict(non_lora_weights, strict=False)
        
        print(f"Loaded non-LoRA weights from {non_lora_path}")

    # 2. 加载 LoRA 权重
    lora_path = os.path.join(checkpoint_path, WEIGHTS_NAME)
    if os.path.exists(lora_path):
        adapters_weights = torch.load(lora_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        set_peft_model_state_dict(model, adapters_weights, adapter_name="default")
        print(f"Loaded LoRA weights from {lora_path}")

    # 3. 如果需要合并 LoRA
    if merge_lora and hasattr(model, 'merge_and_unload'):
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        print("LoRA weights merged successfully")

    return model