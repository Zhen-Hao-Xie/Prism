import os
import torch
def load_from_checkpoint(model, checkpoint_path, merge_lora=False, for_incremental_training=False):
    """从 checkpoint 加载模型权重"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    print(f"  for_incremental_training: {for_incremental_training}")
    
    import os
    import torch
    
    # ========== 找到 PEFT 模型 ==========
    lora_target = model
    if hasattr(model, '_base_model'):
        print("  Detected CLModel wrapper")
        lora_target = model._base_model
    
    print(f"  LoRA target model type: {type(lora_target)}")
    print(f"  has load_adapter: {hasattr(lora_target, 'load_adapter')}")
    
    # ========== 加载 non-LoRA 权重 ==========
    non_lora_path = os.path.join(checkpoint_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        print("  Loading non-LoRA weights...")
        non_lora_weights = torch.load(non_lora_path, map_location='cpu')
        
        target = lora_target
        if hasattr(target, 'base_model'):
            target = target.base_model
        if hasattr(target, 'model'):
            target = target.model
        
        target.load_state_dict(non_lora_weights, strict=False)
        print("    non-LoRA weights loaded")
    
    
    # ========== 加载 LoRA 权重 ==========
    if for_incremental_training:
        # 增量训练：使用 load_adapter
        if hasattr(lora_target, 'load_adapter'):
            if os.path.isdir(checkpoint_path):
                print("\n  Loading LoRA adapter (incremental training mode)...")
                print(f"      checkpoint_path: {checkpoint_path}")
                lora_target.load_adapter(checkpoint_path, adapter_name='default')
                print("    LoRA adapter loaded")
        else:
            print("  Model has no load_adapter method")
    
    # ========== 推理模式：也使用 load_adapter ==========
    else:
        print("\n  Inference mode: trying load_adapter...")
        if hasattr(lora_target, 'load_adapter'):
            if os.path.isdir(checkpoint_path):
                print(f"      load_adapter({checkpoint_path}, adapter_name='default')")
                
                # 检查 adapter_config.json
                config_path = os.path.join(checkpoint_path, 'adapter_config.json')
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    print(f"      adapter_config peft_type: {config.get('peft_type', 'unknown')}")
                
                lora_target.load_adapter(checkpoint_path, adapter_name='default')
                print("    load_adapter finished")
            else:
                print("    Checkpoint is not a directory, falling back to manual LoRA load...")
                _manual_load_lora(lora_target, checkpoint_path)
        else:
            print("    Model has no load_adapter, loading LoRA manually...")
            _manual_load_lora(lora_target, checkpoint_path)

    # ==========================================
    
    # ========== 检查专家权重 ==========
    expert_norms = {}
    for name, param in lora_target.named_parameters():
        if 'lora' in name.lower():
            import re
            # 匹配 loraA.0, loraA.1 等格式
            match = re.search(r'loraA\.(\d+)', name)
            if match:
                expert_id = match.group(1)
                if expert_id not in expert_norms:
                    expert_norms[expert_id] = []
                expert_norms[expert_id].append(param.norm().item())
    
    for expert_id in sorted(expert_norms.keys())[:4]:
        if expert_norms[expert_id]:
            avg_norm = sum(expert_norms[expert_id]) / len(expert_norms[expert_id])
            print(f"    Expert {expert_id}: avg_norm={avg_norm:.6f} (samples={len(expert_norms[expert_id])})")
    
    if '0' in expert_norms and '1' in expert_norms:
        avg0 = sum(expert_norms['0']) / len(expert_norms['0'])
        avg1 = sum(expert_norms['1']) / len(expert_norms['1'])
        if abs(avg0 - avg1) < 1e-6:
            print("    Warning: Expert 0 and Expert 1 weights appear identical")
        else:
            print(f"    Expert 0 vs 1 avg norm: {avg0:.6f} vs {avg1:.6f}")
    # ==================================
    
    print("\nCheckpoint load finished")
    return model


def _manual_load_lora(lora_target, checkpoint_path):
    """手动加载 LoRA 权重"""
    lora_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
    if not os.path.exists(lora_path):
        lora_path = os.path.join(checkpoint_path, 'adapter_model.bin')
    
    if os.path.exists(lora_path):
        print(f"      Manual load: {lora_path}")
        
        if lora_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            lora_weights = load_file(lora_path)
        else:
            lora_weights = torch.load(lora_path, map_location='cpu')
        
        model_keys = set(lora_target.state_dict().keys())
        
        # 键名映射
        remapped_weights = {}
        for key, value in lora_weights.items():
            new_key = key
            # 添加 .default
            if '.lora_A.loraA' in key and '.default.' not in key:
                new_key = key.replace('.lora_A.loraA', '.lora_A.default.loraA')
            elif '.lora_B.loraB' in key and '.default.' not in key:
                new_key = key.replace('.lora_B.loraB', '.lora_B.default.loraB')
            remapped_weights[new_key] = value
        
        matched = set(remapped_weights.keys()) & model_keys
        print(f"      Keys matched after remap: {len(matched)}/{len(remapped_weights)}")
        
        if len(matched) > 0:
            missing, unexpected = lora_target.load_state_dict(remapped_weights, strict=False)
            print("      Manual LoRA load finished")
        else:
            print("      Failed to map LoRA weights to model keys")
    else:
        print("      LoRA weight file not found")
