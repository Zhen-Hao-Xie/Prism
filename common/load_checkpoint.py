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
        print(f"  🔍 检测到 CLModel 包装")
        lora_target = model._base_model
    
    print(f"  ✅ LoRA 目标模型类型：{type(lora_target)}")
    print(f"  📌 是否有 load_adapter 方法: {hasattr(lora_target, 'load_adapter')}")
    
    # ========== 加载 non-LoRA 权重 ==========
    non_lora_path = os.path.join(checkpoint_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        print(f"  📂 加载 non-LoRA 权重...")
        non_lora_weights = torch.load(non_lora_path, map_location='cpu')
        
        target = lora_target
        if hasattr(target, 'base_model'):
            target = target.base_model
        if hasattr(target, 'model'):
            target = target.model
        
        target.load_state_dict(non_lora_weights, strict=False)
        print(f"    ✅ non-LoRA 加载完成")
    
    # ========== 记录加载前的 LoRA 权重 ==========
    print(f"\n🔍 加载前的 LoRA 权重（前3个）:")
    lora_params_before = {}
    count = 0
    for name, param in lora_target.named_parameters():
        if 'lora' in name.lower() and count < 3:
            lora_params_before[name] = param.data.clone().cpu()
            print(f"    {name}: norm={param.norm().item():.6f}")
            count += 1
    # ==========================================
    
    # ========== 加载 LoRA 权重 ==========
    if for_incremental_training:
        # 增量训练：使用 load_adapter
        if hasattr(lora_target, 'load_adapter'):
            if os.path.isdir(checkpoint_path):
                print(f"\n  📂 加载 LoRA adapter (增量训练模式)...")
                print(f"      checkpoint_path: {checkpoint_path}")
                lora_target.load_adapter(checkpoint_path, adapter_name='default')
                print(f"    ✅ LoRA adapter 加载完成")
        else:
            print(f"  ❌ 模型没有 load_adapter 方法")
    
    # ========== 推理模式：也使用 load_adapter ==========
    else:
        print(f"\n  📂 推理模式：尝试使用 load_adapter...")
        if hasattr(lora_target, 'load_adapter'):
            if os.path.isdir(checkpoint_path):
                print(f"      调用 load_adapter({checkpoint_path}, adapter_name='default')")
                
                # 检查 adapter_config.json
                config_path = os.path.join(checkpoint_path, 'adapter_config.json')
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    print(f"      adapter_config peft_type: {config.get('peft_type', 'unknown')}")
                
                lora_target.load_adapter(checkpoint_path, adapter_name='default')
                print(f"    ✅ load_adapter 调用完成")
            else:
                print(f"    ⚠️ Checkpoint 不是目录，回退到手动加载...")
                _manual_load_lora(lora_target, checkpoint_path)
        else:
            print(f"    ⚠️ 模型没有 load_adapter，手动加载...")
            _manual_load_lora(lora_target, checkpoint_path)
    
    # ========== 记录加载后的 LoRA 权重 ==========
    print(f"\n🔍 加载后的 LoRA 权重（前3个）:")
    count = 0
    changed = 0
    for name, param in lora_target.named_parameters():
        if 'lora' in name.lower() and count < 3:
            current_norm = param.norm().item()
            print(f"    {name}: norm={current_norm:.6f}")
            
            if name in lora_params_before:
                diff = (param.data.cpu() - lora_params_before[name]).norm().item()
                if diff > 1e-6:
                    print(f"      ✅ 权重已改变 (diff={diff:.6f})")
                    changed += 1
                else:
                    print(f"      ⚠️ 权重未改变!")
            count += 1
    
    print(f"\n  📊 权重变化: {changed}/{count} 个改变")
    # ==========================================
    
    # ========== 检查专家权重 ==========
    print(f"\n🔍 检查专家权重差异:")
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
            print(f"    ⚠️ 警告：Expert 0 和 Expert 1 权重相同！")
        else:
            print(f"    ✅ Expert 0 vs 1: {avg0:.6f} vs {avg1:.6f}")
    # ==================================
    
    print(f"\n✅ Checkpoint 加载完成")
    return model


def _manual_load_lora(lora_target, checkpoint_path):
    """手动加载 LoRA 权重"""
    lora_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
    if not os.path.exists(lora_path):
        lora_path = os.path.join(checkpoint_path, 'adapter_model.bin')
    
    if os.path.exists(lora_path):
        print(f"      手动加载: {lora_path}")
        
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
        print(f"      映射后匹配: {len(matched)}/{len(remapped_weights)}")
        
        if len(matched) > 0:
            missing, unexpected = lora_target.load_state_dict(remapped_weights, strict=False)
            print(f"      ✅ 手动加载完成")
        else:
            print(f"      ❌ 无法映射权重")
    else:
        print(f"      ⚠️ 未找到 LoRA 权重文件")
# def load_from_checkpoint(model, checkpoint_path, merge_lora=False, for_incremental_training=False):
#     """从 checkpoint 加载模型权重"""
#     print(f"Loading checkpoint from {checkpoint_path}...")
    
#     import os
#     import torch
    
#     # ========== 找到正确的底层模型 ==========
#     target_model = model
    
#     if hasattr(model, '_base_model'):
#         print(f"  🔍 检测到 CLModel 包装")
#         target_model = model._base_model
        
#         if hasattr(target_model, 'base_model'):
#             print(f"  🔍 检测到内部还有 PEFT 包装")
#             target_model = target_model.base_model
    
#     elif hasattr(model, 'base_model'):
#         print(f"  🔍 检测到 PEFT 包装")
#         target_model = model.base_model
        
#         if hasattr(target_model, 'model'):
#             target_model = target_model.model
    
#     else:
#         if hasattr(model, 'model'):
#             target_model = model.model
    
#     print(f"  ✅ 目标模型类型：{type(target_model)}")
    
#     # ========== 加载 non-LoRA 权重 ==========
#     non_lora_path = os.path.join(checkpoint_path, 'non_lora_trainables.bin')
#     if os.path.exists(non_lora_path):
#         print(f"  Loaded non-LoRA weights from {non_lora_path}")
#         non_lora_weights = torch.load(non_lora_path, map_location='cpu')
        
#         if hasattr(target_model, 'model'):
#             target_model.model.load_state_dict(non_lora_weights, strict=False)
#         else:
#             target_model.load_state_dict(non_lora_weights, strict=False)
    
#     # ========== 加载 LoRA 权重（关键修复）==========
#     if for_incremental_training:
#         # 检查是否是 PEFT 模型
#         lora_target = model
#         if hasattr(model, '_base_model'):
#             lora_target = model._base_model
        
#         # 关键修复：load_adapter 期望目录路径，不是文件路径
#         if hasattr(lora_target, 'load_adapter'):
#             # PEFT 模型：使用 load_adapter，传入目录路径（不是文件路径）
#             if os.path.exists(checkpoint_path):
#                 print(f"  Loaded LoRA adapter from {checkpoint_path}")
#                 lora_target.load_adapter(checkpoint_path, adapter_name='default')  # ← 传入目录
#             else:
#                 print(f"  ⚠️  Checkpoint 目录不存在：{checkpoint_path}")
#         else:
#             # 非 PEFT 模型：直接加载状态字典
#             lora_path = os.path.join(checkpoint_path, 'adapter_model.bin')
#             if os.path.exists(lora_path):
#                 print(f"  Loaded LoRA weights from {lora_path}")
#                 lora_weights = torch.load(lora_path, map_location='cpu')
#                 lora_target.load_state_dict(lora_weights, strict=False)
#                   # ========== 诊断：检查权重内容 ==========
#             print(f"\n🔍 LoRA 权重诊断:")
#             print(f"    总键数: {len(lora_weights)}")
            
#             # 检查专家相关权重
#             expert_keys = {}
#             for key in lora_weights.keys():
#                 if 'lora_A' in key or 'lora_B' in key:
#                     # 提取专家 ID
#                     import re
#                     match = re.search(r'expert_(\d+)', key)
#                     if match:
#                         expert_id = match.group(1)
#                         if expert_id not in expert_keys:
#                             expert_keys[expert_id] = []
#                         expert_keys[expert_id].append(key)
            
#             print(f"    发现的专家 ID: {sorted(expert_keys.keys())}")
            
#             # 打印每个专家的权重范数
#             for expert_id in sorted(expert_keys.keys()):
#                 print(f"\n    Expert {expert_id}:")
#                 sample_keys = expert_keys[expert_id][:3]
#                 for key in sample_keys:
#                     weight = lora_weights[key]
#                     print(f"      {key}: shape={weight.shape}, norm={weight.norm().item():.6f}")
            
#             # 检查是否有 task_0 和 task_1 的区别
#             if '0' in expert_keys and '1' in expert_keys:
#                 # 比较 expert_0 和 expert_1 的第一个 lora_A 权重
#                 key0 = [k for k in expert_keys['0'] if 'lora_A' in k][0]
#                 key1 = [k for k in expert_keys['1'] if 'lora_A' in k][0]
#                 diff = (lora_weights[key0] - lora_weights[key1]).norm().item()
#                 print(f"\n    Expert 0 vs Expert 1 差异:")
#                 print(f"      {key0} vs {key1}")
#                 print(f"      L2 差异: {diff:.6f}")
                
#                 if diff < 1e-6:
#                     print(f"      ⚠️ 警告：Expert 0 和 Expert 1 的权重几乎相同！")
#             # =====================================
            
#             # 加载到模型
#             missing, unexpected = lora_target.load_state_dict(lora_weights, strict=False)
#             print(f"\n    ✅ LoRA 加载完成 (missing: {len(missing)}, unexpected: {len(unexpected)})")
    
#     print(f"✅ Checkpoint 加载完成")
#     return model