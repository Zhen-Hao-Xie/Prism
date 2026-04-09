import os
import torch
from PEFT.peft import WEIGHTS_NAME, set_peft_model_state_dict
def _load_cl_extra_state(model, checkpoint_path: str, for_incremental_training: bool = False):
    """
    加载 CL 特定状态（通过通用接口调用）
    
    不关心具体方法，只负责调用 integration.load_extra_state()
    """
    loaded = False
    
    # 优先级 1: CLModel 包装后的模型
    if hasattr(model, 'integration') and model.integration is not None:
        try:
            if hasattr(model.integration, 'load_extra_state'):
                success = model.integration.load_extra_state(checkpoint_path)
                if success:
                    print(f"✅ CL 特定状态已通过 integration 加载")
                    loaded = True
        except Exception as e:
            print(f"⚠️ 通过 integration 加载 CL 状态失败：{e}")
    
    # 优先级 2: 直接挂载的 integration
    if not loaded:
        for attr_name in ['hide_integration', 'sp_integration', 'ranpac_integration']:
            if hasattr(model, attr_name):
                integration = getattr(model, attr_name)
                try:
                    if hasattr(integration, 'load_extra_state'):
                        success = integration.load_extra_state(checkpoint_path)
                        if success:
                            print(f"✅ CL 特定状态已通过 {attr_name} 加载")
                            loaded = True
                            break
                except Exception as e:
                    print(f"⚠️ 通过 {attr_name} 加载 CL 状态失败：{e}")

def load_from_checkpoint(model, checkpoint_path, merge_lora=False, for_incremental_training=False):
    """从 checkpoint 加载模型权重"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    import os
    import torch
    
    # ========== 找到正确的底层模型 ==========
    target_model = model
    
    if hasattr(model, '_base_model'):
        print(f"  🔍 检测到 CLModel 包装")
        target_model = model._base_model
        
        if hasattr(target_model, 'base_model'):
            print(f"  🔍 检测到内部还有 PEFT 包装")
            target_model = target_model.base_model
    
    elif hasattr(model, 'base_model'):
        print(f"  🔍 检测到 PEFT 包装")
        target_model = model.base_model
        
        if hasattr(target_model, 'model'):
            target_model = target_model.model
    
    else:
        if hasattr(model, 'model'):
            target_model = model.model
    
    print(f"  ✅ 目标模型类型：{type(target_model)}")
    
    # ========== 加载 non-LoRA 权重 ==========
    non_lora_path = os.path.join(checkpoint_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        print(f"  Loaded non-LoRA weights from {non_lora_path}")
        non_lora_weights = torch.load(non_lora_path, map_location='cpu')
        
        if hasattr(target_model, 'model'):
            target_model.model.load_state_dict(non_lora_weights, strict=False)
        else:
            target_model.load_state_dict(non_lora_weights, strict=False)
    
    # ========== 加载 LoRA 权重（关键修复）==========
    if for_incremental_training:
        # 检查是否是 PEFT 模型
        lora_target = model
        if hasattr(model, '_base_model'):
            lora_target = model._base_model
        
        # 关键修复：load_adapter 期望目录路径，不是文件路径
        if hasattr(lora_target, 'load_adapter'):
            # PEFT 模型：使用 load_adapter，传入目录路径（不是文件路径）
            if os.path.exists(checkpoint_path):
                print(f"  Loaded LoRA adapter from {checkpoint_path}")
                lora_target.load_adapter(checkpoint_path, adapter_name='default')  # ← 传入目录
            else:
                print(f"  ⚠️  Checkpoint 目录不存在：{checkpoint_path}")
        else:
            # 非 PEFT 模型：直接加载状态字典
            lora_path = os.path.join(checkpoint_path, 'adapter_model.bin')
            if os.path.exists(lora_path):
                print(f"  Loaded LoRA weights from {lora_path}")
                lora_weights = torch.load(lora_path, map_location='cpu')
                lora_target.load_state_dict(lora_weights, strict=False)
    
    print(f"✅ Checkpoint 加载完成")
    return model