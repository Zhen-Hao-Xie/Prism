import os
import torch
import logging
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer, output_dir):
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

# common/save_checkpoint.py
def _save_cl_extra_state(model, output_dir: str):
    """
    保存 CL 特定状态（通过通用接口调用）
    
    不关心具体方法，只负责调用 integration.save_extra_state()
    """
    saved = False
    
    # ========== 调试信息 ==========
    print(f"🔍 调试信息 - model 类型：{type(model)}")
    print(f"🔍 调试信息 - hasattr _integration: {hasattr(model, '_integration')}")
    print(f"🔍 调试信息 - hasattr integration: {hasattr(model, 'integration')}")
    # ===========================
    
    # 优先级 1: CLModel 包装后的模型（有 _integration 属性）← 关键修复：带下划线
    if hasattr(model, '_integration') and model._integration is not None:
        print(f"🔍 找到 _integration 属性")
        try:
            if hasattr(model._integration, 'save_extra_state'):
                success = model._integration.save_extra_state(output_dir, model=model)
                if success:
                    print(f"✅ CL 特定状态已通过 _integration 保存")
                    saved = True
            else:
                print(f"⚠️  _integration 没有 save_extra_state 方法")
        except Exception as e:
            print(f"⚠️  通过 _integration 保存 CL 状态失败：{e}")
            import traceback
            traceback.print_exc()
    
    # 优先级 2: 直接挂载的 integration（兼容旧代码）
    if not saved:
        for attr_name in ['hide_integration', 'sp_integration', 'ranpac_integration', 'integration']:
            if hasattr(model, attr_name):
                integration = getattr(model, attr_name)
                try:
                    if hasattr(integration, 'save_extra_state'):
                        success = integration.save_extra_state(output_dir, model=model)
                        if success:
                            print(f"✅ CL 特定状态已通过 {attr_name} 保存")
                            saved = True
                            break
                except Exception as e:
                    print(f"⚠️  通过 {attr_name} 保存 CL 状态失败：{e}")
    
    if not saved:
        print(f"⚠️  CL 特定状态未保存（未找到 integration）")
    
    
# common/save_checkpoint.py
def save_model(model, training_args, trainer=None, save_extra_state: bool = True):
    """统一的模型保存函数"""
    
    print(f"\n{'='*70}")
    print(f"📦 开始保存模型 | 输出目录：{training_args.output_dir}")
    print(f"{'='*70}\n")
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # ========== 找到真正的 PEFT 模型 ==========
    save_model = model
    if hasattr(model, '_base_model'):
        _base_model = getattr(model, '_base_model')
        if hasattr(_base_model, 'peft_config') or hasattr(_base_model, 'adapter_model'):
            save_model = _base_model
            print(f"🔍 检测到 CLModel 包装，使用 _base_model 保存")
        elif hasattr(_base_model, 'base_model'):
            if hasattr(_base_model.base_model, 'peft_config'):
                save_model = _base_model
                print(f"🔍 检测到嵌套包装，使用 _base_model 保存")
    
    # ========== 调试：检查参数量 ==========
    total_params = sum(p.numel() for p in save_model.parameters())
    trainable_params = sum(p.numel() for p in save_model.parameters() if p.requires_grad)
    print(f"🔍 保存前参数量检查:")
    print(f"  总参数量：{total_params:,}")
    print(f"  可训练参数量：{trainable_params:,}")
    print(f"  可训练比例：{trainable_params / total_params * 100:.4f}%")
    # ===================================
    
    if training_args.lora_enable:
        print("🔧 LoRA 模式：保存 LoRA 权重 + 非 LoRA 权重")
        
        # ========== 关键修复：使用 save_model.named_parameters() ==========
        named_params = list(save_model.named_parameters())
        print(f"🔍 收集到 {len(named_params)} 个参数")
        
        state_dict = get_peft_state_maybe_zero_3(
            named_params, 
            training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            named_params,
            require_grad_only=True
        )
        
        print(f"🔍 提取后:")
        print(f"  LoRA 参数量：{len(state_dict)}")
        print(f"  非 LoRA 参数量：{len(non_lora_state_dict)}")
        # ===================================
        
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            save_model.config.save_pretrained(training_args.output_dir)
            
            if hasattr(save_model, 'save_pretrained'):
                save_model.save_pretrained(
                    training_args.output_dir, 
                    state_dict=state_dict if state_dict else None,
                    safe_serialization=True
                )
                print(f"✅ LoRA 适配器已保存")
            elif state_dict:
                torch.save(
                    state_dict, 
                    os.path.join(training_args.output_dir, 'adapter_model.bin')
                )
                lora_params = sum(p.numel() for p in state_dict.values())
                print(f"✅ LoRA 权重已保存 | 参数量：{lora_params:,}")
            
            if non_lora_state_dict:
                torch.save(
                    non_lora_state_dict, 
                    os.path.join(training_args.output_dir, 'non_lora_trainables.bin')
                )
                non_lora_params = sum(p.numel() for p in non_lora_state_dict.values())
                print(f"✅ 非 LoRA 权重已保存 | 参数量：{non_lora_params:,}")
    else:
        print("🔧 全量微调模式：使用 HF Trainer 保存")
        if trainer is not None:
            safe_save_model_for_hf_trainer(trainer, training_args.output_dir)
        else:
            if hasattr(model, '_base_model'):
                _base_model = getattr(model, '_base_model')
                if hasattr(_base_model, 'save_pretrained'):
                    _base_model.save_pretrained(training_args.output_dir)
                else:
                    model.save_pretrained(training_args.output_dir)
            else:
                model.save_pretrained(training_args.output_dir)
    
    if save_extra_state:
        _save_cl_extra_state(model, training_args.output_dir)
    
    print(f"\n{'='*70}")
    print(f"✅ 模型保存完成")
    print(f"{'='*70}\n")
