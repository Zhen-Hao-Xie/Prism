import torch
import transformers
from typing import Tuple, Optional
import os
import importlib  # ← 在文件顶部添加
from .load_backbone import (
    setup_quantization,
    load_pretrained_model,
    load_tokenizer,
    setup_tokenizer,
    load_clip_tokenizer,
    initialize_multimodal_modules,
    adjust_precision,
    prepare_model_for_kbit,
    setup_gradient_checkpointing,
)
from .load_checkpoint import load_from_checkpoint
from .load_config import ModelArguments, DataArguments, TrainingArguments
from config.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from backbone.llava.model import *
import warnings


# [新增] 延迟导入函数：避免循环依赖，仅在需要时加载 CL 组件
def _try_import_cl_components():
    """延迟导入 CL 组件，避免循环依赖"""
    try:
        from method.base.cl_model import CLModel
        from method.base.integration import CLIntegration
        return CLModel, CLIntegration
    except ImportError:
        return None, None
# common/load_model.py
def load_model_for_train(model_args, data_args, training_args):
    """加载用于训练的模型"""
    
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    bnb_args = setup_quantization(training_args, compute_dtype)

    has_vision = model_args.vision_tower is not None
    model = load_pretrained_model(model_args.model_name_or_path, training_args, bnb_args, has_vision)

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    model = prepare_model_for_kbit(model, training_args, compute_dtype)
    model = setup_gradient_checkpointing(model, training_args)

    tokenizer = load_tokenizer(model_args.model_name_or_path, training_args)
    setup_tokenizer(tokenizer, model, model_args.version)

    # ========== 修改 1: 多模态初始化（在 CL 包装之前）==========
    if model_args.vision_tower is not None:
        print(f"🔧 初始化多模态模块...")
        model, data_args = initialize_multimodal_modules(model, model_args, training_args, data_args, tokenizer)
        print(f"✅ 多模态模块初始化完成\n")

    model = adjust_precision(model, training_args)

    if model_args.text_tower is not None:
        clip_tokenizer = load_clip_tokenizer(model_args.text_tower, training_args)
        model.set_clip_tokenizer(clip_tokenizer)

    model.set_tokenizer(tokenizer)
    if hasattr(model, 'set_cur_task'):
        model.set_cur_task(model_args.cur_task, model_args.expert_num)

    # ========== 修改 2: CL 包装（包含 HiDe LoRA 注入）==========
    method_name = getattr(model_args, 'method', 'base')
    
    if method_name != 'base':
        print(f"\n{'='*70}")
        print(f"🚀 检测到持续学习方法：{method_name}")
        print(f"📌 HiDe 将自行处理 LoRA 注入")
        print(f"{'='*70}\n")
        
        CLModel, CLIntegration = _try_import_cl_components()
        if CLModel is None:
            raise ImportError(f"无法导入 CL 组件")
        
        module = __import__(f"method.{method_name}.integration", fromlist=[''])
        IntegrationClass = getattr(module, f"{method_name.capitalize()}Integration")
        integration = IntegrationClass(model_args)
        
        # 包装模型
        model = CLModel(model, integration)
        print(f"✅ CLModel 包装完成 | 方法：{method_name}\n")
        
        # ========== 修改 3: 加载 checkpoint（在包装后）==========
        # 这样 HiDe 的 LoRA 已经注入，checkpoint 可以正确加载
        if model_args.previous_task_model_path is not None and os.path.exists(model_args.previous_task_model_path):
            print(f"📦 加载前一任务 checkpoint: {model_args.previous_task_model_path}")
            model = load_from_checkpoint(
                model,
                model_args.previous_task_model_path,
                merge_lora=False,
                for_incremental_training=True
            )
            print(f"✅ Checkpoint 加载完成\n")
            
            # 加载 HiDe 状态（anchors 等）
            if hasattr(model, '_integration'):
                integration = model._integration
                if hasattr(integration, 'load_extra_state'):
                    print(f"🔧 加载之前任务的 HiDe 状态...")
                    integration.load_extra_state(model_args.previous_task_model_path, model=model)
                    print(f"✅ 之前任务的 anchors 已恢复")
    else:
        assert(0)
    # =======================================================

    model = model.cuda()
    model.train()
    
    return model, tokenizer, data_args

def _try_import_cl_components():
    """延迟导入 CL 组件，避免循环依赖"""
    try:
        from method.base.cl_model import CLModel
        from method.base.integration import CLIntegration
        return CLModel, CLIntegration
    except ImportError as e:
        print(f"⚠️  CL 组件导入失败：{e}")
        return None, None
    
import importlib  # ← 在文件顶部添加

# common/load_model.py
def load_model_for_inference(
    model_path, 
    model_base, 
    model_name, 
    load_8bit=False, 
    load_4bit=False, 
    device_map="auto", 
    device="cuda", 
    text_tower=None, 
    method: Optional[str] = None,
    **kwargs
):
    """加载用于推理的模型"""
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided.')
        
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if text_tower:
                setattr(lora_cfg_pretrained, 'mm_text_tower', text_tower)
                if not hasattr(lora_cfg_pretrained, 'mm_text_select_layer'):
                    setattr(lora_cfg_pretrained, 'mm_text_select_layer', -1)

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            
            if not text_tower:
                raise ValueError('text_tower must be provided for HiDe routing')

            clip_tokenizer = AutoTokenizer.from_pretrained(
                text_tower,
                cache_dir=None,
                model_max_length=77,
                padding_side="right",
                use_fast=True,
            )

            model.set_clip_tokenizer(clip_tokenizer)
            model.set_tokenizer(tokenizer)
            
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            if method is not None and method != 'base':
                print(f"🚀 推理模式：应用持续学习方法 {method}...")
                CLModel, CLIntegration = _try_import_cl_components()
                if CLModel is not None:
                    module = __import__(f"method.{method}.integration", fromlist=[''])
                    IntegrationClass = getattr(module, f"{method.capitalize()}Integration")
                    
                    class SimpleArgs:
                        def __init__(self, **kw):
                            for k, v in kw.items():
                                setattr(self, k, v)
                    pseudo_args = SimpleArgs(
                        method=method, 
                        cur_task=kwargs.get('cur_task', 0),
                        expert_num=kwargs.get('expert_num', 8),
                        clip_feature_dim=kwargs.get('clip_feature_dim', 768),
                    )
                    
                    integration = IntegrationClass(pseudo_args)
                    
                    # 先包装为 CLModel
                    model = CLModel(model, integration)
                    print(f"✅ 推理模型已包装为 CLModel | 方法：{method}")
                    
                    # ========== 关键修复：使用 load_from_checkpoint 统一加载 ==========
                    if os.path.exists(model_path):
                        print(f"📦 从 checkpoint 加载模型状态...")
                        model = load_from_checkpoint(
                            model,
                            model_path,
                            merge_lora=False,  # ← 不要 merge，保持 PEFT 结构
                            for_incremental_training=False  # ← 推理时不需要增量训练
                        )
                        print(f"✅ Checkpoint 加载完成")
                    
                    # 加载 HiDe 状态（anchors 等）
                    if hasattr(integration, 'load_extra_state'):
                        print(f"🔧 从 hide_state.pt 加载 HiDe 状态...")
                        success = integration.load_extra_state(model_path, model=model)
                        if success:
                            print(f"✅ HiDe 状态已加载 | anchors 可用：{hasattr(model, 'image_anchors')}")
                    
                    # 设置 predicted_task_id
                    predicted_task_id = getattr(pseudo_args, 'cur_task', 0)
                    count = 0
                    for module in model.modules():
                        if module.__class__.__name__ == 'HiDeMOELoraLinear':
                            if hasattr(module, 'predicted_task_id'):
                                module.predicted_task_id = predicted_task_id
                                count += 1
                    if count > 0:
                        print(f"✅ predicted_task_id={predicted_task_id} 已传播到 {count} 个专家层")
            else:
                # 非 CL 方法：手动加载（原有逻辑）
                print('Loading additional LLaVA weights...')
                if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                    if any(k.startswith('model.model.') for k in non_lora_trainables):
                        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                    model.load_state_dict(non_lora_trainables, strict=False)

                from PEFT.peft import PeftModel
                print('Loading LoRA weights...')
                model = PeftModel.from_pretrained(model, model_path)
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
                print('Model is loaded...')
        
        elif model_base is not None:
            # 非 LoRA 模型
            print('Loading LLaVA from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        
        # 加载 vision tower 和 text tower
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

        text_tower_model = model.get_text_tower()
        if text_tower_model is not None:
            if not getattr(text_tower_model, 'is_loaded', True):
                text_tower_model.load_model()
            text_tower_model.to(device=device, dtype=torch.float16)
    else:
        warnings.warn('The model is not llava')
        assert(0)
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len