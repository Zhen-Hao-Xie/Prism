import torch
import transformers
from typing import Tuple, Optional
import os
import importlib  # ← 在文件顶部添加
from backbone.shared.model_loading import (
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
from .config_loader import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    merge_benchmark_task_num_into,
    merge_method_config_into,
)
from config.backbone.llava import CLIP_FEATURE_DIM
from config.backbones.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
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


def load_from_checkpoint(model, checkpoint_path, merge_lora=False, for_incremental_training=False):
    """从 checkpoint 加载模型权重"""
    import json

    print(f"Loading checkpoint from {checkpoint_path}...")
    print(f"  for_incremental_training: {for_incremental_training}")

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


# common/load_model.py
def load_model_for_train(model_args, data_args, training_args):
    """加载用于训练的模型"""
    merge_method_config_into(model_args)
    merge_benchmark_task_num_into(model_args)
    # LoRA 超参以 TrainingArguments（CLI / run.py 覆盖）为准，与 model_args 对齐供 PEFT 方法侧读取
    if getattr(training_args, "lora_enable", False):
        model_args.lora_r = training_args.lora_r
        model_args.lora_alpha = training_args.lora_alpha
        model_args.lora_dropout = training_args.lora_dropout
    _m = str(getattr(model_args, "method", "") or "").lower()
    if _m not in ("", "base", "none", "zeroshot") and getattr(model_args, "task_num", None) is None:
        raise ValueError(
            "Continual learning training requires task_num: pass --benchmark (ucit / coin) "
            "or --task_num explicitly; task counts come from config/benchmarks, not config/methods."
        )

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
        print("Initializing multimodal modules...")
        model, data_args = initialize_multimodal_modules(model, model_args, training_args, data_args, tokenizer)
        print("Multimodal modules initialized.\n")

    model = adjust_precision(model, training_args)

    if model_args.text_tower is not None:
        clip_tokenizer = load_clip_tokenizer(model_args.text_tower, training_args)
        model.set_clip_tokenizer(clip_tokenizer)

    model.set_tokenizer(tokenizer)
    if hasattr(model, 'set_cur_task'):
        model.set_cur_task(model_args.cur_task, model_args.task_num)

    # ========== 修改 2: CL 包装（包含 HiDe LoRA 注入）==========
    method_name = getattr(model_args, 'method', 'base')
    
    if method_name != 'base':
        print(f"\n{'='*70}")
        print(f"Continual learning method: {method_name}")
        if str(method_name).lower() == "zeroshot":
            print("Zeroshot: plain LLaVA, no PEFT injection / no method-side extras")
        else:
            print("Method integration will handle PEFT/LoRA injection")
        print(f"{'='*70}\n")
        
        CLModel, CLIntegration = _try_import_cl_components()
        if CLModel is None:
            raise ImportError("Failed to import CL components (CLModel / CLIntegration)")
        
        module = __import__(f"method.custom.{method_name}.integration", fromlist=[''])
        IntegrationClass = getattr(module, f"{method_name.capitalize()}Integration")
        integration = IntegrationClass(model_args)
        
        # 包装模型
        model = CLModel(model, integration)
        print(f"CLModel wrapper ready | method: {method_name}\n")
        
        # ========== 修改 3: 加载 checkpoint（在包装后）==========
        # 这样 HiDe 的 LoRA 已经注入，checkpoint 可以正确加载
        if model_args.previous_task_model_path is not None and os.path.exists(model_args.previous_task_model_path):
            print(f"Loading previous-task checkpoint: {model_args.previous_task_model_path}")
            model = load_from_checkpoint(
                model,
                model_args.previous_task_model_path,
                merge_lora=False,
                for_incremental_training=True
            )
            print("Checkpoint load finished.\n")
            
            # 加载方法额外状态（如 SAME 的 same_state.bin / HiDe 的 anchors 等）
            if hasattr(model, '_integration'):
                integration = model._integration
                if hasattr(integration, 'load_extra_state'):
                    print("Loading method extra state from previous checkpoint...")
                    ok = integration.load_extra_state(model_args.previous_task_model_path, model=model)
                    if ok:
                        print("Method extra state restored")
                    else:
                        print("Method extra state not found or failed to load (continuing training)")
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
        print(f"CL component import failed: {e}")
        return None, None
    
import importlib  # ← 在文件顶部添加


def _checkpoint_dir_has_peft_adapter(model_path: str) -> bool:
    return bool(model_path) and os.path.isdir(model_path) and os.path.isfile(
        os.path.join(model_path, "adapter_config.json")
    )


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
    benchmark: Optional[str] = None,
    task_num: Optional[int] = None,
    **kwargs
):
    """加载用于推理的模型"""
    kwargs = {"device_map": device_map, **kwargs}
    # 勿传入 HF from_pretrained
    kwargs.pop("task_num", None)
    kwargs.pop("expert_num", None)
    kwargs.pop("benchmark", None)

    if device != "cuda":
        kwargs['device_map'] = {"": device}
    elif not load_8bit and not load_4bit and kwargs.get("device_map") == "auto":
        # device_map="auto" 会在多卡上切分 LLM；随后仅对 vision/text tower 做 .to(device)
        # 与 accelerate 分布不一致时可能触发 CUDA 端段错误。非量化推理与训练一致：整模单卡。
        kwargs["device_map"] = None

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
        use_peft_adapter_layout = model_base is not None and (
            _checkpoint_dir_has_peft_adapter(model_path) or "lora" in model_name.lower()
        )

        if use_peft_adapter_layout:
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
                print(f"Inference: applying continual learning method {method}...")
                CLModel, CLIntegration = _try_import_cl_components()
                if CLModel is not None:
                    module = __import__(f"method.custom.{method}.integration", fromlist=[''])
                    # 若方法提供 PEFT 扩展注册，按需触发（避免 import-time 副作用）
                    if hasattr(module, "ensure_peft_extension_registered"):
                        try:
                            module.ensure_peft_extension_registered()
                        except Exception as e:
                            print(f"PEFT extension registration failed for method {method}: {e}")
                    IntegrationClass = getattr(module, f"{method.capitalize()}Integration")
                    
                    class SimpleArgs:
                        def __init__(self, **kw):
                            for k, v in kw.items():
                                setattr(self, k, v)

                    pseudo_args = SimpleArgs(
                        method=method,
                        cur_task=kwargs.get("cur_task", 0),
                        task_num=task_num,
                        benchmark=benchmark,
                        clip_feature_dim=kwargs.get("clip_feature_dim", CLIP_FEATURE_DIM),
                    )
                    merge_method_config_into(pseudo_args, method=method)
                    merge_benchmark_task_num_into(pseudo_args, benchmark=benchmark)
                    if getattr(pseudo_args, "task_num", None) is None:
                        raise ValueError(
                            "CL inference requires task_num: pass benchmark (e.g. run.py infer sets --benchmark) "
                            "or set task_num explicitly in load_model_for_inference(..., task_num=..., benchmark=...)."
                        )
                    
                    integration = IntegrationClass(pseudo_args)
                    
                    # 先包装为 CLModel
                    model = CLModel(model, integration)
                    print(f"Inference model wrapped with CLModel | method: {method}")
                    
                    if os.path.exists(model_path):
                        print("Loading weights from checkpoint...")
                        model = load_from_checkpoint(
                            model,
                            model_path,
                            merge_lora=False,  # ← 不要 merge，保持 PEFT 结构
                            for_incremental_training=False  # ← 推理时不需要增量训练
                        )
                        print("Checkpoint load finished")
                    
                    # 加载 HiDe 状态（anchors 等）
                    if hasattr(integration, 'load_extra_state'):
                        print("Loading extra state...")
                        success = integration.load_extra_state(model_path, model=model)
                        if success:
                            print(f"Extra state loaded | model has image_anchors: {hasattr(model, 'image_anchors')}")
            else:
                # 非 CL 方法：手动加载（原有逻辑）
                assert(0)
                print('Loading additional LLaVA weights...')
                if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                    if any(k.startswith('model.model.') for k in non_lora_trainables):
                        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                    model.load_state_dict(non_lora_trainables, strict=False)

                from PEFT import PeftModel
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

        if device == "cuda" and not load_8bit and not load_4bit and kwargs.get("device_map") is None:
            model.to(device)
    else:
        warnings.warn('The model is not llava')
        assert(0)
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if isinstance(model, torch.nn.Module):
        model.eval()

    return tokenizer, model, image_processor, context_len