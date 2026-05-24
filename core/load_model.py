import torch
import transformers
from typing import Any, Tuple, Optional
import os
import sys
import importlib
from backbone.shared.model_loading import (
    setup_quantization,
    load_pretrained_model,
    load_tokenizer,
    setup_tokenizer,
    load_clip_tokenizer,
    initialize_multimodal_modules,
    apply_mm_projector_trainability,
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
from config.backbone.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from backbone.llava.model import *
import warnings

from utils.rank import is_local_main_process


def _train_log(*args, **kwargs) -> None:
    """Log only on local main process (LOCAL_RANK 0 or unset)."""
    if is_local_main_process():
        print(*args, **kwargs)


def _resolve_train_cuda_device(training_args) -> torch.device:
    """Map this process to ``cuda:{local_rank}``; single-process launch uses ``cuda``."""
    rank = getattr(training_args, "local_rank", None)
    if rank is None:
        env = os.environ.get("LOCAL_RANK", "-1")
        try:
            rank = int(env)
        except ValueError:
            rank = -1
    else:
        rank = int(rank)
    if rank < 0:
        return torch.device("cuda")
    return torch.device(f"cuda:{rank}")


def _move_model_to_train_device(model: Any, training_args) -> Any:
    """Place the full model on this rank's GPU (avoid 4 ranks stacking on physical GPU0)."""
    device = _resolve_train_cuda_device(training_args)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _train_log(
        f"Moving model to {device} (training_args.local_rank={getattr(training_args, 'local_rank', None)})"
    )
    return model.to(device)


def _cli_sets_training_flag(flag: str) -> bool:
    """True if child ``sys.argv`` contains ``--flag`` or ``--flag=...`` (preserve CLI overrides)."""
    prefix = f"{flag}="
    for a in sys.argv:
        if a == flag or a.startswith(prefix):
            return True
    return False


# Deferred imports for CLModel / CLIntegration (avoid circular imports)
def _try_import_cl_components():
    """Import CLModel / CLIntegration lazily."""
    try:
        from method.base.cl_model import CLModel
        from method.base.integration import CLIntegration
        return CLModel, CLIntegration
    except ImportError as e:
        _train_log(f"CL component import failed: {e}")
        return None, None


def load_from_checkpoint(model, checkpoint_path, merge_lora=False, for_incremental_training=False):
    """Load weights from a checkpoint directory."""
    import json

    _train_log(f"Loading checkpoint from {checkpoint_path}...")
    _train_log(f"  for_incremental_training: {for_incremental_training}")

    # Locate PEFT target
    lora_target = model
    if hasattr(model, '_base_model'):
        _train_log("  Detected CLModel wrapper")
        lora_target = model._base_model

    _train_log(f"  LoRA target model type: {type(lora_target)}")
    _train_log(f"  has load_adapter: {hasattr(lora_target, 'load_adapter')}")

    # non-LoRA weights
    non_lora_path = os.path.join(checkpoint_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        _train_log("  Loading non-LoRA weights...")
        non_lora_weights = torch.load(non_lora_path, map_location='cpu')

        target = lora_target
        if hasattr(target, 'base_model'):
            target = target.base_model
        if hasattr(target, 'model'):
            target = target.model

        target.load_state_dict(non_lora_weights, strict=False)
        _train_log("    non-LoRA weights loaded")

    # LoRA weights
    if for_incremental_training:
        if hasattr(lora_target, 'load_adapter'):
            if os.path.isdir(checkpoint_path):
                _train_log("\n  Loading LoRA adapter (incremental training mode)...")
                _train_log(f"      checkpoint_path: {checkpoint_path}")
                lora_target.load_adapter(checkpoint_path, adapter_name='default')
                _train_log("    LoRA adapter loaded")
        else:
            _train_log("  Model has no load_adapter method")

    # Inference: load_adapter
    else:
        _train_log("\n  Inference mode: trying load_adapter...")
        if hasattr(lora_target, 'load_adapter'):
            if os.path.isdir(checkpoint_path):
                _train_log(f"      load_adapter({checkpoint_path}, adapter_name='default')")

                config_path = os.path.join(checkpoint_path, 'adapter_config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    _train_log(f"      adapter_config peft_type: {config.get('peft_type', 'unknown')}")

                lora_target.load_adapter(checkpoint_path, adapter_name='default')
                _train_log("    load_adapter finished")
            else:
                _train_log("    Checkpoint is not a directory, falling back to manual LoRA load...")
                _manual_load_lora(lora_target, checkpoint_path)
        else:
            _train_log("    Model has no load_adapter, loading LoRA manually...")
            _manual_load_lora(lora_target, checkpoint_path)

    # Optional expert LoRA norm diagnostics
    expert_norms = {}
    for name, param in lora_target.named_parameters():
        if 'lora' in name.lower():
            import re
            match = re.search(r'loraA\.(\d+)', name)
            if match:
                expert_id = match.group(1)
                if expert_id not in expert_norms:
                    expert_norms[expert_id] = []
                expert_norms[expert_id].append(param.norm().item())

    for expert_id in sorted(expert_norms.keys())[:4]:
        if expert_norms[expert_id]:
            avg_norm = sum(expert_norms[expert_id]) / len(expert_norms[expert_id])
            _train_log(f"    Expert {expert_id}: avg_norm={avg_norm:.6f} (samples={len(expert_norms[expert_id])})")

    if '0' in expert_norms and '1' in expert_norms:
        avg0 = sum(expert_norms['0']) / len(expert_norms['0'])
        avg1 = sum(expert_norms['1']) / len(expert_norms['1'])
        if abs(avg0 - avg1) < 1e-6:
            _train_log("    Warning: Expert 0 and Expert 1 weights appear identical")
        else:
            _train_log(f"    Expert 0 vs 1 avg norm: {avg0:.6f} vs {avg1:.6f}")

    _train_log("\nCheckpoint load finished")
    return model


def _manual_load_lora(lora_target, checkpoint_path):
    """Load LoRA weights without HuggingFace ``load_adapter``."""
    lora_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
    if not os.path.exists(lora_path):
        lora_path = os.path.join(checkpoint_path, 'adapter_model.bin')

    if os.path.exists(lora_path):
        _train_log(f"      Manual load: {lora_path}")

        if lora_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            lora_weights = load_file(lora_path)
        else:
            lora_weights = torch.load(lora_path, map_location='cpu')

        model_keys = set(lora_target.state_dict().keys())

        remapped_weights = {}
        for key, value in lora_weights.items():
            new_key = key
            if '.lora_A.loraA' in key and '.default.' not in key:
                new_key = key.replace('.lora_A.loraA', '.lora_A.default.loraA')
            elif '.lora_B.loraB' in key and '.default.' not in key:
                new_key = key.replace('.lora_B.loraB', '.lora_B.default.loraB')
            remapped_weights[new_key] = value

        matched = set(remapped_weights.keys()) & model_keys
        _train_log(f"      Keys matched after remap: {len(matched)}/{len(remapped_weights)}")

        if len(matched) > 0:
            missing, unexpected = lora_target.load_state_dict(remapped_weights, strict=False)
            _train_log("      Manual LoRA load finished")
        else:
            _train_log("      Failed to map LoRA weights to model keys")
    else:
        _train_log("      LoRA weight file not found")


# common/load_model.py
def load_model_for_train(model_args, data_args, training_args):
    """Load model for continual-learning training."""
    if str(getattr(model_args, "method", "") or "").strip().lower() == "zeroshot":
        raise ValueError(
            "method='zeroshot' is inference-only (no continual learning training). "
            "Use eval/infer with --method zeroshot, or set method to a CL method for train."
        )
    merge_method_config_into(model_args)
    merge_benchmark_task_num_into(model_args)
    # Copy METHOD_CONFIG lora_* into training_args unless CLI overrides.
    if getattr(training_args, "lora_enable", False):
        _lora_flag = {
            "lora_r": "--lora_r",
            "lora_alpha": "--lora_alpha",
            "lora_dropout": "--lora_dropout",
        }
        for name, flag in _lora_flag.items():
            if _cli_sets_training_flag(flag):
                continue
            if hasattr(model_args, name):
                v = getattr(model_args, name)
                if v is not None:
                    setattr(training_args, name, v)
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

    # Multimodal (before CL wrapper)
    if model_args.vision_tower is not None:
        _train_log("Initializing multimodal modules...")
        model, data_args = initialize_multimodal_modules(model, model_args, training_args, data_args, tokenizer)
        _train_log("Multimodal modules initialized.\n")

    model = adjust_precision(model, training_args)

    if model_args.text_tower is not None:
        clip_tokenizer = load_clip_tokenizer(model_args.text_tower, training_args)
        model.set_clip_tokenizer(clip_tokenizer)

    model.set_tokenizer(tokenizer)
    if hasattr(model, 'set_cur_task'):
        model.set_cur_task(model_args.cur_task, model_args.task_num)

    # CL wrapper + method integration
    method_name = getattr(model_args, 'method', 'base')
    
    if method_name != 'base':
        _train_log(f"\n{'='*70}")
        _train_log(f"Continual learning method: {method_name}")
        if str(method_name).lower() == "zeroshot":
            _train_log("Zeroshot: plain LLaVA, no PEFT injection / no method-side extras")
        else:
            _train_log("Method integration will handle PEFT/LoRA injection")
        _train_log(f"{'='*70}\n")
        
        CLModel, CLIntegration = _try_import_cl_components()
        if CLModel is None:
            raise ImportError("Failed to import CL components (CLModel / CLIntegration)")
        
        module = __import__(f"method.custom.{method_name}.integration", fromlist=[''])
        IntegrationClass = getattr(module, f"{method_name.capitalize()}Integration")
        integration = IntegrationClass(model_args)

        model = CLModel(model, integration)
        _train_log(f"CLModel wrapper ready | method: {method_name}\n")
        
        # Resume from previous task after PEFT is injected
        prev_path = getattr(model_args, "previous_task_model_path", None)
        if prev_path is not None:
            prev_path = os.path.expanduser(str(prev_path).strip())
            tried_paths = [prev_path]
            # Benchmark configs often say TaskN_llava_lora while train saves TaskN_llava.
            if not os.path.exists(prev_path) and prev_path.endswith("_llava_lora"):
                alt = prev_path[: -len("_lora")]
                tried_paths.append(alt)
                if os.path.exists(alt):
                    _train_log(
                        f"[train] previous_task_model_path {prev_path!r} not found; "
                        f"using {alt!r} (_llava_lora vs _llava naming).\n"
                    )
                    prev_path = alt
                    model_args.previous_task_model_path = alt
            if not os.path.exists(prev_path):
                raise FileNotFoundError(
                    "[train] previous_task_model_path not found (incremental load required). Tried:\n  "
                    + "\n  ".join(repr(p) for p in tried_paths)
                    + "\nFix the path or train the missing previous task before continuing."
                )

        if model_args.previous_task_model_path is not None and os.path.exists(
            model_args.previous_task_model_path
        ):
            _train_log(f"Loading previous-task checkpoint: {model_args.previous_task_model_path}")
            model = load_from_checkpoint(
                model,
                model_args.previous_task_model_path,
                merge_lora=False,
                for_incremental_training=True
            )
            _train_log("Checkpoint load finished.\n")

            if hasattr(model, '_integration'):
                integration = model._integration
                if hasattr(integration, 'load_extra_state'):
                    _train_log("Loading method extra state from previous checkpoint...")
                    ok = integration.load_extra_state(model_args.previous_task_model_path, model=model)
                    if ok:
                        _train_log("Method extra state restored")
                    elif str(method_name).lower() == "same":
                        raise RuntimeError(
                            "SAME incremental training requires carry-over state "
                            "(same_state.bin or prism.same.* keys in adapter_model.safetensors) "
                            f"under {model_args.previous_task_model_path}; load_extra_state returned False."
                        )
                    else:
                        _train_log("Method extra state not found or failed to load (continuing training)")
    else:
        assert(0)
    if hasattr(model, "_integration"):
        _prep = getattr(model._integration, "prepare_training_data", None)
        if callable(_prep):
            _prep(data_args, model_args, training_args)

    apply_mm_projector_trainability(model, training_args)

    model = _move_model_to_train_device(model, training_args)
    model.train()
    
    return model, tokenizer, data_args


def _checkpoint_dir_has_peft_adapter(model_path: str) -> bool:
    return bool(model_path) and os.path.isdir(model_path) and os.path.isfile(
        os.path.join(model_path, "adapter_config.json")
    )


def _checkpoint_has_merged_llava_weights(model_path: str) -> bool:
    """True if directory has merged LLaVA weights (no adapter_config.json)."""
    if not model_path or not os.path.isdir(model_path):
        return False
    return os.path.isfile(os.path.join(model_path, "model.safetensors")) or os.path.isfile(
        os.path.join(model_path, "pytorch_model.bin")
    )


def _load_merged_llava_weights_into_model(model: Any, checkpoint_path: str) -> None:
    """Load ``model.safetensors`` / ``pytorch_model.bin`` into inner LLaVA (``_base_model``)."""
    inner = model._base_model if hasattr(model, "_base_model") else model
    st_path = os.path.join(checkpoint_path, "model.safetensors")
    pt_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.isfile(st_path):
        from safetensors.torch import load_file

        weights: dict = dict(load_file(st_path))
        src = st_path
    elif os.path.isfile(pt_path):
        weights = torch.load(pt_path, map_location="cpu")
        src = pt_path
    else:
        raise FileNotFoundError(
            f"Merged LLaVA checkpoint not found under {checkpoint_path} "
            "(expected model.safetensors or pytorch_model.bin)"
        )
    dtype = next(inner.parameters()).dtype
    weights = {k: v.to(dtype) if torch.is_tensor(v) else v for k, v in weights.items()}
    bad = inner.load_state_dict(weights, strict=False)
    miss = getattr(bad, "missing_keys", None) or []
    unexp = getattr(bad, "unexpected_keys", None) or []
    _train_log(f"  Merged LLaVA weights from {src} | missing_keys={len(miss)} unexpected_keys={len(unexp)}")


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
    """Load model for inference."""
    kwargs = {"device_map": device_map, **kwargs}
    kwargs.pop("task_num", None)
    kwargs.pop("expert_num", None)
    kwargs.pop("benchmark", None)
    same_print_router = kwargs.pop("same_print_router", None)
    same_print_router_max = kwargs.pop("same_print_router_max", None)

    if device != "cuda":
        kwargs['device_map'] = {"": device}
    elif not load_8bit and not load_4bit and kwargs.get("device_map") == "auto":
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
        _zs = str(method or "").strip().lower() == "zeroshot"
        if _zs:
            if not model_base:
                raise ValueError(
                    "method=zeroshot requires model_base (LLaVA weight directory); no CL checkpoint/adapter."
                )
            model_path = os.path.expanduser(str(model_base).strip())
            print("Inference: zeroshot — using model_base weights only (no CL checkpoint).", flush=True)

        use_peft_adapter_layout = model_base is not None and (
            _checkpoint_dir_has_peft_adapter(model_path)
            or "lora" in model_name.lower()
            or (
                str(method).lower() == "hide_llava"
                and os.path.isdir(model_path or "")
                and os.path.isfile(os.path.join(model_path, "hide_state.pt"))
                and _checkpoint_has_merged_llava_weights(model_path)
            )
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
                        same_print_router=bool(same_print_router),
                        same_print_router_max=(
                            int(same_print_router_max)
                            if same_print_router_max is not None
                            else 10_000
                        ),
                    )
                    merge_method_config_into(pseudo_args, method=method)
                    merge_benchmark_task_num_into(pseudo_args, benchmark=benchmark)
                    _is_zeroshot = str(method or "").strip().lower() == "zeroshot"
                    if getattr(pseudo_args, "task_num", None) is None and not _is_zeroshot:
                        raise ValueError(
                            "CL inference requires task_num: pass benchmark (e.g. run.py infer sets --benchmark) "
                            "or set task_num explicitly in load_model_for_inference(..., task_num=..., benchmark=...)."
                        )
                    
                    integration = IntegrationClass(pseudo_args)
                    
                    model = CLModel(model, integration)
                    print(f"Inference model wrapped with CLModel | method: {method}")
                    
                    if os.path.exists(model_path):
                        print("Loading weights from checkpoint...")
                        if _checkpoint_dir_has_peft_adapter(model_path):
                            model = load_from_checkpoint(
                                model,
                                model_path,
                                merge_lora=False,
                                for_incremental_training=False,
                            )
                        elif _checkpoint_has_merged_llava_weights(model_path):
                            _load_merged_llava_weights_into_model(model, model_path)
                        else:
                            model = load_from_checkpoint(
                                model,
                                model_path,
                                merge_lora=False,
                                for_incremental_training=False,
                            )
                        print("Checkpoint load finished")
                    
                    if hasattr(integration, 'load_extra_state'):
                        print("Loading extra state...")
                        success = integration.load_extra_state(model_path, model=model)
                        if success:
                            print(f"Extra state loaded | model has image_anchors: {hasattr(model, 'image_anchors')}")
            else:
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
            print('Loading LLaVA from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_path = os.path.join(model_path, "mm_projector.bin")
            if os.path.isfile(mm_path):
                mm_projector_weights = torch.load(mm_path, map_location="cpu")
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            elif _checkpoint_has_merged_llava_weights(model_path):
                _load_merged_llava_weights_into_model(model, model_path)
            else:
                print(
                    f"Warning: no mm_projector.bin under {model_path}; "
                    "multimodal projector may match base only."
                )
            model.set_tokenizer(tokenizer)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            model.set_tokenizer(tokenizer)
        
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
        integ = getattr(model, "_integration", None)
        if integ is not None:
            from method.custom.specialized_integration import RouterIntegration

            if isinstance(integ, RouterIntegration):
                env_on = os.getenv("SAME_PRINT_ROUTER", "").strip().lower() in ("1", "true", "yes", "on")
                integ._router_mix_log_enabled = (
                    bool(integ._router_mix_log_enabled) or env_on or bool(same_print_router)
                )
                if bool(same_print_router) and same_print_router_max is not None:
                    integ._router_mix_log_max = int(same_print_router_max)
                elif os.getenv("SAME_PRINT_ROUTER_MAX"):
                    integ._router_mix_log_max = int(os.getenv("SAME_PRINT_ROUTER_MAX", "10000"))
                integ._router_mix_log_count = 0
                if integ._router_mix_log_enabled:
                    print(
                        f"[infer] RouterIntegration mixture logging enabled "
                        f"(max_lines={integ._router_mix_log_max}; --same-print-router or SAME_PRINT_ROUTER=1)",
                        flush=True,
                    )
        model.eval()

    return tokenizer, model, image_processor, context_len