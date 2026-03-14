import torch
import transformers
from typing import Tuple, Optional
import os
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
from .peft_utils import create_lora_config, apply_lora
from .load_checkpoint import load_from_checkpoint
from .load_config import ModelArguments, DataArguments, TrainingArguments
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
import warnings


def load_model_for_train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Tuple[torch.nn.Module, transformers.PreTrainedTokenizer, DataArguments]:
    """
    加载用于训练的模型。
    """
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

    # 训练模式：如果需要 LoRA，则添加
    if training_args.lora_enable:
        lora_config = create_lora_config(training_args, model_args, model)
        model = apply_lora(model, lora_config, training_args)

    # 如果是增量训练，加载之前任务的 checkpoint
    if model_args.previous_task_model_path is not None and os.path.exists(model_args.previous_task_model_path):
        model = load_from_checkpoint(
            model,
            model_args.previous_task_model_path,
            merge_lora=False,                # 训练时不合并
            for_incremental_training=True    # 标记为增量训练，以正确加载权重
        )

    if model_args.vision_tower is not None:
        model, data_args = initialize_multimodal_modules(model, model_args, training_args, data_args, tokenizer)

    model = adjust_precision(model, training_args)

    if model_args.text_tower is not None:
        clip_tokenizer = load_clip_tokenizer(model_args.text_tower, training_args)
        model.set_clip_tokenizer(clip_tokenizer)

    model.set_tokenizer(tokenizer)
    if hasattr(model, 'set_cur_task'):
        model.set_cur_task(model_args.cur_task, model_args.expert_num)

    model.train()
    return model, tokenizer, data_args

def load_model_for_inference(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", text_tower=None, **kwargs):
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
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if text_tower:
                # Ensure the base model is constructed with a CLIP text tower for routing.
                setattr(lora_cfg_pretrained, 'mm_text_tower', text_tower)
                if not hasattr(lora_cfg_pretrained, 'mm_text_select_layer'):
                    setattr(lora_cfg_pretrained, 'mm_text_select_layer', -1)

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            if not text_tower:
                raise ValueError('text_tower must be provided for HiDe routing (e.g. openai/clip-vit-large-patch14')

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

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                warnings.warn('There is no corresbonding non_lora_trainables.bin')
                assert(0)
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from PEFT.peft import PeftModel, TaskType, get_peft_model, HiDeMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict
            # else:
            #     from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
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
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

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
