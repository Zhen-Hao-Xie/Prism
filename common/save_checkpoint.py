from __future__ import annotations

import os
import logging

import torch
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


def _is_peft_adapter_param_name(name: str) -> bool:
    """
    是否为 PEFT 适配器侧参数名。

    历史代码仅用 ``\"lora_\" in name``，但 PEFT/HiDe 常见键为 ``lora_A`` / ``lora_B``，
    其中 **不包含** 子串 ``lora_``（``lora_A`` 在 ``o`` 与 ``A`` 之间无下划线），
    会导致 HiDe 等保存时 LoRA 张量为空或行为异常。
    """
    if "lora_" in name:
        return True
    if "lora_A" in name or "lora_B" in name:
        return True
    if "lora_embedding" in name:
        return True
    return False


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if _is_peft_adapter_param_name(k)}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if _is_peft_adapter_param_name(k) or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if _is_peft_adapter_param_name(k):
                to_return[k] = t
                if "lora_" in k:
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
    to_return = {k: t for k, t in named_params if not _is_peft_adapter_param_name(k)}
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


def _save_cl_extra_state(model, output_dir: str):
    """调用 integration.save_extra_state() 写入方法特有状态（如 HiDe anchors）。"""
    saved = False

    if hasattr(model, "_integration") and model._integration is not None:
        try:
            if hasattr(model._integration, "save_extra_state"):
                success = model._integration.save_extra_state(output_dir, model=model)
                if success:
                    saved = True
                else:
                    # 基类默认 False：无单独 CL 状态文件属正常
                    saved = True
        except Exception as e:
            logging.exception("save_extra_state failed: %s", e)

    if not saved:
        logging.debug("No CL extra state saved (no _integration.save_extra_state).")


def _unwrap_for_checkpoint_save(model, trainer):
    """
    训练结束时 ``model`` 可能是 DDP/DeepSpeed 包装；解包后再解析 CLModel / PeftModel，
    否则会误判结构并走错保存分支（例如整模 ``model.safetensors``）。
    """
    m = model
    if trainer is not None and getattr(trainer, "model", None) is not None:
        m = trainer.model
    try:
        from accelerate.utils import unwrap_model

        return unwrap_model(m)
    except Exception:
        return m.module if hasattr(m, "module") else m


def _full_checkpoint_save(trainer, core_model, output_dir: str):
    if trainer is not None:
        safe_save_model_for_hf_trainer(trainer, output_dir)
        return
    if hasattr(core_model, "_base_model"):
        _base_model = getattr(core_model, "_base_model")
        if hasattr(_base_model, "save_pretrained"):
            _base_model.save_pretrained(output_dir)
        else:
            core_model.save_pretrained(output_dir)
    else:
        core_model.save_pretrained(output_dir)


# common/save_checkpoint.py
def save_model(model, training_args, trainer=None, save_extra_state: bool = True):
    """统一的模型保存函数"""
    
    print(f"\n{'='*70}")
    print(f"Saving model | output_dir: {training_args.output_dir}")
    print(f"{'='*70}\n")
    
    os.makedirs(training_args.output_dir, exist_ok=True)

    _lora_on = bool(getattr(training_args, "lora_enable", False))
    _core = _unwrap_for_checkpoint_save(model, trainer)
    # ========== 找到真正的 PEFT 模型（在解包后的核心模块上解析）==========
    save_model = _core
    if hasattr(_core, '_base_model'):
        _base_model = getattr(_core, '_base_model')
        if hasattr(_base_model, 'peft_config') or hasattr(_base_model, 'adapter_model'):
            save_model = _base_model
        elif hasattr(_base_model, 'base_model'):
            if hasattr(_base_model.base_model, 'peft_config'):
                save_model = _base_model

    _peft_cfg = getattr(save_model, "peft_config", None)
    _has_peft = bool(_peft_cfg) if not isinstance(_peft_cfg, dict) else len(_peft_cfg) > 0

    if _lora_on and _has_peft:
        named_params = list(save_model.named_parameters())
        state_dict = get_peft_state_maybe_zero_3(named_params, training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            named_params,
            require_grad_only=True,
        )

        if training_args.local_rank == 0 or training_args.local_rank == -1:
            save_model.config.save_pretrained(training_args.output_dir)

            if hasattr(save_model, 'save_pretrained'):
                adapter_sd = dict(state_dict) if state_dict else {}
                save_model.save_pretrained(
                    training_args.output_dir,
                    state_dict=adapter_sd if adapter_sd else None,
                    safe_serialization=True,
                )
            elif state_dict:
                torch.save(
                    state_dict,
                    os.path.join(training_args.output_dir, 'adapter_model.bin'),
                )

            if non_lora_state_dict:
                torch.save(
                    non_lora_state_dict,
                    os.path.join(training_args.output_dir, 'non_lora_trainables.bin'),
                )

        _lr = getattr(training_args, "local_rank", None)
        _lr = -1 if _lr is None else int(_lr)
        if _lr in (-1, 0):
            adapter_cfg = os.path.join(training_args.output_dir, "adapter_config.json")
            if not os.path.isfile(adapter_cfg):
                logging.warning(
                    "PEFT 保存后未找到 adapter_config.json；请检查 PeftModel.save_pretrained 或 rank0 是否执行保存。"
                )
    else:
        if _lora_on and not _has_peft:
            logging.warning(
                "lora_enable=True 但当前保存目标无 peft_config（LoRA 可能未注入）；将整模保存。"
            )
        _full_checkpoint_save(trainer, _core, training_args.output_dir)
    
    if save_extra_state:
        # 仅主进程写 CL 额外状态，避免 DeepSpeed 多进程同时读改 adapter_model.safetensors / same_state.bin
        _lr = getattr(training_args, "local_rank", None)
        if _lr is None:
            _lr = -1
        else:
            _lr = int(_lr)
        if _lr in (-1, 0):
            _save_cl_extra_state(_core, training_args.output_dir)
    
    print(f"\n{'='*70}")
    print("Model save finished")
    print(f"{'='*70}\n")
