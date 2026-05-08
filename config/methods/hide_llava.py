"""
Defaults for method: hide_llava

PEFT 路径过滤：本文件 ``METHOD_CONFIG`` 已显式设为仅 LLM 主干注入（``PEFT.utils.peft_scope_defaults``）。
若要改为全模型可注入，将 ``exclude_module_path_segments`` 设为 ``[]``；自定义跳过列表见 ``config/methods/README.md``。

注入子层：默认在 LLM 作用域内对**全部 Linear** 注入（``peft_target_modules: linear``）；可用命令行 ``--peft_target_modules`` 覆盖，见 ``PEFT/utils/peft_target_modules.py``。
按 benchmark 的 LoRA 规模：``METHOD_CONFIG`` + ``METHOD_CONFIG_BY_BENCHMARK``；训练命令行覆盖见 ``TRAIN_FLAG_OVERRIDES`` + ``TRAIN_FLAG_OVERRIDES_BY_BENCHMARK``（由 ``run.py`` 合并）。
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

# These are merged on top of config/run_config.py

TRAIN_FLAG_OVERRIDES = {
    # keep consistent with current training command defaults
    "--method": "hide_llava",
    # CoIN 及未在 TRAIN_FLAG_OVERRIDES_BY_BENCHMARK 中出现的 benchmark
    "--lora_r": "64",
    "--lora_alpha": "128",
    "--mm_projector_lr": "2e-5",
    "--num_train_epochs": "1",
    "--learning_rate": "2e-4",
    "--warmup_ratio": "0.03",
    "--lr_scheduler_type": "cosine",
    "--logging_steps": "1",
    "--model_max_length": "2048",
    "--dataloader_num_workers": "4",
}

TRAIN_EXTRA_ARGS: list[str] = []

# 按 benchmark 覆盖 TRAIN_FLAG_OVERRIDES 中的同名项（如 LoRA 规模）
TRAIN_FLAG_OVERRIDES_BY_BENCHMARK = {
    "ucit": {
        "--lora_r": "48",
        "--lora_alpha": "96",
    },
}

INFER_DEFAULTS = {
    # Batch size for `backbone.shared.eval.model_unified` (InferenceEngine)
    "batch_size": 12,
}

# Keys = task index in config/benchmarks/* (same as ``run.py`` / ``train <id>``), not ``cur_task`` alias.
# CoIN: 8 tasks (0–7). UCIT: 6 tasks (0–5).
TRAIN_BATCH_SIZES = {
    "coin": {
        0: 12,
        1: 12,
        2: 12,
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 8,
    },
    "ucit": {
        0: 12,
        1: 12,
        2: 12,
        3: 8,
        4: 8,
        5: 8,
    },
}

# Method parameters (used by method factory / integrations if needed)
METHOD_CONFIG = {
    # CoIN 及未在 METHOD_CONFIG_BY_BENCHMARK 中出现的 benchmark 的默认 LoRA
    "lora_dropout": 0.05,
    # 在 exclude 后的 LLM 范围内，对所有 nn.Linear 子模块注入 LoRA（见 peft_target_modules 预设 ``linear``）
    "peft_target_modules": "linear",
    # 显式：仅 LLM 主干注入（与 PEFT 默认 None 等价，便于各方法配置一致）
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}

# 按 benchmark 覆盖 METHOD_CONFIG 同名字段（推理 merge、与 train 对齐）
METHOD_CONFIG_BY_BENCHMARK = {
    "coin": {
        "lora_r": 64,
        "lora_alpha": 128,
    },
    "ucit": {
        "lora_r": 48,
        "lora_alpha": 96,
    },
}