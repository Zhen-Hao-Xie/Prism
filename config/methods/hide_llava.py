"""
Defaults for method: hide_llava

PEFT 路径过滤：本文件 ``METHOD_CONFIG`` 已显式设为仅 LLM 主干注入（``config.peft_scope_defaults``）。
若要改为全模型可注入，将 ``exclude_module_path_segments`` 设为 ``[]``；自定义跳过列表见 ``config/methods/README.md``。
"""

from config.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

# These are merged on top of config/run_config.py

TRAIN_FLAG_OVERRIDES = {
    # keep consistent with current training command defaults
    "--method": "hide_llava",
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

INFER_DEFAULTS = {
    "clmethod": "hide_llava",
    # Batch size for `backbone.llava.eval.model_unified` (InferenceEngine)
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
    "clip_feature_dim": 768,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    # 显式：仅 LLM 主干注入（与 PEFT 默认 None 等价，便于各方法配置一致）
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}