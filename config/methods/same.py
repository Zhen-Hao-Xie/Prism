"""
Defaults for method: same

可选 ``METHOD_CONFIG["exclude_module_path_segments"]`` 控制 PEFT 注入路径；默认已设为仅 LLM 注入。
"""

from config.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "same",
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

# NOTE: backbone eval currently doesn't accept "same" in --clmethod choices,
# so this is mostly for consistency; set it if/when you add it to CLI.
INFER_DEFAULTS = {
    "clmethod": "same",
    # Batch size for `backbone.llava.eval.model_unified` (InferenceEngine)
    "batch_size": 1,
}

# Keys = task index in config/benchmarks/* (``run.py train <id>``). CoIN: 0–7, UCIT: 0–5.
TRAIN_BATCH_SIZES = {
    "coin": {
        0: 12,
        1: 12,
        2: 12,
        3: 12,
        4: 12,
        5: 12,
        6: 12,
        7: 12,
    },
    "ucit": {
        0: 12,
        1: 12,
        2: 12,
        3: 12,
        4: 12,
        5: 12,
    },
}

# Method parameters (was in same.yaml)
METHOD_CONFIG = {
    "clip_feature_dim": 768,
    "cur_task": 0,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "routing_temperature": 1.0,
    "temparature": 2.0,
    "temparature_2": 1.5,
    "threshold": 0.85,
    "remaining_prob": 0.85,
    "other_total_prob": 0.15,
    "top2_ratio": [3.0, 2.0],
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}


