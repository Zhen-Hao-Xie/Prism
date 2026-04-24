"""
Defaults for method: simple_prompt
"""

from config.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "simple_prompt",
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
    "clmethod": "simple_prompt",
    "batch_size": 12,
}

# Keys = task index in config/benchmarks/* (``run.py train <id>``). CoIN: 8 tasks, UCIT: 6 tasks.
# 每个任务单独一行，便于按数据集改 batch。
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

METHOD_CONFIG = {
    "clip_feature_dim": 768,
    "num_prompt_tokens": 256,
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}
