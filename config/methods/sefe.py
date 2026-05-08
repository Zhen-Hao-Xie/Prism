"""
Defaults for method: sefe（SEFE RegLoRA）
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "sefe",
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
    "batch_size": 8,
}

TRAIN_BATCH_SIZES = {
    "coin": {i: 8 for i in range(8)},
    "ucit": {i: 8 for i in range(6)},
}

METHOD_CONFIG = {
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "sefe_top_p": 0.02,
    "sefe_lambda_reg": 2500.0,
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}
