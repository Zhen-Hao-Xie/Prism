"""
Defaults for method: hide_llava
"""

# These are merged on top of config/run_config.py

TRAIN_FLAG_OVERRIDES = {
    # keep consistent with current training command defaults
    "--method": "hide_llava",
    "--expert_num": "8",
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

INFER_DEFAULTS = {"clmethod": "hide_llava",}

