"""
Central place to edit run defaults.

Rule:
- If a parameter is present here, it becomes the default.
- If you pass the same parameter via CLI, CLI overrides it.
"""

# ===== Argument defaults (train) =====
TRAIN_DEFAULTS = {
    "benchmark": "ucit",
    "gpus": "0,1",
    "port": 29601,
    "debug": False,
    #same,hide_llava
    "method": "same",
    "app_config": "instruct",
    # If True, mirror logs to console; otherwise write only to files under output/
    "console": False,
}

# Extra args appended at the end of the training command.
# Example: ["--save_steps", "100", "--evaluation_strategy", "steps"]
TRAIN_EXTRA_ARGS: list[str] = []

# ===== Argument defaults (infer) =====
INFER_DEFAULTS = {
    "benchmark": "ucit",
    "gpus": "0,1",
    "checkpoint_task": "5",
    "checkpoint_suffix": "_llava_lora",
    "stage": "MoELoRA",
    "method": "hide_llava",
    "app_config": "instruct",
    "clmethod": "hide_llava",
    "temperature": "0",
    "conv_mode": "vicuna_v1",
    # If True, mirror logs to console; otherwise write only to files under output/
    "console": False,
}

"""
Note:
- Train batch sizes are method-specific and live under `config/methods/<method>.py` as TRAIN_BATCH_SIZES.
"""
