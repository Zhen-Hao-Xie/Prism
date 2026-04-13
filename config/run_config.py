"""
Central place to edit run defaults.

Rule:
- If a parameter is present here, it becomes the default.
- If you pass the same parameter via CLI, CLI overrides it.
"""

# ===== Argument defaults (train) =====
TRAIN_DEFAULTS = {
    "benchmark": "coin",
    "gpus": "0,1",
    "port": 29601,
    "debug": False,
    "method": "hide_llava",
    "app_config": "instruct",
    # If True, mirror logs to console; otherwise write only to files under output/
    "console": False,
}

# Extra args appended at the end of the training command.
# Example: ["--save_steps", "100", "--evaluation_strategy", "steps"]
TRAIN_EXTRA_ARGS: list[str] = []

# ===== Argument defaults (infer) =====
INFER_DEFAULTS = {
    "benchmark": "coin",
    "gpus": "0,1",
    "checkpoint_task": "7",
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

# ===== Training batch size (decoupled from benchmark configs) =====
# Mapping: benchmark -> task_id -> batch_size
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
    }
}

