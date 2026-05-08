"""
Central place to edit run defaults.

Rule:
- If a parameter is present here, it becomes the default.
- If you pass the same parameter via CLI, CLI overrides it.
"""

# ===== Argument defaults (train) =====
TRAIN_DEFAULTS = {
    "benchmark": "ucit",
    "gpus": "2,3",
    "port": 29602,
    # True → training subprocess gets PYMCIT_LOG_LEVEL=DEBUG (see run.py / train.py). Does not change batch size.
    "debug": False,
    #same,hide_llava,simple_prompt
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
    "gpus": "2,3",
    "checkpoint_task": "5",
    "checkpoint_suffix": "_llava_lora",
    "stage": "last",
    "method": "same",
    "app_config": "instruct",
    "clmethod": "same",
    "temperature": "0",
    "conv_mode": "vicuna_v1",
    # If True, mirror logs to console; otherwise write only to files under output/
    "console": False,
}



"""
Note:
- Train batch sizes are method-specific: `config/methods/<method>.py` → `TRAIN_BATCH_SIZES`.
- Structure: benchmark name (`coin` / `ucit`) → task index (same as `run.py train <id>`): CoIN 0–7, UCIT 0–5.
"""
