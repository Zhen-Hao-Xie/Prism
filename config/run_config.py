"""
Central place to edit run defaults.

Rule:
- If a parameter is present here, it becomes the default.
- If you pass the same parameter via CLI, CLI overrides it.
"""

# ===== Argument defaults (train) =====
TRAIN_DEFAULTS = {
    "benchmark": "trigap",
    "gpus": "0,1,2,3,4,5",
    "port": 29602,
    # True → training subprocess gets PYMCIT_LOG_LEVEL=DEBUG (see run.py / train.py). Does not change batch size.
    "debug": False,
    # hide_llava, olora, replay_lora, ...
    "method": "same",
    # UCIT: when True, append _sub to train/test/eval *.json paths (canonical names in config/benchmarks/UCIT.py).
    "use_sub_dataset": False,
}

# Extra args appended at the end of the training command.
# Example: ["--save_steps", "100", "--evaluation_strategy", "steps"]
TRAIN_EXTRA_ARGS: list[str] = []

# ===== Argument defaults (infer) =====
INFER_DEFAULTS = {
    "benchmark": "trigap",
    "gpus": "0,1,2,3,4,5",
    "checkpoint_task": "9",
    "checkpoint_suffix": "_llava",
    "stage": "last",
    "method": "same",
    "temperature": "0",
    "use_sub_dataset": False,
}


"""
Note:
- Train batch sizes are method-specific: `config/methods/<method>.py` → `TRAIN_BATCH_SIZES`.
- Structure: benchmark name (`coin` / `ucit`) → task index (same as `run.py train <id>`): CoIN 0–7, UCIT 0–5.
- Default conversation template: `config/backbone/llava.py` → `DEFAULT_CONV_MODE`; override at infer with `--conv-mode`.
- `use_sub_dataset`: for **ucit** only, runtime add/remove `_sub` on `train_data_path` / `test_data_path` / `eval_annotation_path` (`.json`); see `utils/sub_dataset.py`.
"""
