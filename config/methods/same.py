"""
Defaults for method: same

可选 ``METHOD_CONFIG["exclude_module_path_segments"]`` 控制 PEFT 注入路径；默认已设为仅 LLM 注入。

``peft_target_modules``：本方法默认 **仅 FFN**（``gate_proj`` / ``up_proj`` / ``down_proj``）；可用 ``METHOD_CONFIG`` 或 ``--peft_target_modules`` 覆盖，见 ``PEFT/utils/peft_target_modules.py``。
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "same",
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
    # Batch size for `backbone.shared.eval.model_unified` (InferenceEngine)
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

METHOD_CONFIG = {
    "lora_dropout": 0.05,
    "peft_target_modules": "ffn",
    # SAME PEFT 层内谱 / 曲率相关超参（见 PEFT.tuners.custom.same.SAMELinear）
    "tau_score": 0.1,
    "curvature_mu": 0.9,
    "window_size": 3,
    "max_components": 64,
    # 累积奇异值能量占比阈值，用于选取主方向（原默认 0.9）
    "cumulative_energy_ratio": 0.9,
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}


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