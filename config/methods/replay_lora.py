"""
replay_lora：标准 LoRA（默认 attention+FFN）+ 任务分区经验回放。

回放超参见 ``METHOD_CONFIG``：

- ``replay_buffer_size``：总容量目标（按 ``task_num-1`` 槽均分上限）。
- ``replay_sample_prob``：对每个 **当前 batch 中的样本**（``cl_raw_example``）独立伯努利，决定是否写入 buffer；
  重写 ``Replay_loraIntegration.should_store_training_example`` 可按特征/loss 等自定义（见 ``method/base/integration.py`` 默认接口）。

``peft_target_modules`` 默认 ``attn_and_ffn``（见 ``PEFT/utils/peft_target_modules.py``）。
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "replay_lora",
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
    "batch_size": 12,
}

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

METHOD_CONFIG = {
    "lora_dropout": 0.05,
    "peft_target_modules": "attn_and_ffn",
    "exclude_module_path_segments": EXCLUDE_FOR_LLM_ONLY_INJECTION,
    "replay_buffer_size": 5000,
    "replay_sample_prob": 0.7,
}

METHOD_CONFIG_BY_BENCHMARK = {
    "ucit": {"lora_r": 72, "lora_alpha": 144},#we set r to 64 as it introduces extra samples which should be stored.
}
