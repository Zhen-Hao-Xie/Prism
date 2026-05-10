"""
Defaults for method: olora (O-LoRA, Wang et al., arXiv:2310.14152).

与论文 / 官方实现（https://github.com/cmnfriend/O-LoRA）对齐要点：

**注入模块（§3.2 末）**：原文沿用 Hu et al. (2021)，LoRA **仅加在注意力 W_q、W_v** 上；LLaMA/Vicuna 中对应 ``q_proj``、``v_proj``（本配置使用预设 ``attn_qv``）。

**损失（式 (7)(8)）**：CE + λ₁·Σ L_orth，``olora_lambda`` 对应 λ₁。官方脚本 ``scripts/order_1.sh`` 为 ``lamda_1=0.5``、``lamda_2=0``。

**优化**：官方 T5 脚本 ``learning_rate=1e-3``、``lr_scheduler_type=constant``、``warmup_steps=0``（等价于不在此处做 warmup）；``num_train_epochs=1``。

**LoRA 秩与缩放**：官方 ``run_uie_lora.py`` 默认 ``lora_dim=8``（即秩 r=8）、构造适配器时 ``lora_alpha=32``、``lora_dropout=0.1``，故 **α/r = 4**。
本仓库 O-LoRA 将总秩 ``lora_r`` **按任务槽数 task_num 均分**，为使**每槽有效秩仍为 8**，取 ``lora_r = 8 × task_num``；
并保持 ``lora_alpha / lora_r = 4``（与官方一致）：``lora_alpha = 4 × lora_r``。

多模态 batch、序列长等与原文 T5 不同，见 ``TRAIN_BATCH_SIZES`` / ``run.py``；若显存允许可参考原文 ``per_device_train_batch_size=8`` 适当加大。
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

# This is the official setting
# TRAIN_FLAG_OVERRIDES = {
#     "--method": "olora",
#     "--mm_projector_lr": "1e-3",
#     "--num_train_epochs": "1",
#     "--learning_rate": "1e-03",
#     "--warmup_ratio": "0",
#     "--lr_scheduler_type": "constant",
#     "--logging_steps": "1",
#     "--model_max_length": "2048",
#     "--dataloader_num_workers": "4",
# }

TRAIN_FLAG_OVERRIDES = {
    "--method": "olora",
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
        0: 6,
        1: 6,
        2: 6,
        3: 6,
        4: 6,
        5: 6,
        6: 6,
        7: 6,
    },
    "ucit": {
        0: 6,
        1: 4,
        2: 4,
        3: 4,
        4: 4,
        5: 4,
    },
}

METHOD_CONFIG = {
    # 官方 run_uie_lora.py：lora_dropout=0.1
    "lora_dropout": 0.1,
    "peft_target_modules": "attn_and_ffn",
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
    "olora_lambda": 0.5,
    # 训练时每 N 次 forward 打印 CE/orth/total；0 关闭
    "olora_orthogonal_log_interval": 50,
}

METHOD_CONFIG_BY_BENCHMARK = {
    "coin": {
        "lora_r": 64,
        "lora_alpha": 256,
    },
    "ucit": {
        "lora_r": 96,
        "lora_alpha": 384,
    },
}
