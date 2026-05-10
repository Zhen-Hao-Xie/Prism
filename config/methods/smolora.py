"""
默认配置：方法 ``smolora``（SMoLoRA）。

指令侧优先级：``ins_emb_path``（自定义 .pkl）>
``smolora_builtin_sentence_ins_emb``（框架内建 MiniLM，等价 ``ins_gen.py`` single）>
否则 CLIP ``text_tower``（特征维见 ``config.backbone.llava.CLIP_FEATURE_DIM``，通常 768）。

PEFT 注入位置：``METHOD_CONFIG["peft_target_modules"]`` 或 ``--peft_target_modules``（``attn`` / ``ffn`` / ``linear`` 等），默认 ``attention``；见 ``PEFT/utils/peft_target_modules.py``。
"""

from PEFT.utils.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

TRAIN_FLAG_OVERRIDES = {
    "--method": "smolora",
    "--lora_r": "64",
    "--lora_alpha": "128",
    "--num_train_epochs": "1",
    "--learning_rate": "2e-4",
    "--warmup_ratio": "0.03",
    "--lr_scheduler_type": "cosine",
    "--model_max_length": "2048",
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
    "cur_task": 0,
    "smolora_expert_num": 8,
    "ins_emb_path": None,
    "smolora_builtin_sentence_ins_emb": True,
    "smolora_sentence_transformer_model": "/root/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2",
    "smolora_clip_no_grad": True,
    "ins_type": 0,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}
