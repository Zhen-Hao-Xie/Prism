"""
默认配置：方法 ``smolora``（SMoLoRA）。

指令侧优先级：``ins_emb_path``（自定义 .pkl）>
``smolora_builtin_sentence_ins_emb``（框架内建 MiniLM，等价 ``ins_gen.py`` single）>
否则 CLIP ``text_tower``（需 ``clip_feature_dim``，通常 768）。
"""

from config.peft_scope_defaults import EXCLUDE_FOR_LLM_ONLY_INJECTION

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
    "clmethod": "smolora",
    "batch_size": 8,
}

TRAIN_BATCH_SIZES = {
    "coin": {i: 8 for i in range(8)},
    "ucit": {i: 8 for i in range(6)},
}

METHOD_CONFIG = {
    "cur_task": 0,
    # 总专家数 = VU 半区 + IF 半区；每侧路由在 expert_num/2 里 top-1。要「各 4 个」→ 填 8（且 lora_r 可被 8 整除）
    "smolora_expert_num": 8,
    # 自定义 .pkl 时优先于内建；留空且 ``smolora_builtin_sentence_ins_emb`` 为 True 时在 initialize_model 内算 MiniLM 矩阵
    "ins_emb_path": None,
    # True：不跑作者脚本，与 ins_gen.py single 同列表 + mean pooling（默认 True 以对齐论文）
    "smolora_builtin_sentence_ins_emb": True,
    # 本机目录（须含 config.json）；框架会 local_files_only=True，不向 huggingface.co 发请求。换机器请改为你的缓存路径。
    "smolora_sentence_transformer_model": "/root/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2",
    "clip_feature_dim": 768,
    "smolora_clip_no_grad": True,
    "ins_type": 0,
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "exclude_module_path_segments": list(EXCLUDE_FOR_LLM_ONLY_INJECTION),
}
