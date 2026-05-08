"""LLaVA 骨干相关常量（与 Vicuna-7B / 32 层 Llama 等对齐）。"""

BACKBONE_ID = "llava"
DEFAULT_CONV_MODE = "vicuna_v1"

# CLIP / 视觉塔输出特征维度（锚点、任务路由、Smolora 实例嵌入等与视觉对齐时使用）
CLIP_FEATURE_DIM = 768

# 语言模型 Transformer 层数（HiDe 推理：仅最后一层 block 上 LoRA 按 predicted_task_id 路由，其余层 fuse）
NUM_HIDDEN_LAYERS = 32
LAST_LORA_BLOCK_INDEX = NUM_HIDDEN_LAYERS - 1
