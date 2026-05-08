"""LLaVA backbone 专用默认值（与 config/paths 及评测 CLI 对齐）。"""

from config.backbone.llava import (
    BACKBONE_ID,
    CLIP_FEATURE_DIM,
    DEFAULT_CONV_MODE,
    LAST_LORA_BLOCK_INDEX,
    NUM_HIDDEN_LAYERS,
)

__all__ = [
    "BACKBONE_ID",
    "CLIP_FEATURE_DIM",
    "DEFAULT_CONV_MODE",
    "LAST_LORA_BLOCK_INDEX",
    "NUM_HIDDEN_LAYERS",
]
