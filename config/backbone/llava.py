"""LLaVA backbone constants (aligned with Vicuna-7B / 32-layer Llama)."""

BACKBONE_ID = "llava"
DEFAULT_CONV_MODE = "vicuna_v1"

# CLIP / vision tower hidden size (anchors, routing, SmoLoRA instruction alignment)
CLIP_FEATURE_DIM = 768

# LM transformer depth (HiDe: route LoRA on last block only by predicted_task_id; fuse on others)
NUM_HIDDEN_LAYERS = 32
LAST_LORA_BLOCK_INDEX = NUM_HIDDEN_LAYERS - 1
