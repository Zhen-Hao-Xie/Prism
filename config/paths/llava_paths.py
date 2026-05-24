"""Central path configuration for the project."""
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent.parent.absolute())

BASE_MODEL_PATH = "/root/autodl-tmp/LLaVa"
CLIP_PATH = "/root/autodl-tmp/CLIP"
PRISM_ROOT = "/root/autodl-tmp/MCIT"
# Legacy alias (pre-rename checkpoints / docs)
MCIT_ROOT = PRISM_ROOT

PRETRAIN_MM_PROJECTOR = f"{BASE_MODEL_PATH}/mm_projector.bin"
INSTRUCTION_DIR = f"{PRISM_ROOT}/instructions"
IMAGE_FOLDER = f"{PRISM_ROOT}/datasets"

CHECKPOINT_DIR = f"{PROJECT_ROOT}/checkpoints"
RESULT_DIR = f"{PROJECT_ROOT}/results"
LOG_DIR = f"{PROJECT_ROOT}/logs"

DEEPSPEED_CONFIG = f"{PROJECT_ROOT}/config/deepspeed/zero2.json"
