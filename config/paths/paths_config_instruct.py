# config/paths_config.py
from pathlib import Path

# 项目根目录（通常不需要改）
PROJECT_ROOT = str(Path(__file__).parent.parent.parent.absolute())

# 基础模型路径
BASE_MODEL_PATH = "/root/autodl-tmp/LLaVa"

# CLIP模型路径
CLIP_PATH = "/root/autodl-tmp/CLIP"

# MCIT数据集根目录
MCIT_ROOT = "/root/autodl-tmp/MCIT"

# 预训练的mm_projector
PRETRAIN_MM_PROJECTOR = f"{BASE_MODEL_PATH}/mm_projector.bin"

# 指令文件目录
INSTRUCTION_DIR = f"{MCIT_ROOT}/instructions"

# 图像文件夹
IMAGE_FOLDER = f"{MCIT_ROOT}/datasets"

# 输出目录
CHECKPOINT_DIR = f"{PROJECT_ROOT}/checkpoints"
RESULT_DIR = f"{PROJECT_ROOT}/results"
LOG_DIR = f"{PROJECT_ROOT}/logs"

# Deepspeed配置
DEEPSPEED_CONFIG = f"{PROJECT_ROOT}/config/deepspeed/zero2.json"