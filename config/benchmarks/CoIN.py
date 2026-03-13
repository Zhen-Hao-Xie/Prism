# config/coin.py
from ..paths_config import INSTRUCTION_DIR, CHECKPOINT_DIR,PRETRAIN_MM_PROJECTOR

# CoIN Benchmark 8个任务的配置
COIN_TASKS = [
    {
        "cur_task": 0,
        "name": "ScienceQA",
        "data_path": f"{INSTRUCTION_DIR}/ScienceQA/train.json",
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task0",
        "batch_size": 1,
        "pretrain_mm_mlp_adapter": f"{PRETRAIN_MM_PROJECTOR}",
        "previous_task":None,
    },
    {
        "cur_task": 1,
        "name": "TextVQA",
        "data_path": f"{INSTRUCTION_DIR}/TextVQA/train.json",
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task1",
        "batch_size": 1,
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task0",
    },
    {
        "cur_task": 2,
        "name": "ImageNet",
        "data_path": f"{INSTRUCTION_DIR}/ImageNet/train.json",
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task2",
        "batch_size": 8,
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task1",
    },
    {
        "cur_task": 3,
        "name": "GQA",
        "data_path": f"{INSTRUCTION_DIR}/GQA/train.json",
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task3",
        "batch_size": 8,
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task2",
    },
    {
        "cur_task": 4,
        "name": "VizWiz",
        "data_path": f"{INSTRUCTION_DIR}/VizWiz/train.json",
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task4",
        "batch_size": 8,
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task3",
    },
    {
        "cur_task": 5,
        "name": "Grounding",
        "data_path": f"{INSTRUCTION_DIR}/Grounding/train.json",
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task5",
        "batch_size": 8,
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task4",
    },
    {
        "cur_task": 6,
        "name": "VQAv2",
        "data_path": f"{INSTRUCTION_DIR}/VQAv2/train.json",
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task6",
        "batch_size": 8,
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task5",
    },
    {
        "cur_task": 7,
        "name": "OCRVQA",
        "data_path": f"{INSTRUCTION_DIR}/OCRVQA/train.json",
        "output_dir": f"{CHECKPOINT_DIR}/CoIN/Task7",
        "batch_size": 8,
        "previous_task": f"{CHECKPOINT_DIR}/CoIN/Task6",
    },
]