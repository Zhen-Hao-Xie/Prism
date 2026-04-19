import os

BACKBONE_CONFIG = os.getenv("APP_CONFIG", "instruct")

# 动态加载对应的配置
if BACKBONE_CONFIG == 'base':
    from ..paths.paths_config_base import INSTRUCTION_DIR, CHECKPOINT_DIR, PRETRAIN_MM_PROJECTOR
elif BACKBONE_CONFIG == 'instruct':
    from ..paths.paths_config_instruct import INSTRUCTION_DIR, CHECKPOINT_DIR, PRETRAIN_MM_PROJECTOR
else:
    raise ValueError(f"Unknown config: {BACKBONE_CONFIG}")

# 由于UCIT的指令数据在/root/autodl-tmp/UCIT/instructions下，直接配置根路径
UCIT_INSTRUCTION_DIR = "/root/autodl-tmp/UCIT/instructions"
UCIT_IMAGE_DIR = "/root/autodl-tmp/UCIT/datasets"

UCIT_TASKS = [
    {
        "cur_task": 0,
        "name": "ImageNet-R",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/ImageNet-R/train.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/ImageNet-R/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/ImageNet-R/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task0_llava_lora",
        "batch_size": 12,  # From per_device_train_batch_size in Task1.sh
        "pretrain_mm_mlp_adapter": f"{PRETRAIN_MM_PROJECTOR}",
        "image_folder": UCIT_IMAGE_DIR,
        "previous_task": None,
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--annotation-file", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 1,
        "name": "ArxivQA",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/ArixvQA/train_4w.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/ArixvQA/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/ArixvQA/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task1_llava_lora",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task0_llava_lora",
        "image_folder": UCIT_IMAGE_DIR,
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--annotation-file", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 2,
        "name": "Vizcap",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/VizWiz/train.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/VizWiz/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/VizWiz/val_coco_type_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task2_llava_lora",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task1_llava_lora",
        "image_folder": UCIT_IMAGE_DIR,
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--annotation-file", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 3,
        "name": "IconQA",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/IconQA/train.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/IconQA/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/IconQA/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task3_llava_lora",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task2_llava_lora",
        "image_folder": UCIT_IMAGE_DIR,
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--annotation-file", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 4,
        "name": "CLEVR",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/CLEVR/train_4w.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/CLEVR/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/CLEVR/test_3000.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task4_llava_lora",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task3_llava_lora",
        "image_folder": UCIT_IMAGE_DIR,
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--annotation-file", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    },
    {
        "cur_task": 5,
        "name": "Flickr30k",
        "train_data_path": f"{UCIT_INSTRUCTION_DIR}/Flickr30k/train_brief_4w.json",
        "test_data_path": f"{UCIT_INSTRUCTION_DIR}/Flickr30k/test_3000.json",
        "eval_annotation_path": f"{UCIT_INSTRUCTION_DIR}/Flickr30k/val_coco_type.json",
        "output_dir": f"{CHECKPOINT_DIR}/UCIT/Task5_llava_lora",
        "batch_size": 12,
        "previous_task": f"{CHECKPOINT_DIR}/UCIT/Task4_llava_lora",
        "image_folder": UCIT_IMAGE_DIR,
        "eval": {
            "inference_args": [],
            "eval_args": [
                "--annotation-file", "{eval_annotation_path}",
                "--result-file", "{result_file}",
                "--output-dir", "{output_dir}"
            ],
            "needs_conversion": False,
        }
    }
]
