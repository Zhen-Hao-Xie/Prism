#!/usr/bin/env python
# run_task.py
import os
import subprocess
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from config.paths_config import (
    BASE_MODEL_PATH, CLIP_PATH, PRETRAIN_MM_PROJECTOR,
    IMAGE_FOLDER, DEEPSPEED_CONFIG
)
from config.benchmarks.CoIN import COIN_TASKS

# 可以在这里添加更多benchmark
BENCHMARKS = {
    "coin": COIN_TASKS,
    # "other": OTHER_TASKS,  # 可以添加其他benchmark
}

def build_command(task, gpus="0,1", port=29601, debug=False):
    """构建训练命令"""
    
    cmd = [
        "deepspeed",
        f"--include=localhost:{gpus}",
        f"--master_port={port}",
        "backbone/llava/train/train_mem.py",
        "--deepspeed", DEEPSPEED_CONFIG,
        "--lora_enable", "True",
        "--lora_r", "64",
        "--lora_alpha", "128",
        "--mm_projector_lr", "2e-5",
        "--expert_num", "8",
        "--model_name_or_path", BASE_MODEL_PATH,
        "--freeze_mm_mlp_adapter", "True",
        "--version", "v1",
        "--data_path", task["train_data_path"],
        "--image_folder", IMAGE_FOLDER,
        "--vision_tower", CLIP_PATH,
        "--text_tower", CLIP_PATH,
        "--cur_task", str(task["cur_task"]),
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--group_by_modality_length", "True",
        "--bf16", "True",
        "--output_dir", task["output_dir"],
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", str(1 if debug else task["batch_size"]),
        "--per_device_eval_batch_size", str(1 if debug else task["batch_size"]),
        "--gradient_accumulation_steps", "1",
        "--evaluation_strategy", "no",
        "--save_strategy", "epoch",
        "--learning_rate", "2e-4",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "4",
        "--lazy_preprocess", "True",
        "--report_to", "none",
        "--method","hide_llava"
    ]

    # 如果任务指定了pretrain_mm_mlp_adapter，使用任务特定的
    if "pretrain_mm_mlp_adapter" in task:
        cmd.extend(["--pretrain_mm_mlp_adapter", task["pretrain_mm_mlp_adapter"]])
    # 添加前一个任务的路径（如果不是第一个任务）
    elif task["previous_task"]:
        cmd.extend(["--previous_task_model_path", task["previous_task"]])
    else:
        assert(0)
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description='Run a task from a benchmark')
    parser.add_argument('--benchmark', type=str, default='coin',
                       choices=BENCHMARKS.keys(),
                       help='Benchmark name (default: coin)')
    parser.add_argument('--task', type=int, required=True,
                       help='Task ID to run')
    parser.add_argument('--gpus', type=str, default='0,1',
                       help='GPUs to use (default: 0,1)')
    parser.add_argument('--port', type=int, default=29601,
                       help='Master port (default: 29601)')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (batch size=4)')
    
    args = parser.parse_args()
    
    # 获取benchmark的任务列表
    tasks = BENCHMARKS[args.benchmark]
    
    # 检查任务ID是否有效
    if args.task < 0 or args.task >= len(tasks):
        print(f"❌ Error: Task {args.task} not in {args.benchmark} benchmark (0-{len(tasks)-1})")
        sys.exit(1)
    
    # 获取任务配置
    task = tasks[args.task]
    
    print(f"\n{'='*60}")
    print(f"Training on benchmark{args.benchmark} Task {args.task}: {task['name']}")
    print(f"{'='*60}")
    print(f"Data: {task['train_data_path']}")
    print(f"Batch Size: {task['batch_size']}")
    print(f"Output: {task['output_dir']}")
    if task['previous_task']:
        print(f"🔄 Loading from: {task['previous_task']}")
    else:
        print(f"✨ First task - no previous checkpoint")
    print(f"🎮 GPUs: {args.gpus}")
    print(f"{'='*60}\n")
    
    # 构建并执行命令
    cmd = build_command(task, args.gpus, args.port, args.debug)
    
    print("📝 Command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Task {args.task} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Task {args.task} failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()