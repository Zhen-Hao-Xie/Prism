#!/usr/bin/env python
# tools/eval_task.py
import os
import sys
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from config.paths_config import (
    BASE_MODEL_PATH, CLIP_PATH, IMAGE_FOLDER, RESULT_DIR
)
from config.benchmarks import BENCHMARKS


def get_task_config(benchmark_name, task_id):
    """获取指定benchmark中某个任务的配置"""
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Benchmark '{benchmark_name}' not found. Available: {list(BENCHMARKS.keys())}")
    tasks = BENCHMARKS[benchmark_name]
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Task {task_id} not in benchmark '{benchmark_name}' (0-{len(tasks)-1})")
    return tasks[task_id]


def format_args(template, **kwargs):
    """格式化参数列表，替换模板中的变量"""
    formatted = []
    for arg in template:
        if isinstance(arg, str) and "{" in arg and "}" in arg:
            try:
                formatted.append(arg.format(**kwargs))
            except KeyError:
                formatted.append(arg)
        else:
            formatted.append(arg)
    return formatted


def run_inference(task, model_path, gpu_list, chunks, chunk_idx, output_file):
    """对单个chunk执行推理"""
    eval_config = task["eval"]
    
    # 构建推理命令
    cmd = ["python", "-m", eval_config["inference_module"]]
    
    # 添加通用参数
    common_args = [
        "--model-path", model_path,
        "--model-base", BASE_MODEL_PATH,
        "--question-file", task["test_data_path"],
        "--image-folder", IMAGE_FOLDER,
        "--text-tower", CLIP_PATH,
        "--answers-file", output_file,
        "--num-chunks", str(chunks),
        "--chunk-idx", str(chunk_idx),
        "--temperature", "0",
        "--conv-mode", "vicuna_v1"
    ]
    cmd.extend(common_args)
    
    # 添加数据集特定参数
    if eval_config.get("inference_args"):
        specific_args = format_args(
            eval_config["inference_args"],
            **task
        )
        cmd.extend(specific_args)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_list[chunk_idx])
    subprocess.run(cmd, env=env, check=True)


def run_conversion(task, result_file, output_dir):
    """运行数据格式转换（如果需要）"""
    eval_config = task.get("eval", {})
    
    if not eval_config.get("needs_conversion", False):
        return
    
    convert_cmd = ["python", eval_config["conversion_script"]]
    convert_args = format_args(
        eval_config["conversion_args"],
        result_file=result_file,
        output_dir=output_dir,
        **task
    )
    convert_cmd.extend(convert_args)
    subprocess.run(convert_cmd, check=True)


def run_evaluation(task, result_file, output_dir):
    """运行评估脚本"""
    eval_config = task["eval"]
    
    # 创建评估命令
    cmd = ["python", "-m", eval_config["eval_module"]]
    
    # 格式化评估参数
    eval_args = format_args(
        eval_config["eval_args"],
        result_file=result_file,
        output_dir=output_dir,
        **task
    )
    cmd.extend(eval_args)
    
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run inference and evaluation for a single task")
    parser.add_argument("--benchmark", type=str, default="coin", help="Benchmark name")
    parser.add_argument("--task", type=int, required=True, help="Task ID")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--gpus", type=str, default="0,1", help="GPUs to use (e.g., 0,1,2)")
    parser.add_argument("--stage", type=str, default="MoELoRA", help="Stage name for results folder")
    args = parser.parse_args()

    # 获取任务配置
    task = get_task_config(args.benchmark, args.task)
    
    # 验证任务有评估配置
    if "eval" not in task:
        raise ValueError(f"Task {args.task} ({task['name']}) has no evaluation configuration")
    
    gpu_list = [int(x.strip()) for x in args.gpus.split(",")]
    chunks = len(gpu_list)

    print(f"\n{'='*60}")
    print(f"Evaluating {args.benchmark} Task {args.task}: {task['name']}")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Test data: {task['test_data_path']}")
    print(f"GPUs: {args.gpus} ({chunks} chunks)")
    print(f"Stage: {args.stage}")
    print(f"Inference module: {task['eval']['inference_module']}")
    print(f"Evaluation module: {task['eval']['eval_module']}")
    print(f"{'='*60}\n")

    # 创建结果目录
    result_dir = Path(RESULT_DIR) / task["name"] / args.stage
    result_dir.mkdir(parents=True, exist_ok=True)

    # 并行推理
    print("🚀 Running inference...")
    for idx in range(chunks):
        output_file = result_dir / f"{chunks}_{idx}.jsonl"
        run_inference(task, args.model_path, gpu_list, chunks, idx, output_file)

    # 合并结果
    print("\n📊 Merging results...")
    merged_file = result_dir / "merge.jsonl"
    with open(merged_file, "w") as outfile:
        for idx in range(chunks):
            chunk_file = result_dir / f"{chunks}_{idx}.jsonl"
            with open(chunk_file) as infile:
                outfile.write(infile.read())

    # 格式转换（如果需要）
    run_conversion(task, str(merged_file), str(result_dir))

    # 运行评估
    print("\n📈 Running evaluation...")
    run_evaluation(task, str(merged_file), str(result_dir))
    
    print(f"\n✅ {task['name']} evaluation completed!")
    print(f"📁 Results saved in: {result_dir}")


if __name__ == "__main__":
    main()