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

# 任务名到推理方法的映射（根据你的第二个文件定义）
INFERENCE_METHOD_MAP = {
    "ScienceQA": "scienceqa",
}

# 任务名到评估子命令的映射（根据你的第三个文件定义）
EVAL_TASK_MAP = {
    "ScienceQA": "scienceqa",
    "TextVQA": "textvqa",
    "ImageNet": "imagenet",
    "GQA": "gqa",
    "VizWiz": "vizwiz",
    "Grounding": "grounding",
    "VQAv2": "vqav2",
    "OCRVQA": "ocrvqa",
}

def get_task_config(benchmark_name, task_id):
    """获取指定benchmark中某个任务的配置"""
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"Benchmark '{benchmark_name}' not found. Available: {list(BENCHMARKS.keys())}")
    tasks = BENCHMARKS[benchmark_name]
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"Task {task_id} not in benchmark '{benchmark_name}' (0-{len(tasks)-1})")
    return tasks[task_id]

def format_args(template, **kwargs):
    """格式化参数列表，确保所有值转为字符串"""
    formatted = []
    for arg in template:
        if isinstance(arg, str) and "{" in arg and "}" in arg:
            try:
                # 将替换后的值转为字符串
                val = arg.format(**{k: str(v) for k, v in kwargs.items()})
                formatted.append(val)
            except KeyError:
                formatted.append(arg)
        else:
            formatted.append(str(arg))
    return formatted

def run_inference(task, model_path, gpu_list, chunks, chunk_idx, output_file):
    """使用统一推理入口执行推理"""
    eval_config = task["eval"]
    
    # 确定推理方法
    method = INFERENCE_METHOD_MAP.get(task["name"], "default")
    
    cmd = [
        "python", "-m", "llava.eval.model_unified",  # 根据实际文件名调整
        method,
        "--model-path", str(model_path),
        "--model-base", str(BASE_MODEL_PATH),
        "--question-file", str(task["test_data_path"]),
        "--image-folder", str(IMAGE_FOLDER),
        "--answers-file", str(output_file),
        "--num-chunks", str(chunks),
        "--chunk-idx", str(chunk_idx),
        "--temperature", "0",
        "--conv-mode", "vicuna_v1"
    ]
    if CLIP_PATH:
        cmd.extend(["--text-tower", str(CLIP_PATH)])
    
    # 添加任务特定推理参数
    if eval_config.get("inference_args"):
        specific_args = format_args(eval_config["inference_args"], **task)
        cmd.extend(specific_args)
    
    # 确保所有元素为字符串
    cmd = [str(item) for item in cmd]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_list[chunk_idx])
    subprocess.run(cmd, env=env, check=True)

def run_evaluation(task, result_file, output_dir):
    """使用统一评估入口执行评估"""
    eval_config = task["eval"]
    
    # 确定评估子命令
    eval_task = EVAL_TASK_MAP.get(task["name"])
    if eval_task is None:
        raise ValueError(f"No evaluation task mapping for {task['name']}")

    cmd = [
        "python", "-m", "llava.eval.eval_unified",  # 根据实际文件名调整
        eval_task,
        "--result-file", str(result_file),
        "--output-dir", str(output_dir)
    ]
    
    # 添加任务特定评估参数
    if eval_config.get("eval_args"):
        # 注意：这里需要将 result_file 和 output_dir 作为额外变量传入
        eval_args = format_args(
            eval_config["eval_args"],
            result_file=str(result_file),
            output_dir=str(output_dir),
            **task
        )
        cmd.extend(eval_args)
    
    cmd = [str(item) for item in cmd]
    subprocess.run(cmd, check=True)

# 其余函数（run_conversion、main）基本保持不变，但也要确保路径转换
def run_conversion(task, result_file, output_dir):
    eval_config = task.get("eval", {})
    if not eval_config.get("needs_conversion", False):
        return
    
    convert_cmd = ["python", eval_config["conversion_script"]]
    task_copy = task.copy()
    task_copy.pop('output_dir', None)
    task_copy.pop('result_file', None)
    convert_args = format_args(
        eval_config["conversion_args"],
        result_file=str(result_file),
        output_dir=str(output_dir),
        **task_copy
    )
    convert_cmd.extend(convert_args)
    convert_cmd = [str(item) for item in convert_cmd]
    subprocess.run(convert_cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run inference and evaluation for a single task")
    parser.add_argument("--benchmark", type=str, default="coin", help="Benchmark name")
    parser.add_argument("--task", type=int, required=True, help="Task ID")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--gpus", type=str, default="0", help="GPUs to use (e.g., 0,1,2)")
    parser.add_argument("--stage", type=str, default="MoELoRA", help="Stage name for results folder")
    parser.add_argument("--method", type=str, default="default", choices=["default", "answer", "scienceqa"], help="Inference method to use")
    
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
    print(f"{'='*60}\n")

    # 创建结果目录
    result_dir = Path(RESULT_DIR) / task["name"] / args.stage
    result_dir.mkdir(parents=True, exist_ok=True)

    # 并行推理
    print("Running inference...")
    for idx in range(chunks):
        output_file = result_dir / f"{chunks}_{idx}.jsonl"
        run_inference(task, args.model_path, gpu_list, chunks, idx, output_file)

    # 合并结果
    print("\nMerging results...")
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