#!/bin/bash
# scripts/run_eval_benchmark.sh

# 动态设置PYTHONPATH
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

# 从 paths_config.py 读取 CHECKPOINT_DIR
CHECKPOINT_DIR=$(python -c "from config.paths_config import CHECKPOINT_DIR; print(CHECKPOINT_DIR)")

# 默认值
BENCHMARK="coin"
GPUS="0,1"
MODEL_PATH=""  # 如果指定，则使用这个路径；否则使用默认的 Task7_llava_lora
CHECKPOINT_SUFFIX="_llava_lora"  # checkpoint 后缀
DEFAULT_TASK="0"  # 默认使用 Task0 的checkpoint
STAGE="MoELoRA"
TASKS=()  # 要评估的任务列表

# 显示帮助信息
show_help() {
    echo "Usage: $0 [OPTIONS] [TASK_ID ...]"
    echo ""
    echo "Options:"
    echo "  --benchmark NAME      Benchmark name (default: coin)"
    echo "  --gpus GPUS           GPUs to use (default: 0,1)"
    echo "  --checkpoint-task TASK Task number to use for checkpoint (default: 7, gives Task7_llava_lora)"
    echo "  --checkpoint-suffix SUFFIX Checkpoint suffix (default: _llava_lora)"
    echo "  --model-path PATH     Full model path (overrides checkpoint-dir + checkpoint-task + suffix)"
    echo "  --stage NAME          Stage name for results folder (default: MoELoRA)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 0 1 2                      # Test tasks 0,1,2 using default Task7_llava_lora checkpoint"
    echo "  $0 --checkpoint-task 8 0 1     # Test tasks 0,1 using Task8_llava_lora checkpoint"
    echo "  $0 --model-path /custom/path 0 # Test task 0 using custom model path"
    echo "  $0 0 1 2 3 4 5 6 7            # Test all tasks using default Task7_llava_lora"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --checkpoint-task)
            DEFAULT_TASK="$2"
            shift 2
            ;;
        --checkpoint-suffix)
            CHECKPOINT_SUFFIX="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            TASKS+=("$1")
            shift
            ;;
    esac
done

# 如果没有指定任务，报错
if [ ${#TASKS[@]} -eq 0 ]; then
    echo "❌ Error: No tasks specified for evaluation"
    echo "Please specify which tasks to evaluate (e.g., 0 1 2)"
    show_help
    exit 1
fi

# 将benchmark名称转换为对应的目录名
BENCHMARK_DIR=""
case $BENCHMARK in
    coin|CoIN)
        BENCHMARK_DIR="CoIN"  # 你的目录名是 CoIN
        ;;
    *)
        BENCHMARK_DIR="$BENCHMARK"  # 其他benchmark保持原样
        ;;
esac

# 确定模型路径
if [ -n "$MODEL_PATH" ]; then
    # 用户指定了完整路径
    FULL_MODEL_PATH="$MODEL_PATH"
    echo "📌 Using custom model path: $FULL_MODEL_PATH"
else
    # 使用 checkpoint-dir + benchmark-dir + Task{task}_llava_lora
    CHECKPOINT_NAME="Task${DEFAULT_TASK}${CHECKPOINT_SUFFIX}"
    FULL_MODEL_PATH="${CHECKPOINT_DIR}/${BENCHMARK_DIR}/${CHECKPOINT_NAME}"
    echo "📌 Using default model path: $FULL_MODEL_PATH (from Task${DEFAULT_TASK})"
fi

# 检查模型路径是否存在
if [ ! -d "$FULL_MODEL_PATH" ]; then
    echo "⚠️  Warning: Model path does not exist: $FULL_MODEL_PATH"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 切换到项目根目录
cd "$(dirname "$0")/.." || { echo "❌ Failed to cd to project root"; exit 1; }

# 打印配置信息
echo "=========================================="
echo "Evaluation Configuration"
echo "=========================================="
echo "Benchmark: $BENCHMARK"
echo "Benchmark directory: $BENCHMARK_DIR"
echo "Checkpoint root: $CHECKPOINT_DIR"
echo "Checkpoint name: Task${DEFAULT_TASK}${CHECKPOINT_SUFFIX}"
echo "Full model path: $FULL_MODEL_PATH"
echo "GPUs: $GPUS"
echo "Stage: $STAGE"
echo "Tasks to evaluate: ${TASKS[*]}"
echo "=========================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 依次运行指定的任务
FAILED_TASKS=()
for task_id in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "🚀 Evaluating $BENCHMARK Task $task_id"
    echo "=========================================="
    echo "Using model: $FULL_MODEL_PATH"
    
    # 运行评估
    python tools/eval_task.py \
        --benchmark "$BENCHMARK" \
        --task "$task_id" \
        --model-path "$FULL_MODEL_PATH" \
        --gpus "$GPUS" \
        --stage "$STAGE"
    
    if [ $? -ne 0 ]; then
        echo "❌ Task $task_id failed"
        FAILED_TASKS+=("$task_id")
    else
        echo "✅ Task $task_id completed successfully"
    fi
done

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

# 打印总结
echo ""
echo "=========================================="
echo "🎉 Evaluation Complete!"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Tasks evaluated: ${#TASKS[@]}"
echo "Model used: $FULL_MODEL_PATH"

if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "✅ All tasks succeeded!"
else
    echo "❌ Failed tasks: ${FAILED_TASKS[*]}"
fi

# 如果有失败的任务，退出码为1
[ ${#FAILED_TASKS[@]} -eq 0 ] || exit 1