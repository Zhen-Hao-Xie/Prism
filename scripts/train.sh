#!/bin/bash
# scripts/run_benchmark.sh

# 动态设置PYTHONPATH
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH"

# 默认值
BENCHMARK="coin"
GPUS="0,1"
TASKS=()

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
        *)
            TASKS+=("$1")
            shift
            ;;
    esac
done

# 如果没有指定任务，报错
if [ ${#TASKS[@]} -eq 0 ]; then
    echo "❌ Error: No tasks specified"
    echo "Usage: $0 [--benchmark NAME] [--gpus GPUS] <task_id> [task_id ...]"
    echo "Example: $0 --benchmark coin 1 2 3"
    echo "Example: $0 --benchmark coin --gpus 2,3 0 1 2 3 4 5 6 7"
    exit 1
fi

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 依次运行指定的任务
for task_id in "${TASKS[@]}"; do
    echo "=========================================="
    echo "🚀 Running $BENCHMARK Task $task_id"
    echo "=========================================="
    
    python scripts/run_task.py --benchmark "$BENCHMARK" --task "$task_id" --gpus "$GPUS"
    
    if [ $? -ne 0 ]; then
        echo "❌ Task $task_id failed"
        exit 1
    fi
    
    echo "✅ Task $task_id completed"
    echo ""
done

echo "🎉 All specified tasks completed successfully!"