#!/bin/bash
# ====== 路径配置 ======
export BASE_MODEL_PATH="autodl-tmp/LLaVa"           # 基础模型
export CHECKPOINT_PATH="./checkpoints/Task8_llava_lora_ours"  # 最终 checkpoint
export CLIP_PATH="autodl-tmp/CLIP"                 # CLIP 路径
export SAME_SOURCE="autodl-tmp/tangjt/PyMCIT"        # SAME 源码路径

# ====== GPU 配置 ======
export CUDA_VISIBLE_DEVICES=0  # 指定使用的GPU

# ====== 显示配置信息 ======
echo "========================================="
echo "🚀 启动配置"
echo "========================================="
echo "📁 基础模型: $BASE_MODEL_PATH"
echo "📁 Checkpoint: $CHECKPOINT_PATH"
echo "📁 CLIP路径: $CLIP_PATH"
echo "📁 SAME源码: $SAME_SOURCE"
echo "🎮 使用GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# ====== 运行Python脚本 ======
python chat_robot/chat_main.py