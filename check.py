# check_hide_state.py
"""
检查 hide_state.pt 文件内容，打印每个 anchor 的范数
"""
import torch
import os
import sys

def check_hide_state(checkpoint_path):
    """检查 hide_state.pt 文件"""
    
    hide_path = os.path.join(checkpoint_path, 'hide_state.pt')
    
    if not os.path.exists(hide_path):
        print(f"❌ hide_state.pt 不存在：{hide_path}")
        print(f"\n📁 目录内容：")
        if os.path.exists(checkpoint_path):
            for f in os.listdir(checkpoint_path):
                size = os.path.getsize(os.path.join(checkpoint_path, f)) / 1024 / 1024
                print(f"  {f}: {size:.2f} MB")
        return False
    
    print(f"{'='*70}")
    print(f"📦 检查 hide_state.pt")
    print(f"{'='*70}")
    print(f"文件路径：{hide_path}")
    print(f"文件大小：{os.path.getsize(hide_path) / 1024 / 1024:.2f} MB\n")
    
    # 加载文件
    state = torch.load(hide_path, map_location='cpu')
    
    print(f"📋 所有键名：{list(state.keys())}\n")
    
    # 打印每个 anchor/boundary 的详细信息
    for key in ['image_anchors', 'text_anchors', 'image_boundary', 'text_boundary']:
        if key not in state:
            print(f"⚠️  {key} 不存在")
            continue
        
        tensors = state[key]
        
        if not isinstance(tensors, list):
            print(f"⚠️  {key} 不是列表类型：{type(tensors)}")
            continue
        
        print(f"{'='*70}")
        print(f"🎯 {key}: {len(tensors)} 个任务")
        print(f"{'='*70}")
        
        for i, t in enumerate(tensors):
            if isinstance(t, torch.Tensor):
                l2_norm = t.norm(p=2).item()
                l1_norm = t.norm(p=1).item()
                mean_val = t.mean().item()
                std_val = t.std().item() if t.numel() > 1 else 0.0
                min_val = t.min().item()
                max_val = t.max().item()
                
                print(f"\n  Task {i}:")
                print(f"    Shape: {tuple(t.shape)}")
                print(f"    L2 范数：{l2_norm:.6f}")
                print(f"    L1 范数：{l1_norm:.6f}")
                print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
                print(f"    Min: {min_val:.6f}, Max: {max_val:.6f}")
            else:
                print(f"\n  Task {i}: ⚠️ 不是 Tensor 类型：{type(t)}")
        
        print()
    
    # 打印元数据
    print(f"{'='*70}")
    print(f"📊 元数据")
    print(f"{'='*70}")
    for key in ['expert_num', 'num_tasks', '_last_predicted_task_id']:
        if key in state:
            print(f"  {key}: {state[key]}")
    print()
    
    return True


if __name__ == '__main__':
    # 默认路径
    default_path = '/root/autodl-tmp/tjt/PyMCIT/checkpoints/CoIN/Task1_llava_lora'
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = default_path
    
    print(f"🔍 检查路径：{checkpoint_path}\n")
    check_hide_state(checkpoint_path)