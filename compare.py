# compare_checkpoints.py
"""
对比两个 checkpoint 中的专家权重
用法: python compare_checkpoints.py /path/to/Task0 /path/to/Task1
"""

import os
import sys
import torch
import numpy as np
from safetensors.torch import load_file


def load_weights(checkpoint_path):
    """加载 checkpoint 中的权重"""
    # 尝试 safetensors
    lora_path = os.path.join(checkpoint_path, 'adapter_model.safetensors')
    if os.path.exists(lora_path):
        return load_file(lora_path)
    
    # 尝试 bin
    lora_path = os.path.join(checkpoint_path, 'adapter_model.bin')
    if os.path.exists(lora_path):
        return torch.load(lora_path, map_location='cpu')
    
    raise FileNotFoundError(f"No adapter_model found in {checkpoint_path}")


def analyze_expert_weights(weights, checkpoint_name):
    """分析专家权重"""
    print(f"\n{'='*70}")
    print(f"📊 {checkpoint_name}")
    print(f"{'='*70}")
    
    # 按专家分组
    expert_weights = {}
    for key in weights.keys():
        if 'lora_A' not in key and 'lora_B' not in key:
            continue
        
        # 提取专家 ID
        import re
        # 匹配 loraA.0, loraA.1 等
        match = re.search(r'loraA\.(\d+)', key)
        if not match:
            match = re.search(r'loraB\.(\d+)', key)
        if not match:
            match = re.search(r'expert_(\d+)', key)
        
        if match:
            expert_id = match.group(1)
            if expert_id not in expert_weights:
                expert_weights[expert_id] = []
            expert_weights[expert_id].append((key, weights[key]))
    
    print(f"发现的专家 ID: {sorted(expert_weights.keys())}")
    
    # 计算每个专家的统计信息
    expert_stats = {}
    for expert_id, items in expert_weights.items():
        norms = [w.norm().item() for _, w in items]
        means = [w.mean().item() for _, w in items]
        stds = [w.std().item() for _, w in items]
        
        expert_stats[expert_id] = {
            'count': len(items),
            'norm_mean': np.mean(norms),
            'norm_std': np.std(norms),
            'mean_mean': np.mean(means),
            'std_mean': np.mean(stds),
        }
        
        print(f"\n  Expert {expert_id} ({len(items)} 个参数):")
        print(f"    norm:  mean={expert_stats[expert_id]['norm_mean']:.6f}, std={expert_stats[expert_id]['norm_std']:.6f}")
        print(f"    mean:  {expert_stats[expert_id]['mean_mean']:.6f}")
        print(f"    std:   {expert_stats[expert_id]['std_mean']:.6f}")
        
        # 打印前 3 个参数的范数
        print(f"    前3个参数范数:")
        for key, w in items[:3]:
            print(f"      {key.split('.')[-2]}.{key.split('.')[-1]}: {w.norm().item():.6f}")
    
    return expert_weights, expert_stats


def compare_experts(weights0, weights1, expert_id, checkpoint0_name, checkpoint1_name):
    """对比两个 checkpoint 中同一个专家的权重"""
    print(f"\n{'='*70}")
    print(f"🔍 对比 Expert {expert_id}")
    print(f"{'='*70}")
    
    # 找到两个 checkpoint 中共有的键
    keys0 = set(k for k in weights0.keys() if f'loraA.{expert_id}' in k or f'loraB.{expert_id}' in k)
    keys1 = set(k for k in weights1.keys() if f'loraA.{expert_id}' in k or f'loraB.{expert_id}' in k)
    
    common_keys = keys0 & keys1
    print(f"共有的键: {len(common_keys)}")
    
    if len(common_keys) == 0:
        print("⚠️ 没有共有的键")
        return
    
    # 计算差异
    diffs = []
    same_count = 0
    diff_count = 0
    
    for key in sorted(common_keys)[:10]:  # 只对比前10个
        w0 = weights0[key]
        w1 = weights1[key]
        diff = (w0 - w1).norm().item()
        diffs.append(diff)
        
        norm0 = w0.norm().item()
        norm1 = w1.norm().item()
        
        if diff < 1e-6:
            same_count += 1
            status = "✅ 相同"
        else:
            diff_count += 1
            status = f"❌ 不同 (diff={diff:.6f})"
        
        print(f"  {key}:")
        print(f"    {checkpoint0_name}: norm={norm0:.6f}")
        print(f"    {checkpoint1_name}: norm={norm1:.6f}")
        print(f"    {status}")
    
    if len(common_keys) > 10:
        # 统计所有键
        all_same = 0
        all_diff = 0
        all_diffs = []
        for key in common_keys:
            diff = (weights0[key] - weights1[key]).norm().item()
            all_diffs.append(diff)
            if diff < 1e-6:
                all_same += 1
            else:
                all_diff += 1
        
        print(f"\n📊 完整统计 ({len(common_keys)} 个键):")
        print(f"  相同的键: {all_same}")
        print(f"  不同的键: {all_diff}")
        if all_diffs:
            print(f"  差异统计: min={min(all_diffs):.6f}, max={max(all_diffs):.6f}, mean={np.mean(all_diffs):.6f}")


def main():
    if len(sys.argv) < 3:
        print("用法: python compare_checkpoints.py /path/to/Task0 /path/to/Task1")
        sys.exit(1)
    
    path0 = sys.argv[1]
    path1 = sys.argv[2]
    
    name0 = os.path.basename(path0)
    name1 = os.path.basename(path1)
    
    print(f"\n📂 Checkpoint 0: {path0}")
    print(f"📂 Checkpoint 1: {path1}")
    
    # 加载权重
    weights0 = load_weights(path0)
    weights1 = load_weights(path1)
    
    print(f"\n📦 Checkpoint 0: {len(weights0)} 个键")
    print(f"📦 Checkpoint 1: {len(weights1)} 个键")
    
    # 分析每个 checkpoint 的专家权重
    expert_weights0, stats0 = analyze_expert_weights(weights0, name0)
    expert_weights1, stats1 = analyze_expert_weights(weights1, name1)
    
    # 对比 Expert 0
    if '0' in expert_weights0 and '0' in expert_weights1:
        compare_experts(weights0, weights1, '0', name0, name1)
    
    # 对比 Expert 1
    if '1' in expert_weights0 and '1' in expert_weights1:
        compare_experts(weights0, weights1, '1', name0, name1)
    
    # 检查 Expert 0 vs Expert 1 在同一 checkpoint 中的差异
    print(f"\n{'='*70}")
    print(f"🔍 同一 checkpoint 内 Expert 0 vs Expert 1")
    print(f"{'='*70}")
    
    for name, weights in [(name0, weights0), (name1, weights1)]:
        print(f"\n  {name}:")
        
        # 找 Expert 0 和 Expert 1 的第一个 lora_A 权重
        key0 = None
        key1 = None
        for key in weights.keys():
            if 'lora_A' in key or 'loraA.0' in key:
                if key0 is None and ('loraA.0' in key or 'expert_0' in key):
                    key0 = key
                if key1 is None and ('loraA.1' in key or 'expert_1' in key):
                    key1 = key
            if key0 and key1:
                break
        
        if key0 and key1:
            w0 = weights[key0]
            w1 = weights[key1]
            diff = (w0 - w1).norm().item()
            print(f"    Expert 0 vs Expert 1 diff: {diff:.6f}")
            if diff < 1e-6:
                print(f"    ⚠️ 警告：Expert 0 和 Expert 1 权重相同！")
            else:
                print(f"    ✅ Expert 0 和 Expert 1 权重不同")
    
    # 总结
    print(f"\n{'='*70}")
    print(f"📋 总结")
    print(f"{'='*70}")
    
    # 检查是否有专家权重在所有 checkpoint 中都相同
    if '0' in stats0 and '0' in stats1:
        norm_diff = abs(stats0['0']['norm_mean'] - stats1['0']['norm_mean'])
        if norm_diff > 0.01:
            print(f"  ⚠️ Expert 0 在 {name0} 和 {name1} 中不同 (diff={norm_diff:.6f})")
            print(f"     这可能是灾难性遗忘！")
        else:
            print(f"  ✅ Expert 0 在两个 checkpoint 中相同")
    
    if '1' in stats0 and '1' in stats1:
        norm_diff = abs(stats0['1']['norm_mean'] - stats1['1']['norm_mean'])
        if norm_diff > 0.01:
            print(f"  ✅ Expert 1 在 {name0} 和 {name1} 中不同 (diff={norm_diff:.6f})")
            print(f"     说明 task_1 训练时更新了 Expert 1")
        else:
            print(f"  ⚠️ Expert 1 在两个 checkpoint 中相同 (diff={norm_diff:.6f})")
            print(f"     说明 task_1 训练时没有更新 Expert 1！")


if __name__ == "__main__":
    main()