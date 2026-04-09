#!/usr/bin/env python
# tree_ignore.py
import os
import sys
from pathlib import Path

# 要忽略的目录和文件模式
IGNORE_DIRS = {
    '.git', '__pycache__', 'checkpoints', 'checkpoint',
    '.idea', '.vscode', 'node_modules', 'dist', 'build',
    '*.egg-info', '.ipynb_checkpoints', 'wandb', 'runs',
    'logs', 'results'  # 假设你想忽略results文件夹
}

IGNORE_FILES = {
    '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.dylib',
    '*.txt', '*.log', '*.tmp', '*.temp',
    '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.ico',
    '*.pdf', '*.doc', '*.docx', '*.xls', '*.xlsx',
    '*.zip', '*.tar', '*.gz', '*.rar',
    '*.bin', '*.pkl', '*.pt', '*.pth', '*.ckpt', '*.safetensors',
    '.DS_Store', 'Thumbs.db'
}

def should_ignore(name, is_dir=False):
    """检查文件/目录是否应该被忽略"""
    if is_dir:
        # 检查目录名
        if name in IGNORE_DIRS:
            return True
        # 检查目录名模式（如 *.egg-info）
        for pattern in IGNORE_DIRS:
            if pattern.startswith('*.') and name.endswith(pattern[1:]):
                return True
    else:
        # 检查文件名
        if name in IGNORE_FILES:
            return True
        # 检查文件扩展名
        for pattern in IGNORE_FILES:
            if pattern.startswith('*.') and name.endswith(pattern[1:]):
                return True
    return False

def count_items(path):
    """统计目录中的项目数量（用于判断是否是空目录）"""
    try:
        return len([x for x in os.listdir(path) if not should_ignore(x, os.path.isdir(os.path.join(path, x)))])
    except PermissionError:
        return 0

def list_directory(path, prefix="", is_last=True, root_depth=0, max_depth=None):
    """递归列出目录结构，忽略指定文件和目录"""
    if max_depth is not None and root_depth > max_depth:
        return
    
    try:
        items = sorted([x for x in os.listdir(path) 
                       if not should_ignore(x, os.path.isdir(os.path.join(path, x)))])
    except PermissionError:
        print(prefix + "└── [权限不足]")
        return
    
    for i, item in enumerate(items):
        item_path = os.path.join(path, item)
        is_dir = os.path.isdir(item_path)
        is_last_item = i == len(items) - 1
        
        # 当前项的前缀
        if is_last:
            current_prefix = prefix + ("└── " if is_last_item else "├── ")
            next_prefix = prefix + ("    " if is_last_item else "│   ")
        else:
            current_prefix = prefix + ("└── " if is_last_item else "├── ")
            next_prefix = prefix + ("    " if is_last_item else "│   ")
        
        # 如果是目录，显示统计信息
        if is_dir:
            sub_count = count_items(item_path)
            count_str = f" ({sub_count} items)" if sub_count > 0 else " (empty)"
            print(f"{current_prefix}{item}/{count_str}")
            
            # 递归进入子目录
            list_directory(item_path, next_prefix, is_last_item, root_depth + 1, max_depth)
        else:
            # 获取文件大小
            try:
                size = os.path.getsize(item_path)
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size/(1024*1024):.1f} MB"
                print(f"{current_prefix}{item} ({size_str})")
            except:
                print(f"{current_prefix}{item}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='列出目录结构，忽略指定文件和文件夹')
    parser.add_argument('path', nargs='?', default='.', help='要列出的路径')
    parser.add_argument('-L', '--max-depth', type=int, help='最大递归深度')
    parser.add_argument('--show-hidden', action='store_true', help='显示隐藏文件（以.开头的文件）')
    parser.add_argument('--no-size', action='store_true', help='不显示文件大小')
    
    args = parser.parse_args()
    
    root_path = os.path.abspath(args.path)
    print(f"📁 {root_path}")
    print("=" * 60)
    
    list_directory(root_path, max_depth=args.max_depth)
    print("=" * 60)
    
    # 打印忽略统计
    print(f"\n📊 忽略规则:")
    print(f"  忽略目录: {', '.join(sorted(IGNORE_DIRS)[:5])}{'...' if len(IGNORE_DIRS) > 5 else ''}")
    print(f"  忽略文件类型: {', '.join(sorted(IGNORE_FILES)[:5])}{'...' if len(IGNORE_FILES) > 5 else ''}")

if __name__ == "__main__":
    main()