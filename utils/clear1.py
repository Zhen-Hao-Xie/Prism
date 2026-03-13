#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

def remove_chinese_comments_simple():
    """
    简单但有效的删除中文注释
    """
    for root, dirs, files in os.walk('.'):

        if '.git' in root or '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    modified = False
                    
                    for line in lines:

                        if '#' in line:
                            parts = line.split('#', 1)
                            if len(parts) == 2:
                                code_part, comment_part = parts

                                if re.search(r'[\u4e00-\u9fa5]', comment_part):

                                    new_line = code_part.rstrip()
                                    if new_line or line.strip():
                                        new_lines.append(new_line)
                                    modified = True
                                    continue
                        
                        new_lines.append(line.rstrip('\n'))
                    
                    if modified:

                        backup_path = filepath + '.bak'
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                        

                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(new_lines))
                        
                        print(f"✅ 已处理: {filepath}")
                
                except Exception as e:
                    print(f"❌ 处理失败 {filepath}: {e}")

if __name__ == "__main__":
    print("⚠️  警告：这将删除所有包含中文的注释")
    confirm = input("确认继续吗？(输入 yes 继续): ")
    
    if confirm.lower() == 'yes':
        remove_chinese_comments_simple()
        print("\n✅ 处理完成！原始文件已备份为 .bak 文件")
    else:
        print("操作取消")