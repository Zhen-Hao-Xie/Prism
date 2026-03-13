import json
import re
from collections import defaultdict

def is_valid_option(answer: str) -> bool:
    """检查是否为有效的选项（A/B/C/D或其小写形式）"""
    if not answer:
        return False
    

    cleaned = re.sub(r'[\(\)\.\s]', '', answer.strip()).upper()
    return cleaned in ['A', 'B', 'C', 'D']

def normalize_option(answer: str) -> str:
    """规范化选项：提取出A/B/C/D的大写形式"""
    if not answer:
        return ""
    

    cleaned = re.sub(r'[^A-Za-z]', '', answer)
    if cleaned and cleaned[0].upper() in ['A', 'B', 'C', 'D']:
        return cleaned[0].upper()
    

    match = re.search(r'\b([A-Da-d])\b', answer)
    if match:
        return match.group(1).upper()
    
    return ""

def categorize_prediction(ground_truth: str, pred: str) -> str:
    """
    分类预测结果
    1. 完全正确：内容完全一致（不考虑格式）
    2. 形式轻微错误：内容正确但格式有差异
    3. 任务内形式正确：预测是A/B/C/D但答错，格式与真实答案一致
    4. 任务内形式错误：预测是A/B/C/D但答错，格式与真实答案不一致
    5. 严重错误：预测不在A/B/C/D范围内
    """

    gt_content = normalize_option(ground_truth)
    pred_content = normalize_option(pred)
    

    is_pred_valid = is_valid_option(pred)
    

    if not is_pred_valid:
        return '严重错误'
    

    if not gt_content:
        return '严重错误'
    

    content_match = gt_content == pred_content
    

    format_match = ground_truth.strip() == pred.strip()
    

    if content_match and format_match:
        return '完全正确'
    

    if content_match and not format_match:
        return '形式轻微错误'
    

    if is_pred_valid:

        if format_match:
            return '任务内形式正确'

        else:
            return '任务内形式错误'
    
    return '严重错误'

def analyze_json_file(filepath: str, analyze_correct: bool = True, analyze_incorrect: bool = True):
    """
    分析JSON文件
    analyze_correct: 是否分析correct数组
    analyze_incorrect: 是否分析incorrect数组
    """
    print(f"正在分析文件: {filepath}")
    
    categories = {
        '完全正确': 0,
        '形式轻微错误': 0,
        '任务内形式正确': 0,
        '任务内形式错误': 0,
        '严重错误': 0
    }
    
    total = 0
    details = defaultdict(list)
    
    try:

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件结构: {list(data.keys())}")
        

        if analyze_correct and 'correct' in data:
            correct_items = data['correct']
            print(f"correct数组样本数: {len(correct_items)}")
            
            for item in correct_items:
                if not isinstance(item, dict):
                    continue
                
                total += 1
                gt = str(item.get('ground_truth', ''))
                pred = str(item.get('pred', ''))
                question_id = item.get('question_id', f'C{total}')
                

                category = categorize_prediction(gt, pred)
                categories[category] += 1
                

                details[category].append({
                    'type': 'correct',
                    'question_id': question_id,
                    'ground_truth': gt,
                    'pred': pred,
                    'content_match': normalize_option(gt) == normalize_option(pred),
                    'format_match': gt.strip() == pred.strip()
                })
        

        if analyze_incorrect and 'incorrect' in data:
            incorrect_items = data['incorrect']
            print(f"incorrect数组样本数: {len(incorrect_items)}")
            
            for item in incorrect_items:
                if not isinstance(item, dict):
                    continue
                
                total += 1
                gt = str(item.get('ground_truth', ''))
                pred = str(item.get('pred', ''))
                question_id = item.get('question_id', f'I{total}')
                

                category = categorize_prediction(gt, pred)
                categories[category] += 1
                

                details[category].append({
                    'type': 'incorrect',
                    'question_id': question_id,
                    'ground_truth': gt,
                    'pred': pred,
                    'content_match': normalize_option(gt) == normalize_option(pred),
                    'format_match': gt.strip() == pred.strip()
                })
        
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败: {e}")
        print("文件可能不是有效的JSON格式")
        return None, None
    except FileNotFoundError:
        print(f"错误: 文件不存在: {filepath}")
        return None, None
    except Exception as e:
        print(f"错误: 读取文件时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    return categories, details, total

def print_statistics(categories: dict, details: dict, total: int):
    """打印统计结果"""
    print("\n" + "=" * 60)
    print("预测结果详细统计")
    print("=" * 60)
    
    if total == 0:
        print("没有找到有效样本")
        return
    
    print(f"总样本数: {total}")
    print("-" * 60)
    

    category_definitions = {
        '完全正确': "内容完全一致，格式也完全一致",
        '形式轻微错误': "内容正确但格式有差异（如大小写不同）",
        '任务内形式正确': "预测是A/B/C/D但答错，格式与真实答案一致",
        '任务内形式错误': "预测是A/B/C/D但答错，格式与真实答案不一致",
        '严重错误': "预测不是A/B/C/D中的任何一个"
    }
    

    print("分类统计:")
    print("-" * 60)
    for category in ['完全正确', '形式轻微错误', '任务内形式正确', '任务内形式错误', '严重错误']:
        count = categories[category]
        percent = count / total * 100
        definition = category_definitions.get(category, "")
        
        print(f"{category:15} ({definition})")
        print(f"  数量: {count:6d} | 百分比: {percent:6.2f}%")
        

        if details[category]:
            correct_count = sum(1 for d in details[category] if d['type'] == 'correct')
            incorrect_count = sum(1 for d in details[category] if d['type'] == 'incorrect')
            print(f"  来源: correct={correct_count}, incorrect={incorrect_count}")
        
        print()
    
    print("-" * 60)
    

    total_correct = categories['完全正确'] + categories['形式轻微错误']
    total_task_error = categories['任务内形式正确'] + categories['任务内形式错误']
    
    print("汇总指标:")
    print(f"1. 总正确率: {total_correct/total*100:.2f}%")
    print(f"   - 完全正确: {categories['完全正确']/total*100:.2f}%")
    print(f"   - 形式轻微错误: {categories['形式轻微错误']/total*100:.2f}%")
    print()
    
    if total_task_error > 0:
        print(f"2. 任务内错误率: {total_task_error/total*100:.2f}%")
        print(f"   - 任务内形式正确: {categories['任务内形式正确']/total_task_error*100:.2f}% ({categories['任务内形式正确']}/{total_task_error})")
        print(f"   - 任务内形式错误: {categories['任务内形式错误']/total_task_error*100:.2f}% ({categories['任务内形式错误']}/{total_task_error})")
    else:
        print("2. 任务内错误率: 0.00% (无任务内错误)")
    print()
    
    print(f"3. 严重错误率: {categories['严重错误']/total*100:.2f}%")
    
    print("-" * 60)
    

    print("错误类型分析:")
    print()
    

    if categories['严重错误'] > 0:
        print(f"严重错误 ({categories['严重错误']}个):")

        pred_types = defaultdict(int)
        for detail in details['严重错误']:
            pred = detail['pred']
            if pred == "":
                pred_types["空字符串"] += 1
            elif len(pred) == 1 and pred.upper() not in ['A', 'B', 'C', 'D']:
                pred_types[f"单字符但非A/B/C/D: '{pred}'"] += 1
            elif re.search(r'[A-Za-z]', pred) and not re.search(r'[A-Da-d]', pred):
                pred_types[f"字母但非A/B/C/D: '{pred}'"] += 1
            elif re.search(r'\d', pred):
                pred_types[f"数字: '{pred}'"] += 1
            else:
                pred_types[f"其他: '{pred}'"] += 1
        
        for pred_type, count in sorted(pred_types.items(), key=lambda x: x[1], reverse=True):
            percent = count / categories['严重错误'] * 100
            print(f"  {pred_type}: {count}个 ({percent:.1f}%)")
        

        print(f"\n严重错误示例 (前5个):")
        for i, detail in enumerate(details['严重错误'][:5]):
            print(f"  {i+1}. ID={detail['question_id']}, 类型={detail['type']}")
            print(f"     真实: '{detail['ground_truth']}'")
            print(f"     预测: '{detail['pred']}'")
    

    if total_task_error > 0:
        print(f"\n任务内错误 ({total_task_error}个):")
        

        confusion_matrix = defaultdict(lambda: defaultdict(int))
        for category in ['任务内形式正确', '任务内形式错误']:
            for detail in details[category]:
                gt = normalize_option(detail['ground_truth'])
                pred = normalize_option(detail['pred'])
                if gt and pred:
                    confusion_matrix[gt][pred] += 1
        
        print("混淆矩阵 (真实答案 → 预测答案):")
        options = ['A', 'B', 'C', 'D']
        print("    ", end="")
        for p in options:
            print(f"{p:>6}", end="")
        print()
        
        for gt in options:
            print(f"{gt:4}", end=" ")
            for pred in options:
                if gt == pred:
                    print(f"{'-':>6}", end="")
                else:
                    count = confusion_matrix[gt].get(pred, 0)
                    print(f"{count:>6}", end="")
            print()
        

        all_errors = []
        for gt in confusion_matrix:
            for pred in confusion_matrix[gt]:
                count = confusion_matrix[gt][pred]
                if gt != pred:
                    all_errors.append((f"{gt}→{pred}", count))
        
        if all_errors:
            all_errors.sort(key=lambda x: x[1], reverse=True)
            print(f"\n常见错误模式 (前10):")
            for error_pattern, count in all_errors[:10]:
                percent = count / total_task_error * 100
                print(f"  {error_pattern}: {count}次 ({percent:.1f}%)")
    
    print("-" * 60)
    

    print("各类别示例:")
    for category in ['完全正确', '形式轻微错误', '任务内形式正确', '任务内形式错误', '严重错误']:
        if details[category]:
            print(f"\n{category}示例 (前2个):")
            for i, detail in enumerate(details[category][:2]):
                print(f"  {i+1}. ID={detail['question_id']}, 类型={detail['type']}")
                print(f"     真实: '{detail['ground_truth']}'")
                print(f"     预测: '{detail['pred']}'")
                if category in ['形式轻微错误', '任务内形式正确', '任务内形式错误']:
                    print(f"     内容匹配: {detail['content_match']}, 格式匹配: {detail['format_match']}")

def quick_analyze(filepath: str):
    """
    快速分析版本
    """
    print(f"分析文件: {filepath}")
    print("-" * 50)
    
    categories, details, total = analyze_json_file(filepath)
    
    if total is None or total == 0:
        print("分析失败或没有数据")
        return
    
    print_statistics(categories, details, total)

def compare_correct_incorrect(filepath: str):
    """
    分别比较correct和incorrect数组的统计
    """
    print(f"分别分析correct和incorrect数组: {filepath}")
    print("=" * 80)
    

    for array_name in ['correct', 'incorrect']:
        print(f"\n分析{array_name.upper()}数组:")
        print("-" * 40)
        
        categories, details, total = analyze_json_file(
            filepath, 
            analyze_correct=(array_name == 'correct'),
            analyze_incorrect=(array_name == 'incorrect')
        )
        
        if total is None or total == 0:
            print(f"  {array_name}数组没有数据或分析失败")
            continue
        
        print(f"  {array_name}数组统计:")
        for category in ['完全正确', '形式轻微错误', '任务内形式正确', '任务内形式错误', '严重错误']:
            count = categories[category]
            percent = count / total * 100
            print(f"    {category}: {count} ({percent:.1f}%)")
        

        total_correct = categories['完全正确'] + categories['形式轻微错误']
        print(f"    正确率: {total_correct/total*100:.1f}%")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("请输入JSON文件路径: ").strip()
    
    if not filepath:
        print("错误: 请输入文件路径")
        return
    
    print("选择分析模式:")
    print("1. 快速分析（默认）")
    print("2. 分别比较correct和incorrect")
    print("3. 只分析correct数组")
    print("4. 只分析incorrect数组")
    
    choice = input("请输入选择 (1-4, 默认1): ").strip()
    
    if choice == '2':
        compare_correct_incorrect(filepath)
    elif choice == '3':
        categories, details, total = analyze_json_file(filepath, analyze_correct=True, analyze_incorrect=False)
        if total and total > 0:
            print_statistics(categories, details, total)
    elif choice == '4':
        categories, details, total = analyze_json_file(filepath, analyze_correct=False, analyze_incorrect=True)
        if total and total > 0:
            print_statistics(categories, details, total)
    else:
        quick_analyze(filepath)

if __name__ == "__main__":

    quick_analyze('/root/autodl-tmp/tangjt/PyMCIT/results/ScienceQA/MoELoRA/output.jsonl')