import json

def load_jsonl(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_questions(file_path):
    """统一加载问题文件（自动识别 json/jsonl）"""
    if file_path.endswith('.jsonl'):
        return load_jsonl(file_path)
    return load_json(file_path)