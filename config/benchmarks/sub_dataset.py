"""UCIT 等 benchmark：在 JSON 指令路径上自动加/去 ``_sub`` 后缀（见 ``run_config.use_sub_dataset``）。"""
from __future__ import annotations

import copy
from typing import Any, Dict

# 任务 dict 中需要处理的字符串路径键（仅对以 .json 结尾且非目录的路径改写）
_TASK_JSON_PATH_KEYS = ("train_data_path", "test_data_path", "eval_annotation_path")


def json_path_with_sub_suffix(path: str) -> str:
    """``.../foo.json`` → ``.../foo_sub.json``；已是 ``*_sub.json`` 则不变。"""
    if not path or not isinstance(path, str):
        return path
    if not path.endswith(".json"):
        return path
    if path.endswith("_sub.json"):
        return path
    return f"{path[:-5]}_sub.json"


def json_path_without_sub_suffix(path: str) -> str:
    """``.../foo_sub.json`` → ``.../foo.json``；否则不变。"""
    if not path or not isinstance(path, str):
        return path
    if path.endswith("_sub.json"):
        return f"{path[:-9]}.json"
    return path


def apply_use_sub_dataset_to_task(
    task: Dict[str, Any],
    *,
    use_sub_dataset: bool,
    benchmark: str,
) -> Dict[str, Any]:
    """
    仅在 ``benchmark == ucit`` 时改写路径（与 CoIN 等全量 json 文件名兼容）。
    配置文件中应使用**不含** ``_sub`` 的规范名；``use_sub_dataset=True`` 时自动加 ``_sub``。
    """
    t = copy.deepcopy(task)
    if str(benchmark).strip().lower() != "ucit":
        return t
    for key in _TASK_JSON_PATH_KEYS:
        if key not in t or t[key] is None:
            continue
        p = t[key]
        if not isinstance(p, str):
            continue
        t[key] = json_path_with_sub_suffix(p) if use_sub_dataset else json_path_without_sub_suffix(p)
    return t
