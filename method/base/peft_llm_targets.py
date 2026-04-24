"""
与 ``PEFT.peft.utils.llava_peft_scope`` 对齐，供各 method 在扫描 ``named_modules``（如 ``_find_target_modules``）时
与 PEFT 注入使用同一套 ``exclude_module_path_segments`` 规则。
"""

from __future__ import annotations

from typing import Any, Optional

from PEFT.peft.utils.llava_peft_scope import DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS, should_skip_peft_path


def should_skip_module_for_peft_scan(full_module_name: str, method_config: Any) -> bool:
    """按 ``method_config.exclude_module_path_segments`` 判断是否跳过该模块路径。"""
    segs = getattr(method_config, "exclude_module_path_segments", None)
    return should_skip_peft_path(full_module_name, segs)


def module_path_outside_llm_for_peft(full_module_name: str) -> bool:
    """等价于 ``exclude_module_path_segments is None``（仅 LLaVA 默认排除集）。"""
    return should_skip_peft_path(full_module_name, None)


module_path_under_clip_tower = module_path_outside_llm_for_peft

__all__ = [
    "DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS",
    "should_skip_module_for_peft_scan",
    "module_path_outside_llm_for_peft",
    "module_path_under_clip_tower",
]
