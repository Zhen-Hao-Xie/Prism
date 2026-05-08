"""
与 ``PEFT.utils.llava_peft_scope`` 对齐，供各 method 在扫描 ``named_modules``（如 ``_find_target_modules``）时
与 PEFT 注入使用同一套 ``exclude_module_path_segments`` 规则。
"""

from __future__ import annotations

from typing import Any, List

import torch.nn as nn

from PEFT.utils.llava_peft_scope import DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS, should_skip_peft_path
from PEFT.utils.peft_target_modules import resolve_peft_target_spec_to_allowed_suffixes


def should_skip_module_for_peft_scan(full_module_name: str, method_config: Any) -> bool:
    segs = getattr(method_config, "exclude_module_path_segments", None)
    return should_skip_peft_path(full_module_name, segs)


def module_path_outside_llm_for_peft(full_module_name: str) -> bool:
    return should_skip_peft_path(full_module_name, None)


module_path_under_clip_tower = module_path_outside_llm_for_peft


def collect_peft_target_linear_suffixes(model: nn.Module, method_config: Any) -> List[str]:
    _root = getattr(model, "_base_model", None) or model
    found: set = set()
    for name, module in _root.named_modules():
        if should_skip_module_for_peft_scan(name, method_config):
            continue
        if not isinstance(module, nn.Linear):
            continue
        found.add(name.split(".")[-1])
    spec = getattr(method_config, "peft_target_modules", None)
    if spec is None:
        spec = getattr(method_config, "lora_target_modules", None)
    if spec is None:
        spec = "attention"
    allow = resolve_peft_target_spec_to_allowed_suffixes(spec)
    if allow is None:
        out = sorted(found)
        if not out:
            print("    Warning: no in-scope nn.Linear for PEFT; check exclude_module_path_segments / model")
        else:
            print(f"    PEFT target_modules (all in-scope Linears, spec={spec!r}): {out}")
        return out
    out = sorted(found & allow)
    if not out and found:
        raise ValueError(
            f"peft_target_modules={spec!r} 与当前 LLM 中在 scope 内的 Linear 子模块名无交集。"
            f" 已扫描到后缀(示例)={sorted(found)[:32]}，允许集={allow}"
        )
    if not out and not found:
        raise ValueError("模型中在 scope 内未找到任何 nn.Linear，请检查与 exclude 配置。")
    print(f"    PEFT target_modules (spec={spec!r}): {out}")
    return out


collect_lora_target_linear_suffixes = collect_peft_target_linear_suffixes

__all__ = [
    "DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS",
    "should_skip_module_for_peft_scan",
    "module_path_outside_llm_for_peft",
    "module_path_under_clip_tower",
    "collect_peft_target_linear_suffixes",
    "collect_lora_target_linear_suffixes",
]
