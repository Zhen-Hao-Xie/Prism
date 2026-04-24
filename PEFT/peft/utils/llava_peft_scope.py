# -*- encoding: utf-8 -*-
"""
LLaVA 等多模态模型上 PEFT 的 **路径级** 过滤：按 ``named_modules`` 前缀的分段名跳过某些子树。

``exclude_module_path_segments``（见 ``PeftConfig``）语义：

- ``None``：使用本模块内置的 LLaVA 默认集合（CLIP 双塔 + ``mm_projector`` + ``vision_resampler``）。
- ``[]``：不做路径过滤，仅按各 tuner 的 ``target_modules`` 规则匹配（可注入到 CLIP 等任意子模块）。
- 非空 ``list``：仅当路径分段中出现列表中的任一名称时跳过（各方法自行配置）。
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Sequence

# ``LlavaMetaModel`` / ``LlavaLlamaModel`` 下常见、且常与 LLM 子层同名的非解码器分支
DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS: FrozenSet[str] = frozenset(
    {
        "text_tower",
        "vision_tower",
        "mm_projector",
        "vision_resampler",
    }
)


def should_skip_peft_path(module_key: str, exclude_module_path_segments: Optional[Sequence[str]]) -> bool:
    """
    若应在该 ``module_key`` 上跳过 PEFT 注入，返回 True。

    ``exclude_module_path_segments`` 与 ``PeftConfig`` / 训练侧 ``ModelArguments`` 字段对齐。
    """
    if not module_key:
        return False
    if exclude_module_path_segments is None:
        active: FrozenSet[str] = DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS
    elif len(exclude_module_path_segments) == 0:
        return False
    else:
        active = frozenset(exclude_module_path_segments)
    parts = module_key.split(".")
    return any(p in active for p in parts)


def should_skip_peft_on_llava_path(module_key: str) -> bool:
    """兼容旧 API：等价于 ``exclude_module_path_segments is None``（仅 LLaVA 默认排除）。"""
    return should_skip_peft_path(module_key, None)
