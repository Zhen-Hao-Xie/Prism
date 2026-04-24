"""
各 CL 方法 ``METHOD_CONFIG`` 中共用的 PEFT 路径过滤：与 PEFT 内置 LLaVA 默认集合一致，避免两处漂移。
"""

from __future__ import annotations

from typing import Final, List

from PEFT.peft.utils.llava_peft_scope import DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS

EXCLUDE_FOR_LLM_ONLY_INJECTION: Final[List[str]] = list(DEFAULT_LLAVA_EXCLUDE_PATH_SEGMENTS)
