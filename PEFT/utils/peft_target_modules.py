"""
PEFT ``target_modules`` 的预设与解析（按子模块名后缀，如 ``q_proj``、``gate_proj``）。

在 ``config/methods/<method>.py`` 的 ``METHOD_CONFIG`` 中主要使用 ``peft_target_modules``；旧配置中的
``lora_target_modules`` 仍被识别，含义相同。

- **预设字符串**（不区分大小写）：
  - ``attention`` 或 ``attn``：注意力 Linear（Vicuna/LLaMA 上一般为 ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``）；
  - ``ffn`` 或 ``mlp``：FFN（``gate_proj`` / ``up_proj`` / ``down_proj``）；
  - ``attn_and_ffn``（别名 ``attn_ffn`` / ``attn+ffn`` / ``transformer``）：上述 attention ∪ FFN（**不含** ``lm_head`` 等词表投影；HiDe 等建议用此，避免对 ``lm_head`` 注入导致路径/形状问题）；
  - ``linear`` / ``all`` / ``full``：在 ``exclude_module_path_segments`` 过滤后，对 LLM 内**全部** ``nn.Linear`` 收集子模块名（与参考 SAME ``find_all_linear_names`` 一类行为）。
- **或** 子模块名列表，如 ``["q_proj", "o_proj", "down_proj"]``，会与当前模型中实际出现名**取交集**。

**未在配置中设置时**，与仓库内原各 PEFT/LoRA 方法习惯一致，默认 ``attention``。

命令行：``--peft_target_modules ffn`` 或子模块名逗号分隔、JSON 列表
``["q_proj","v_proj"]``。旧名 ``lora_target_modules`` 在 ``METHOD_CONFIG`` 或动态属性中仍有效。
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Set

# Vicuna-7B / LLaMA 2 等常见命名
PEFT_TARGET_MODULE_PRESETS: dict[str, List[str]] = {
    "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "ffn": ["gate_proj", "up_proj", "down_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}

# 历史别名（旧代码或说明里可能写 LoRA 专用名）
PEFT_LORA_TARGET_PRESETS = PEFT_TARGET_MODULE_PRESETS


def resolve_peft_target_spec_to_allowed_suffixes(raw: Any) -> Optional[Set[str]]:
    """
    将 ``peft_target_modules`` 配置解析为**允许的子模块名（后缀）集合**（多作用于 LoRA/同类 adapter）。

    返回 ``None`` 表示不筛选：除 path exclude 外，对扫描到的所有 Linear 后缀都允许。
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set, frozenset)):
        s = {str(x).strip() for x in raw if str(x).strip()}
        if not s:
            raise ValueError("peft_target_modules 空列表不合法，请显式使用预设 'all' 或子模块名")
        return s
    s0 = str(raw).strip()
    if not s0:
        return None
    low = s0.lower()
    if low in ("all", "full", "linear", "*", "any"):
        return None
    if low == "attn":
        return set(PEFT_TARGET_MODULE_PRESETS["attention"])
    if low in ("attn_and_ffn", "attn_ffn", "attn+ffn", "transformer", "transformer_blocks"):
        return set(PEFT_TARGET_MODULE_PRESETS["attention"]) | set(PEFT_TARGET_MODULE_PRESETS["ffn"])
    if low in PEFT_TARGET_MODULE_PRESETS:
        return set(PEFT_TARGET_MODULE_PRESETS[low])
    if s0.lstrip().startswith("["):
        try:
            arr = json.loads(s0)
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 peft_target_modules 为 JSON 列表: {e}") from e
        if not isinstance(arr, (list, tuple)):
            raise ValueError("peft_target_modules 的 JSON 需为 list")
        s = {str(x).strip() for x in arr if str(x).strip()}
        if not s:
            raise ValueError("peft_target_modules 的 JSON 列表为空，请显式使用预设 'all'")
        return s
    if "," in s0 or re.search(r"\s{2,}", s0):
        return {p.strip() for p in re.split(r"[\s,]+", s0) if p.strip()}
    return {s0}


# 历史函数名
resolve_lora_target_spec_to_allowed_suffixes = resolve_peft_target_spec_to_allowed_suffixes


__all__ = [
    "PEFT_TARGET_MODULE_PRESETS",
    "PEFT_LORA_TARGET_PRESETS",
    "resolve_peft_target_spec_to_allowed_suffixes",
    "resolve_lora_target_spec_to_allowed_suffixes",
]
