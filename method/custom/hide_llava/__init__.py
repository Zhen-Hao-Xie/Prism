"""
HiDe-LLaVA 方法包。

保持轻量，避免 import-time 副作用（例如提前导入 PEFT/torch）。
真正的注册在 `method/hide_llava/integration.py` 中按需触发。
"""
