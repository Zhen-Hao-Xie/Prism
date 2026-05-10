"""
Defaults for method: zeroshot（纯 LLaVA 基线，仅推理评估）

本方法不参与持续学习训练：勿使用 ``run.py train --method zeroshot``。
训练相关字段（``TRAIN_FLAG_OVERRIDES`` / ``TRAIN_BATCH_SIZES``）故意留空；
仅保留推理侧 ``INFER_DEFAULTS`` 供 ``run.py infer`` 等读取。
"""

TRAIN_FLAG_OVERRIDES: dict[str, str] = {}
TRAIN_EXTRA_ARGS: list[str] = []
TRAIN_BATCH_SIZES: dict = {}

INFER_DEFAULTS = {
    "batch_size": 12,
}

METHOD_CONFIG: dict = {}
