# config/benchmarks/__init__.py
from .CoIN import COIN_TASKS

# 可以继续添加其他benchmark
# from .other import OTHER_TASKS

BENCHMARKS = {
    "coin": COIN_TASKS,
    # "other": OTHER_TASKS,
}