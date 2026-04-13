# config/benchmarks/__init__.py
from .CoIN import COIN_TASKS
from .UCIT import UCIT_TASKS

# 可以继续添加其他benchmark
# from .other import OTHER_TASKS

BENCHMARKS = {
    "coin": COIN_TASKS,
    "ucit": UCIT_TASKS,
}