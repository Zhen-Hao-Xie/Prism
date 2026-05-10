# config/benchmarks/__init__.py
from .CoIN import COIN_TASKS
from .UCIT import UCIT_TASKS

# Add more benchmarks here if needed.
# from .other import OTHER_TASKS

BENCHMARKS = {
    "coin": COIN_TASKS,
    "ucit": UCIT_TASKS,
}

# Must match task list lengths in BENCHMARKS.
BENCHMARK_TASK_NUM: dict[str, int] = {name: len(tasks) for name, tasks in BENCHMARKS.items()}