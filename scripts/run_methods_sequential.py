#!/usr/bin/env python3
"""
Run ``run.py train`` then ``run.py infer`` for several CL methods in sequence.

Defaults for benchmark / GPUs / checkpoint suffix / etc. come from ``config/run_config.py``
(TRAIN_DEFAULTS + INFER_DEFAULTS); CLI overrides those. Intended for unattended overnight runs.

Examples::

    # Fixed TriGap recipe: same trains 2–9 then infer 0–9; others train+infer 0–9.
    # By default the script continues to the next method if one step fails (--no-continue-on-error to disable).
    python scripts/run_methods_sequential.py --preset trigap_four_methods

    # Shared schedule for arbitrary methods
    python scripts/run_methods_sequential.py -m same hide_llava olora

    python scripts/run_methods_sequential.py -m same replay_lora --train-tasks 0 1 2 \\
        --log-file /tmp/overnight.txt

    python scripts/run_methods_sequential.py --preset trigap_four_methods --dry-run
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (method_name, train_task_ids, infer_task_ids). Infer uses checkpoint from last train_task_id.
PRESETS: dict[str, list[tuple[str, list[int], list[int]]]] = {
    "trigap_four_methods": [
        ("same", [0,1,2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("hide_llava", list(range(10)), list(range(10))),
        ("replay_lora", list(range(10)), list(range(10))),
        ("ft_lora", list(range(10)), list(range(10))),
    ],
}

# Running as ``python scripts/this.py`` puts ``scripts/`` first on sys.path; repo imports need project root.
_ROOT = str(PROJECT_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_run_config_maps() -> tuple[dict, dict]:
    try:
        from config import run_config  # type: ignore

        td = getattr(run_config, "TRAIN_DEFAULTS", None) or {}
        inf = getattr(run_config, "INFER_DEFAULTS", None) or {}
        return dict(td), dict(inf)
    except Exception:
        return {}, {}


def _benchmark_task_count(benchmark: str) -> int:
    from config.benchmarks import BENCHMARKS  # type: ignore

    key = benchmark.lower()
    tasks = BENCHMARKS.get(key)
    if tasks is None:
        raise SystemExit(f"Unknown benchmark {benchmark!r}; known: {sorted(BENCHMARKS.keys())}")
    return len(tasks)


def _method_module_path(method: str) -> Path:
    return PROJECT_ROOT / "config" / "methods" / f"{method.strip().lower()}.py"


def _normalize_task_ids(values: Sequence[int] | None, benchmark: str) -> list[int]:
    if values is None or len(values) == 0:
        n = _benchmark_task_count(benchmark)
        return list(range(n))
    return list(values)


def _log_banner(logf, msg: str) -> None:
    line = f"\n{'=' * 72}\n{msg}\n{'=' * 72}\n"
    sys.stdout.write(line)
    logf.write(line)
    logf.flush()


def _run_command(cmd: list[str], logf) -> int:
    quoted = shlex.join(cmd)
    header = f"$ {quoted}\n"
    sys.stdout.write(header)
    logf.write(header)
    logf.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        logf.write(line)
        logf.flush()
    return proc.wait()


def _build_parser() -> argparse.ArgumentParser:
    train_d, infer_d = _load_run_config_maps()

    p = argparse.ArgumentParser(
        description="Sequentially train+infer multiple methods (uses run_config defaults)."
    )
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help=(
            "Use a built-in per-method train/infer schedule (overrides --methods / "
            "--train-tasks / --infer-tasks). See PRESETS in this file."
        ),
    )
    p.add_argument(
        "--list-presets",
        action="store_true",
        help="Print preset names and schedules, then exit.",
    )
    p.add_argument(
        "--methods",
        "-m",
        nargs="+",
        default=["same", "hide_llava", "olora"],
        help="Method names (ignored when --preset is set). Default: same hide_llava olora.",
    )
    p.add_argument(
        "--benchmark",
        "-b",
        default=str(train_d.get("benchmark", infer_d.get("benchmark", "trigap"))),
    )
    p.add_argument("--gpus", default=str(train_d.get("gpus", infer_d.get("gpus", "0,1"))))
    p.add_argument("--port", type=int, default=int(train_d.get("port", 29601)))
    p.add_argument(
        "--train-tasks",
        nargs="*",
        type=int,
        default=None,
        metavar="ID",
        help="Task IDs to train (default: all tasks for benchmark).",
    )
    p.add_argument(
        "--infer-tasks",
        nargs="*",
        type=int,
        default=None,
        metavar="ID",
        help="Task IDs for infer+eval (default: same as train tasks).",
    )
    p.add_argument(
        "--checkpoint-task",
        default=None,
        help="Checkpoint task index for infer (default: last ID in train tasks).",
    )
    p.add_argument(
        "--checkpoint-suffix",
        default=str(infer_d.get("checkpoint_suffix", "_llava")),
    )
    p.add_argument("--stage", default=str(infer_d.get("stage", "last")))
    p.add_argument("--temperature", default=str(infer_d.get("temperature", "0")))
    p.add_argument(
        "--use-sub-dataset",
        action=argparse.BooleanOptionalAction,
        default=bool(train_d.get("use_sub_dataset", infer_d.get("use_sub_dataset", False))),
    )
    p.add_argument("--debug", action="store_true", default=bool(train_d.get("debug", False)))
    p.add_argument("--train-only", action="store_true", help="Skip infer step.")
    p.add_argument("--infer-only", action="store_true", help="Skip train step.")
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a failed train/infer step, still run the remaining methods "
            "(default: true). Use --no-continue-on-error to stop on first failure."
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only; do not execute.")
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append full stdout log (default: output/<BACKBONE>/overnight/run_methods_<stamp>.txt).",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.list_presets:
        for name in sorted(PRESETS.keys()):
            print(f"[{name}]")
            for method, tr, inf in PRESETS[name]:
                print(f"  {method}: train {tr} | infer {inf} | ckpt_task={tr[-1]}")
        return 0

    jobs: list[tuple[str, list[int], list[int], str]]

    if args.preset:
        jobs = []
        for method, train_ids, infer_ids in PRESETS[args.preset]:
            m = method.strip().lower()
            if args.checkpoint_task is not None:
                ck = str(args.checkpoint_task)
            else:
                ck = str(train_ids[-1])
            jobs.append((m, train_ids, infer_ids, ck))
    else:
        train_tasks = _normalize_task_ids(args.train_tasks, args.benchmark)
        if args.infer_tasks is None or len(args.infer_tasks) == 0:
            infer_tasks = list(train_tasks)
        else:
            infer_tasks = list(args.infer_tasks)

        ck_shared = args.checkpoint_task
        if ck_shared is None:
            ck_shared = str(train_tasks[-1])
        else:
            ck_shared = str(ck_shared)

        jobs = [
            (m.strip().lower(), train_tasks, infer_tasks, ck_shared)
            for m in args.methods
        ]

    try:
        from config.backbone.llava import BACKBONE_ID  # type: ignore
    except Exception:
        BACKBONE_ID = "llava"

    stamp = datetime.now().strftime("%m-%d_%H-%M")
    log_path = args.log_file
    if log_path is None:
        log_path = (
            PROJECT_ROOT
            / "output"
            / BACKBONE_ID
            / "overnight"
            / f"run_methods_{stamp}.txt"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, str, int]] = []

    with open(log_path, "a", encoding="utf-8") as logf:
        job_summary = "; ".join(
            f"{jm}: train={jtr} infer={jinf} ck={jck}" for jm, jtr, jinf, jck in jobs
        )
        _log_banner(
            logf,
            f"run_methods_sequential start {datetime.now().isoformat()}\n"
            f"log_file={log_path}\nbenchmark={args.benchmark}\n"
            f"preset={args.preset!s}\n"
            f"jobs: {job_summary}",
        )

        for m, train_tasks, infer_tasks, ck_task in jobs:
            train_ids_str = [str(x) for x in train_tasks]
            infer_ids_str = [str(x) for x in infer_tasks]

            if not _method_module_path(m).is_file():
                msg = f"Skip unknown method (missing {_method_module_path(m)})"
                _log_banner(logf, msg)
                failures.append((m, "missing_method_config", 127))
                if not args.continue_on_error:
                    logf.write(f"EXIT non-zero due to {msg}\n")
                    return 127
                continue

            common_train = [
                sys.executable,
                str(PROJECT_ROOT / "run.py"),
                "train",
                "--benchmark",
                args.benchmark,
                "--method",
                m,
                "--gpus",
                args.gpus,
                "--port",
                str(args.port),
            ]
            if args.debug:
                common_train.append("--debug")
            if args.use_sub_dataset:
                common_train.append("--use-sub-dataset")
            else:
                common_train.append("--no-use-sub-dataset")
            train_cmd = common_train + train_ids_str

            common_infer = [
                sys.executable,
                str(PROJECT_ROOT / "run.py"),
                "infer",
                "--benchmark",
                args.benchmark,
                "--method",
                m,
                "--gpus",
                args.gpus,
                "--checkpoint-task",
                ck_task,
                "--checkpoint-suffix",
                args.checkpoint_suffix,
                "--stage",
                args.stage,
                "--temperature",
                args.temperature,
            ]
            if args.use_sub_dataset:
                common_infer.append("--use-sub-dataset")
            else:
                common_infer.append("--no-use-sub-dataset")
            infer_cmd = common_infer + infer_ids_str

            if m == "zeroshot":
                _log_banner(logf, f"Method {m}: train skipped (zeroshot is infer-only).")
                train_cmd = None

            steps: list[tuple[str, list[str] | None]] = []
            if not args.infer_only:
                steps.append(("train", train_cmd))
            if not args.train_only:
                steps.append(("infer", infer_cmd))

            for step_name, cmd in steps:
                if cmd is None:
                    continue
                _log_banner(logf, f"method={m} step={step_name}")
                if args.dry_run:
                    sys.stdout.write(shlex.join(cmd) + "\n")
                    logf.write(shlex.join(cmd) + "\n")
                    logf.flush()
                    continue
                rc = _run_command(cmd, logf)
                if rc != 0:
                    failures.append((m, step_name, rc))
                    logf.write(f"\nFAILED method={m} step={step_name} rc={rc}\n")
                    logf.flush()
                    if not args.continue_on_error:
                        _log_banner(logf, f"Stopping after failure rc={rc}")
                        return rc

        _log_banner(logf, f"run_methods_sequential end {datetime.now().isoformat()}")

    if failures:
        sys.stderr.write(f"Completed with failures: {failures}\n")
        return min(255, max(f[2] for f in failures))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
