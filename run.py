#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import shlex
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent


def _run_stamp() -> str:
    # month/day/hour/min to distinguish runs
    return datetime.now().strftime("%m-%d-%H-%M")


def _load_run_config() -> dict:
    try:
        from config import run_config  # type: ignore

        return {
            "TRAIN_DEFAULTS": getattr(run_config, "TRAIN_DEFAULTS", {}) or {},
            "TRAIN_FLAG_OVERRIDES": getattr(run_config, "TRAIN_FLAG_OVERRIDES", {}) or {},
            "TRAIN_EXTRA_ARGS": getattr(run_config, "TRAIN_EXTRA_ARGS", []) or [],
            "INFER_DEFAULTS": getattr(run_config, "INFER_DEFAULTS", {}) or {},
            "TRAIN_BATCH_SIZES": getattr(run_config, "TRAIN_BATCH_SIZES", {}) or {},
        }
    except Exception:
        return {
            "TRAIN_DEFAULTS": {},
            "TRAIN_FLAG_OVERRIDES": {},
            "TRAIN_EXTRA_ARGS": [],
            "INFER_DEFAULTS": {},
            "TRAIN_BATCH_SIZES": {},
        }


def _load_method_config(method: str) -> dict:
    try:
        mod = __import__(f"config.methods.{method}", fromlist=["*"])
        return {
            "TRAIN_FLAG_OVERRIDES": getattr(mod, "TRAIN_FLAG_OVERRIDES", {}) or {},
            "TRAIN_EXTRA_ARGS": getattr(mod, "TRAIN_EXTRA_ARGS", []) or [],
            "INFER_DEFAULTS": getattr(mod, "INFER_DEFAULTS", {}) or {},
        }
    except Exception:
        return {
            "TRAIN_FLAG_OVERRIDES": {},
            "TRAIN_EXTRA_ARGS": [],
            "INFER_DEFAULTS": {},
        }


def _parse_task_ids(values: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for v in values:
        try:
            ids.append(int(v))
        except ValueError as e:
            raise SystemExit(f"Invalid task id: {v}") from e
    if not ids:
        raise SystemExit("No tasks specified.")
    return ids


def _benchmark_dir_name(benchmark: str) -> str:
    return "CoIN" if benchmark.lower() == "coin" else benchmark


def _resolve_paths(app_config: str):
    if app_config == "base":
        from config.paths.paths_config_base import (  # type: ignore
            BASE_MODEL_PATH,
            CLIP_PATH,
            IMAGE_FOLDER,
            DEEPSPEED_CONFIG,
            CHECKPOINT_DIR,
            RESULT_DIR,
        )
    elif app_config == "instruct":
        from config.paths.paths_config_instruct import (  # type: ignore
            BASE_MODEL_PATH,
            CLIP_PATH,
            IMAGE_FOLDER,
            DEEPSPEED_CONFIG,
            CHECKPOINT_DIR,
            RESULT_DIR,
        )
    else:
        raise SystemExit(f"Unknown --app-config: {app_config}")

    return {
        "BASE_MODEL_PATH": str(BASE_MODEL_PATH),
        "CLIP_PATH": str(CLIP_PATH),
        "IMAGE_FOLDER": str(IMAGE_FOLDER),
        "DEEPSPEED_CONFIG": str(DEEPSPEED_CONFIG),
        "CHECKPOINT_DIR": str(CHECKPOINT_DIR),
        "RESULT_DIR": str(RESULT_DIR),
    }


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _log_line(msg: str, *, log_file: Path, mirror: bool, lock: threading.Lock | None = None) -> None:
    _ensure_parent(log_file)
    line = msg if msg.endswith("\n") else msg + "\n"
    if lock:
        with lock:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line)
            if mirror:
                sys.stdout.write(line)
                sys.stdout.flush()
    else:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line)
        if mirror:
            sys.stdout.write(line)
            sys.stdout.flush()


def _apply_flag_overrides(cmd: list[str], overrides: dict) -> list[str]:
    """
    Override flags in a flat argv list.

    - For "--flag value": replace value if flag exists, else append.
    - For value==True: ensure flag exists (no value).
    - For value==False/None: do nothing.
    """
    out = list(cmd)
    for flag, value in (overrides or {}).items():
        if value is False or value is None:
            continue
        if value is True:
            if flag not in out:
                out.append(str(flag))
            continue

        value_s = str(value)
        try:
            i = out.index(str(flag))
        except ValueError:
            out.extend([str(flag), value_s])
            continue

        if i + 1 >= len(out):
            out.append(value_s)
        else:
            out[i + 1] = value_s
    return out


def _append_args(cmd: list[str], extra: list[str]) -> list[str]:
    if not extra:
        return cmd
    return [*cmd, *[str(x) for x in extra]]


def _tee_run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_file: Path, mirror: bool) -> int:
    """
    Run a command and tee *all* stdout/stderr into log_file.

    For multi-process launchers (e.g. deepspeed), piping at the OS level is more reliable
    than capturing via Python's stdout=PIPE.
    """
    _ensure_parent(log_file)
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
    log_str = shlex.quote(str(log_file))
    if mirror:
        bash_cmd = f"set -o pipefail; ({cmd_str}) 2>&1 | tee -a {log_str}"
    else:
        bash_cmd = f"set -o pipefail; ({cmd_str}) >> {log_str} 2>&1"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("COMMAND: " + cmd_str + "\n\n")

    proc = subprocess.run(
        ["bash", "-lc", bash_cmd],
        cwd=str(cwd),
        env=env,
    )
    return int(proc.returncode)


def _stream_run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    log_file: Path,
    lock: threading.Lock | None = None,
    mirror: bool = False,
) -> int:
    """
    Stream a subprocess' stdout/stderr to both console and log_file.

    This is used for python inference/eval subprocesses where stdout=PIPE is reliable.
    """
    _ensure_parent(log_file)
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\nCOMMAND: " + cmd_str + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if lock:
                with lock:
                    if mirror:
                        sys.stdout.write(line)
                    f.write(line)
                    f.flush()
            else:
                if mirror:
                    sys.stdout.write(line)
                f.write(line)
                f.flush()
        return int(proc.wait())


def _build_env(app_config: str) -> dict[str, str]:
    env = os.environ.copy()
    env["APP_CONFIG"] = app_config
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    # Sanitize thread env vars to avoid libgomp warnings.
    def _sanitize_threads(name: str, default: str = "1") -> None:
        v = env.get(name)
        if v is None or v == "":
            env[name] = default
            return
        if not str(v).strip().isdigit():
            env[name] = default

    _sanitize_threads("OMP_NUM_THREADS", "8")
    _sanitize_threads("MKL_NUM_THREADS", env.get("OMP_NUM_THREADS", "8"))
    return env


def _get_task_config(benchmark: str, task_id: int) -> dict:
    from config.benchmarks import BENCHMARKS  # type: ignore

    if benchmark not in BENCHMARKS:
        raise SystemExit(f"Benchmark '{benchmark}' not found. Available: {list(BENCHMARKS.keys())}")
    tasks = BENCHMARKS[benchmark]
    if task_id < 0 or task_id >= len(tasks):
        raise SystemExit(f"Task {task_id} not in benchmark '{benchmark}' (0-{len(tasks)-1})")
    return tasks[task_id]


def _build_train_command(
    task: dict,
    *,
    gpus: str,
    port: int,
    debug: bool,
    method: str,
    paths: dict,
    batch_size: int,
) -> list[str]:
    cmd = [
        "deepspeed",
        f"--include=localhost:{gpus}",
        f"--master_port={port}",
        "backbone/llava/train/train_mem.py",
        "--deepspeed",
        paths["DEEPSPEED_CONFIG"],
        "--lora_enable",
        "True",
        "--lora_r",
        "64",
        "--lora_alpha",
        "128",
        "--mm_projector_lr",
        "2e-5",
        "--expert_num",
        "8",
        "--model_name_or_path",
        paths["BASE_MODEL_PATH"],
        "--freeze_mm_mlp_adapter",
        "True",
        "--version",
        "v1",
        "--data_path",
        str(task["train_data_path"]),
        "--image_folder",
        paths["IMAGE_FOLDER"],
        "--vision_tower",
        paths["CLIP_PATH"],
        "--text_tower",
        paths["CLIP_PATH"],
        "--cur_task",
        str(task["cur_task"]),
        "--mm_projector_type",
        "mlp2x_gelu",
        "--mm_vision_select_layer",
        "-2",
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "pad",
        "--group_by_modality_length",
        "True",
        "--bf16",
        "True",
        "--output_dir",
        str(task["output_dir"]),
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        str(1 if debug else batch_size),
        "--per_device_eval_batch_size",
        str(1 if debug else batch_size),
        "--gradient_accumulation_steps",
        "1",
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "epoch",
        "--learning_rate",
        "2e-4",
        "--weight_decay",
        "0.",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "1",
        "--tf32",
        "True",
        "--model_max_length",
        "2048",
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        "4",
        "--lazy_preprocess",
        "True",
        "--report_to",
        "none",
        "--method",
        method,
    ]

    # Task0 uses pretrain projector; others load previous task checkpoint
    if "pretrain_mm_mlp_adapter" in task:
        cmd.extend(["--pretrain_mm_mlp_adapter", str(task["pretrain_mm_mlp_adapter"])])
    elif task.get("previous_task"):
        cmd.extend(["--previous_task_model_path", str(task["previous_task"])])
    else:
        raise SystemExit("Task config missing 'pretrain_mm_mlp_adapter' or 'previous_task'.")

    return [str(x) for x in cmd]


# ===== Inference/Evaluation (ported from tools/eval_task.py) =====
INFERENCE_METHOD_MAP = {"ScienceQA": "scienceqa"}
EVAL_TASK_MAP = {
    "ScienceQA": "scienceqa",
    "TextVQA": "textvqa",
    "ImageNet": "imagenet",
    "GQA": "gqa",
    "VizWiz": "vizwiz",
    "Grounding": "grounding",
    "VQAv2": "vqav2",
    "OCRVQA": "ocrvqa",
}


def _format_args(template: list, **kwargs) -> list[str]:
    formatted: list[str] = []
    for arg in template:
        if isinstance(arg, str) and "{" in arg and "}" in arg:
            try:
                formatted.append(arg.format(**{k: str(v) for k, v in kwargs.items()}))
            except KeyError:
                formatted.append(arg)
        else:
            formatted.append(str(arg))
    return formatted


def _run_inference_one_chunk(
    task: dict,
    *,
    model_path: str,
    paths: dict,
    gpu_id: int,
    chunks: int,
    chunk_idx: int,
    output_file: Path,
    clmethod: str,
    temperature: str,
    conv_mode: str,
):
    eval_config = task["eval"]
    method = INFERENCE_METHOD_MAP.get(task["name"], "default")

    cmd = [
        sys.executable,
        "-m",
        "backbone.llava.eval.model_unified",
        method,
        "--clmethod",
        str(clmethod),
        "--model-path",
        str(model_path),
        "--model-base",
        paths["BASE_MODEL_PATH"],
        "--question-file",
        str(task["test_data_path"]),
        "--image-folder",
        paths["IMAGE_FOLDER"],
        "--answers-file",
        str(output_file),
        "--num-chunks",
        str(chunks),
        "--chunk-idx",
        str(chunk_idx),
        "--temperature",
        str(temperature),
        "--conv-mode",
        str(conv_mode),
    ]
    if paths["CLIP_PATH"]:
        cmd.extend(["--text-tower", paths["CLIP_PATH"]])

    if eval_config.get("inference_args"):
        cmd.extend(_format_args(eval_config["inference_args"], **task))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Note: output capture happens at caller level via _stream_run.
    subprocess.run([str(x) for x in cmd], env=env, cwd=str(PROJECT_ROOT), check=True)


def _run_inference_parallel(
    task: dict,
    *,
    model_path: str,
    paths: dict,
    gpus: list[int],
    result_dir: Path,
    clmethod: str,
    temperature: str,
    conv_mode: str,
):
    chunks = len(gpus)
    print_lock = threading.Lock()
    failed: list[int] = []

    def run_chunk(idx: int) -> bool:
        output_file = result_dir / f"{chunks}_{idx}.jsonl"
        try:
            # Build the same command as _run_inference_one_chunk but stream its output.
            eval_config = task["eval"]
            method = INFERENCE_METHOD_MAP.get(task["name"], "default")
            cmd = [
                sys.executable,
                "-m",
                "backbone.llava.eval.model_unified",
                method,
                "--clmethod",
                str(clmethod),
                "--model-path",
                str(model_path),
                "--model-base",
                paths["BASE_MODEL_PATH"],
                "--question-file",
                str(task["test_data_path"]),
                "--image-folder",
                paths["IMAGE_FOLDER"],
                "--answers-file",
                str(output_file),
                "--num-chunks",
                str(chunks),
                "--chunk-idx",
                str(idx),
                "--temperature",
                str(temperature),
                "--conv-mode",
                str(conv_mode),
            ]
            if paths["CLIP_PATH"]:
                cmd.extend(["--text-tower", paths["CLIP_PATH"]])
            if eval_config.get("inference_args"):
                cmd.extend(_format_args(eval_config["inference_args"], **task))

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[idx])

            rc = _stream_run(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                log_file=result_dir / "_infer_process_log.txt",
                lock=print_lock,
            )
            if rc != 0:
                raise RuntimeError(f"inference subprocess failed with code {rc}")
            with print_lock:
                print(f"  ✅ Chunk {idx} completed on GPU {gpus[idx]}")
            return True
        except Exception as e:
            with print_lock:
                print(f"  ❌ Chunk {idx} failed on GPU {gpus[idx]}: {e}")
            return False

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=chunks) as ex:
        futs = {ex.submit(run_chunk, i): i for i in range(chunks)}
        for fut in concurrent.futures.as_completed(futs):
            idx = futs[fut]
            try:
                ok = fut.result()
                if not ok:
                    failed.append(idx)
            except Exception:
                failed.append(idx)

    if failed:
        raise RuntimeError(f"Failed chunks: {failed}")


def _run_evaluation(task: dict, *, result_file: Path, output_dir: Path):
    eval_config = task["eval"]
    eval_task = EVAL_TASK_MAP.get(task["name"])
    if eval_task is None:
        raise SystemExit(f"No evaluation task mapping for {task['name']}")

    cmd = [sys.executable, "-m", "backbone.llava.eval.eval_unified", eval_task]
    if eval_task != "gqa":
        cmd.extend(["--result-file", str(result_file), "--output-dir", str(output_dir)])

    if eval_config.get("eval_args"):
        task_copy = dict(task)
        task_copy.pop("output_dir", None)
        task_copy.pop("result_file", None)
        cmd.extend(
            _format_args(
                eval_config["eval_args"],
                result_file=str(result_file),
                output_dir=str(output_dir),
                **task_copy,
            )
        )

    print("Evaluation command:", " ".join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], cwd=str(PROJECT_ROOT), check=True)


def _infer_y_label(model_path: str, checkpoint_task: str) -> str:
    return checkpoint_task if model_path == "" else "custom"


def cmd_train(args: argparse.Namespace) -> int:
    # benchmark configs are imported in *this* process
    os.environ["APP_CONFIG"] = str(args.app_config)
    env = _build_env(args.app_config)
    paths = _resolve_paths(args.app_config)
    task_ids = _parse_task_ids(args.tasks)
    stamp = _run_stamp()
    cfg = _load_run_config()
    method_cfg = _load_method_config(args.method)
    mirror = bool(getattr(args, "console", False))

    flag_overrides = dict(cfg.get("TRAIN_FLAG_OVERRIDES", {}))
    flag_overrides.update(method_cfg.get("TRAIN_FLAG_OVERRIDES", {}))
    extra_args = list(cfg.get("TRAIN_EXTRA_ARGS", [])) + list(method_cfg.get("TRAIN_EXTRA_ARGS", []))
    batch_sizes = cfg.get("TRAIN_BATCH_SIZES", {})

    for task_id in task_ids:
        task = _get_task_config(args.benchmark, task_id)
        bs = None
        if isinstance(batch_sizes, dict):
            bs = (batch_sizes.get(args.benchmark) or {}).get(task_id)
        if bs is None:
            bs = task.get("batch_size")
        if bs is None:
            raise SystemExit(
                f"Missing batch_size for benchmark={args.benchmark} task={task_id}. "
                f"Set in config/run_config.py TRAIN_BATCH_SIZES."
            )
        log_path = (
            PROJECT_ROOT
            / "output"
            / "train"
            / "text"
            / args.method
            / args.benchmark
            / f"task{task_id:02d}"
            / f"run_{stamp}.txt"
        )

        # Write header to file; mirror only if requested
        _log_line(f"TRAIN method={args.method} benchmark={args.benchmark} task={task_id}", log_file=log_path, mirror=mirror)
        _log_line(f"log: {log_path}", log_file=log_path, mirror=mirror)

        cmd = _build_train_command(
            task,
            gpus=args.gpus,
            port=args.port,
            debug=args.debug,
            method=args.method,
            paths=paths,
            batch_size=int(bs),
        )
        cmd = _apply_flag_overrides(cmd, flag_overrides)
        cmd = _append_args(cmd, extra_args)

        rc = _tee_run(cmd, cwd=PROJECT_ROOT, env=env, log_file=log_path, mirror=mirror)
        if rc != 0:
            return rc

    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    # benchmark configs are imported in *this* process
    os.environ["APP_CONFIG"] = str(args.app_config)
    env = _build_env(args.app_config)
    paths = _resolve_paths(args.app_config)
    task_ids = _parse_task_ids(args.tasks)
    stamp = _run_stamp()
    mirror = bool(getattr(args, "console", False))

    method_cfg = _load_method_config(args.method)
    if "--clmethod" not in sys.argv:
        m_cl = method_cfg.get("INFER_DEFAULTS", {}).get("clmethod")
        if m_cl:
            args.clmethod = m_cl

    if args.model_path:
        model_path = args.model_path
        checkpoint_task = args.checkpoint_task
    else:
        checkpoint_dir = paths["CHECKPOINT_DIR"]
        benchmark_dir = _benchmark_dir_name(args.benchmark)
        ckpt_name = f"Task{args.checkpoint_task}{args.checkpoint_suffix}"
        model_path = str(Path(checkpoint_dir) / benchmark_dir / ckpt_name)
        checkpoint_task = args.checkpoint_task

    y = _infer_y_label(args.model_path, checkpoint_task)
    failed: list[int] = []

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        raise SystemExit("Invalid --gpus. Example: --gpus 0,1")

    for task_id in task_ids:
        task = _get_task_config(args.benchmark, task_id)

        log_path = (
            PROJECT_ROOT
            / "output"
            / "infer"
            / "text"
            / args.method
            / args.benchmark
            / f"{y}_to_{task_id}_{stamp}.txt"
        )

        def _do_infer_eval() -> int:
            # mirror original behavior: results go to RESULT_DIR/<task_name>/<stage>
            result_dir = Path(paths["RESULT_DIR"]) / task["name"] / args.stage
            result_dir.mkdir(parents=True, exist_ok=True)

            print("Running inference in parallel...")
            _run_inference_parallel(
                task,
                model_path=model_path,
                paths=paths,
                gpus=gpus,
                result_dir=result_dir,
                clmethod=args.clmethod,
                temperature=args.temperature,
                conv_mode=args.conv_mode,
            )

            print("\nMerging results...")
            merged_file = result_dir / "merge.jsonl"
            with open(merged_file, "w", encoding="utf-8") as out:
                chunks = len(gpus)
                for idx in range(chunks):
                    chunk_file = result_dir / f"{chunks}_{idx}.jsonl"
                    if chunk_file.exists():
                        out.write(chunk_file.read_text(encoding="utf-8"))
                    else:
                        print(f"Warning: {chunk_file} not found")

            print("\n📈 Running evaluation...")
            _run_evaluation(task, result_file=merged_file, output_dir=result_dir)
            return 0

        _ensure_parent(log_path)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"MODEL: {model_path}\n")
            f.write(f"BENCHMARK: {args.benchmark}\n")
            f.write(f"TASK: {task_id}\n")
            f.write(f"GPUS: {args.gpus}\n")
            f.write(f"STAGE: {args.stage}\n\n")

        # Dedicated lock for this task's logging (avoid interleaving across chunks)
        log_lock = threading.Lock()

        try:
            _log_line("=" * 60, log_file=log_path, mirror=mirror, lock=log_lock)
            _log_line(
                f"INFER method={args.method} benchmark={args.benchmark} y={y} -> x={task_id}",
                log_file=log_path,
                mirror=mirror,
                lock=log_lock,
            )
            _log_line(f"model: {model_path}", log_file=log_path, mirror=mirror, lock=log_lock)
            _log_line(f"log: {log_path}", log_file=log_path, mirror=mirror, lock=log_lock)
            _log_line("=" * 60, log_file=log_path, mirror=mirror, lock=log_lock)

            # mirror original behavior: results go to RESULT_DIR/<task_name>/<stage>
            result_dir = Path(paths["RESULT_DIR"]) / task["name"] / args.stage
            result_dir.mkdir(parents=True, exist_ok=True)

            print("Running inference in parallel...")
            # Stream each chunk subprocess into the same log file
            # by pointing the per-chunk streaming log to `log_path`.
            def _run_infer_chunks():
                chunks = len(gpus)
                failed_chunks: list[int] = []

                def run_chunk(idx: int) -> bool:
                    output_file = result_dir / f"{chunks}_{idx}.jsonl"
                    eval_config = task["eval"]
                    method = INFERENCE_METHOD_MAP.get(task["name"], "default")
                    cmd = [
                        sys.executable,
                        "-m",
                        "backbone.llava.eval.model_unified",
                        method,
                        "--clmethod",
                        str(args.clmethod),
                        "--model-path",
                        str(model_path),
                        "--model-base",
                        paths["BASE_MODEL_PATH"],
                        "--question-file",
                        str(task["test_data_path"]),
                        "--image-folder",
                        paths["IMAGE_FOLDER"],
                        "--answers-file",
                        str(output_file),
                        "--num-chunks",
                        str(chunks),
                        "--chunk-idx",
                        str(idx),
                        "--temperature",
                        str(args.temperature),
                        "--conv-mode",
                        str(args.conv_mode),
                    ]
                    if paths["CLIP_PATH"]:
                        cmd.extend(["--text-tower", paths["CLIP_PATH"]])
                    if eval_config.get("inference_args"):
                        cmd.extend(_format_args(eval_config["inference_args"], **task))

                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpus[idx])
                    rc = _stream_run(cmd, cwd=PROJECT_ROOT, env=env, log_file=log_path, lock=log_lock, mirror=mirror)
                    return rc == 0

                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
                    futs = {ex.submit(run_chunk, i): i for i in range(len(gpus))}
                    for fut in concurrent.futures.as_completed(futs):
                        idx = futs[fut]
                        ok = False
                        try:
                            ok = fut.result()
                        except Exception:
                            ok = False
                        if ok:
                            with log_lock:
                                _log_line(
                                    f"  ✅ Chunk {idx} completed on GPU {gpus[idx]}",
                                    log_file=log_path,
                                    mirror=mirror,
                                    lock=log_lock,
                                )
                        else:
                            with log_lock:
                                _log_line(
                                    f"  ❌ Chunk {idx} failed on GPU {gpus[idx]}",
                                    log_file=log_path,
                                    mirror=mirror,
                                    lock=log_lock,
                                )
                            failed_chunks.append(idx)

                if failed_chunks:
                    raise RuntimeError(f"Failed chunks: {failed_chunks}")

            _run_infer_chunks()

            _log_line("\nMerging results...", log_file=log_path, mirror=mirror, lock=log_lock)
            merged_file = result_dir / "merge.jsonl"
            with open(merged_file, "w", encoding="utf-8") as out:
                chunks = len(gpus)
                for idx in range(chunks):
                    chunk_file = result_dir / f"{chunks}_{idx}.jsonl"
                    if chunk_file.exists():
                        out.write(chunk_file.read_text(encoding="utf-8"))
                    else:
                        _log_line(f"Warning: {chunk_file} not found", log_file=log_path, mirror=mirror, lock=log_lock)

            _log_line("\n📈 Running evaluation...", log_file=log_path, mirror=mirror, lock=log_lock)
            eval_task = EVAL_TASK_MAP.get(task["name"])
            if eval_task is None:
                raise RuntimeError(f"No evaluation task mapping for {task['name']}")
            eval_cmd = [sys.executable, "-m", "backbone.llava.eval.eval_unified", eval_task]
            if eval_task != "gqa":
                eval_cmd.extend(["--result-file", str(merged_file), "--output-dir", str(result_dir)])
            if task["eval"].get("eval_args"):
                task_copy = dict(task)
                task_copy.pop("output_dir", None)
                task_copy.pop("result_file", None)
                eval_cmd.extend(
                    _format_args(
                        task["eval"]["eval_args"],
                        result_file=str(merged_file),
                        output_dir=str(result_dir),
                        **task_copy,
                    )
                )
            rc = _stream_run(eval_cmd, cwd=PROJECT_ROOT, env=os.environ.copy(), log_file=log_path, lock=log_lock, mirror=mirror)
            if rc != 0:
                raise RuntimeError(f"evaluation subprocess failed with code {rc}")

            rc = 0
        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n❌ Task {task_id} failed: {e}\n")
            if mirror:
                print(f"❌ Task {task_id} failed: {e}")
            rc = 1

        if rc != 0:
            failed.append(task_id)

    if failed:
        if mirror:
            print(f"❌ Failed tasks: {failed}")
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="run.py", description="Single entrypoint: train / infer")
    sub = parser.add_subparsers(dest="command", required=True)

    cfg = _load_run_config()

    p_train = sub.add_parser("train", help="Train tasks")
    p_train.add_argument("tasks", nargs="*", help="Task IDs to train (e.g. 0 1 2)")
    p_train.add_argument("--benchmark", default="coin")
    p_train.add_argument("--gpus", default="0,1")
    p_train.add_argument("--port", type=int, default=29601)
    p_train.add_argument("--debug", action="store_true")
    p_train.add_argument("--method", default="hide_llava")
    p_train.add_argument("--console", action="store_true", help="Mirror output to console (default: write only to output/ logs)")
    p_train.add_argument(
        "--app-config",
        default=os.environ.get("APP_CONFIG", "instruct"),
        choices=["base", "instruct"],
    )
    p_train.set_defaults(**{k: v for k, v in cfg["TRAIN_DEFAULTS"].items() if k != "handler"})
    p_train.set_defaults(train_flag_overrides=cfg["TRAIN_FLAG_OVERRIDES"])
    p_train.set_defaults(train_extra_args=cfg["TRAIN_EXTRA_ARGS"])
    p_train.set_defaults(handler=cmd_train)

    p_infer = sub.add_parser("infer", help="Infer+eval tasks")
    p_infer.add_argument("tasks", nargs="*", help="Task IDs to evaluate (e.g. 0 1 2)")
    p_infer.add_argument("--benchmark", default="coin")
    p_infer.add_argument("--gpus", default="0,1")
    p_infer.add_argument("--checkpoint-task", default="7")
    p_infer.add_argument("--checkpoint-suffix", default="_llava_lora")
    p_infer.add_argument("--model-path", default="")
    p_infer.add_argument("--stage", default="MoELoRA")
    p_infer.add_argument("--clmethod", default="hide_llava")
    p_infer.add_argument("--temperature", default="0")
    p_infer.add_argument("--conv-mode", default="vicuna_v1")
    p_infer.add_argument("--method", default="hide_llava")
    p_infer.add_argument("--console", action="store_true", help="Mirror output to console (default: write only to output/ logs)")
    p_infer.add_argument(
        "--app-config",
        default=os.environ.get("APP_CONFIG", "instruct"),
        choices=["base", "instruct"],
    )
    p_infer.set_defaults(**{k: v for k, v in cfg["INFER_DEFAULTS"].items() if k != "handler"})
    p_infer.set_defaults(handler=cmd_infer)

    args = parser.parse_args()
    rc = int(args.handler(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

