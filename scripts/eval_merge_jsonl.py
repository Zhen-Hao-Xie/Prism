#!/usr/bin/env python3
"""
仅根据 merge.jsonl（或 merge.json）做评测，不跑推理。

路径约定（与 run.py 一致；自动兼容少一层 method 的旧路径）::

    规范: <RESULT_DIR>/<BACKBONE>/<BenchmarkDir>/<method>/<数据集任务名>/<stage>/merge.jsonl
    兼容: <RESULT_DIR>/<BACKBONE>/<BenchmarkDir>/<数据集任务名>/<stage>/merge.jsonl

例如::

    .../results/llava/UCIT/same/Flickr30k/last/merge.jsonl

由此解析 benchmark（UCIT/CoIN → ucit/coin）、数据集（TaskName，对应 BENCHMARKS 里任务的 ``name``），
再选择 ``EVAL_TASK_MAP`` 中的评测方式。

Flickr30k / Vizcap（COCO captions）：用 ``question_id`` 对齐标注里的 ``images[].file_name``  stem → ``image_id``，
并只对预测中出现的 ``image_id`` 过滤 GT，避免子集预测与全量 3000 条标注键不一致导致 pycocoevalcap 断言失败。

用法::

    python scripts/eval_merge_jsonl.py /path/to/merge.jsonl
    python scripts/eval_merge_jsonl.py /path/to/result_dir   # 自动找 merge.jsonl
    python scripts/eval_merge_jsonl.py merge.jsonl --benchmark ucit --task-id 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# 项目根：scripts/ 上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.benchmarks import BENCHMARKS  # noqa: E402
from config.benchmarks.sub_dataset import apply_use_sub_dataset_to_task  # noqa: E402


# 与 run.py 中 EVAL_TASK_MAP 保持一致
EVAL_TASK_MAP: Dict[str, str] = {
    "ScienceQA": "scienceqa",
    "TextVQA": "textvqa",
    "ImageNet": "imagenet",
    "GQA": "gqa",
    "VizWiz": "vizwiz",
    "Grounding": "grounding",
    "VQAv2": "vqav2",
    "OCRVQA": "ocrvqa",
    "ImageNet-R": "imagenetr",
    "ArxivQA": "arxivqa",
    "IconQA": "iconqa",
    "CLEVR": "clevr",
    "Flickr30k": "flickr30k",
    "Vizcap": "vizcap",
}


def _format_args(template: List[str], **kwargs: Any) -> List[str]:
    formatted: List[str] = []
    for arg in template:
        if isinstance(arg, str) and "{" in arg and "}" in arg:
            try:
                formatted.append(arg.format(**{k: str(v) for k, v in kwargs.items()}))
            except KeyError:
                formatted.append(arg)
        else:
            formatted.append(str(arg))
    return formatted


def _benchmark_key_from_dir(folder: str) -> str:
    f = folder.strip()
    if f.lower() == "coin":
        return "coin"
    if f.lower() == "ucit":
        return "ucit"
    return f.lower()


def _is_benchmark_dir_segment(name: str) -> bool:
    return name.strip().lower() in ("ucit", "coin")


def resolve_merge_path(arg: Path) -> Path:
    p = arg.expanduser().resolve()
    if p.is_dir():
        for name in ("merge.jsonl", "merge.json"):
            c = p / name
            if c.is_file():
                return c
        raise FileNotFoundError(f"目录下未找到 merge.jsonl / merge.json: {p}")
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def parse_result_layout(merge_path: Path) -> Tuple[str, str, str, Path]:
    """
    从 merge 文件路径解析 (benchmark_key, task_name, stage, result_dir)。

    自 ``stage`` 目录向上：规范为 ``…/<Benchmark>/<method>/<dataset>/<stage>``；
    若 ``…/<Benchmark>/<dataset>/<stage>``（少一层 method）仍可解析。
    """
    result_dir = merge_path.parent.resolve()
    stage = result_dir.name
    if not stage:
        raise ValueError("无效的 merge 父目录名（stage）。")

    up: List[str] = []
    cur = result_dir.parent
    for _ in range(12):
        if not cur or not cur.name:
            break
        up.append(cur.name)
        cur = cur.parent

    if len(up) < 2:
        raise ValueError(
            "无法从路径解析：父目录层级过浅。\n"
            "期望: .../<BACKBONE>/<UCIT|CoIN>/<method>/<TaskName>/<stage>/merge.jsonl\n"
            "  或: .../<BACKBONE>/<UCIT|CoIN>/<TaskName>/<stage>/merge.jsonl\n"
            "或使用 --benchmark 与 --task-id / --task-name。"
        )

    task_name: str
    bench_folder: str

    if len(up) >= 3 and _is_benchmark_dir_segment(up[2]):
        # …/Benchmark/<method>/<TaskName>/stage
        task_name = up[0]
        bench_folder = up[2]
    elif _is_benchmark_dir_segment(up[1]):
        # …/Benchmark/<TaskName>/stage（无 method，兼容旧落盘）
        task_name = up[0]
        bench_folder = up[1]
    else:
        raise ValueError(
            "无法从路径解析：在父路径上未找到 UCIT 或 CoIN 目录。\n"
            f"  自 stage 向上的片段: {up!r}\n"
            "请确认结果目录结构，或使用 --benchmark / --task-id。"
        )

    benchmark = _benchmark_key_from_dir(bench_folder)
    return benchmark, task_name, stage, result_dir


def find_task(
    benchmark: str,
    *,
    task_name: Optional[str] = None,
    task_id: Optional[int] = None,
) -> Dict[str, Any]:
    tasks = BENCHMARKS.get(benchmark)
    if not tasks:
        raise SystemExit(f"未知 benchmark: {benchmark!r}，可用: {list(BENCHMARKS.keys())}")

    if task_id is not None:
        if task_id < 0 or task_id >= len(tasks):
            raise SystemExit(f"task_id 越界: {task_id}（{benchmark} 共 {len(tasks)} 个任务，下标 0..{len(tasks) - 1}）")
        return dict(tasks[task_id])

    if task_name is None:
        raise SystemExit("未指定任务：请提供路径解析或 --task-name / --task-id")

    for t in tasks:
        if t["name"] == task_name:
            return dict(t)
    for t in tasks:
        if t["name"].lower() == task_name.lower():
            return dict(t)
    names = [t["name"] for t in tasks]
    raise SystemExit(f"在 benchmark={benchmark!r} 中找不到任务 {task_name!r}。可选: {names}")


def resolve_annotation_task(
    task: Dict[str, Any],
    benchmark: str,
    use_sub_dataset: Optional[bool],
) -> Dict[str, Any]:
    """对 UCIT 应用 _sub 路径；use_sub_dataset 为 None 时按文件是否存在自动选择。"""
    if benchmark != "ucit":
        return apply_use_sub_dataset_to_task(
            task, use_sub_dataset=False, benchmark=benchmark
        )

    t_false = apply_use_sub_dataset_to_task(
        task, use_sub_dataset=False, benchmark=benchmark
    )
    t_true = apply_use_sub_dataset_to_task(
        task, use_sub_dataset=True, benchmark=benchmark
    )
    p_false = t_false.get("eval_annotation_path")
    p_true = t_true.get("eval_annotation_path")

    if use_sub_dataset is True:
        return t_true
    if use_sub_dataset is False:
        return t_false

    # auto
    if isinstance(p_true, str) and os.path.isfile(p_true):
        return t_true
    if isinstance(p_false, str) and os.path.isfile(p_false):
        return t_false
    raise FileNotFoundError(
        f"找不到评测标注（已尝试 _sub 与非 _sub）：\n  {p_false}\n  {p_true}"
    )


def _build_qid_to_image_id(coco: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for im in coco.get("images", []):
        fn = str(im.get("file_name", ""))
        stem = os.path.splitext(fn)[0]
        out[str(stem)] = int(im["id"])
    return out


def _filter_coco_to_image_ids(data: Dict[str, Any], image_ids: set) -> Dict[str, Any]:
    imgs = [im for im in data.get("images", []) if int(im["id"]) in image_ids]
    iids = {int(im["id"]) for im in imgs}
    anns = [a for a in data.get("annotations", []) if int(a["image_id"]) in iids]
    return {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": imgs,
        "annotations": anns,
    }


def eval_caption_merge_subset(
    merge_path: Path,
    annotation_path: str,
    output_dir: Path,
) -> None:
    """Flickr30k / Vizcap：question_id → image_id，过滤 GT 后走 eval_caption。"""
    from backbone.shared.eval.eval_caption import eval_single, merge_captions

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco_full = json.load(f)

    qid_to_iid = _build_qid_to_image_id(coco_full)
    pred_rows: List[Dict[str, Any]] = []
    used_iids: set = set()
    missing_q: List[str] = []

    with open(merge_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("question_id", ""))
            iid = qid_to_iid.get(qid)
            if iid is None:
                missing_q.append(qid)
                continue
            pred_rows.append({"image_id": int(iid), "caption": str(obj.get("text", ""))})
            used_iids.add(int(iid))

    if missing_q:
        print(
            f"[warn] {len(missing_q)} 条预测的 question_id 未在 COCO images 中找到（示例 {missing_q[:5]}）"
        )

    if not pred_rows:
        raise SystemExit("没有可用的预测（question_id 均无法对齐到 COCO image）。")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "pred_coco_type_eval_subset.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_rows, f, indent=2, ensure_ascii=False)

    filtered = _filter_coco_to_image_ids(coco_full, used_iids)
    filtered_ann_path = output_dir / "gt_coco_filtered_for_eval.json"
    with open(filtered_ann_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    total = len(pred_rows)
    ans_gt_path = output_dir / "ans_gt.json"
    merge_captions(str(pred_path), str(filtered_ann_path), str(ans_gt_path))

    args = SimpleNamespace(output_dir=str(output_dir))
    eval_single(str(pred_path), str(filtered_ann_path), total, args)
    print(f"[ok] caption 评测完成 | samples={total} | output_dir={output_dir}")


def eval_ucit_deepseek_style_merge(
    merge_path: Path,
    annotation_path: str,
    output_dir: Path,
) -> None:
    """ImageNet-R / ArxivQA / IconQA / CLEVR：与 eval_deepseek_r1.eval_single 一致，question_id 统一为 str。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(annotation_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise SystemExit(f"标注应为 JSON 数组: {annotation_path}")
    annotations = {str(a["question_id"]): a for a in raw}

    results: List[Dict[str, Any]] = []
    with open(merge_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))

    total = len(results)
    right = 0
    answer_gt_file: List[Dict[str, Any]] = []
    missing = 0
    for result in results:
        qid = str(result.get("question_id", ""))
        ann = annotations.get(qid)
        if ann is None:
            missing += 1
            continue
        pred = str(result.get("text", ""))
        ground_truth = str(ann.get("answer", ""))
        if pred.upper() == ground_truth.upper():
            right += 1
        answer_gt_file.append({"pred": pred, "ground_truth": ground_truth})

    if missing:
        print(f"[warn] {missing} 条预测在标注中找不到 question_id")

    ans_gt_file = output_dir / "ans_gt.json"
    with open(ans_gt_file, "w", encoding="utf-8") as f:
        json.dump(answer_gt_file, f, ensure_ascii=False, indent=4)

    acc = 100.0 * right / total if total else 0.0
    summary = f"Samples: {total}\nAccuracy: {acc:.2f}%\n"
    print(summary)
    result_text = output_dir / "Result.text"
    with open(result_text, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[ok] 已写入 {result_text}")


def run_eval_unified_subprocess(
    task: Dict[str, Any],
    merge_path: Path,
    output_dir: Path,
) -> None:
    """走 backbone.shared.eval.eval_unified（与 run._run_evaluation 一致）。"""
    eval_task = EVAL_TASK_MAP.get(task["name"])
    if eval_task is None:
        raise SystemExit(f"无评测映射: task name={task['name']!r}")

    cmd = [sys.executable, "-m", "backbone.shared.eval.eval_unified", eval_task]
    if eval_task != "gqa":
        cmd.extend(["--result-file", str(merge_path), "--output-dir", str(output_dir)])

    eval_config = task.get("eval") or {}
    if eval_config.get("eval_args"):
        task_copy = dict(task)
        task_copy.pop("output_dir", None)
        task_copy.pop("result_file", None)
        cmd.extend(
            _format_args(
                eval_config["eval_args"],
                result_file=str(merge_path),
                output_dir=str(output_dir),
                **task_copy,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print("COMMAND:", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="根据 merge.jsonl 路径解析 benchmark/任务并评测（不推理）")
    parser.add_argument(
        "merge_or_dir",
        type=str,
        help="merge.jsonl 路径，或其所在结果目录（含 .../TaskName/stage/）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="评测中间产物与 Result.text 输出目录（默认与 merge 同目录）",
    )
    parser.add_argument("--benchmark", type=str, default=None, help="ucit / coin，覆盖路径解析")
    parser.add_argument("--task-name", type=str, default=None, help="任务名，如 Flickr30k（覆盖路径解析）")
    parser.add_argument("--task-id", type=int, default=None, help="任务下标 0..N-1（与 run train 下标一致）")
    parser.add_argument(
        "--use-sub-dataset",
        type=str,
        default=None,
        choices=("true", "false", "auto"),
        help="UCIT 标注是否使用 _sub 后缀；默认 auto 按文件是否存在",
    )

    args = parser.parse_args()
    merge_path = resolve_merge_path(Path(args.merge_or_dir))
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else merge_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark: str
    stage = ""

    if args.task_id is not None and args.benchmark:
        benchmark = str(args.benchmark).strip().lower()
        task = find_task(benchmark, task_id=args.task_id)
    elif args.benchmark and (args.task_id is not None or args.task_name):
        benchmark = str(args.benchmark).strip().lower()
        task = find_task(
            benchmark,
            task_name=args.task_name,
            task_id=args.task_id,
        )
    elif not args.benchmark and not args.task_name and args.task_id is None:
        benchmark, task_name, stage, _ = parse_result_layout(merge_path)
        task = find_task(benchmark, task_name=task_name)
        print(
            f"路径解析: benchmark={benchmark} task={task_name!r} stage={stage!r}\n"
            f"  merge={merge_path}"
        )
    else:
        raise SystemExit(
            "请任选其一：\n"
            "  (1) merge 路径符合 .../<BACKBONE>/<UCIT|CoIN>/<method>/<TaskName>/<stage>/merge.jsonl；或\n"
            "  (2) 显式 --benchmark 与 (--task-id 或 --task-name)。"
        )

    use_sub: Optional[bool] = None
    if args.use_sub_dataset == "true":
        use_sub = True
    elif args.use_sub_dataset == "false":
        use_sub = False
    elif args.use_sub_dataset in (None, "auto"):
        use_sub = None

    task_resolved = resolve_annotation_task(task, benchmark, use_sub)
    ann = task_resolved.get("eval_annotation_path")
    if not ann or not isinstance(ann, str):
        raise SystemExit("任务配置缺少 eval_annotation_path")

    eval_task = EVAL_TASK_MAP.get(task["name"])
    if eval_task is None:
        raise SystemExit(f"EVAL_TASK_MAP 未注册: {task['name']!r}")

    print(f"评测任务: {eval_task} | 标注: {ann}")

    if eval_task in ("flickr30k", "vizcap"):
        eval_caption_merge_subset(merge_path, ann, out_dir)
    elif eval_task in ("imagenetr", "arxivqa", "iconqa", "clevr"):
        eval_ucit_deepseek_style_merge(merge_path, ann, out_dir)
    else:
        run_eval_unified_subprocess(task_resolved, merge_path, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
