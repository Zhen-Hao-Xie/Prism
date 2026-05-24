# PRISM: Multimodal Continual Instruction Tuning Toolbox

---

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#methods-implemented">Methods Implemented</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#license">License</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#contact">Contact</a>
</p>

---

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square&logo=python&color=3776AB&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-latest-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ready-green?style=flat-square)](https://www.deepspeed.ai/)

</div>

Welcome to **PRISM**, a PyTorch codebase for training and evaluating **multimodal large language models** (built around **LLaVA**) under **continual-learning** settings: multi-task instruction tuning with benchmarks such as **UCIT** and **CoIN**. Methods are organized under `method/custom/` and wired through a shared integration layer (`method/base/`) and factory (`method/factory.py`). Training and inference are driven by a single CLI entrypoint: `run.py`.

## Introduction

Many deployments require models to absorb **new tasks or domains over time** without full retraining from scratch. This repository provides an experimental framework for **continual instruction tuning** on vision-language models: PEFT adapters (LoRA-style tuners and variants), replay-style pipelines, regularization-based objectives, and mixture-of-experts style extensions, all registered as named **methods** and combined with benchmark-specific data paths and DeepSpeed-backed training scripts under `backbone/shared/`.

Typical workflow:

1. Point paths (base LLaVA weights, CLIP, datasets, checkpoints, logs) at your machine via `config/paths/paths.py`.
2. Choose **benchmark** (`ucit` / `coin`), **method**, and **task ids** in `config/run_config.py` or on the command line.
3. Run **`python run.py train …`** for sequential tasks, then **`python run.py infer …`** for evaluation; merge or analyze prediction JSONL with `scripts/eval_merge_jsonl.py` when needed.

## Methods Implemented

Each row is the **`--method`** string (folder under `method/custom/<name>/`). Implementations live in `integration.py` unless noted.

| Method id | Role |
|-----------|------|
| `hide_llava` | HiDe-style continual tuning integration for LLaVA. |
| `replay_lora` | Replay-assisted LoRA continual learning. |
| `ft_lora` | Full fine-tuning style training with LoRA hooks. |
| `olora` | Orthogonal / structured LoRA variant (`O-LoRA`-style integration). |
| `smolora` | Small LoRA configuration path. |
| `moelora` | Mixture-of-experts style LoRA routing. |
| `clmoe` | Continual learning with MoE-oriented wiring. |
| `modal_prompt` | Modal / prompt-based adaptation. |
| `ewc` | Elastic Weight Consolidation–style penalty on trainable parameters. |
| `disco` | Custom PEFT tuner integration (`PEFT/tuners/custom/disco.py`). |
| `same` | Same-task / baseline-style integration for comparisons. |
| `zeroshot` | Zero-shot evaluation path without incremental updates. |

New methods can be added by creating `method/custom/<your_method>/integration.py` and registering with `@CLMethodFactory.register("your_method")`.

## How To Use

### Clone

```bash
git clone <YOUR_REPO_URL> PRISM
cd PRISM
```

### Environment

Dependencies are listed under **`requirements/`** (see [`requirements/README.md`](requirements/README.md)).

```bash
# PyTorch (CUDA 11.8 example) then full train + eval stack
pip install -r requirements/torch.txt
pip install -r requirements.txt
```

Conda users: `conda env create -f environment.yml && conda activate prism`.

Align checkpoint paths with your LLaVA / CLIP weights after install.

### Paths and assets

Edit **`config/paths/`** so that at minimum these resolve on your system:

- **`BASE_MODEL_PATH`** — LLaVA (or compatible) base weights.
- **`CLIP_PATH`** — CLIP weights used by the multimodal stack.
- **`PRISM_ROOT`** — root that contains instructions and dataset layout expected by the benchmarks.
- **`CHECKPOINT_DIR`**, **`RESULT_DIR`**, **`LOG_DIR`** — outputs under the project (defaults point inside the repo).

Benchmark JSON annotations and image roots are configured under **`config/benchmarks/`** (e.g. UCIT / CoIN task lists).

### Train

Defaults live in **`config/run_config.py`** (`TRAIN_DEFAULTS`, `TRAIN_EXTRA_ARGS`). Method-specific flags and batch sizes are in **`config/methods/<method>.py`**.

```bash
python run.py train 0 1 2 --benchmark ucit --method ewc --gpus 0,1,2,3
```

- **`tasks`**: numeric task indices defined per benchmark (CoIN typically `0`–`7`, UCIT `0`–`5`).
- **`--use-sub-dataset` / `--no-use-sub-dataset`**: for UCIT, toggles `_sub` suffix on dataset JSON paths (see `utils/sub_dataset.py`).

Training invokes the backbone train pipeline (`backbone/shared/train/`) with DeepSpeed config from **`config/deepspeed/`** (see `DEEPSPEED_CONFIG` in `config/paths/paths.py`).

### Infer and evaluate

```bash
python run.py infer 5 --benchmark ucit --method ewc --checkpoint-task 5 --stage last --gpus 0,1
```

Adjust **`--checkpoint-task`**, **`--checkpoint-suffix`**, **`--stage`**, **`--conv-mode`**, and **`--temperature`** as needed; inference defaults are merged from **`config/run_config.py`** (`INFER_DEFAULTS`) and **`config/methods/<method>.py`** (`INFER_DEFAULTS`).

For aggregating or comparing JSONL outputs, use **`scripts/eval_merge_jsonl.py`** (see that script’s CLI for merge modes and metrics).

## Configuration

| File / area | Purpose |
|-------------|---------|
| `config/run_config.py` | Global CLI defaults for `train` / `infer`. |
| `config/methods/<method>.py` | Per-method training overrides, batch sizes, inference defaults. |
| `config/benchmarks/` | Benchmark definitions (tasks, paths, eval hooks). |
| `config/backbone/llava.py` | Backbone id and default conversation template (`DEFAULT_CONV_MODE`). |
| `config/paths/paths.py` | All filesystem roots for models, data, and outputs. |

Key training knobs (memory size, task schedule, etc.) follow each benchmark’s JSON/Python config; optimization hyperparameters are usually split between **`config/methods/*.py`** and the backbone train scripts.

## License



## Acknowledgments


## Contact

If there are any questions, please feel free to  propose new features by opening an issue or contact with the author: **Jun-Tao Tang**([juntao_tang@outlook.com](mailto:juntao_tang@outlook.com)) and **Shi-Yu Cheng**(). Enjoy the code.
