import transformers
from dataclasses import dataclass, field
from typing import Any, Optional


def merge_method_config_into(obj: Any, method: Optional[str] = None) -> None:
    """
    将 `config/methods/<method>.py` 里的 `METHOD_CONFIG` 写入 obj：
    - 仅当 obj 上不存在该属性，或当前值为 None 时写入（命令行 / 训练脚本显式传入优先）。
    这样方法专属字段（如 simple_prompt 的 num_prompt_tokens）不必出现在共享的 ModelArguments 里。
    注意：`task_num` / `expert_num` 由 benchmark 决定，见 `merge_benchmark_task_num_into`，此处不再从 METHOD_CONFIG 合并。
    """
    m = (method or getattr(obj, "method", None) or "").strip().lower()
    if not m or m == "base":
        return
    try:
        mod = __import__(f"config.methods.{m}", fromlist=["METHOD_CONFIG"])
        mc = getattr(mod, "METHOD_CONFIG", None)
        if not isinstance(mc, dict):
            return
        mc = dict(mc)
        for key, val in mc.items():
            if key in ("expert_num", "task_num"):
                continue
            if not hasattr(obj, key):
                setattr(obj, key, val)
                continue
            if getattr(obj, key) is None:
                setattr(obj, key, val)
    except Exception:
        return


def merge_benchmark_task_num_into(obj: Any, benchmark: Optional[str] = None) -> None:
    """
    当未显式提供 `task_num`（命令行为 None）时，用 `config.benchmarks.BENCHMARK_TASK_NUM[benchmark]` 填充。
    显式 `--task_num` 优先，不覆盖。
    """
    if getattr(obj, "task_num", None) is not None:
        return
    bm = (benchmark or getattr(obj, "benchmark", None) or "").strip().lower()
    if not bm:
        return
    try:
        from config.benchmarks import BENCHMARK_TASK_NUM  # type: ignore
    except Exception:
        return
    n = BENCHMARK_TASK_NUM.get(bm)
    if n is None:
        return
    setattr(obj, "task_num", int(n))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    text_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_text_select_layer: Optional[int] = field(default=-1)
    cur_task: Optional[int] = field(default=0)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    task_embedding_dim: Optional[int] = field(default=64)
    benchmark: Optional[str] = field(
        default=None,
        metadata={
            "help": "Benchmark：ucit / coin 等；与任务数一致，用于在未传 --task_num 时自动设置 task_num",
        },
    )
    task_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "持续学习场景下的任务/专家数量；不传时由 --benchmark 对应 config/benchmarks 中的任务数补全",
        },
    )
    method: str = field(
        default="hide_llava",
        metadata={"help": "CL method: hide_llava / same / simple_prompt / smolora 等"},
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    memory_data_path: str = field(default=None, metadata={"help": "Path to the memory data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(default=512)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    

def load_config():
    """解析命令行参数，返回三个配置对象"""
    import sys

    for i, arg in enumerate(sys.argv):
        if arg == "--expert_num" and i + 1 < len(sys.argv):
            sys.argv[i] = "--task_num"
        elif arg.startswith("--expert_num="):
            sys.argv[i] = "--task_num=" + arg.split("=", 1)[1]

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    return parser.parse_args_into_dataclasses()