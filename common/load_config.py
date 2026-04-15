import transformers
from dataclasses import dataclass, field
from typing import Any, Optional


def merge_method_config_into(obj: Any, method: Optional[str] = None) -> None:
    """
    将 `config/methods/<method>.py` 里的 `METHOD_CONFIG` 写入 obj：
    - 仅当 obj 上不存在该属性，或当前值为 None 时写入（命令行 / 训练脚本显式传入优先）。
    这样方法专属字段（如 simple_prompt 的 num_prompt_tokens）不必出现在共享的 ModelArguments 里。
    METHOD_CONFIG 中旧键 expert_num 会映射为 task_num（仅当未提供 task_num 时）。
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
        if "task_num" not in mc and "expert_num" in mc:
            mc["task_num"] = mc["expert_num"]
        for key, val in mc.items():
            if key == "expert_num":
                continue
            if not hasattr(obj, key):
                setattr(obj, key, val)
                continue
            if getattr(obj, key) is None:
                setattr(obj, key, val)
    except Exception:
        return


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
    task_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "持续学习场景下的任务数量；不传时由 config/methods/<method>.py 的 METHOD_CONFIG 补全",
        },
    )
    method: str = field(
        default="hide_llava",
        metadata={"help": "CL method: hide_llava / same / simple_prompt 等"},
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