"""
与 ``SMoLoRA/scripts/SMoLoRA/ins_gen.py`` 中 **single** 分支（``list_instruction`` + mean pooling）一致，
在训练进程内用 ``transformers`` 加载 MiniLM 并生成 ``[T, D]`` 指令矩阵，无需单独跑作者脚本。

若训练环境**不能访问** ``huggingface.co``：请把 ``smolora_sentence_transformer_model`` 设为**本机模型目录**
（内含 ``config.json`` 与 tokenizer 文件），或改用 ``ins_emb_path`` / 关闭内建走 CLIP（见 ``compute_instruction_embeddings`` 抛错说明）。
"""

from __future__ import annotations

import os
from typing import List, Sequence

import torch

# 与 ins_gen.py 中 ``list_instruction``（生成 ``ins_emb_single.pkl`` 的默认列表）逐字一致
INS_EMB_SINGLE_INSTRUCTIONS: List[str] = [
    "Answer with the option's letter from the given choices directly.",
    "Answer the question using a single word or phrase.",
    "What is happening in the presented picture?\nPlease describe it in one complete sentence.",
    "What is the object in the image?\nAnswer the question using a single word or phrase.",
    "Answer the question using a single word or phrase.",
    "Answer the question using a single word or phrase.",
    "What is the background of the image?\nAnswer the question using a single word or phrase.",
]


def _is_local_transformers_model_dir(model_name: str) -> bool:
    p = os.path.abspath(os.path.expanduser(model_name.strip()))
    return bool(os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")))


def _load_pretrained_or_raise(model_name: str, *, local_files_only: bool):
    from transformers import AutoModel, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as e:
        hint = (
            "加载 Sentence-Transformer / MiniLM 失败（常见于训练机无法访问 huggingface.co）。可选方案：\n"
            "  1) 在能联网的机器上用 ``huggingface-cli download sentence-transformers/all-MiniLM-L6-v2`` "
            "或浏览器下载快照，拷到训练机后，把 ``smolora_sentence_transformer_model`` 设为**该快照目录的绝对路径**"
            "（目录内需有 ``config.json`` 与 tokenizer 相关文件，此时框架会自动 ``local_files_only=True``，不再请求外网）；\n"
            "  2) 若可用镜像：启动前 ``export HF_ENDPOINT=https://hf-mirror.com``（以你环境可用的镜像为准）；\n"
            "  3) 配置 ``ins_emb_path`` 指向已生成的 ``ins_emb_single.pkl``，可跳过内建句向量；\n"
            "  4) 配置 ``smolora_builtin_sentence_ins_emb: False``，改用 CLIP ``text_tower`` 作为指令特征。\n"
            f"  原始错误: {type(e).__name__}: {e}"
        )
        raise RuntimeError(hint) from e
    return tokenizer, model


def mean_pooling(model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def compute_instruction_embeddings(
    sentences: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_only = _is_local_transformers_model_dir(model_name)
    if local_only:
        model_name = os.path.abspath(os.path.expanduser(model_name.strip()))
    tokenizer, model = _load_pretrained_or_raise(model_name, local_files_only=local_only)
    model.eval()
    model.to(device)
    encoded = tokenizer(
        list(sentences),
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out = model(**encoded)
    emb = mean_pooling(out, encoded["attention_mask"])
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return emb.cpu().float()


def compute_default_ins_emb_matrix(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: torch.device | None = None,
) -> torch.Tensor:
    return compute_instruction_embeddings(INS_EMB_SINGLE_INSTRUCTIONS, model_name=model_name, device=device)
