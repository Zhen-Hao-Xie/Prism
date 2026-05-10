import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from backbone.shared.multimodal.data_processor import get_model_name_from_path

from backbone.shared.data import InferenceDataset


@dataclass
class InferenceContext:
    args: Any
    tokenizer: Any
    model: Any
    image_processor: Any
    model_name: str


class BaseInferenceAdapter(ABC):
    @abstractmethod
    def on_model_ready(self, context: InferenceContext) -> None:
        pass

    @abstractmethod
    def infer_one(self, sample: Dict[str, Any], context: InferenceContext) -> Dict[str, str]:
        pass


class InferenceEngine:
    def __init__(self, adapter: BaseInferenceAdapter) -> None:
        self.adapter = adapter

    def _load_model(self, args: Any) -> Tuple[Any, Any, Any, str]:
        from common.load_model import load_model_for_inference

        _m = str(getattr(args, "method", "") or "").strip().lower()
        if _m == "zeroshot":
            mb = getattr(args, "model_base", None)
            if not mb:
                raise ValueError(
                    "method=zeroshot 需要 --model-base（LLaVA 权重）；不加载 CL adapter checkpoint，无需 --model-path。"
                )
            mp = os.path.expanduser(str(mb).strip())
            if str(getattr(args, "model_path", "") or "").strip():
                print(
                    "[infer] method=zeroshot：忽略 --model-path，仅从 --model-base 加载纯 LLaVA。",
                    flush=True,
                )
            model_name = getattr(args, "model_name", None) or get_model_name_from_path(mp)
        else:
            model_path = os.path.expanduser(args.model_path)
            mp = model_path
            model_name = getattr(args, "model_name", None) or get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_model_for_inference(
            mp,
            args.model_base,
            model_name,
            method=args.method,
            text_tower=getattr(args, "text_tower", None),
            benchmark=getattr(args, "benchmark", None),
            task_num=getattr(args, "cl_task_num", None),
        )
        return tokenizer, model, image_processor, model_name

    def _build_dataset(self, args: Any) -> InferenceDataset:
        return InferenceDataset(
            question_file=args.question_file,
            num_chunks=args.num_chunks,
            chunk_idx=args.chunk_idx,
        )

    def run(self, args: Any) -> None:
        tokenizer, model, image_processor, model_name = self._load_model(args)
        context = InferenceContext(
            args=args,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            model_name=model_name,
        )
        self.adapter.on_model_ready(context)

        dataset = self._build_dataset(args)
        samples = dataset.samples
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

        batch_size = getattr(args, "batch_size", 1)
        print(
            f"Evaluating {len(samples)} questions (batch size {batch_size}), saving to {answers_file}")

        with open(answers_file, "w", encoding="utf-8") as file:
            for i in tqdm(range(0, len(samples), batch_size)):
                batch_samples = samples[i:i + batch_size]
                if hasattr(self.adapter, "infer_batch"):
                    outputs = self.adapter.infer_batch(batch_samples, context)
                else:
                    outputs = [self.adapter.infer_one(
                        sample, context) for sample in batch_samples]

                for sample, output in zip(batch_samples, outputs):
                    result = {
                        "question_id": sample["question_id"],
                        "prompt": output["prompt"],
                        "text": output["text"],
                        "answer_id": shortuuid.uuid(),
                        "model_id": context.model_name,
                        "metadata": {},
                    }
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")
                if getattr(args, "flush_each_line", False):
                    file.flush()


def read_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def default_generate(model: Any, input_ids: torch.Tensor, images: Optional[torch.Tensor], args: Any, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    generate_kwargs = {
        "do_sample": getattr(args, "temperature", 0.0) > 0,
        "temperature": getattr(args, "temperature", 0.0),
        "top_p": getattr(args, "top_p", None),
        "num_beams": getattr(args, "num_beams", 1),
        "max_new_tokens": getattr(args, "max_new_tokens", 128),
        "use_cache": True,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
    generate_kwargs.update(kwargs)

    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,  # 使用关键字参数
            images=images,
            **generate_kwargs,
        )


def decode_new_tokens(tokenizer: Any, output_ids: torch.Tensor, input_ids: torch.Tensor) -> str:
    input_len = input_ids.shape[1]
    return tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
