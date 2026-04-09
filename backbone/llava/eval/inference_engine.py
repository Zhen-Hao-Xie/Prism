import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from common.load_model import load_model_for_inference
from common.data_processor import get_model_name_from_path

from common.data_manager import InferenceDataset


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
        model_path = os.path.expanduser(args.model_path)
        model_name = getattr(args, "model_name",
                             None) or get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_model_for_inference(
            model_path,
            args.model_base,
            model_name,
            method=args.clmethod,  # ← 关键：传入 method 参数
            text_tower=getattr(args, "text_tower", None),
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

        print(f"Evaluating {len(samples)} questions, saving to {answers_file}")

        with open(answers_file, "w", encoding="utf-8") as file:
            for sample in tqdm(samples):
                output = self.adapter.infer_one(sample, context)
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


def default_generate(model: Any, input_ids: torch.Tensor, images: Optional[torch.Tensor], args: Any, **kwargs: Any) -> torch.Tensor:
    generate_kwargs = {
        "do_sample": getattr(args, "temperature", 0.0) > 0,
        "temperature": getattr(args, "temperature", 0.0),
        "top_p": getattr(args, "top_p", None),
        "num_beams": getattr(args, "num_beams", 1),
        "max_new_tokens": getattr(args, "max_new_tokens", 128),
        "use_cache": True,
    }
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
