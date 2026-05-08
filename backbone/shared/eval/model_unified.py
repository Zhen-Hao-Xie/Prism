import argparse
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

try:
    from config.backbones.llava import DEFAULT_CONV_MODE
except Exception:
    DEFAULT_CONV_MODE = "vicuna_v1"


def _eval_preload_llama_tokenizer_from_argv() -> None:
    """在导入 torch 之前预热 LLaMA SPM tokenizer，避免与 PyTorch inductor 后台线程并发导致 sentencepiece 段错误。"""
    if os.environ.get("MCIT_SKIP_SPM_PRELOAD", "").strip() in ("1", "true", "yes"):
        return
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--model-base", type=str, default=None)
    ns, _ = probe.parse_known_args(sys.argv[1:])
    mb = ns.model_base
    if not mb:
        return
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(os.path.expanduser(mb), use_fast=False)


_eval_preload_llama_tokenizer_from_argv()

from backbone.shared.eval.inference_engine import InferenceEngine
from backbone.shared.eval.task_adapters import DefaultTaskAdapter, ScienceQATaskAdapter


class BaseModelMethod(ABC):
    name: str = ""
    help_text: str = ""

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    @abstractmethod
    def prepare_args(self, args: argparse.Namespace) -> None:
        pass

    @abstractmethod
    def build_engine(self, args: argparse.Namespace) -> InferenceEngine:
        pass


class ModelMethodRegistry:
    def __init__(self) -> None:
        self._methods: Dict[str, BaseModelMethod] = {}

    def register(self, method: BaseModelMethod) -> None:
        if method.name in self._methods:
            raise ValueError(f"Duplicated model method: {method.name}")
        self._methods[method.name] = method

    def add_subparsers(self, subparsers: Any) -> None:
        for name, method in self._methods.items():
            parser = subparsers.add_parser(name, help=method.help_text)
            add_common_arguments(parser)
            method.add_arguments(parser)
            parser.set_defaults(_method=name)

    def run(self, args: argparse.Namespace) -> None:
        method_name = getattr(args, "_method", None)
        if method_name not in self._methods:
            raise ValueError(f"Unknown model method: {method_name}")
        method = self._methods[method_name]
        method.prepare_args(args)
        engine = method.build_engine(args)
        engine.run(args)

    def available_methods(self) -> List[str]:
        return list(self._methods.keys())


class DefaultMethod(BaseModelMethod):
    name = "default"
    help_text = "General multimodal inference"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model-name", type=str,
                            default=None, help="Override model name")
        parser.add_argument("--num-workers", type=int, default=0)

    def prepare_args(self, args: argparse.Namespace) -> None:
        args.flush_each_line = True

    def build_engine(self, args: argparse.Namespace) -> InferenceEngine:
        return InferenceEngine(
            adapter=DefaultTaskAdapter(
                recursive_image_search=True,
                auto_mmtag_for_plain=False,
            )
        )


class AnswerMethod(BaseModelMethod):
    name = "answer"
    help_text = "Legacy answer inference mode"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--cur-task", type=int, default=0)

    def prepare_args(self, args: argparse.Namespace) -> None:
        args.model_name = None
        args.flush_each_line = False

    def build_engine(self, args: argparse.Namespace) -> InferenceEngine:
        return InferenceEngine(
            adapter=DefaultTaskAdapter(
                recursive_image_search=False,
                auto_mmtag_for_plain=True,
            )
        )


class ScienceQAMethod(BaseModelMethod):
    name = "scienceqa"
    help_text = "ScienceQA-specific inference mode"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--mm-text-select-layer", type=str, default="")
        parser.add_argument("--answer-prompter", action="store_true")
        parser.add_argument("--single-pred-prompt", action="store_true")

    def prepare_args(self, args: argparse.Namespace) -> None:
        args.model_name = None
        args.flush_each_line = True
        if not hasattr(args, "max_new_tokens") or args.max_new_tokens is None:
            args.max_new_tokens = 1024

    def build_engine(self, args: argparse.Namespace) -> InferenceEngine:
        return InferenceEngine(adapter=ScienceQATaskAdapter())


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=DEFAULT_CONV_MODE)
    parser.add_argument("--text-tower", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num-beams", dest="num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens",
                        dest="max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=[
            "base",
            "hide_llava",
            "same",
            "simple_prompt",
            "smolora",
            "moelora",
            "sefe",
            "zeroshot",
        ],
        help="持续学习方法（须与 checkpoint 训练时 method 一致；与 common/load_model.load_model_for_inference 对齐）",
    )
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="ucit / coin 等；用于与 checkpoint 对齐的 task_num（与 run.py infer 一致）",
    )
    parser.add_argument(
        "--task-num",
        dest="cl_task_num",
        type=int,
        default=None,
        help="显式指定 CL 的 task_num；若设置则优先于 --benchmark 推导",
    )


def build_registry() -> ModelMethodRegistry:
    registry = ModelMethodRegistry()
    registry.register(DefaultMethod())
    registry.register(AnswerMethod())
    registry.register(ScienceQAMethod())
    return registry


def build_parser(registry: ModelMethodRegistry) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified inference entrypoint for multiple model methods"
    )
    subparsers = parser.add_subparsers(dest="method", required=True)
    registry.add_subparsers(subparsers)
    return parser


def run_method_cli(method: BaseModelMethod, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser)
    method.add_arguments(parser)
    args = parser.parse_args()
    method.prepare_args(args)
    engine = method.build_engine(args)
    engine.run(args)


def main() -> None:
    registry = build_registry()
    parser = build_parser(registry)
    args = parser.parse_args()
    registry.run(args)


if __name__ == "__main__":
    main()
