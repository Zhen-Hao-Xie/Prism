from .eval_default import BaseEvaluator
from .eval_caption import eval_single as eval_caption_single
from .eval_caption import create_coco_type, merge_captions
from .eval_deepseek_r1 import eval_single, deepseek_chat_final

import os
from typing import Any, Dict

import sys
sys.path.append(os.path.dirname(__file__))


class DeepSeekR1Evaluator(BaseEvaluator):
    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        ans_gt_file = eval_single(args.annotation_file, args.result_file, args)
        # Note: API call is commented out in original script, so we just run the local accuracy check
        return {"result_file": ans_gt_file}


class ImageNetREvaluator(DeepSeekR1Evaluator):
    name = "imagenetr"
    help_text = "Evaluate ImageNet-R results using DeepSeek R1 script"


class ArxivQAEvaluator(DeepSeekR1Evaluator):
    name = "arxivqa"
    help_text = "Evaluate ArxivQA results using DeepSeek R1 script"


class IconQAEvaluator(DeepSeekR1Evaluator):
    name = "iconqa"
    help_text = "Evaluate IconQA results using DeepSeek R1 script"


class CLEVREvaluator(DeepSeekR1Evaluator):
    name = "clevr"
    help_text = "Evaluate CLEVR results using DeepSeek R1 script"


class CaptionEvaluator(BaseEvaluator):
    def add_arguments(self, parser: Any) -> None:
        parser.add_argument("--annotation-file", required=True)
        parser.add_argument("--result-file", required=True)
        parser.add_argument("--output-dir", type=str, default=None)

    def evaluate(self, args: Any) -> Dict[str, Any]:
        output_file, total = create_coco_type(
            args.annotation_file, args.result_file, args.output_dir)
        ans_gt_file = os.path.join(args.output_dir, 'ans_gt.json')
        merge_captions(output_file, args.annotation_file, ans_gt_file)
        eval_caption_single(output_file, args.annotation_file, total, args)
        return {"total": total}


class Flickr30kEvaluator(CaptionEvaluator):
    name = "flickr30k"
    help_text = "Evaluate Flickr30k results using caption eval"


class VizcapEvaluator(CaptionEvaluator):
    name = "vizcap"
    help_text = "Evaluate VizWiz caption results using caption eval"

# VizWiz already has an evaluator in eval_unified using m4c, but if we want to use the UCIT caption evaluator,
# We can subclass it here or let the user decide.
