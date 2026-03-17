import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from common.data_manager import ImageFinder
from config.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from common.conversation import SeparatorStyle, conv_templates
from common.data_processor import process_images, tokenizer_image_token
from common.generation_utils import KeywordsStoppingCriteria
from common.logging import disable_torch_init

from .inference_engine import (
    BaseInferenceAdapter,
    InferenceContext,
    decode_new_tokens,
    default_generate,
    read_image,
)


class DefaultTaskAdapter(BaseInferenceAdapter):
    def __init__(self, recursive_image_search: bool = True, auto_mmtag_for_plain: bool = False) -> None:
        self.recursive_image_search = recursive_image_search
        self.auto_mmtag_for_plain = auto_mmtag_for_plain
        self.image_finder: Optional[ImageFinder] = None

    def on_model_ready(self, context: InferenceContext) -> None:
        disable_torch_init()
        if self.recursive_image_search and getattr(context.args, "image_folder", None):
            self.image_finder = ImageFinder(context.args.image_folder)

        if self.auto_mmtag_for_plain:
            model_name = context.model_name
            conv_mode = context.args.conv_mode
            if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in conv_mode:
                context.args.conv_mode = conv_mode + "_mmtag"
                print(
                    f"It seems that this is a plain model, but it is not using a mmtag prompt, "
                    f"auto switching to {context.args.conv_mode}."
                )

    def _build_question_with_image_token(self, question: str, model_config: Any) -> str:
        if getattr(model_config, "mm_use_im_start_end", False):
            return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        return DEFAULT_IMAGE_TOKEN + "\n" + question

    def _resolve_image_path(self, image_file: str, image_folder: str) -> str:
        if self.image_finder is not None:
            return self.image_finder.find(image_file)
        return os.path.join(image_folder, image_file)

    def infer_one(self, sample: Dict[str, Any], context: InferenceContext) -> Dict[str, str]:
        question = sample["text"]
        qs = self._build_question_with_image_token(
            question, context.model.config)

        conv = conv_templates[context.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_file = sample["image"]
        image_path = self._resolve_image_path(
            image_file, context.args.image_folder)
        image = read_image(image_path)
        image_tensor = process_images(
            [image], context.image_processor, context.model.config)[0]

        input_ids = tokenizer_image_token(
            prompt, context.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(device="cuda", non_blocking=True)
        images = image_tensor.unsqueeze(0).to(
            dtype=torch.float16, device="cuda", non_blocking=True)

        output_ids = default_generate(
            context.model, input_ids, images, context.args)
        output_text = decode_new_tokens(
            context.tokenizer, output_ids, input_ids)

        return {
            "prompt": question,
            "text": output_text,
        }


class ScienceQATaskAdapter(BaseInferenceAdapter):
    def on_model_ready(self, context: InferenceContext) -> None:
        disable_torch_init()
        context.model.model.initialize_text_modules(context.args)

        model_path = os.path.expanduser(context.args.model_path)
        anchor_path = os.path.join(model_path, "non_lora_trainables.bin")

        if (not dist.is_initialized()) or (dist.get_rank() == 0):
            self._load_anchors_to_model(context.model, anchor_path)

        if dist.is_initialized():
            dist.barrier()
            self._sync_anchors_across_gpus(context.model)

    def _load_anchors_to_model(self, model: Any, anchor_path: str) -> None:
        if not os.path.exists(anchor_path):
            print(f"⚠️ Warning: {anchor_path} not found!")
            return

        state_dict = torch.load(anchor_path, map_location="cpu")
        prefix = "base_model.model."
        param_groups = ["image_anchors", "text_anchors",
                        "image_boundary", "text_boundary"]

        for group in param_groups:
            if not hasattr(model, group):
                print(f"  -> Model has no attribute {group}")
                continue

            param_list = getattr(model, group)
            if not isinstance(param_list, torch.nn.ParameterList):
                print(f"  -> {group} is not a ParameterList, skipping")
                continue

            loaded_count = 0
            for index in range(len(param_list)):
                key = f"{prefix}{group}.{index}"
                if key in state_dict:
                    param_list[index].data.copy_(
                        state_dict[key].to(param_list[index].device))
                    loaded_count += 1
                else:
                    print(f"  ⚠️ Key not found: {key}")
            print(f"  -> Loaded {loaded_count} parameters into {group}")

    def _sync_anchors_across_gpus(self, model: Any) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return

        for group_name in ["image_anchors", "text_anchors", "image_boundary", "text_boundary"]:
            if hasattr(model, group_name):
                group = getattr(model, group_name)
                if isinstance(group, torch.nn.ParameterList):
                    for param in group:
                        dist.broadcast(param.data, src=0)

    def infer_one(self, sample: Dict[str, Any], context: InferenceContext) -> Dict[str, str]:
        question = sample["text"]
        qs = question.replace("<image>", "").strip()
        cur_prompt = qs

        images = None
        if "image" in sample:
            image = read_image(os.path.join(
                context.args.image_folder, sample["image"]))
            image_tensor = context.image_processor.preprocess(
                image, return_tensors="pt")["pixel_values"][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            if getattr(context.model.config, "mm_use_im_start_end", False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                    DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            cur_prompt = "<image>\n" + cur_prompt

        if context.args.single_pred_prompt:
            suffix = "Answer with the option's letter from the given choices directly."
            qs = qs + "\n" + suffix
            cur_prompt = cur_prompt + "\n" + suffix

        conv = conv_templates[context.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, context.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(
            keywords, context.tokenizer, input_ids)] if conv.version == "v0" else None

        output_ids = default_generate(
            context.model,
            input_ids,
            images,
            context.args,
            max_new_tokens=getattr(context.args, "max_new_tokens", 1024),
            stopping_criteria=stopping_criteria,
        )
        outputs = decode_new_tokens(
            context.tokenizer, output_ids, input_ids).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        if context.args.answer_prompter:
            outputs_reasoning = outputs
            prompt_with_answer = prompt + outputs_reasoning + " ###\nANSWER:"
            answer_input_ids = tokenizer_image_token(
                prompt_with_answer,
                context.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).cuda()

            answer_output_ids = default_generate(
                context.model,
                answer_input_ids,
                images,
                context.args,
                max_new_tokens=64,
                stopping_criteria=stopping_criteria,
            )
            answer_text = decode_new_tokens(
                context.tokenizer, answer_output_ids, answer_input_ids).strip()
            if answer_text.endswith(stop_str):
                answer_text = answer_text[:-len(stop_str)]
            answer_text = answer_text.strip()
            outputs = outputs_reasoning + "\n The answer is " + answer_text

        return {
            "prompt": cur_prompt,
            "text": outputs,
        }
