import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from common.load_model import load_model_for_inference
from common.file_manager import load_questions
from common.utils import get_chunk
from common.data_manager import ImageFinder
from llava.mm_utils import tokenizer_image_token, process_images
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class EvalDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_finder = ImageFinder(image_folder) if image_folder else None
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, idx):
        line = self.questions[idx]
        image_file = line["image"]
        qs = line["text"]

        # 构建 prompt
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 加载图像
        image_path = self.image_finder.find(image_file) if self.image_finder else None
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


@torch.inference_mode()
def generate_outputs(model, input_ids, image_tensor, args):
    """封装生成逻辑"""
    return model.generate(
        input_ids,
        images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True
    )


def eval_model(args):
    # 加载模型
    model_path = os.path.expanduser(args.model_path)
    model_name = args.model_name or os.path.basename(model_path)
    tokenizer, model, image_processor, context_len = load_model_for_inference(
        model_path, args.model_base, model_name, text_tower=args.text_tower
    )

    # 加载问题并按 chunk 分割
    questions = load_questions(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # 准备输出文件
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # 创建数据集和 DataLoader
    dataset = EvalDataset(
        questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    # 开始推理
    with open(answers_file, 'w', encoding='utf-8') as f:
        for (input_ids, image_tensor), sample in tqdm(zip(dataloader, questions), total=len(questions)):
            input_ids = input_ids.to('cuda', non_blocking=True)

            output_ids = generate_outputs(model, input_ids, image_tensor, args)

            # 解码输出
            input_len = input_ids.shape[1]
            output_text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()

            # 写入结果
            result = {
                "question_id": sample["question_id"],
                "prompt": sample["text"],
                "text": output_text,
                "answer_id": shortuuid.uuid(),
                "model_id": model_name,
                "metadata": {}
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None, help="Override model name")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--text-tower", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)