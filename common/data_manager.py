"""
数据管理模块：包含数据集类、DataLoader、Sampler等
"""

import json
import random
import os
import copy
from typing import Any, Dict, Iterator, List, Optional
import torch
from PIL import Image
import transformers
from dataclasses import dataclass
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from config.constants import IGNORE_INDEX
from common.conversation import Conversation as conversation_lib
from common.file_manager import load_questions
from common.data_processor import get_chunk
from common.data_processor import expand2square, tokenizer_image_token, preprocess_multimodal, preprocess


class LazySupervisedDataset(Dataset):
    """懒加载的有监督数据集"""
    
    def __init__(self, data_path: str, tokenizer, data_args):
        super().__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        if data_args.memory_data_path is not None:
            memory_data = json.load(open(data_args.memory_data_path, "r"))
            self.list_data_dict.extend(memory_data)
            random.shuffle(self.list_data_dict)
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        """返回样本长度（用于采样）"""
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        """返回带模态信息的样本长度（正数为有图像，负数为无图像）"""
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        
        has_image = 'image' in sources[0]
        
        if has_image:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            
            if self.data_args.image_aspect_ratio == 'pad':
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        
        if has_image:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset:
    """有监督数据集的DataCollator"""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        return batch


def make_supervised_data_module(tokenizer, data_args):
    """创建有监督数据模块"""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


class ImageFinder:
    """在文件夹中递归查找图像文件，支持通过文件名或相对路径查找"""
    
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self._index = None

    def build_index(self):
        """构建文件名到完整路径的索引（用于递归查找）"""
        index = {}
        if not self.image_folder:
            self._index = index
            return
        print(f"Building image index for folder: {self.image_folder}")
        for dirpath, _, filenames in tqdm(os.walk(self.image_folder)):
            for filename in filenames:
                lower = filename.lower()
                if not lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    continue
                index.setdefault(filename, os.path.join(dirpath, filename))
        self._index = index

    def find(self, image_file):
        """查找图像文件的完整路径"""
        # 情况1：直接拼接
        direct_path = os.path.join(self.image_folder, image_file)
        if os.path.isfile(direct_path):
            return direct_path

        # 情况2：递归查找（基于文件名）
        if self._index is None:
            self.build_index()
        candidate = self._index.get(os.path.basename(image_file))
        if candidate and os.path.isfile(candidate):
            return candidate

        raise FileNotFoundError(
            f"Image not found: {image_file} under image_folder={self.image_folder}"
        )


class InferenceDataset:
    """推理数据集类"""
    
    def __init__(self, question_file: str, num_chunks: int = 1, chunk_idx: int = 0) -> None:
        self.question_file = os.path.expanduser(question_file)
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        self._samples: List[Dict[str, Any]] | None = None

    def load(self) -> List[Dict[str, Any]]:
        if self._samples is None:
            samples = load_questions(self.question_file)
            self._samples = get_chunk(samples, self.num_chunks, self.chunk_idx)
        return self._samples

    def __len__(self) -> int:
        return len(self.load())

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.load())

    @property
    def samples(self) -> List[Dict[str, Any]]:
        return self.load()


def split_to_even_chunks(indices, lengths, num_chunks):
    """将索引列表分割成大致等长的块"""
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """根据长度分组的索引"""
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """根据模态（图像/文本）和长度分组的索引"""
    assert all(l != 0 for l in lengths), "Should not have zero length."
    
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # 所有样本都属于同一模态
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    """按长度分组的采样器"""
    
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)