"""
replay_lora：在主干上注入普通 LoRA（默认 ``attn_and_ffn``），并配合固定容量、按任务均分的回放缓冲区。

**Loss**：经 ``memory_data_path`` 混入 ``LazySupervisedDataset`` 的回放行与当前任务行一样走 ``preprocess`` → forward，
参与同一个 LM loss；**没有**单独「无 loss 的回放前向」。``on_training_batch_end`` 里写入 buffer 的只是 JSON 副本，
用于**后续任务**；该步 loss 已在同一步对**当前 batch 张量**算过。

- 训练任务 ``k`` 开始前：槽 ``0..k-1`` → 侧车 JSON + ``data_args.memory_data_path``，与主数据 extend 后 shuffle。
- 训练中：``LLaVATrainer`` 在每个 micro-batch 的 backward 之后调用 ``on_training_batch_end``；
  对 ``cl_raw_example`` 逐条 ``should_store_training_example``，为真则写入当前任务槽。仅 rank0 改 buffer。
  最后一项任务不写槽。
"""

from __future__ import annotations

import copy
import json
import os
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.factory import CLMethodFactory

_STATE_NAME = "replay_buffer_state.json"
_MEMORY_SIDECAR = "_cl_replay_memory.json"


def _count_json_list_samples(path: Optional[str]) -> Optional[int]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else None
    except Exception:
        return None


class TaskPartitionedReplayBuffer:
    """总容量按 ``task_num - 1`` 均分；训练任务 ``k`` 只写入第 ``k`` 槽；回放仅使用 ``0..k-1``。"""

    def __init__(self, task_num: int, total_capacity: int):
        self.task_num = int(task_num)
        n_slot = max(0, self.task_num - 1)
        self.n_slot = n_slot
        if n_slot <= 0:
            cap_each = 0
        else:
            cap_each = max(1, int(total_capacity) // n_slot)
        self.per_slot_cap = cap_each
        self.slots: List[List[Dict[str, Any]]] = [[] for _ in range(n_slot)]

    def maybe_add_from_task(self, task_id: int, sample: Dict[str, Any]) -> bool:
        tid = int(task_id)
        if self.n_slot <= 0 or tid < 0 or tid >= self.n_slot:
            return False
        slot = self.slots[tid]
        if len(slot) >= self.per_slot_cap:
            return False
        slot.append(copy.deepcopy(sample))
        return True

    def flatten_past_slots(self, cur_task: int) -> List[Dict[str, Any]]:
        k = int(cur_task)
        if k <= 0 or self.n_slot <= 0:
            return []
        out: List[Dict[str, Any]] = []
        upper = min(k, self.n_slot)
        for i in range(upper):
            for s in self.slots[i]:
                out.append(copy.deepcopy(s))
        return out

    def counts_mixed_into_training(self, cur_task: int) -> List[int]:
        """训练 ``cur_task`` 时，自历史槽 ``0..cur_task-1`` 各混入的条数（槽 ``t`` 对应来源任务 ``t``）。"""
        k = int(cur_task)
        if k <= 0 or self.n_slot <= 0:
            return []
        upper = min(k, self.n_slot)
        return [len(self.slots[i]) for i in range(upper)]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "task_num": self.task_num,
            "per_slot_cap": self.per_slot_cap,
            "slots": [[copy.deepcopy(x) for x in slot] for slot in self.slots],
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            return
        slots = d.get("slots")
        if not isinstance(slots, list):
            return
        n_expected = len(self.slots)
        for i in range(min(n_expected, len(slots))):
            row = slots[i]
            if not isinstance(row, list):
                continue
            cap = self.per_slot_cap
            self.slots[i] = [copy.deepcopy(x) for x in row[:cap] if isinstance(x, dict)]


@CLMethodFactory.register("replay_lora")
class Replay_loraIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num = int(getattr(config, "task_num", 8))
        self.cur_task = int(getattr(config, "cur_task", 0))
        cap = int(getattr(config, "replay_buffer_size", 512))
        self.replay_sample_prob = float(getattr(config, "replay_sample_prob", 0.02))
        self.buffer = TaskPartitionedReplayBuffer(self.task_num, cap)
        self._replay_source_path: Optional[str] = None

    def initialize_model(self, model: nn.Module) -> None:
        for _, p in model.named_parameters():
            p.requires_grad = False
        self._setup_lora(model)
        print(
            f"[replay_lora] LoRA | buffer_slots={self.buffer.n_slot} "
            f"per_slot_cap={self.buffer.per_slot_cap} | "
            f"store_prob={self.replay_sample_prob} (per-sample via should_store_training_example; current cl_raw_example)",
            flush=True,
        )

    def _find_target_modules(self, model: nn.Module) -> List[str]:
        return collect_peft_target_linear_suffixes(model, self.config)

    def _setup_lora(self, model: nn.Module) -> None:
        from PEFT import LoraConfig, get_peft_model

        target_modules = self._find_target_modules(model)
        lora_config = LoraConfig(
            r=int(getattr(self.config, "lora_r", 64)),
            lora_alpha=int(getattr(self.config, "lora_alpha", 128)),
            lora_dropout=float(getattr(self.config, "lora_dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            exclude_module_path_segments=self.peft_exclude_module_path_segments,
        )
        _base = getattr(model, "_base_model", None)
        if _base is not None:
            peft_model = get_peft_model(_base, lora_config)
            object.__setattr__(model, "_base_model", peft_model)
        else:
            get_peft_model(model, lora_config)
        peft_wrapped = getattr(model, "_base_model", None)
        if peft_wrapped is not None and hasattr(peft_wrapped, "print_trainable_parameters"):
            peft_wrapped.print_trainable_parameters()

    def prepare_training_data(self, data_args: Any, model_args: Any, training_args: Any = None) -> None:
        self._replay_source_path = getattr(data_args, "data_path", None)
        cur = int(getattr(model_args, "cur_task", self.cur_task))
        rank = -1
        if training_args is not None:
            rank = int(getattr(training_args, "local_rank", -1))

        counts = self.buffer.counts_mixed_into_training(cur)
        main_n = _count_json_list_samples(self._replay_source_path)
        if rank <= 0:
            main_hint = f" | current_task_json_samples={main_n}" if main_n is not None else ""
            if cur <= 0 or self.buffer.n_slot <= 0:
                print(
                    f"[replay_lora] before task {cur}: no past-task replay mixed "
                    f"(train set = current task JSON only){main_hint}",
                    flush=True,
                )
            else:
                per = [f"task{t}={c}" for t, c in enumerate(counts)]
                total = sum(counts)
                merged = (main_n + total) if main_n is not None else None
                merged_hint = f" | merged_list_len≈{merged} after extend+shuffle" if merged is not None else ""
                print(
                    f"[replay_lora] before task {cur}: replay by source — {', '.join(per)} | "
                    f"total_replay={total}{main_hint}{merged_hint}",
                    flush=True,
                )

        mem = self.buffer.flatten_past_slots(cur)
        if not mem:
            data_args.memory_data_path = None
            return
        out_dir = "."
        if training_args is not None:
            out_dir = getattr(training_args, "output_dir", None) or "."
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, _MEMORY_SIDECAR)
        if rank <= 0:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(mem, f, ensure_ascii=False)
            if main_n is not None:
                print(
                    f"[replay_lora] data module: main={main_n} + replay={len(mem)} => "
                    f"LazySupervisedDataset list len {main_n + len(mem)} (then shuffled) | sidecar {path}",
                    flush=True,
                )
            else:
                print(
                    f"[replay_lora] replay sidecar written: {len(mem)} samples -> {path}",
                    flush=True,
                )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        data_args.memory_data_path = path

    @staticmethod
    def _is_primary_rank_for_cl_state() -> bool:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return int(torch.distributed.get_rank()) == 0

    def should_store_training_example(
        self,
        model: nn.Module,
        context: CLContext,
        raw_example: Dict[str, Any],
        batch: Dict[str, Any],
        *,
        example_index: int,
        loss: Optional[torch.Tensor] = None,
    ) -> bool:
        """默认：以 ``replay_sample_prob`` 对当前样本伯努利抽样。可重写以接入特征/难度等。"""
        if self.replay_sample_prob <= 0:
            return False
        return random.random() < self.replay_sample_prob

    def on_training_batch_end(
        self,
        model: nn.Module,
        context: CLContext,
        batch: Dict[str, Any],
        *,
        loss: Optional[torch.Tensor] = None,
        trainer: Any = None,
    ) -> None:
        if not model.training:
            return
        if not self._is_primary_rank_for_cl_state():
            return
        raws = batch.get("cl_raw_example")
        if not isinstance(raws, list) or not raws:
            return
        tid = context.task_id
        if tid is None:
            tid = int(getattr(self.config, "cur_task", self.cur_task))
        else:
            tid = int(tid)
        for i, raw in enumerate(raws):
            if not isinstance(raw, dict):
                continue
            if not self.should_store_training_example(
                model, context, raw, batch, example_index=i, loss=loss
            ):
                continue
            if not self.buffer.maybe_add_from_task(tid, raw):
                break

    def on_input_prep(self, model: nn.Module, args: tuple, kwargs: dict, context: CLContext) -> None:
        return

    def on_forward_start(self, model: nn.Module, context: CLContext) -> None:
        return

    def on_forward_end(self, model: nn.Module, outputs: Any, context: CLContext) -> Any:
        return outputs

    def on_task_end(self, model: nn.Module, context: CLContext, task_id: int) -> None:
        print(f"[replay_lora] task {task_id} end | slot_sizes={[len(s) for s in self.buffer.slots]}", flush=True)

    def get_inference_config(self) -> Dict[str, Any]:
        return {}

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, _STATE_NAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.buffer.state_dict(), f, ensure_ascii=False)
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        path = os.path.join(load_dir, _STATE_NAME)
        if not os.path.isfile(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return False
        self.buffer.load_state_dict(d)
        print(f"[replay_lora] loaded buffer from {path}", flush=True)
        return True
