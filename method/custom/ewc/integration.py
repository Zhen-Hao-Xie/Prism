"""
EWC：在 LLM 的 attention + FFN 上注入标准 LoRA；学习任务 t 结束后估计对角 Fisher（仅 LoRA），
保存该任务的参数快照 θ*；任务 t+1 起在 CE 损失上增加 (λ/2) Σ F_i (θ_i - θ*_i)²（对所有已完成任务求和）。

参见仓库根目录 ``ewc.md``。
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from config.backbones.constants import IGNORE_INDEX
from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
from method.base.context import CLContext
from method.base.integration import CLIntegration
from method.factory import CLMethodFactory

_LOG = logging.getLogger(__name__)


def _infer_batch_size(batch: Dict[str, Any]) -> int:
    for key in ("input_ids", "labels"):
        v = batch.get(key)
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.shape[0])
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.shape[0])
    return 1


def _split_batch_rows(
    batch: Dict[str, Any], start: int, end: int, batch_size: int
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
            out[k] = v[start:end]
        elif isinstance(v, list) and len(v) == batch_size:
            out[k] = v[start:end]
        else:
            out[k] = v
    return out


def _microbatch_chunks(batch: Dict[str, Any], micro_bs: int) -> List[Dict[str, Any]]:
    bs = _infer_batch_size(batch)
    if micro_bs <= 0 or bs <= micro_bs:
        return [batch]
    return [
        _split_batch_rows(batch, i, min(i + micro_bs, bs), bs)
        for i in range(0, bs, micro_bs)
    ]


def _masked_label_count(labels: torch.Tensor) -> torch.Tensor:
    return (labels != IGNORE_INDEX).sum().to(dtype=torch.float32).clamp(min=1.0)


def _fisher_chunk_loss_scale(batch_full: Dict[str, Any], chunk: Dict[str, Any], *, device: torch.device) -> torch.Tensor:
    """将子 batch 上 HF 返回的 mean(loss) 换算成对「整 batch 标量 loss」的贡献：scale = T_chunk / T_full。"""
    lf = batch_full.get("labels")
    lc = chunk.get("labels")
    if isinstance(lf, torch.Tensor) and isinstance(lc, torch.Tensor):
        full_d = _masked_label_count(lf)
        chunk_d = _masked_label_count(lc)
        return (chunk_d / full_d).to(device=device, dtype=torch.float32)
    full_bs = max(1, _infer_batch_size(batch_full))
    chunk_bs = max(1, _infer_batch_size(chunk))
    return torch.tensor(chunk_bs / float(full_bs), device=device, dtype=torch.float32)


def _unwrap_peft_root(model: nn.Module) -> nn.Module:
    """CLModel → PeftModel / 内层 LLaVA。"""
    inner = getattr(model, "_base_model", model)
    return inner


def _iter_lora_named_parameters(model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
    root = _unwrap_peft_root(model)
    for name, p in root.named_parameters():
        if "lora" not in name.lower():
            continue
        yield name, p


@CLMethodFactory.register("ewc")
class EwcIntegration(CLIntegration):
    def __init__(self, config: Any):
        super().__init__(config)
        self.task_num: int = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.cur_task: int = int(getattr(config, "cur_task", 0))
        self.ewc_lambda: float = float(getattr(config, "ewc_lambda", 5000.0))
        self.ewc_fisher_batches: int = int(getattr(config, "ewc_fisher_batches", 50))
        # None：不拆 micro-batch（与旧行为一致）；建议 1～4 以降低任务结束时 Fisher 反向的峰值显存
        self.ewc_fisher_micro_batch_size: Optional[int] = getattr(
            config, "ewc_fisher_micro_batch_size", None
        )
        if self.ewc_fisher_micro_batch_size is not None:
            self.ewc_fisher_micro_batch_size = int(self.ewc_fisher_micro_batch_size)

        # task_id -> { full_param_name -> tensor CPU }
        self._anchors: Dict[int, Dict[str, torch.Tensor]] = {}
        self._fisher: Dict[int, Dict[str, torch.Tensor]] = {}

    def initialize_model(self, model: nn.Module) -> None:
        for _, p in model.named_parameters():
            p.requires_grad = False
        self._setup_lora(model)
        _LOG.info(
            "[EWC] LoRA injected | λ=%s | Fisher batches/task_end=%s | Fisher micro-batch=%s",
            self.ewc_lambda,
            self.ewc_fisher_batches,
            self.ewc_fisher_micro_batch_size,
        )

    def _find_target_modules(self, model: nn.Module) -> List[str]:
        return collect_peft_target_linear_suffixes(model, self.config)

    def _setup_lora(self, model: nn.Module) -> None:
        from PEFT import LoraConfig, get_peft_model

        target_modules = self._find_target_modules(model)
        r = int(getattr(self.config, "lora_r", 64))
        lora_config = LoraConfig(
            r=r,
            lora_alpha=int(getattr(self.config, "lora_alpha", max(r * 2, 32))),
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
        wrapped = getattr(model, "_base_model", None)
        if wrapped is not None and hasattr(wrapped, "print_trainable_parameters"):
            wrapped.print_trainable_parameters()

    def _snapshot_anchors(self, model: nn.Module, task_id: int) -> None:
        snap: Dict[str, torch.Tensor] = {}
        for name, p in _iter_lora_named_parameters(model):
            snap[name] = p.detach().float().cpu().clone()
        self._anchors[int(task_id)] = snap
        _LOG.info("[EWC] saved θ* anchors for task %s (%d tensors)", task_id, len(snap))

    def _estimate_fisher_diagonal(
        self, model: nn.Module, trainer: Any, task_id: int
    ) -> None:
        n_batches = max(1, self.ewc_fisher_batches)
        loader = trainer.get_train_dataloader()
        accum: Dict[str, torch.Tensor] = {}
        count = 0

        was_training = model.training
        model.eval()

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        micro_bs = self.ewc_fisher_micro_batch_size
        if micro_bs is not None and micro_bs > 0:
            _LOG.info(
                "[EWC] Fisher diagonal: micro-batch=%s (train batch size does not apply here — "
                "OOM at task end is often this pass; lower micro-batch if needed)",
                micro_bs,
            )

        for name, p in _iter_lora_named_parameters(model):
            accum[name] = torch.zeros(p.numel(), dtype=torch.float32)

        it = iter(loader)
        for _ in range(n_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = trainer._prepare_inputs(batch)

            chunks = (
                _microbatch_chunks(batch, micro_bs)
                if micro_bs is not None and micro_bs > 0
                else [batch]
            )

            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                if len(chunks) == 1:
                    outputs = model(**chunks[0])
                    loss = getattr(outputs, "loss", None)
                    if loss is None:
                        continue
                    loss.backward()
                else:
                    bad = False
                    for ch in chunks:
                        outputs = model(**ch)
                        loss = getattr(outputs, "loss", None)
                        if loss is None:
                            bad = True
                            break
                        scale = _fisher_chunk_loss_scale(batch, ch, device=loss.device)
                        (loss * scale).backward()
                    if bad:
                        model.zero_grad(set_to_none=True)
                        continue

            for name, p in _iter_lora_named_parameters(model):
                if p.grad is None:
                    continue
                g = p.grad.detach().float().cpu().view(-1)
                accum[name] += g.pow(2)

            model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            count += 1

        if count == 0:
            _LOG.warning("[EWC] Fisher: no batch succeeded for task %s; Fisher left empty", task_id)
            self._fisher[int(task_id)] = {k: torch.zeros_like(v) for k, v in accum.items()}
        else:
            for name in accum:
                accum[name] /= float(count)
            self._fisher[int(task_id)] = accum
            _LOG.info("[EWC] Fisher diagonal estimated for task %s (%d batches)", task_id, count)

        if was_training:
            model.train()

    def on_train_task_finished(
        self,
        model: nn.Module,
        context: CLContext,
        task_id: int,
        *,
        trainer: Optional[Any] = None,
    ) -> None:
        """单个任务训练跑完时调用：先存 θ*，再在一部分 batch 上估计 Fisher（见 ``ewc.md``）。"""
        tid = int(task_id)
        self._snapshot_anchors(model, tid)
        if trainer is not None:
            self._estimate_fisher_diagonal(model, trainer, tid)
        else:
            _LOG.warning(
                "[EWC] trainer ref missing; Fisher not computed (anchors still saved). "
                "Ensure CLTrainerCallback.trainer is set in train.py."
            )
            self._fisher[tid] = {
                n: torch.zeros(p.numel(), dtype=torch.float32)
                for n, p in _iter_lora_named_parameters(model)
            }

    def _ewc_penalty(self, model: nn.Module, cur_task: int) -> torch.Tensor:
        device = next(model.parameters()).device
        total = torch.zeros((), device=device, dtype=torch.float32)
        ct = int(cur_task)

        for tid in sorted(self._anchors.keys()):
            if tid >= ct:
                continue
            anchors = self._anchors.get(tid)
            fisher = self._fisher.get(tid)
            if not anchors or not fisher:
                continue
            for name, p in _iter_lora_named_parameters(model):
                if name not in anchors or name not in fisher:
                    continue
                theta_star = anchors[name].to(device=p.device, dtype=torch.float32).reshape(-1)
                fdiag = fisher[name].to(device=p.device, dtype=torch.float32)
                theta = p.reshape(-1).float()
                diff = theta - theta_star
                total = total + (fdiag * diff * diff).sum()
        return total

    def on_input_prep(self, model: nn.Module, args: tuple, kwargs: dict, context: CLContext) -> None:
        return

    def on_forward_start(self, model: nn.Module, context: CLContext) -> None:
        return

    def on_forward_end(self, model: nn.Module, outputs: Any, context: CLContext) -> Any:
        if not model.training:
            return outputs
        loss = getattr(outputs, "loss", None)
        if loss is None:
            return outputs

        ct = int(getattr(context, "task_id", getattr(self.config, "cur_task", self.cur_task)))
        if ct < 1:
            return outputs

        pen = self._ewc_penalty(model, ct)
        reg = 0.5 * float(self.ewc_lambda) * pen.to(dtype=loss.dtype, device=loss.device)
        outputs.loss = loss + reg
        return outputs

    def on_task_end(self, model: nn.Module, context: CLContext, task_id: int) -> None:
        _LOG.debug("[EWC] on_task_end task_id=%s (anchors=%s)", task_id, list(self._anchors.keys()))

    def get_inference_config(self) -> Dict[str, Any]:
        return {
            "ewc_lambda": self.ewc_lambda,
            "ewc_fisher_batches": self.ewc_fisher_batches,
            "ewc_fisher_micro_batch_size": self.ewc_fisher_micro_batch_size,
            "lora_r": int(getattr(self.config, "lora_r", 64)),
        }

    def save_extra_state(self, output_dir: str, model=None) -> bool:
        import os

        path = os.path.join(output_dir, "ewc_state.pt")
        payload = {
            "anchors": {k: v for k, v in self._anchors.items()},
            "fisher": {k: v for k, v in self._fisher.items()},
            "ewc_lambda": self.ewc_lambda,
        }
        torch.save(payload, path)
        return True

    def load_extra_state(self, load_dir: str, model=None) -> bool:
        import os

        path = os.path.join(load_dir, "ewc_state.pt")
        if not os.path.isfile(path):
            return False
        blob = torch.load(path, map_location="cpu")
        if not isinstance(blob, dict):
            return False
        a = blob.get("anchors")
        f = blob.get("fisher")
        if isinstance(a, dict):
            self._anchors = {int(k): v for k, v in a.items()}
        if isinstance(f, dict):
            self._fisher = {int(k): v for k, v in f.items()}
        _LOG.info("[EWC] loaded state from %s (tasks: %s)", path, sorted(self._anchors.keys()))
        return bool(self._anchors)
