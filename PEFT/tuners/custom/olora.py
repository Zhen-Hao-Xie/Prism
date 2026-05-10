# -*- encoding: utf-8 -*-
"""
O-LoRA（Orthogonal Low-Rank Adaptation）多任务 LoRA 实现。

- 每层维护 ``expert_num`` 组低秩适配器（对应最多 ``expert_num`` 个顺序任务）。
- **训练**：仅当前任务 ``cur_task`` 对应的 A/B 参与前向；与论文一致可对历史 A 施加正交正则（见 ``compute_olora_orthogonal_loss``）。
- **推理**（``eval``）：将任务 ``0..cur_task`` 的 A、B 在秩维上拼接，等价于累计 LoRA 作用（与 CoIN / 原 OLoRA 推理方式一致）。

无 token 级 Router，与工具包内 ``MoELoRA``（软路由）正交。

正交项在 ``OLoRALinear.forward`` 内计算并 **fuse** 进本层输出 ``result``（极小系数），
再经 ``OLoRAModel._olora_orth_sum`` 汇总后写入 ``loss``；与在 ``forward`` 外对 Parameter
再建一条 loss 边相比，可与 CE、activation checkpoint、DeepSpeed ZeRO-2 共用同一反向路径。
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ...import_utils import is_bnb_4bit_available, is_bnb_available
from ...utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from ..standard.lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)

if is_bnb_available():
    import bitsandbytes as bnb

# 将正交标量 fuse 进线性输出，使 d(loss)/d(A) 经 ``result`` 主路径回传（系数极小，对 logits 影响可忽略）
OLORA_ORTH_FUSE_SCALE: float = 1e-8


def _olora_linear_orthogonal_fragment(linear: "OLoRALinear", adapter_name: str) -> Optional[torch.Tensor]:
    """单层：``sum_{i<t} |A_t A_i^T|``（L1）；历史 ``A_i`` detach。无历史时返回 None。"""
    if linear.cur_task < 1:
        return None
    if adapter_name not in linear.lora_A:
        return None
    la = linear.lora_A[adapter_name]
    ct = int(linear.cur_task)
    a_cur = la.loraA[ct].mlp.weight
    total: Optional[torch.Tensor] = None
    for i in range(ct):
        a_prev = la.loraA[i].mlp.weight
        term = torch.abs(torch.mm(a_cur, a_prev.detach().T)).sum()
        total = term if total is None else total + term
    return total


@dataclass
class OLoRAConfig(LoraConfig):
    """O-LoRA：在标准 LoRA 超参上增加任务槽位数与当前任务索引。"""

    expert_num: int = field(default=8, metadata={"help": "任务槽 / 专家数（顺序 CL 最大任务数）。"})
    cur_task: int = field(default=0, metadata={"help": "当前正在训练的任务 id，0-based。"})
    task_embedding_dim: int = field(default=64, metadata={"help": "占位字段，与旧 CoIN 配置兼容。"})

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_OLORA


class OLoRAModel(LoraModel):
    """将目标 Linear 替换为 ``OLoRALinear``（多任务槽 + 当前任务单路前向）。"""

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    @staticmethod
    def _prepare_olora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify target_modules in peft_config")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_olora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError("OLoRAModel supports only 1 adapter with bias.")

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        self._olora_orth_sum: Optional[torch.Tensor] = None
        self._olora_orth_hook_handle = None
        self._register_olora_orth_reset_hook()

    def _register_olora_orth_reset_hook(self) -> None:
        """每个 LM forward 开始时清空正交累加器（与 decoder 前向对齐）。"""
        if getattr(self, "_olora_orth_hook_handle", None) is not None:
            return
        inner = self.model
        decoder = getattr(inner, "model", None)
        if decoder is None:
            return

        def _hook(_mod, _args):
            self._olora_orth_sum = None

        self._olora_orth_hook_handle = decoder.register_forward_pre_hook(_hook)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name, layer = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target, self.model.training, layer)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(f"Target modules {lora_config.target_modules} not found in base model.")

    def _create_new_module(self, lora_config, adapter_name, target, training, layer):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_embedding_dim": getattr(lora_config, "task_embedding_dim", 64),
            "expert_num": lora_config.expert_num,
            "cur_task": lora_config.cur_task,
            "layer_id": layer,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn("fan_in_fan_out set to True for Linear; setting to False.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn("fan_in_fan_out set to False for Conv1D; setting to True.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(f"Target module {target} not supported.")
            kwargs["olora_tuner_parent"] = self
            new_module = OLoRALinear(
                adapter_name,
                in_features,
                out_features,
                bias=bias,
                train_signal=training,
                **kwargs,
            )
        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name, _ = _get_submodules(self.model, key)
            except IndexError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])
        return self.model


class OLoRALayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int, expert_num: int, cur_task: int, training: bool, layer_id: int):
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.training = training
        self.layer_id = layer_id

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        lora_dropout_layer = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        if r > 0:
            self.lora_A.update(
                nn.ModuleDict(
                    {
                        adapter_name: OLoRALinearA(
                            self.in_features,
                            r,
                            self.expert_num,
                            self.cur_task,
                            self.layer_id,
                        )
                    }
                )
            )
            self.lora_B.update(
                nn.ModuleDict(
                    {
                        adapter_name: OLoRALinearB(
                            r,
                            self.out_features,
                            self.expert_num,
                            self.cur_task,
                            self.layer_id,
                        )
                    }
                )
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)


class OLoRALinear(nn.Linear, OLoRALayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        train_signal: bool = False,
        layer_id: int = 0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", 8)
        kwargs.pop("task_embedding_dim", None)
        olora_tuner_parent = kwargs.pop("olora_tuner_parent", None)
        self.cur_task = kwargs.pop("cur_task", 0)
        self.layer_id = layer_id
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        OLoRALayer.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            expert_num=self.expert_num,
            cur_task=self.cur_task,
            training=train_signal,
            layer_id=layer_id,
        )

        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        # OLoRAModel 是 nn.Module，不可在 Module.__init__ 完成前赋值给 self（会被注册为子模块）
        object.__setattr__(self, "_olora_tuner_parent", olora_tuner_parent)

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x_d = self.lora_dropout[self.active_adapter](x)
            if self.training:
                lora_a_out = self.lora_A[self.active_adapter](x_d, None)
                lora_b_out = self.lora_B[self.active_adapter](lora_a_out, None)
            else:
                lora_a_out = self.lora_A[self.active_adapter](x_d, self.cur_task)
                lora_b_out = self.lora_B[self.active_adapter](lora_a_out, self.cur_task)
            result = result + lora_b_out * self.scaling[self.active_adapter]
            if self._olora_tuner_parent is not None and self.training and self.cur_task >= 1:
                frag = _olora_linear_orthogonal_fragment(self, self.active_adapter)
                if frag is not None:
                    tp = self._olora_tuner_parent
                    tp._olora_orth_sum = frag if tp._olora_orth_sum is None else tp._olora_orth_sum + frag
                    fuse = result.new_tensor(OLORA_ORTH_FUSE_SCALE)
                    result = result + fuse * frag * torch.ones_like(result)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = result.to(previous_dtype)
        return result


class OLoRALinearA(nn.Module):
    """各任务槽的 LoRA A 侧；训练时只激活 ``cur_task``。"""

    def __init__(self, in_features, r_total, expert_num, cur_task, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.layer_id = layer_id
        r_per = r_total // expert_num
        if r_per * expert_num != r_total:
            raise ValueError(f"LoRA rank r={r_total} 必须能被 expert_num={expert_num} 整除。")
        self.r_per = r_per
        self.in_features = in_features
        self.loraA = nn.ModuleList([OLoRAExpert(in_features, r_per) for _ in range(expert_num)])
        # 推理路径避免每层每步 new nn.Linear（自回归会爆炸）；按 (eval_max_task, device, dtype) 缓存拼接权重
        self._eval_weight_cache: Dict[tuple, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, eval_max_task: Optional[int]) -> torch.Tensor:
        if eval_max_task is None:
            assert 0 <= self.cur_task < self.expert_num
            return self.loraA[self.cur_task](x)
        cache_key = (int(eval_max_task), x.device, x.dtype)
        w = self._eval_weight_cache.get(cache_key)
        if w is None:
            parts = [self.loraA[i].mlp.weight for i in range(eval_max_task + 1)]
            w = torch.cat(parts, dim=0).to(device=x.device, dtype=x.dtype)
            self._eval_weight_cache[cache_key] = w
        return F.linear(x, w)


class OLoRALinearB(nn.Module):
    """各任务槽的 LoRA B 侧。"""

    def __init__(self, r_total, out_features, expert_num, cur_task, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.layer_id = layer_id
        r_per = r_total // expert_num
        if r_per * expert_num != r_total:
            raise ValueError(f"LoRA rank r={r_total} 必须能被 expert_num={expert_num} 整除。")
        self.r_per = r_per
        self.out_features = out_features
        self.loraB = nn.ModuleList([OLoRAExpert(r_per, out_features) for _ in range(expert_num)])
        self._eval_weight_cache: Dict[tuple, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, eval_max_task: Optional[int]) -> torch.Tensor:
        if eval_max_task is None:
            assert 0 <= self.cur_task < self.expert_num
            return self.loraB[self.cur_task](x)
        cache_key = (int(eval_max_task), x.device, x.dtype)
        w = self._eval_weight_cache.get(cache_key)
        if w is None:
            parts = [self.loraB[i].mlp.weight for i in range(eval_max_task + 1)]
            w = torch.cat(parts, dim=1).to(device=x.device, dtype=x.dtype)
            self._eval_weight_cache[cache_key] = w
        return F.linear(x, w)


class OLoRAExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.mlp(x)


def sync_olora_cur_task(model: nn.Module, cur_task: int) -> None:
    """将 ``cur_task`` 写入所有 ``OLoRALinear`` 及其 A/B 子模块（训练单路 / 推理累计范围）。"""
    for m in model.modules():
        if isinstance(m, OLoRALinear):
            m.cur_task = int(cur_task)
            for child in m.lora_A.values():
                if isinstance(child, OLoRALinearA):
                    child.cur_task = int(cur_task)
            for child in m.lora_B.values():
                if isinstance(child, OLoRALinearB):
                    child.cur_task = int(cur_task)


def apply_olora_expert_trainable_mask(
    model: nn.Module,
    cur_task: int,
    adapter_name: str = "default",
) -> None:
    """
    仅当前任务槽 ``cur_task`` 的 LoRA 参数可训练；其余任务槽与未来槽冻结。
    应在 ``mark_only_lora_as_trainable`` 之后调用。
    """
    # 兼容 ``...loraA.i.mlp.weight``（常规）与历史 ``...loraA.i.weight``（曾对 mlp.weight 重复注册）
    pat = re.compile(rf"lora_[AB]\.{re.escape(adapter_name)}\.lora[AB]\.(\d+)\.(?:mlp\.)?weight$")
    for name, p in model.named_parameters():
        m = pat.search(name)
        if not m:
            continue
        tid = int(m.group(1))
        p.requires_grad = tid == int(cur_task)


def compute_olora_orthogonal_loss(
    model: nn.Module,
    cur_task: int,
    adapter_name: str = "default",
) -> torch.Tensor:
    """
    O-LoRA 正交正则：对每层前缀，累加 ``sum |A_t @ A_i^T|``（``i < t``），``A`` 为 ``lora_A`` 权重矩阵。

    与论文中 ``||A_i^T A_t||_F^2`` 不同之处在于使用 L1 式绝对值和（与原 LLaVA OLoRA 脚本一致）。
    """
    if cur_task < 1:
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0)
        z = torch.zeros((), device=p.device, dtype=p.dtype)
        return z

    pattern = re.compile(rf"lora_A\.{re.escape(adapter_name)}\.loraA\.(\d+)\.(?:mlp\.)?weight$")
    current_by_prefix: Dict[str, torch.nn.Parameter] = {}
    prev_by_prefix: Dict[str, Dict[int, torch.nn.Parameter]] = {}

    for name, param in model.named_parameters():
        m = pattern.search(name)
        if not m:
            continue
        tid = int(m.group(1))
        prefix = name[: m.start()]
        if tid == cur_task:
            current_by_prefix[prefix] = param
        elif 0 <= tid < cur_task:
            prev_by_prefix.setdefault(prefix, {})[tid] = param

    if not current_by_prefix:
        p = next(model.parameters(), None)
        return torch.zeros((), device=p.device, dtype=p.dtype) if p is not None else torch.tensor(0.0)

    total: Optional[torch.Tensor] = None
    for prefix, a_cur in current_by_prefix.items():
        prevs = prev_by_prefix.get(prefix)
        if not prevs:
            continue
        for i in range(cur_task):
            a_prev = prevs.get(i)
            if a_prev is None:
                continue
            term = torch.abs(torch.mm(a_cur, a_prev.detach().T)).sum()
            total = term if total is None else total + term

    if total is None:
        p = next(model.parameters(), None)
        return torch.zeros((), device=p.device, dtype=p.dtype) if p is not None else torch.tensor(0.0)
    return total
