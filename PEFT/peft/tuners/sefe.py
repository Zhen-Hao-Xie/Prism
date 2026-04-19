# PEFT/peft/tuners/sefe.py
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import PeftType, PeftConfig, _freeze_adapter, transpose, _get_submodules
from .lora import LoraConfig, LoraModel, LoraLayer, mark_only_lora_as_trainable


@dataclass
class SefeConfig(LoraConfig):
    """
    Configuration mapped to PEFT's CAUSAL_LM_SEFE or similar structure.
    Used for SEFE RegLoRA implementation.
    """
    num_tasks: int = field(default=8, metadata={
                           "help": "Number of continual tasks."})
    sefe_top_p: float = field(default=0.02, metadata={
                              "help": "Top percentage of Delta W to protect as key elements."})
    sefe_lambda_reg: float = field(
        default=2500.0, metadata={"help": "Regularization strength."})

    def __post_init__(self):
        self.peft_type = PeftType.SEFE
        # default to empty if not set
        if getattr(self, "target_modules", None) is None:
            pass


class SefeModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        # Must initialize PEFT base
        super().__init__(model, config, adapter_name)

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = getattr(self.model, "config", {
                                   "model_type": "custom"})
            if hasattr(model_config, "to_dict"):
                model_config = model_config.to_dict()
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config

        self._find_and_replace(adapter_name)

        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "SefeModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(
            self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue
            is_target_modules_in_base_model = True

            # Unpack 4 values to match workspace's modified _get_submodules
            ret = _get_submodules(self.model, key)
            if len(ret) == 4:
                parent, target, target_name, _ = ret
            else:
                parent, target, target_name = ret

            if isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    getattr(lora_config, "init_lora_weights", True),
                )
            else:
                new_module = self._create_new_module(
                    lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": getattr(lora_config, "init_lora_weights", True),
            "sefe_top_p": getattr(lora_config, "sefe_top_p", 0.02),
            "sefe_lambda_reg": getattr(lora_config, "sefe_lambda_reg", 1.0),
        }

        if isinstance(target, torch.nn.Linear):
            in_features, out_features = target.in_features, target.out_features
            if kwargs["fan_in_fan_out"]:
                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            new_module = SefeLinear(
                adapter_name, in_features, out_features, bias=bias, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported by SEFE.")
        return new_module


class SefeLayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int, sefe_top_p: float = 0.02, **kwargs):
        super().__init__(in_features=in_features, out_features=out_features, **kwargs)
        self.sefe_top_p = sefe_top_p
        # Persistent mask for important elements initialized to 0
        self.register_buffer("weight_mask", torch.zeros(
            out_features, in_features, dtype=torch.bool), persistent=True)

    def update_mask(self, adapter_name):
        if adapter_name not in self.lora_A:
            return

        if self.r[adapter_name] > 0:
            # 1. Compute delta W
            delta_w = self.get_delta_weight(adapter_name)

            # 2. Identify top elements
            num_elements = delta_w.numel()
            k = max(1, int(num_elements * self.sefe_top_p))

            # Use abs value for magnitude
            abs_delta_w = torch.abs(delta_w).flatten()
            if k > 0 and abs_delta_w.max() > 0:
                threshold = torch.topk(abs_delta_w, k).values[-1]
                new_mask = torch.abs(delta_w) >= threshold
                # 3. Update persistent boolean mask (logical OR)
                self.weight_mask.logical_or_(new_mask)

    def get_reg_loss(self, adapter_name):
        loss = 0.0
        if self.weight_mask is not None and self.weight_mask.any():
            delta_w = self.get_delta_weight(adapter_name)
            # RegLoss = sum((Mask * delta_W)^2)
            loss = (self.weight_mask * delta_w).pow(2).sum()
        return loss


class InjectRegLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A_weight, B_weight, mask, scaling, lambda_reg):
        ctx.save_for_backward(A_weight, B_weight, mask)
        ctx.scaling = scaling
        ctx.lambda_reg = lambda_reg
        return x

    @staticmethod
    def backward(ctx, grad_output):
        A_weight, B_weight, mask = ctx.saved_tensors
        if mask is not None and mask.any() and ctx.lambda_reg > 0:
            with torch.enable_grad():
                A_dummy = A_weight.detach().requires_grad_(True)
                B_dummy = B_weight.detach().requires_grad_(True)
                delta_w = (B_dummy @ A_dummy) * ctx.scaling
                reg_loss = (mask * delta_w).abs().mean() * ctx.lambda_reg
                grad_A, grad_B = torch.autograd.grad(
                    reg_loss, (A_dummy, B_dummy))
                print("Loss: ", reg_loss)
        else:
            grad_A = None
            grad_B = None

        return grad_output, grad_A, grad_B, None, None, None


class SefeLinear(nn.Linear, SefeLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        sefe_top_p: float = 0.02,
        sefe_lambda_reg: float = 2500.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SefeLayer.__init__(self, in_features=in_features,
                           out_features=out_features, sefe_top_p=sefe_top_p)
        self.weight.requires_grad = False
        self.sefe_lambda_reg = sefe_lambda_reg

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha,
                          lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        # Prevent default deepspeed / PEFT merging as RegLoRA does this custom at task end.
        pass

    def unmerge(self):
        pass

    def get_delta_weight(self, adapter):
        return (
            transpose(
                self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = F.linear(x, transpose(
            self.weight, self.fan_in_fan_out), bias=self.bias)
        x = x.to(self.lora_A[self.active_adapter].weight.dtype)

        if self.r[self.active_adapter] > 0:
            lora_A_mod = self.lora_A[self.active_adapter]
            lora_B_mod = self.lora_B[self.active_adapter]
            x_dropped = self.lora_dropout[self.active_adapter](x)

            if self.training and getattr(self, "sefe_lambda_reg", 0.0) > 0 and self.weight_mask is not None and self.weight_mask.any():
                x_dropped = InjectRegLoss.apply(x_dropped, lora_A_mod.weight, lora_B_mod.weight,
                                                self.weight_mask, self.scaling[self.active_adapter], self.sefe_lambda_reg)

            result += lora_B_mod(lora_A_mod(x_dropped)) * \
                self.scaling[self.active_adapter]

        result = result.to(previous_dtype)
        return result
