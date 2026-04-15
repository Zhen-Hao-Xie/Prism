# -*- encoding: utf-8 -*-
import warnings
import math
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from typing import Optional
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Linear8bitLt,
    Linear4bit,
    Embedding,
    Conv2d,
)
from ..import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb

@dataclass
class SAMEConfig(LoraConfig):
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=8)
    cur_task: int = field(default=0)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_SAME


class SAMEModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_SAME_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError("MMOELoraModel supports only 1 adapter with bias.")

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
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
            parent, target, target_name, layer = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name, lora_config.r, lora_config.lora_alpha,
                    lora_config.lora_dropout, lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name, lora_config.r, lora_config.lora_alpha,
                    lora_config.lora_dropout, lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name, lora_config.r, lora_config.lora_alpha,
                    lora_config.lora_dropout, lora_config.init_lora_weights,
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
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
            "cur_task": lora_config.cur_task,
            "layer_id": layer,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update({
                "has_fp16_weights": target.state.has_fp16_weights,
                "memory_efficient_backward": target.state.memory_efficient_backward,
                "threshold": target.state.threshold,
                "index": target.index,
            })
            new_module = Linear8bitLt(adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs)
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update({
                "compute_dtype": target.compute_dtype,
                "compress_statistics": target.weight.compress_statistics,
                "quant_type": target.weight.quant_type,
            })
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
                    warnings.warn("fan_in_fan_out set to True for Linear layer; setting to False.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape)
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn("fan_in_fan_out set to False for Conv1D; setting to True.")
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(f"Target module {target} not supported.")
            new_module = SAMELinear(
                adapter_name, in_features, out_features, bias=bias, 
                train_signal=training, **kwargs
            )
        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_SAME_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify target_modules in peft_config")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LoRA layers when model is loaded in 8-bit mode")
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
                        target.in_channels, target.out_channels,
                        kernel_size=target.kernel_size, stride=target.stride,
                        padding=target.padding, dilation=target.dilation,
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


class SAMELayer(LoraLayer):
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
            self.lora_A.update(nn.ModuleDict({adapter_name: SAMELinearA(
                self.in_features, r, self.expert_num, self.cur_task, self.training, self.layer_id)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: SAMELinearB(
                r, self.out_features, self.expert_num, self.cur_task, self.training, self.layer_id)}))
            self.scaling[adapter_name] = lora_alpha / r

            # ✅ 新增：显式设置所有专家参数为可训练
            for expert_id in range(self.expert_num):
                for param in self.lora_A[adapter_name].loraA[expert_id].parameters():
                    param.requires_grad_(True)
                for param in self.lora_B[adapter_name].loraB[expert_id].parameters():
                    param.requires_grad_(True)


        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)





class SAMELinear(nn.Linear, SAMELayer):
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
        self.te_dim = kwargs.pop("task_embedding_dim", 64)
        self.cur_task = kwargs.pop("cur_task", 0)
        self.layer_id = layer_id
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SAMELayer.__init__(
            self, in_features=in_features, out_features=out_features,
            expert_num=self.expert_num, cur_task=self.cur_task,
            training=train_signal, layer_id=layer_id,
        )

        self.max_components = 64
        self.window_size = 3
        self.training_signal = train_signal

        self.router = torch.ones(self.expert_num)  
        self.lora_router = nn.ModuleDict({adapter_name: SAMEExpert(self.in_features, self.expert_num)})

        self.weight.requires_grad = False
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

        self.register_buffer(f"cov_U_{adapter_name}", torch.zeros(in_features, self.max_components))
        self.register_buffer(f"cov_S_{adapter_name}", torch.zeros(self.max_components))
        self.register_buffer(f"cov_alpha_{adapter_name}", torch.tensor(0.0))
        self.register_buffer(f"cov_U_prev_{adapter_name}", torch.zeros(in_features, self.max_components))
        self.register_buffer(f"cov_S_prev_{adapter_name}", torch.zeros(self.max_components))
        self.register_buffer(f"cov_prev_valid_{adapter_name}", torch.tensor(False))
        self.register_buffer(f"utilization_{adapter_name}", torch.ones(self.expert_num) / self.expert_num)
        self.register_buffer(f"importance_{adapter_name}", torch.ones(self.expert_num) / self.expert_num)
        self.register_buffer(f"expert_masks_{adapter_name}", torch.ones(self.expert_num))

        self.current_step = 0
        self.start_step = 10
        self.all_frozen = False
        self.router_scaling_factor = 0.2
        self.expert_scaling_factor = 1e-3
        self.perp_suppression_factor = 1-self.expert_scaling_factor
        self.curvature_mu=0.9
        self.tau_score=0.1
        self._register_same_hooks(adapter_name)

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


    def _register_same_hooks(self, adapter_name):
        router_weight = self.lora_router[adapter_name].weight
        router_weight.register_hook(lambda grad: self._spectral_aware_router_hook(grad, adapter_name))


        for name in ['loraA']:
            module_dict = getattr(self, f'lora_{name[4]}')
            if adapter_name not in module_dict:
                continue
            experts = module_dict[adapter_name].loraA if name == 'loraA' else module_dict[adapter_name].loraB
            for expert_id in range(len(experts)):
                weight = experts[expert_id].mlp.weight
                if not weight.requires_grad:
                    print(f"[WARN] Layer{self.layer_id} Expert{expert_id} {name}.weight requires_grad=False, hook skipped")
                    assert(0)
                    continue
                hook_fn = self._make_curvature_hook(name, expert_id, adapter_name)
                weight.register_hook(hook_fn)


    def _spectral_aware_router_hook(self, grad, adapter):
        if not self.training or self.current_step < self.start_step:
            return grad


        U = getattr(self, f"cov_U_{adapter}")
        S = getattr(self, f"cov_S_{adapter}")
            # ✅ 改为能量阈值 0.95
        energy = S ** 2
        total_energy = energy.sum()
        
        if total_energy == 0:
            return grad
        
        # 计算累积能量占比
        cumsum = torch.cumsum(energy, dim=0)
        ratio = cumsum / total_energy
        
        # 保留累积能量达到 90% 的方向
        k = (ratio <= 0.9).sum().item() + 1
        k = min(k, len(S))  # 确保不越界
        
        if k == 0:
            return grad

        V_parallel = U[:, :k] 
        grad_parallel = grad @ V_parallel @ V_parallel.t()

        # 按奇异值从大到小累加能量
        S_smooth = torch.zeros_like(S[:k])
        for i in range(k):
            start = max(0, i - self.window_size + 1)
            S_smooth[i] = S[start:i+1].mean()
        

        scaling = 1.0 / (S_smooth + 1e-6)
        scaling = torch.clamp(scaling / scaling.max(), min=1.0 - self.router_scaling_factor,max=1.0 + self.router_scaling_factor)

        grad_parallel_scaled = (grad_parallel @ V_parallel) * scaling.unsqueeze(0)
        grad_parallel_scaled = grad_parallel_scaled @ V_parallel.t()


        grad_perp = grad - grad_parallel
        grad_perp_suppressed = grad_perp * self.perp_suppression_factor  


        return grad_parallel_scaled + grad_perp_suppressed
    
    # def _make_curvature_hook(self, name, expert_id, adapter):
    #     """正交投影到零空间：Lu et al. (AAAI-25) Consistent MoE Prompt"""
    #     def hook(grad):
    #         if not self.training or self.cur_task == 0:
    #             return grad
            
    #         cov_prev_valid = getattr(self, f"cov_prev_valid_{adapter}")
    #         if not cov_prev_valid:
    #             assert(0)
    #             return grad

    #         device = grad.device
    #         # U_prev: 协方差矩阵的特征向量 (对应论文中的 V), S_prev: 特征值 (对应奇异值平方)
    #         U_prev = getattr(self, f"cov_U_prev_{adapter}").to(device)
    #         S_prev = getattr(self, f"cov_S_prev_{adapter}").to(device)
            
    #         # ✅ 论文方法：基于奇异值阈值确定零空间 (而非能量占比)
    #         # 找出非零奇异值的最小值
    #         nonzero_mask = S_prev > 1e-8 * S_prev.max()
    #         if nonzero_mask.sum() == 0:
    #             return grad
    #         lambda_min = S_prev[nonzero_mask].min()
            
    #         # 零空间判定阈值 (alpha 通常设为 1e-3)
    #         alpha = 1e-3
    #         null_mask = S_prev < (alpha * lambda_min)
            
    #         # 获取零空间基向量 V_null (对应论文中的 \tilde{V})
    #         U_null = U_prev[:, null_mask]
            
    #         if U_null.size(1) == 0:
    #             return grad
            
    #         # ✅ 计算投影矩阵 H = V_null @ V_null.T (论文 Eq. 23)
    #         # 为了效率，直接计算 grad @ H 而不是构造完整矩阵
    #         grad_t = grad.t()
    #         # 投影到零空间: grad_proj = U_null @ (U_null.T @ grad_t)
    #         projected_grad_t = U_null @ (U_null.t() @ grad_t)
            
    #         # ✅ 专家松弛约束 (论文 Eq. 21 后文): Delta P = [eta * H + (1-eta) * I] @ Grad
    #         # 如果是专家参数，应用 eta 松弛 (eta 接近 1，如 0.99)；如果是 Router，eta=1.0
    #         eta = 0.99 
    #         scaled_grad_t = eta * projected_grad_t + (1 - eta) * grad_t
            
    #         scaled_grad = scaled_grad_t.t()

    #         # 更新梯度
    #         grad_norm_before = grad.norm().item()
    #         grad.data.copy_(scaled_grad.data)
    #         grad_norm_after = grad.norm().item()
    #             #         grad_norm_after = grad.norm().item()

    #         if grad_norm_before > 1e-10:
    #             ratio = grad_norm_after / grad_norm_before
    #             print(f"[HOOK] Layer{self.layer_id} {name} Expert{expert_id},ratio={ratio:.2f}x")

    #         return scaled_grad
    #     return hook
    
    # def _make_curvature_hook(self, name, expert_id, adapter):
    #     """硬正交投影：保留 95% 能量的方向"""
    #     def hook(grad):
    #         if not self.training or self.cur_task == 0:
    #             return grad
            
    #         cov_prev_valid = getattr(self, f"cov_prev_valid_{adapter}")
    #         if not cov_prev_valid:
    #             assert(0);
    #             return grad

    #         device = grad.device
    #         U_prev = getattr(self, f"cov_U_prev_{adapter}").to(device)
    #         S_prev = getattr(self, f"cov_S_prev_{adapter}").to(device)
            
    #         # ✅ 计算累积能量占比，找到保留的维度数 k
    #         energy = S_prev ** 2
    #         total_energy = energy.sum()
            
    #         # 计算累积和
    #         cumsum = torch.cumsum(energy, dim=0)
    #         ratio = cumsum / total_energy
            
    #         # 找到第一个超过 0.9 的位置
    #         k = (ratio <= 0.9).sum().item() + 1
    #         k = min(k, len(S_prev))  # 确保不越界
            
    #         if k == 0:
    #             assert(0);
    #             return grad
    #         grad_t = grad.t()
    #         proj_parallel = U_prev[:, :k] @ (U_prev[:, :k].t() @ grad_t)
    #         proj_perp = grad_t - proj_parallel
    #         scaled_grad = proj_perp.t()
    #         grad_norm_before = grad.norm().item()
    #         grad.data.copy_(scaled_grad.data)

    #         grad_norm_after = grad.norm().item()

    #         # if grad_norm_before > 1e-10:
    #         #     ratio = grad_norm_after / grad_norm_before
    #         #     print(f"[HOOK] Layer{self.layer_id} {name} Expert{expert_id} | "
    #         #           f"k={k}/{len(S_prev)}, ratio={ratio:.2f}x")
    #         return scaled_grad
    #     return hook
    

    def _make_curvature_hook(self, name, expert_id, adapter):
        def hook(grad):
            if not self.training or self.cur_task == 0:
                return grad
            cov_prev_valid = getattr(self, f"cov_prev_valid_{adapter}", False)
            if not cov_prev_valid:
                assert(0)
                return grad

            device = grad.device
            U_prev = getattr(self, f"cov_U_prev_{adapter}").to(device)
            S_prev = getattr(self, f"cov_S_prev_{adapter}").to(device)
            
            # 1. 确定有效秩 k（能量阈值法）
            energy = S_prev ** 2
            total_energy = energy.sum()
            if total_energy < 1e-10:
                assert(0)
                return grad
                
            cumsum = torch.cumsum(energy, dim=0)
            ratio = cumsum / (total_energy + 1e-10)
            k = (ratio <= 0.9).sum().item() + 1
            k = max(1, min(k, len(S_prev)))  # ✅ 确保 1 ≤ k ≤ len(S_prev)
            
            # 2. 曲率感知缩放因子
            mu = self.curvature_mu  # 建议设为 1e-3
            inv_S = 1.0 / (S_prev[:k] + mu)      # 平行方向缩放
            perp_scale = 1.0 / mu                 # 垂直方向缩放
            
            # 3. 梯度投影与缩放
            grad_t = grad.t()  # [d_in, d_out]
            U_k = U_prev[:, :k]  # [d_in, k]
            
            # 平行分量: V_k @ diag(inv_S) @ V_k.T @ grad.T
            proj_parallel = U_k @ (torch.diag(inv_S) @ (U_k.t() @ grad_t))
            
            # 垂直分量: (1/μ) * (I - V_k @ V_k.T) @ grad.T
            proj_perp = grad_t - U_k @ (U_k.t() @ grad_t)
            proj_perp_scaled = proj_perp * perp_scale
            
            # 4. 合并并返回
            scaled_grad_t = proj_parallel + proj_perp_scaled

            grad_norm_before = grad.norm().item()
            scaled_grad = scaled_grad_t.t()

            grad.data.copy_(scaled_grad.data)
            grad_norm_after = grad.norm().item()

            # if  grad_norm_before > 1e-10 and int(expert_id)!=self.cur_task:
            #     print(f"[HOOK] Layer{self.layer_id} {name} Expert{expert_id} | "
            #         f"k={k}/{len(S_prev)}, mu={mu:.1e} | "
            #         f"grad_norm: {grad_norm_before:.4f} → {grad_norm_after:.4f} "
            #         f"(ratio={grad_norm_after/grad_norm_before:.2f}x)")
            return scaled_grad
        return hook
    

    
    def _apply_topk_sparsification(self, router_probs, router, k=2):
        probs = router_probs * (router.to(router_probs.device) ** 2)
        #probs = router_probs
        probs = probs / (probs.sum() + 1e-8)
        if k < self.expert_num:
            topk_vals, topk_idx = torch.topk(probs, k)
            final_probs = torch.zeros_like(probs)
            final_probs[topk_idx] = probs[topk_idx]
            final_probs = final_probs / (final_probs.sum() + 1e-8)
        else:
            final_probs = probs
        return final_probs
    
    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
            if self.training:
                x_flat = x.view(-1, x.size(-1))
                self._update_router_covariance(x_flat, self.active_adapter)
                router_logits = self.lora_router[self.active_adapter](x.view(-1, x.size(-1)))
                g_phi = F.softmax(router_logits, dim=-1)
                g_phi_mean = g_phi.mean(dim=0)
                g_phi_mean = self._get_masked_routing(g_phi_mean)
                final_routing=g_phi_mean
            else:
                router_logits = self.lora_router[self.active_adapter](x.view(-1, x.size(-1)))/self.expert_num
                g_phi = F.softmax(router_logits, dim=-1)
                g_phi_mean = g_phi.mean(dim=0)
                final_routing = self._apply_topk_sparsification(g_phi_mean, self.router, k=1)
            
            final_routing = final_routing.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)
            lora_a_output = self.lora_A[self.active_adapter](x, final_routing)
            lora_b_output = self.lora_B[self.active_adapter](lora_a_output, final_routing)
            result += lora_b_output * self.scaling[self.active_adapter]

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        result = result.to(previous_dtype)

        if self.training:
            self.current_step += 1
            with torch.no_grad():
                self._update_activation_metrics(g_phi, x.view(-1, x.size(-1)))
                if not self.all_frozen:
                    self._apply_adaptive_freezing()

        return result

    def _update_router_covariance(self, x_flat, adapter):
        device = x_flat.device
        batch_size = x_flat.size(0)
        
        cov_alpha = getattr(self, f"cov_alpha_{adapter}")
        
        if cov_alpha == 0 or self.current_step % 20 == 0:
            d = x_flat.size(-1)
            k = min(self.max_components, d)

            x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)

            original_dtype = x_flat.dtype
            if original_dtype in [torch.bfloat16, torch.float16]:
                compute_dtype = torch.float32
            else:
                compute_dtype = original_dtype

            Q = torch.randn(d, k, device=device, dtype=compute_dtype)
            x_centered_comp = x_centered.to(compute_dtype)
            Y = x_centered_comp.t() @ (x_centered_comp @ Q)
            
            U_approx, _ = torch.linalg.qr(Y, mode='reduced')
            U_approx = U_approx.to(original_dtype)
            
            proj_x = x_centered @ U_approx
            S_approx = torch.sqrt(torch.sum(proj_x ** 2, dim=0) / (batch_size - 1 + 1e-8))
            S_approx = S_approx.to(original_dtype)

            setattr(self, f"cov_U_{adapter}", U_approx)
            setattr(self, f"cov_S_{adapter}", S_approx)


        setattr(self, f"cov_alpha_{adapter}", cov_alpha + batch_size)

    def _update_activation_metrics(self, routing_probs, x_flat):
        adapter = self.active_adapter
        device = x_flat.device
        

        batch_util = routing_probs.mean(dim=0)
        util = getattr(self, f"utilization_{adapter}").to(device)
        new_util = 0.95 * util + 0.05 * batch_util
        setattr(self, f"utilization_{adapter}", new_util.detach())


        input_energy = torch.norm(x_flat, dim=1, keepdim=True) ** 2
        weighted_energy = routing_probs * input_energy
        batch_importance = weighted_energy.mean(dim=0)
        importance = getattr(self, f"importance_{adapter}").to(device)
        new_importance = 0.95 * importance + 0.05 * batch_importance
        setattr(self, f"importance_{adapter}", new_importance.detach())

    def _apply_adaptive_freezing(self):
        adapter = self.active_adapter
        scores = self._compute_activation_scores(adapter)
        tau_score = self.tau_score
        current_task_expert = min(self.cur_task, self.expert_num - 1)
        masks = torch.ones(self.expert_num, device=scores.device)
        low_score_mask = scores < tau_score
        #make sure a least one expert is activate
        low_score_mask[current_task_expert] = False
        masks[low_score_mask] = 0.0
        other_experts_mask = torch.ones(self.expert_num, dtype=torch.bool, device=scores.device)
        other_experts_mask[current_task_expert] = False
        all_other_frozen = low_score_mask[other_experts_mask].all()
        if all_other_frozen:
            self.all_frozen = True
        setattr(self, f"expert_masks_{adapter}", masks.detach())

    def _compute_activation_scores(self, adapter):
        util = getattr(self, f"utilization_{adapter}")
        imp = getattr(self, f"importance_{adapter}")
        
        device = util.device
        
        def normalize(t):
            t = t.to(device)
            if t.max() == t.min():
                return torch.zeros_like(t)
            return (t - t.min()) / (t.max() - t.min() + 1e-8)
        
        util_norm = normalize(util)
        imp_norm = normalize(imp)
        return util_norm - imp_norm

    def _get_masked_routing(self, routing_probs):
        adapter = self.active_adapter
        device = routing_probs.device

        masks = getattr(self, f"expert_masks_{adapter}").to(device)
        
        masked = routing_probs * masks
        masked_sum = masked.sum(dim=-1, keepdim=True)
        if masked_sum.min() == 0:
            current_task_expert = min(self.cur_task, self.expert_num - 1)
            masked[:, current_task_expert] = 1.0
            masked_sum = masked.sum(dim=-1, keepdim=True)
        
        return masked / masked_sum

    def save_task_covariance_snapshot(self, adapter):
        U_curr = getattr(self, f"cov_U_{adapter}").clone()
        S_curr = getattr(self, f"cov_S_{adapter}").clone()
        k = (S_curr > 1e-6).sum().item()
        
        if k > 0:
            U_prev = torch.zeros_like(getattr(self, f"cov_U_prev_{adapter}"))
            S_prev = torch.zeros_like(getattr(self, f"cov_S_prev_{adapter}"))
            U_prev[:, :k] = U_curr[:, :k]
            S_prev[:k] = S_curr[:k]
            setattr(self, f"cov_U_prev_{adapter}", U_prev)
            setattr(self, f"cov_S_prev_{adapter}", S_prev)
            setattr(self, f"cov_prev_valid_{adapter}", torch.tensor(True))
        else:
            setattr(self, f"cov_prev_valid_{adapter}", torch.tensor(False))
            print(f"[Layer {self.layer_id}] ❌ No valid covariance to save")

    def reset_for_new_task(self, adapter):
        setattr(self, f"cov_alpha_{adapter}", torch.tensor(0.0))
        setattr(self, f"utilization_{adapter}", torch.ones(self.expert_num) / self.expert_num)
        setattr(self, f"expert_masks_{adapter}", torch.ones(self.expert_num))
        
        if self.training:
            lora_a_module = self.lora_A[adapter]
            lora_b_module = self.lora_B[adapter]
            for expert_id in range(self.expert_num):
                for param in lora_a_module.loraA[expert_id].parameters():
                    param.requires_grad_(True)
                for param in lora_b_module.loraB[expert_id].parameters():
                    param.requires_grad_(True)


class SAMELinearA(nn.Module):
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([SAMEExpert(in_features, out_features // expert_num) for _ in range(expert_num)])
        self.layer_id = layer_id

    def forward(self, x, routing_weights):
        output = self.loraA[0](x) * routing_weights[0]
        for i in range(1, self.expert_num):
            output += self.loraA[i](x) * routing_weights[i]
        return output

class SAMELinearB(nn.Module):
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer_id):
        super().__init__()
        self.expert_num = expert_num
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([SAMEExpert(in_features // expert_num, out_features) for _ in range(expert_num)])
        self.layer_id = layer_id

    def forward(self, x, routing_weights):
        output = self.loraB[0](x) * routing_weights[0]
        for i in range(1, self.expert_num):
            output += self.loraB[i](x) * routing_weights[i]
        return output

class SAMEExpert(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features, bias=False)
        self.weight = self.mlp.weight

    def forward(self, x):
        return self.mlp(x)