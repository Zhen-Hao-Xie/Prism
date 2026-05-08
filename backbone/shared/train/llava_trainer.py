import os
import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from typing import Optional

# 从 common 导入已解耦的功能
from common.save_checkpoint import get_mm_adapter_state_maybe_zero_3
from backbone.shared.data import LengthGroupedSampler


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
                generator=self.args.data_seed if self.args.data_seed is not None else None,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer with optional separate LR for mm_projector.
        """
        if is_sagemaker_mp_enabled() or self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model

        # 获取需要衰减的参数
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # 如果设置了 mm_projector 单独的学习率
        if self.args.mm_projector_lr is not None:
            projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
            optimizer_grouped_parameters = [
                # 非 projector 参数（衰减）
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_parameters and n not in projector_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                # 非 projector 参数（不衰减）
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_parameters and n not in projector_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
                # projector 参数（衰减）- 使用单独的学习率
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_parameters and n in projector_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.mm_projector_lr,
                },
                # projector 参数（不衰减）- 使用单独的学习率
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_parameters and n in projector_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_projector_lr,
                },
            ]
        else:
            # 标准分组
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_parameters and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        # 处理 ShardedDDP
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            self.optimizer = ZeroRedundancyOptimizer(
                params=optimizer_grouped_parameters,
                optimizer_class=optimizer_cls,
                **optimizer_kwargs,
            )
        else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # 特殊处理 Adam8bit（可选，可考虑移至 common）
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank in [0, -1]:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, 'mm_projector.bin'))
        else:
            super()._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if not getattr(self.args, 'tune_mm_mlp_adapter', False):
            super()._save(output_dir, state_dict)