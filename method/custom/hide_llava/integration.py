# method/hide/integration.py
"""
HiDe-LLaVA 方法实现
基于原型匹配的多模态任务路由 + HiDe MOE-LoRA
"""
from config.backbone.llava import CLIP_FEATURE_DIM
from method.base.integration import CLIntegration
from method.base.context import CLContext
from method.base.hooks import HookManager
from method.factory import CLMethodFactory
from method.base.peft_extension import register_peft_extension
from backbone.shared.peft_llm_targets import collect_peft_target_linear_suffixes
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os


_PEFT_EXT_REGISTERED = False


def ensure_peft_extension_registered() -> None:
    """
    按需把 HiDe 的 PEFT 扩展注入映射表。
    这个函数应当在真正需要 HiDe PEFT 的时候调用，避免 import-time 副作用。
    """
    global _PEFT_EXT_REGISTERED
    if _PEFT_EXT_REGISTERED:
        return

    # 延迟导入，避免提前拉起 PEFT/torch 依赖
    from PEFT.peft_model import PeftModelForCausalLMLORAMOE
    from PEFT.tuners.custom.hidellava import HiDeMOELoraConfig, HiDeMOELoraModel

    register_peft_extension(
        peft_type="MOE_LORA_HiDe",
        config_cls=HiDeMOELoraConfig,
        tuner_model_cls=HiDeMOELoraModel,
        task_type="CAUSAL_LM_HiDe",
        task_peft_model_cls=PeftModelForCausalLMLORAMOE,
    )
    _PEFT_EXT_REGISTERED = True


@CLMethodFactory.register("hide_llava", "hide")
class Hide_llavaIntegration(CLIntegration):
    """
    HiDe-LLaVA 集成类
    实现基于 CLIP 原型匹配的任务路由逻辑
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        self.hook_manager = HookManager()
        
        # === HiDe 配置参数（任务数；PEFT 内部仍称 expert_num）===
        self.task_num = int(getattr(config, "task_num", getattr(config, "expert_num", 8)))
        self.feature_dim = int(CLIP_FEATURE_DIM)
        
        # === 原型存储（任务数 × 特征维度）===
        # 使用 ParameterList 便于保存/加载，但不参与梯度更新
        self.image_anchors: Optional[torch.nn.ParameterList] = None
        self.text_anchors: Optional[torch.nn.ParameterList] = None
        self.image_boundary: Optional[torch.nn.ParameterList] = None  # 样本计数
        self.text_boundary: Optional[torch.nn.ParameterList] = None
        
        # === 状态缓存 ===
        self._last_predicted_task_id: Optional[int] = None


    # method/hide_llava/integration.py
    def initialize_model(self, model):
        """
        初始化 HiDe 相关组件
        - 加载/创建 anchors（原型）
        - 配置属性挂载
        - 配置 HiDe MOE-LoRA
        - 冻结 backbone 参数
        """
        device = next(model.parameters()).device
        for name, param in model.named_parameters():
            param.requires_grad = False

        print(f"\n{'='*70}")
        print("[HiDe] initialize_model start")
        
        # ========== 步骤 1: 加载/初始化 anchors ==========
        # 如果 anchors 已经有值（从 checkpoint 加载），不要重新初始化
        if self.image_anchors is None:
            self.image_anchors = torch.nn.ParameterList([
                torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                for _ in range(self.task_num)
            ]).to(device)
            print("  image_anchors initialized (random)")
        else:
            print("  image_anchors already present (from checkpoint)")
            for i, p in enumerate(self.image_anchors):
                print(f"    task_{i}: L2_norm={p.norm().item():.4f}")
        
        if self.text_anchors is None:
            self.text_anchors = torch.nn.ParameterList([
                torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                for _ in range(self.task_num)
            ]).to(device)
            print("  text_anchors initialized (random)")
        else:
            print("  text_anchors already present (from checkpoint)")
        
        if self.image_boundary is None:
            self.image_boundary = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                for _ in range(self.task_num)
            ]).to(device)
        if self.text_boundary is None:
            self.text_boundary = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                for _ in range(self.task_num)
            ]).to(device)
        
        # 挂载到模型
        model.image_anchors = self.image_anchors
        model.text_anchors = self.text_anchors
        model.image_boundary = self.image_boundary
        model.text_boundary = self.text_boundary
        model.task_num = self.task_num
    
        self._setup_hide_lora(model)


        # ========== 关键：在 LoRA 配置完成后，强制冻结 anchors ==========
        # 因为 PEFT 可能会改变 requires_grad 状态
        print("\nFreezing HiDe state parameters (anchors/boundaries)...")
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in ['image_anchors', 'text_anchors', 'image_boundary', 'text_boundary']):
                param.requires_grad = False
                frozen_count += 1
        print(f"  Froze {frozen_count} HiDe state parameters")
        # ===========================================================

        # ========== 步骤 4: 验证参数量 ==========
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n[HiDe] parameter count check:")
        print(f"  total parameters: {total_params:,}")
        print(f"  trainable parameters: {trainable_params:,}")
        print(f"  trainable ratio: {trainable_params / total_params * 100:.4f}%")
        
        print(f"{'='*70}\n")
        print(f"HiDe initialization done | num_tasks: {self.task_num} | feature_dim: {self.feature_dim}")
    
    def _setup_hide_lora(self, model):
        """配置 HiDe MOE-LoRA"""
        try:
            ensure_peft_extension_registered()
            from PEFT import HiDeMOELoraConfig, get_peft_model
            
            target_modules = self._find_target_modules(model)
            print(f"    target modules found: {len(target_modules)}")
            
            lora_config = HiDeMOELoraConfig(
                target_modules=target_modules,
                r=getattr(self.config, 'lora_r', 64),
                lora_alpha=getattr(self.config, 'lora_alpha', 128),
                lora_dropout=getattr(self.config, 'lora_dropout', 0.05),
                expert_num=self.task_num,
                cur_task=getattr(self.config, 'cur_task', 0),
                task_type="CAUSAL_LM",
                exclude_module_path_segments=self.peft_exclude_module_path_segments,
            )
            
            # 获取 CLModel 的_base_model 来应用 PEFT
            _base_model = getattr(model, '_base_model', None)
            if _base_model is not None:
                print("    applying PEFT to _base_model...")
                peft_model = get_peft_model(_base_model, lora_config)
                # 更新 CLModel 的_base_model 引用
                object.__setattr__(model, '_base_model', peft_model)
            else:
                print("    applying PEFT to model directly...")
                peft_model = get_peft_model(model, lora_config)
            
            # 打印 PEFT 信息
            if hasattr(peft_model, 'print_trainable_parameters'):
                peft_model.print_trainable_parameters()
            
            print("  HiDe MOE-LoRA configured")
            
        except ImportError as e:
            print(f"  HiDeMOELoraConfig not available, skipping LoRA setup: {e}")
        except Exception as e:
            print(f"  LoRA configuration failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_target_modules(self, model) -> List[str]:
        """PEFT 目标子模块名（后缀），由 ``peft_target_modules`` / ``lora_target_modules`` 控制，见 ``PEFT/utils/peft_target_modules``。"""
        return collect_peft_target_linear_suffixes(model, self.config)
    
    def on_input_prep(self, model, args, kwargs, context: CLContext):
        """输入准备阶段：实现 HiDe 的 CLIP 路由逻辑"""
        
        images = kwargs.get('images', None)
        input_ids = args[0] if args else None
        
        # ========== 获取 clip_tokenizer 和 text_tower ==========
        clip_tokenizer = getattr(model, 'clip_tokenizer', None)
        text_tower = getattr(model, 'text_tower', None)
        
        # 如果没找到，尝试从 base_model 获取
        if clip_tokenizer is None or text_tower is None:
            _base_model = getattr(model, '_base_model', None)
            if _base_model is not None:
                if clip_tokenizer is None:
                    clip_tokenizer = getattr(_base_model, 'clip_tokenizer', None)
                if text_tower is None:
                    text_tower = getattr(_base_model, 'text_tower', None)
        
        # 如果还是没找到，尝试从 base_model.base_model 获取
        if clip_tokenizer is None or text_tower is None:
            if _base_model is not None and hasattr(_base_model, 'base_model'):
                if clip_tokenizer is None:
                    clip_tokenizer = getattr(_base_model.base_model, 'clip_tokenizer', None)
                if text_tower is None:
                    text_tower = getattr(_base_model.base_model, 'text_tower', None)
        
        if clip_tokenizer is None or text_tower is None:
            print("[HiDe] skip routing: clip_tokenizer or text_tower is missing")
            return
        
        # ========== 路由策略 ==========
        # 训练：使用 context.task_id 作为当前样本任务（真值），不写死为 CLIP 预测。
        # 推理：仅文本走 _text_only_routing；图文走 _predict_task；均不使用 context.task_id。
        # 1) 仅文本 — 训练：传播 gold task_id；推理：_text_only_routing（CLIP vs text_anchors）
        # 2) 图像+文本 — 训练：_update_prototypes(..., context.task_id)；推理：_predict_task（0.5/0.5）
        if images is None:
            if model.training:
                tid = getattr(context, "task_id", None)
                if tid is not None:
                    try:
                        tid_i = int(tid)
                        if 0 <= tid_i < self.task_num:
                            self._propagate_task_id(model, tid_i)
                        else:
                            print(
                                f"[HiDe] training text-only: task_id={tid_i} out of range [0,{self.task_num}), "
                                "fallback to CLIP text routing"
                            )
                            self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
                    except (TypeError, ValueError):
                        self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
                else:
                    print("[HiDe] training text-only: context.task_id is None, using CLIP text routing")
                    self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
            else:
                self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
            return
        # 多模态（有图；通常也有 input_ids 供 CLIP 文本）
        else:
            # 提取 CLIP 特征
            # print("[HiDe] multimodal sample: extract features...")
            image_guide_features, text_guide_features = self._extract_clip_features(model, images, input_ids, clip_tokenizer, text_tower)
            
            if model.training:
                # 训练模式：更新原型
                self._update_prototypes(
                    image_guide_features, text_guide_features,
                    context.task_id, context,
                )
            else:
                print("Inference: predicting task for multimodal batch")
                predicted_task_id = self._predict_task(
                    image_guide_features, text_guide_features, context
                )
                self._propagate_task_id(model, predicted_task_id)


    def _extract_clip_features(
        self, model, images, input_ids, clip_tokenizer, text_tower
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = images.device

        # === 图像特征提取 ===
        vision_tower = getattr(model, 'vision_tower', None)
        if vision_tower and hasattr(vision_tower, 'is_loaded') and vision_tower.is_loaded:
            with torch.no_grad():
                raw_output = vision_tower(images)
                if isinstance(raw_output, tuple):
                    raw_features = raw_output[0]
                else:
                    raw_features = raw_output
                if raw_features.dim() == 3:
                    image_guide_features = raw_features.mean(dim=1)
                else:
                    image_guide_features = raw_features
        else:
            assert 0, "vision_tower not loaded"

        text_guide_features: Optional[torch.Tensor] = None

        # === 文本特征提取（无 input_ids / tokenizer 时不造随机向量，供路由走「仅图像」分支）===
        if input_ids is not None and clip_tokenizer is not None:
            # 获取主 tokenizer
            main_tokenizer = getattr(model, 'tokenizer', None)
            if main_tokenizer is None:
                _base_model = getattr(model, '_base_model', None)
                if _base_model is not None:
                    main_tokenizer = getattr(_base_model, 'tokenizer', None)
            
            if main_tokenizer is not None:
                input_pad = np.where(
                    input_ids.cpu().detach().numpy() != -200,
                    input_ids.cpu().detach().numpy(),
                    main_tokenizer.pad_token_id,
                )
                decoded_inputs = main_tokenizer.batch_decode(input_pad, skip_special_tokens=True)
                decoded_hidden = ['\n'.join(d.split('\n')[1:]) for d in decoded_inputs]
                decoded_clip = [d.split(' ASSISTANT')[0] for d in decoded_hidden]
            else:
                # 回退：直接用 clip_tokenizer 解码（可能乱码，但比空字符串好）
                input_pad = np.where(
                    input_ids.cpu().detach().numpy() != -200,
                    input_ids.cpu().detach().numpy(),
                    clip_tokenizer.pad_token_id,
                )
                decoded_inputs = clip_tokenizer.batch_decode(input_pad, skip_special_tokens=True)
                decoded_hidden = ['\n'.join(d.split('\n')[1:]) for d in decoded_inputs]
                decoded_clip = [d.split(' ASSISTANT')[0] for d in decoded_hidden]
                print("Warning: using clip_tokenizer fallback for decode")
            
            # 打印解码文本（调试）
            #print(f"    decoded_clip[0][:100]: {decoded_clip[0][:100]}")
            
            clip_inputs = clip_tokenizer(
                decoded_clip,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                text_guide_features = text_tower(clip_inputs)
                if isinstance(text_guide_features, tuple):
                    text_guide_features = text_guide_features[0]

        if text_guide_features is not None and text_guide_features.dim() == 1:
            text_guide_features = text_guide_features.unsqueeze(0)

        return image_guide_features, text_guide_features
    

    def _update_prototypes(
        self,
        image_feat: torch.Tensor,
        text_feat: Optional[torch.Tensor],
        task_id: Optional[int],
        context: CLContext,
    ):
        """
        训练时更新原型（滑动平均）
        公式: anchor_new = (anchor_old * count + feat_sum) / (count + batch_size)
        无 CLIP 文本特征时只更新图像原型。
        """
        if task_id is None or task_id >= self.task_num:
            return

        batch_size = image_feat.shape[0]
        if batch_size == 0:
            return

        task_idx = task_id

        with torch.no_grad():
            old_img_anchor = self.image_anchors[task_idx].data.clone()
            old_img_count = self.image_boundary[task_idx].data.clone()

            image_sum = old_img_anchor * old_img_count + image_feat.sum(dim=0)
            new_img_count = old_img_count + batch_size

            self.image_anchors[task_idx].data.copy_(image_sum / new_img_count)
            self.image_boundary[task_idx].data.copy_(new_img_count)

            self.image_anchors[task_idx].requires_grad = False
            self.image_boundary[task_idx].requires_grad = False

            if text_feat is not None:
                old_txt_anchor = self.text_anchors[task_idx].data.clone()
                old_txt_count = self.text_boundary[task_idx].data.clone()

                text_sum = old_txt_anchor * old_txt_count + text_feat.sum(dim=0)
                new_txt_count = old_txt_count + batch_size

                self.text_anchors[task_idx].data.copy_(text_sum / new_txt_count)
                self.text_boundary[task_idx].data.copy_(new_txt_count)

                self.text_anchors[task_idx].requires_grad = False
                self.text_boundary[task_idx].requires_grad = False

    def _task_cosine_max_vs_anchor(
        self, feat: torch.Tensor, anchors: torch.nn.ParameterList, t: int
    ) -> float:
        anchor = anchors[t].to(feat.device)
        return F.cosine_similarity(
            feat.unsqueeze(1),
            anchor.unsqueeze(0),
            dim=2,
        ).max().item()

    def _predict_task(
        self,
        image_feat: torch.Tensor,
        text_feat: Optional[torch.Tensor],
        context: CLContext,
    ) -> int:
        """
        多模态 batch 的路由（pre_generate / forward 中带图路径）：
        - 有 CLIP 文本特征：图像相似度与文本相似度各占 0.5 再 argmax。
        - 无 CLIP 文本特征：回退为仅用图像 vs image_anchors（非「纯文本」场景；纯文本走 _text_only_routing）。
        """
        device = image_feat.device

        image_sims = [
            self._task_cosine_max_vs_anchor(image_feat, self.image_anchors, t)
            for t in range(self.task_num)
        ]

        if text_feat is None:
            sim = torch.tensor(image_sims, device=device)
            predicted_task_id = int(torch.argmax(sim).item())
            print(
                f"    routing: multimodal fallback (no CLIP text) | image sims: {[f'{s:.4f}' for s in image_sims]}"
            )
            print(f"    argmax: {predicted_task_id}")
            return predicted_task_id

        text_sims = [
            self._task_cosine_max_vs_anchor(text_feat, self.text_anchors, t)
            for t in range(self.task_num)
        ]
        combined = [0.5 * img + 0.5 * txt for img, txt in zip(image_sims, text_sims)]
        sim = torch.tensor(combined, device=device)
        predicted_task_id = int(torch.argmax(sim).item())

        print(f"    routing: image + text (0.5 / 0.5) | image: {[f'{s:.4f}' for s in image_sims]}")
        print(f"    text: {[f'{s:.4f}' for s in text_sims]}")
        print(f"    combined: {[f'{s:.4f}' for s in combined]}")
        print(f"    argmax: {predicted_task_id}")

        return predicted_task_id
    
    def _propagate_task_id(self, model, task_id: int):
        """
        将任务 ID 写入 HiDeMOELoraLinear；并缓存供增量解码复用。
        训练时可传 gold ``context.task_id``；推理时应为 CLIP 路由结果，勿用 gold task_id。
        """
        for module in model.modules():
            if module.__class__.__name__ == 'HiDeMOELoraLinear':
                if hasattr(module, 'predicted_task_id'):
                    module.predicted_task_id = task_id

        model._last_predicted_task_id = task_id

    def _inference_routing_fallback_task_id(self, model) -> int:
        """推理失败回退：禁止用训练 gold task_id；仅用上一轮预测或 0。"""
        last = getattr(model, "_last_predicted_task_id", None)
        if last is not None and isinstance(last, int) and 0 <= last < self.task_num:
            return last
        return 0

    def _text_only_routing(self, model, input_ids, clip_tokenizer, text_tower):
        """
        仅文本样本的路由：CLIP 文本特征与各任务 text_anchors 的余弦相似度，argmax（纯文本预测）。
        """
        if input_ids is None or input_ids.shape[1] <= 1:
            # 增量解码阶段：复用之前的预测结果
            if hasattr(model, '_last_predicted_task_id') and model._last_predicted_task_id is not None:
                self._propagate_task_id(model, model._last_predicted_task_id)
            return
        
        # 获取设备
        device = next(model.parameters()).device
        
        # 解码并编码文本
        input_pad = np.where(
            input_ids.cpu().detach().numpy() != -200,
            input_ids.cpu().detach().numpy(),
            clip_tokenizer.pad_token_id,
        )
        decoded_inputs = clip_tokenizer.batch_decode(input_pad, skip_special_tokens=True)
        decoded_hidden = ['\n'.join(d.split('\n')[1:]) for d in decoded_inputs]
        decoded_clip = [d.split(' ASSISTANT')[0] for d in decoded_hidden]
        
        clip_inputs = clip_tokenizer(
            decoded_clip,
            padding="longest",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            text_feat = text_tower(clip_inputs)  # [B, 768]
            text_feat = text_feat.to(device)  # 确保在正确的设备上
        
        # 确保 text_anchors 在正确的设备上
        for t in range(self.task_num):
            if hasattr(self.text_anchors[t], 'to'):
                self.text_anchors[t] = self.text_anchors[t].to(device)
        
        # 预测任务（仅文本相似度）
        text_sims = []
        for t in range(self.task_num):
            # 确保两个 tensor 在同一设备
            text_feat_device = text_feat.device
            anchor_device = self.text_anchors[t].device
            
            if text_feat_device != anchor_device:
                self.text_anchors[t] = self.text_anchors[t].to(text_feat_device)
            
            sim = F.cosine_similarity(
                text_feat.unsqueeze(1),      # [B, 1, D]
                self.text_anchors[t].unsqueeze(0),  # [1, D]
                dim=2
            ).max().item()
            text_sims.append(sim)
        
        predicted_task_id = int(torch.argmax(torch.tensor(text_sims)).item())
        print(f"Text-only routing: predicted task_id={predicted_task_id}")
        self._propagate_task_id(model, predicted_task_id)
    
    def on_forward_start(self,model, context: CLContext):
        """Forward 开始前：可选的清理操作"""
        pass
    
    def on_forward_end(self, model, outputs, context: CLContext):
        """Forward 结束后：可选的后处理"""
        return outputs
    
    def on_step_end(self, model, context: CLContext, loss=None):
        """训练步结束后：可选的状态更新"""
        # HiDe 的原型更新已在 on_input_prep 中完成，这里无需额外操作
        pass
    
    def on_task_end(self, model, context: CLContext, task_id: int):
        """
        任务训练结束后：冻结当前任务的原型（可选）
        并保存状态
        """
        print(f"HiDe task {task_id} finished | prototypes updated in training loop")
    
    def get_inference_config(self) -> Dict:
        """返回推理时需要的配置"""
        return {
            "task_num": self.task_num,
            "feature_dim": self.feature_dim,
            "task_to_category": self.task_to_category,
            "category_to_tasks": self.category_to_tasks,
        }
    

    # method/hide_llava/integration.py
    def save_extra_state(self, output_dir: str, model=None):
        """保存 HiDe 特定状态（所有任务的原型）"""
        import os
        import torch
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[HiDe] saving extra state to {output_dir}...")
        
        # 收集 anchors 和 boundaries
        state = {}
        
        # 图像 anchors - 保存所有 8 个任务
        if self.image_anchors is not None:
            state['image_anchors'] = [p.cpu().clone() for p in self.image_anchors]
            print(f"  image_anchors: {len(self.image_anchors)} tasks")
            for i, p in enumerate(self.image_anchors):
                print(f"    task_{i}: norm={p.norm().item():.4f}")
        
        # 文本 anchors - 保存所有 8 个任务
        if self.text_anchors is not None:
            state['text_anchors'] = [p.cpu().clone() for p in self.text_anchors]
            print(f"  text_anchors: {len(self.text_anchors)} tasks")
            for i, p in enumerate(self.text_anchors):
                print(f"    task_{i}: norm={p.norm().item():.4f}")
        
        # 图像 boundaries
        if self.image_boundary is not None:
            state['image_boundary'] = [p.cpu().clone() for p in self.image_boundary]
            print(f"  image_boundary: {len(self.image_boundary)} tasks")
        
        # 文本 boundaries
        if self.text_boundary is not None:
            state['text_boundary'] = [p.cpu().clone() for p in self.text_boundary]
            print(f"  text_boundary: {len(self.text_boundary)} tasks")
        
        # 保存元数据（保留 expert_num 键以兼容旧 checkpoint 读取逻辑）
        state["task_num"] = self.task_num
        state["expert_num"] = self.task_num
        state['_last_predicted_task_id'] = self._last_predicted_task_id
        
        # 保存为单独文件
        if state:
            save_path = os.path.join(output_dir, 'hide_state.pt')
            torch.save(state, save_path)
            print(f"HiDe state saved: {save_path}")
            print(f"   file size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
            return True
        else:
            print("HiDe state dict empty; nothing saved")
            return False
    
    # method/hide_llava/integration.py

    def load_extra_state(self, load_dir: str, model=None):
        """加载 HiDe 特定状态"""
        import os
        import torch
        
        load_path = os.path.join(load_dir, 'hide_state.pt')
        if not os.path.exists(load_path):
            print(f"HiDe state file not found: {load_path}")
            return False
        
        print(f"\n{'='*70}")
        print(f"[HiDe] loading extra state from {load_path}...")
        print(f"{'='*70}")
        
        state = torch.load(load_path, map_location='cpu')
        
        # ========== 打印加载的 anchors 范数 ==========
        print("\nLoaded image_anchors L2 norms:")
        if 'image_anchors' in state:
            for i, p in enumerate(state['image_anchors']):
                norm = torch.norm(p).item()
                print(f"    task_{i}: {norm:.4f}")
        else:
            print("    no image_anchors in checkpoint")
        
        print("\nLoaded text_anchors L2 norms:")
        if 'text_anchors' in state:
            for i, p in enumerate(state['text_anchors']):
                norm = torch.norm(p).item()
                print(f"    task_{i}: {norm:.4f}")
        else:
            print("    no text_anchors in checkpoint")
        
        print("\nLoaded image_boundary:")
        if 'image_boundary' in state:
            for i, b in enumerate(state['image_boundary']):
                print(f"    task_{i}: {b.item():.2f}")
        
        print("\nLoaded text_boundary:")
        if 'text_boundary' in state:
            for i, b in enumerate(state['text_boundary']):
                print(f"    task_{i}: {b.item():.2f}")
        # ==========================================
        
        # 恢复图像 anchors
        if 'image_anchors' in state and self.image_anchors is not None:
            for i, p in enumerate(state['image_anchors']):
                if i < len(self.image_anchors):
                    self.image_anchors[i].data.copy_(p)
            print("\n  image_anchors restored from checkpoint")
        
        # 恢复文本 anchors
        if 'text_anchors' in state and self.text_anchors is not None:
            for i, p in enumerate(state['text_anchors']):
                if i < len(self.text_anchors):
                    self.text_anchors[i].data.copy_(p)
            print("  text_anchors restored from checkpoint")
        
        # 恢复 boundaries
        if 'image_boundary' in state and self.image_boundary is not None:
            for i, p in enumerate(state['image_boundary']):
                if i < len(self.image_boundary):
                    self.image_boundary[i].data.copy_(p)
        if 'text_boundary' in state and self.text_boundary is not None:
            for i, p in enumerate(state['text_boundary']):
                if i < len(self.text_boundary):
                    self.text_boundary[i].data.copy_(p)
        
        # 设置到 model
        if model is not None:
            print("\n  attaching anchors to model...")
            if self.image_anchors is not None:
                object.__setattr__(model, 'image_anchors', self.image_anchors)
            if self.text_anchors is not None:
                object.__setattr__(model, 'text_anchors', self.text_anchors)
            if self.image_boundary is not None:
                object.__setattr__(model, 'image_boundary', self.image_boundary)
            if self.text_boundary is not None:
                object.__setattr__(model, 'text_boundary', self.text_boundary)
            print("  anchors attached on model")
        
        print(f"\nHiDe state loaded: {load_path}")
        print(f"{'='*70}\n")
        return True

    def pre_generate_hook(self, model, input_ids, images, context) -> bool:
        """
        HiDe 的 generate 前钩子：进行路由预测。
        纯文本：与 on_input_prep 一致，走 _text_only_routing（纯文本 vs text_anchors）。
        图文：_extract_clip_features + _predict_task（0.5/0.5）。
        """
        clip_tokenizer = getattr(model, "clip_tokenizer", None)
        text_tower = getattr(model, "text_tower", None)
        _base_model = getattr(model, "_base_model", None)
        if _base_model is not None:
            if clip_tokenizer is None:
                clip_tokenizer = getattr(_base_model, "clip_tokenizer", None)
            if text_tower is None:
                text_tower = getattr(_base_model, "text_tower", None)
        if _base_model is not None and hasattr(_base_model, "base_model"):
            if clip_tokenizer is None:
                clip_tokenizer = getattr(_base_model.base_model, "clip_tokenizer", None)
            if text_tower is None:
                text_tower = getattr(_base_model.base_model, "text_tower", None)

        if images is None:
            if clip_tokenizer is None or text_tower is None:
                fb = self._inference_routing_fallback_task_id(model)
                self._propagate_task_id(model, fb)
                print(
                    f"[HiDe] text-only generate: clip missing, inference fallback={fb} "
                    "(not using training task_id)"
                )
                return True
            self._text_only_routing(model, input_ids, clip_tokenizer, text_tower)
            return True

        if clip_tokenizer is None or text_tower is None:
            fb = self._inference_routing_fallback_task_id(model)
            print(
                f"[HiDe] multimodal generate: clip missing, inference fallback={fb} "
                "(not using training task_id)"
            )
            self._propagate_task_id(model, fb)
            return True

        image_guide_features, text_guide_features = self._extract_clip_features(
            model, images, input_ids, clip_tokenizer, text_tower
        )
        predicted_task_id = self._predict_task(
            image_guide_features, text_guide_features, context
        )
        self._propagate_task_id(model, predicted_task_id)
        return True