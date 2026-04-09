# method/hide/integration.py
"""
HiDe-LLaVA 方法实现
基于原型匹配的多模态任务路由 + HiDe MOE-LoRA
"""
from method.base.integration import CLIntegration
from method.base.context import CLContext
from method.base.hooks import HookManager
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import os


class Hide_llavaIntegration(CLIntegration):
    """
    HiDe-LLaVA 集成类
    实现基于 CLIP 原型匹配的任务路由逻辑
    """
    
    def __init__(self, config: Any):
        super().__init__(config)
        self.hook_manager = HookManager()
        
        # === HiDe 配置参数 ===
        self.num_tasks = getattr(config, 'expert_num', 8)
        self.feature_dim = getattr(config, 'clip_feature_dim', 768)  # CLIP 特征维度
        self.expert_num = getattr(config, 'expert_num', 8)
        
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
        print(f"🔧 [HiDe] initialize_model 开始")
        
        # ========== 步骤 1: 加载/初始化 anchors ==========
        # 如果 anchors 已经有值（从 checkpoint 加载），不要重新初始化
        if self.image_anchors is None:
            self.image_anchors = torch.nn.ParameterList([
                torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                for _ in range(self.num_tasks)
            ]).to(device)
            print(f"  ✅ image_anchors 已初始化（随机）")
        else:
            print(f"  ✅ image_anchors 已存在（从 checkpoint 加载）")
            for i, p in enumerate(self.image_anchors):
                print(f"    task_{i}: L2_norm={p.norm().item():.4f}")
        
        if self.text_anchors is None:
            self.text_anchors = torch.nn.ParameterList([
                torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
                for _ in range(self.num_tasks)
            ]).to(device)
            print(f"  ✅ text_anchors 已初始化（随机）")
        else:
            print(f"  ✅ text_anchors 已存在（从 checkpoint 加载）")
        
        if self.image_boundary is None:
            self.image_boundary = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                for _ in range(self.num_tasks)
            ]).to(device)
        if self.text_boundary is None:
            self.text_boundary = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
                for _ in range(self.num_tasks)
            ]).to(device)
        
        # 挂载到模型
        model.image_anchors = self.image_anchors
        model.text_anchors = self.text_anchors
        model.image_boundary = self.image_boundary
        model.text_boundary = self.text_boundary
        model.expert_num = self.expert_num
    
        self._setup_hide_lora(model)
        # ========== 步骤 4: 验证参数量 ==========
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n🔍 [HiDe] 参数量验证:")
        print(f"  总参数量：{total_params:,}")
        print(f"  可训练参数量：{trainable_params:,}")
        print(f"  可训练比例：{trainable_params / total_params * 100:.4f}%")
        
        if trainable_params < 1000000:
            print(f"  ⚠️  警告：可训练参数过少！")
        elif trainable_params > 100000000:
            print(f"  ⚠️  警告：可训练参数过多！")
        else:
            print(f"  ✅ 参数量正常")
        
        print(f"{'='*70}\n")
        print(f"✅ HiDe 初始化完成 | 任务数：{self.num_tasks} | 特征维度：{self.feature_dim}")
    # def initialize_model(self, model):
    #     """
    #     初始化 HiDe 相关组件
    #     - 创建 anchors（原型）
    #     - 配置属性挂载
    #     - 可选：配置 HiDe MOE-LoRA
    #     """
    #     device = next(model.parameters()).device
    #     print(f"  冻结 backbone 参数...")
    #     for name, param in model.named_parameters():
    #         param.requires_grad = False
    #     # 1. 初始化原型参数（随机初始化，训练时逐步更新）
    #     self.image_anchors = torch.nn.ParameterList([
    #         torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
    #         for _ in range(self.num_tasks)
    #     ]).to(device)
        
    #     self.text_anchors = torch.nn.ParameterList([
    #         torch.nn.Parameter(0.1 * torch.randn(1, self.feature_dim), requires_grad=False)
    #         for _ in range(self.num_tasks)
    #     ]).to(device)
        
    #     self.image_boundary = torch.nn.ParameterList([
    #         torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
    #         for _ in range(self.num_tasks)
    #     ]).to(device)
        
    #     self.text_boundary = torch.nn.ParameterList([
    #         torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
    #         for _ in range(self.num_tasks)
    #     ]).to(device)
        
    #     # 2. 挂载到模型（方便其他模块访问）
    #     model.image_anchors = self.image_anchors
    #     model.text_anchors = self.text_anchors
    #     model.image_boundary = self.image_boundary
    #     model.text_boundary = self.text_boundary
    #     model.expert_num = self.expert_num
        
    #     # 3. 初始化状态缓存
    #     if not hasattr(model, '_last_predicted_task_id'):
    #         model._last_predicted_task_id = None
        
    #     self._setup_hide_lora(model)
        
    #     print(f"✅ HiDe 初始化完成 | 任务数: {self.num_tasks} | 特征维度: {self.feature_dim}")
    
    def _setup_hide_lora(self, model):
        """配置 HiDe MOE-LoRA"""
        try:
            from PEFT.peft import HiDeMOELoraConfig, get_peft_model
            
            target_modules = self._find_target_modules(model)
            print(f"    找到目标模块：{len(target_modules)} 个")
            
            lora_config = HiDeMOELoraConfig(
                target_modules=target_modules,
                r=getattr(self.config, 'lora_r', 64),
                lora_alpha=getattr(self.config, 'lora_alpha', 128),
                lora_dropout=getattr(self.config, 'lora_dropout', 0.05),
                expert_num=self.expert_num,
                cur_task=getattr(self.config, 'cur_task', 0),
                task_type="CAUSAL_LM",
            )
            
            # 获取 CLModel 的_base_model 来应用 PEFT
            _base_model = getattr(model, '_base_model', None)
            if _base_model is not None:
                print(f"    对 _base_model 应用 PEFT...")
                peft_model = get_peft_model(_base_model, lora_config)
                # 更新 CLModel 的_base_model 引用
                object.__setattr__(model, '_base_model', peft_model)
            else:
                print(f"    对 model 直接应用 PEFT...")
                peft_model = get_peft_model(model, lora_config)
            
            # 打印 PEFT 信息
            if hasattr(peft_model, 'print_trainable_parameters'):
                peft_model.print_trainable_parameters()
            
            print(f"  ✅ HiDe MOE-LoRA 配置完成")
            
        except ImportError as e:
            print(f"  ⚠️  未找到 HiDeMOELoraConfig，跳过 LoRA 配置：{e}")
        except Exception as e:
            print(f"  ❌ LoRA 配置失败：{e}")
            import traceback
            traceback.print_exc()
    
    def _find_target_modules(self, model) -> List[str]:
        """查找需要注入 LoRA 的目标模块（只返回 Linear 层类型名）"""
        target_modules = set()
        
        # 获取真正的 base_model
        _base_model = getattr(model, '_base_model', None)
        if _base_model is None:
            _base_model = model
        
        # 只查找 Linear 层
        for name, module in _base_model.named_modules():
            # 关键修复：只处理 Linear 层
            if isinstance(module, torch.nn.Linear):
                if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                    # 只保留模块类型名
                    module_type = name.split('.')[-1]
                    target_modules.add(module_type)
        
        print(f"    找到目标模块类型：{list(target_modules)}")
        return list(target_modules)
    
    def on_input_prep(self, model, args, kwargs, context: CLContext):
        """输入准备阶段：实现 HiDe 的 CLIP 路由逻辑"""
        
        
        images = kwargs.get('images', None)
        input_ids = args[0] if args else None
        
        # ========== 关键修复：多方式获取 clip_tokenizer 和 text_tower ==========
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
        
        # 如果还是没找到，尝试从 base_model.base_model 获取（PEFT 嵌套）
        if clip_tokenizer is None or text_tower is None:
            if _base_model is not None and hasattr(_base_model, 'base_model'):
                if clip_tokenizer is None:
                    clip_tokenizer = getattr(_base_model.base_model, 'clip_tokenizer', None)
                if text_tower is None:
                    text_tower = getattr(_base_model.base_model, 'text_tower', None)
        
        
        if clip_tokenizer is None or text_tower is None:
            print(f"⚠️ [HiDe] 跳过：clip_tokenizer 或 text_tower 为空")
            return
        
        # === 场景 1: 纯文本样本（无图像）===
        if images is None and not model.training:
            self._text_only_routing(model, input_ids, clip_tokenizer, text_tower, context)
            return
        
        # === 场景 2: 多模态样本 ===
        if images is not None:
            
            # 提取 CLIP 特征
            image_guide_features, text_guide_features = self._extract_clip_features(
                model, images, input_ids, clip_tokenizer, text_tower
            )
            
            if model.training:
                
                # 训练模式：更新原型
                self._update_prototypes(
                    image_guide_features, text_guide_features, 
                    context.task_id, context
                )
            else:
                # 推理模式：预测任务并传播
                predicted_task_id = self._predict_task(
                    image_guide_features, text_guide_features, context
                )
                self._propagate_task_id(model, predicted_task_id)
        else:
            print(f"⚠️ [HiDe] images is None，跳过原型更新")

    def _extract_clip_features(self, model, images, input_ids, clip_tokenizer, text_tower):
        """
        提取 CLIP 的图像和文本特征
        返回: (image_features: [B, D], text_features: [B, D])
        """
        device = images.device
        
        # === 图像特征提取 ===
        if hasattr(model, 'encode_images'):
            # 使用模型的 encode_images 方法（LLaVA 标准接口）
            image_guide_features, _ = model.encode_images(images)
            # image_guide_features: [B, num_patches, D] -> [B, D]
            if image_guide_features.dim() == 3:
                image_guide_features = image_guide_features[:, 0]  # 取 first patch 或 mean
        else:
            # 备用方案：直接通过 vision tower
            vision_tower = getattr(model, 'vision_tower', None)
            if vision_tower and hasattr(vision_tower, 'is_loaded') and vision_tower.is_loaded:
                with torch.no_grad():
                    raw_features = vision_tower(images)
                image_guide_features = raw_features.mean(dim=1)  # [B, D]
            else:
                # 兜底：随机特征（仅用于调试）
                image_guide_features = torch.randn(
                    images.shape[0], self.feature_dim, device=device
                )
        
        # === 文本特征提取 ===
        if input_ids is not None and clip_tokenizer is not None:
            # 解码 input_ids 为文本
            input_pad = np.where(
                input_ids.cpu().detach().numpy() != -200,
                input_ids.cpu().detach().numpy(),
                clip_tokenizer.pad_token_id,
            )
            decoded_inputs = clip_tokenizer.batch_decode(input_pad, skip_special_tokens=True)
            # 提取 CLIP 输入部分（参考原代码逻辑）
            decoded_hidden = ['\n'.join(d.split('\n')[1:]) for d in decoded_inputs]
            decoded_clip = [d.split(' ASSISTANT')[0] for d in decoded_hidden]
            
            # CLIP 编码
            clip_inputs = clip_tokenizer(
                decoded_clip,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            with torch.no_grad():
                text_guide_features = text_tower(clip_inputs)  # [B, 768]
        else:
            text_guide_features = torch.randn(
                1, self.feature_dim, device=device
            )
        
        return image_guide_features, text_guide_features
    
    def _update_prototypes(self, image_feat: torch.Tensor, text_feat: torch.Tensor, task_id: Optional[int], context: CLContext):
        """
        训练时更新原型（滑动平均）
        公式: anchor_new = (anchor_old * count + feat_sum) / (count + batch_size)
        """
        if task_id is None or task_id >= self.num_tasks:
            return
            
        batch_size = image_feat.shape[0]
        if batch_size == 0:
            return
        
        # 更新图像原型
        image_sum = self.image_anchors[task_id] * self.image_boundary[task_id] + image_feat.sum(dim=0)
        self.image_boundary[task_id].data += batch_size
        self.image_anchors[task_id] = image_sum / self.image_boundary[task_id]
        
        # 更新文本原型
        text_sum = self.text_anchors[task_id] * self.text_boundary[task_id] + text_feat.sum(dim=0)
        self.text_boundary[task_id].data += batch_size
        self.text_anchors[task_id] = text_sum / self.text_boundary[task_id]
    
    def _predict_task(self, image_feat: torch.Tensor, text_feat: torch.Tensor, context: CLContext ) -> int:
        """
        推理时预测任务 ID
        通过计算与每个任务原型的余弦相似度，取最大值
        """
        device = image_feat.device
        
        # 计算相似度
        image_sims = []
        text_sims = []
        
        for t in range(self.expert_num):
            # 图像相似度: [B, 1, D] vs [1, D] -> [B, 1] -> scalar
            img_sim = F.cosine_similarity(
                image_feat.unsqueeze(1),  # [B, 1, D]
                self.image_anchors[t],     # [1, D]
                dim=2
            ).max().item()
            image_sims.append(img_sim)
            
            # 文本相似度
            txt_sim = F.cosine_similarity(
                text_feat.unsqueeze(1).to(device),
                self.text_anchors[t],
                dim=2
            ).max().item()
            text_sims.append(txt_sim)
        
        sim = (torch.tensor(image_sims, device=device) + torch.tensor(text_sims, device=device)) / 2
        sim = torch.tensor(text_sims, device=device)  # 参考原代码
        
        predicted_task_id = int(torch.argmax(sim).item())
        return predicted_task_id
    
    def _propagate_task_id(self, model, task_id: int):
        """
        将预测的任务 ID 传播到所有 HiDeMOELoraLinear 层
        确保推理时每个样本使用正确的专家
        """
        count = 0
        for module in model.modules():
            if module.__class__.__name__ == 'HiDeMOELoraLinear':
                if hasattr(module, 'predicted_task_id'):
                    module.predicted_task_id = task_id
                    count += 1
        
        # 缓存预测结果（供增量解码阶段复用）
        model._last_predicted_task_id = task_id
        
        if count > 0:
            print(f"✅ 任务 ID 传播 | task={task_id} | HiDe 层数={count}")
    
    def _text_only_routing(self, model, input_ids, clip_tokenizer, text_tower):
        """
        纯文本样本的路由（ScienceQA 等无图像场景）
        仅使用文本特征进行任务预测
        """
        if input_ids is None or input_ids.shape[1] <= 1:
            # 增量解码阶段：复用之前的预测结果
            if hasattr(model, '_last_predicted_task_id') and model._last_predicted_task_id is not None:
                self._propagate_task_id(model, model._last_predicted_task_id)
            return
        
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
        ).to(next(model.parameters()).device)
        
        with torch.no_grad():
            text_feat = text_tower(clip_inputs)  # [B, 768]
        
        # 预测任务（仅文本相似度）
        text_sims = []
        for t in range(self.expert_num):
            sim = F.cosine_similarity(
                text_feat.unsqueeze(1),
                self.text_anchors[t],
                dim=2
            ).max().item()
            text_sims.append(sim)
        
        predicted_task_id = int(torch.argmax(torch.tensor(text_sims)).item())
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
        print(f"✅ HiDe 任务 {task_id} 完成 | 原型已更新")
    
    def get_inference_config(self) -> Dict:
        """返回推理时需要的配置"""
        return {
            'expert_num': self.expert_num,
            'feature_dim': self.feature_dim,
            'task_to_category': self.task_to_category,
            'category_to_tasks': self.category_to_tasks,
        }
    

    # method/hide_llava/integration.py
    def save_extra_state(self, output_dir: str):
        """保存 HiDe 特定状态（所有任务的原型）"""
        import os
        import torch
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔍 [HiDe] 保存额外状态到 {output_dir}...")
        
        # 收集 anchors 和 boundaries
        state = {}
        
        # 图像 anchors - 保存所有 8 个任务
        if self.image_anchors is not None:
            state['image_anchors'] = [p.cpu().clone() for p in self.image_anchors]
            print(f"  ✅ image_anchors: {len(self.image_anchors)} 个任务")
            for i, p in enumerate(self.image_anchors):
                print(f"    task_{i}: norm={p.norm().item():.4f}")
        
        # 文本 anchors - 保存所有 8 个任务
        if self.text_anchors is not None:
            state['text_anchors'] = [p.cpu().clone() for p in self.text_anchors]
            print(f"  ✅ text_anchors: {len(self.text_anchors)} 个任务")
            for i, p in enumerate(self.text_anchors):
                print(f"    task_{i}: norm={p.norm().item():.4f}")
        
        # 图像 boundaries
        if self.image_boundary is not None:
            state['image_boundary'] = [p.cpu().clone() for p in self.image_boundary]
            print(f"  ✅ image_boundary: {len(self.image_boundary)} 个任务")
        
        # 文本 boundaries
        if self.text_boundary is not None:
            state['text_boundary'] = [p.cpu().clone() for p in self.text_boundary]
            print(f"  ✅ text_boundary: {len(self.text_boundary)} 个任务")
        
        # 保存元数据
        state['expert_num'] = self.expert_num
        state['num_tasks'] = self.num_tasks
        state['_last_predicted_task_id'] = self._last_predicted_task_id
        
        # 保存为单独文件
        if state:
            save_path = os.path.join(output_dir, 'hide_state.pt')
            torch.save(state, save_path)
            print(f"✅ HiDe 状态已保存：{save_path}")
            print(f"   文件大小：{os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"⚠️  HiDe 状态为空，未保存")
            return False
    

    # method/hide_llava/integration.py
    def load_extra_state(self, load_dir: str, model=None):
        """加载 HiDe 特定状态"""
        import os
        import torch
        
        load_path = os.path.join(load_dir, 'hide_state.pt')
        if not os.path.exists(load_path):
            print(f"⚠️  未找到 HiDe 状态文件：{load_path}")
            return False
        
        print(f"🔍 [HiDe] 加载额外状态从 {load_path}...")
        state = torch.load(load_path, map_location='cpu')
        
        # 恢复图像 anchors
        if 'image_anchors' in state and self.image_anchors is not None:
            for i, p in enumerate(state['image_anchors']):
                if i < len(self.image_anchors):
                    self.image_anchors[i].data.copy_(p)
            print(f"  ✅ image_anchors 已恢复")
        
        # 恢复文本 anchors
        if 'text_anchors' in state and self.text_anchors is not None:
            for i, p in enumerate(state['text_anchors']):
                if i < len(self.text_anchors):
                    self.text_anchors[i].data.copy_(p)
            print(f"  ✅ text_anchors 已恢复")
        
        # 恢复 boundaries
        if 'image_boundary' in state and self.image_boundary is not None:
            for i, p in enumerate(state['image_boundary']):
                if i < len(self.image_boundary):
                    self.image_boundary[i].data.copy_(p)
        if 'text_boundary' in state and self.text_boundary is not None:
            for i, p in enumerate(state['text_boundary']):
                if i < len(self.text_boundary):
                    self.text_boundary[i].data.copy_(p)
        
        # 关键：如果传入了 model，直接设置到 model
        if model is not None:
            print(f"  🔧 设置 anchors 到 model...")
            if self.image_anchors is not None:
                object.__setattr__(model, 'image_anchors', self.image_anchors)
            if self.text_anchors is not None:
                object.__setattr__(model, 'text_anchors', self.text_anchors)
            if self.image_boundary is not None:
                object.__setattr__(model, 'image_boundary', self.image_boundary)
            if self.text_boundary is not None:
                object.__setattr__(model, 'text_boundary', self.text_boundary)
            print(f"  ✅ anchors 已直接设置到 model")
        
        print(f"✅ HiDe 状态已加载：{load_path}")
        return True