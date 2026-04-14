# method/base/cl_model.py
"""
持续学习模型包装器

关键：
1. 直接操作 _modules 字典注册 base_model，避免__setattr__ 循环依赖
2. 在 forward 中手动调用 prepare_inputs_labels_for_multimodal
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import CausalLMOutputWithPast

from .context import CLContext
from .integration import CLIntegration


class CLModel(nn.Module):
    """持续学习模型包装器"""
    
    def __init__(self, base_model, integration: CLIntegration):
        # 关键 1: 调用 nn.Module.__init__
        nn.Module.__init__(self)
        
        # 关键 2: 设置内部属性
        object.__setattr__(self, '_integration', integration)
        object.__setattr__(self, '_cl_context', CLContext())
        
        # 关键 3: 直接操作 _modules 字典注册 base_model
        # 这样 DeepSpeed 能正确遍历参数，且避免__setattr__ 循环依赖
        object.__setattr__(self, '_base_model', base_model)
        self._modules['_base_model'] = base_model  # ← 关键：手动注册到 _modules
        
        # 关键 5: 复制其他属性（非 nn.Module 属性）
        if hasattr(base_model, 'config'):
            object.__setattr__(self, 'config', base_model.config)
        if hasattr(base_model, 'vocab_size'):
            object.__setattr__(self, 'vocab_size', base_model.vocab_size)
        if hasattr(base_model, 'clip_tokenizer'):
            object.__setattr__(self, 'clip_tokenizer', base_model.clip_tokenizer)
        if hasattr(base_model, 'tokenizer'):
            object.__setattr__(self, 'tokenizer', base_model.tokenizer)
        if hasattr(base_model, 'cur_task'):
            self._cl_context.task_id = base_model.cur_task
        
        # 关键 6: 初始化 integration
        self._integration.initialize_model(self)
        
        # 关键 7: 验证参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params * 100 if total_params > 0 else 0

        print(f"✅ CLModel 初始化完成 | 方法：{integration.__class__.__name__}")
        print(f"✅ CLModel 总参数量：{total_params:,} 个参数")
        print(f"✅ CLModel 可训练参数量：{trainable_params:,} 个参数")
        print(f"✅ 可训练参数占比：{trainable_ratio:.4f}%")
    

    def __getattr__(self, name: str):
        """委托属性访问到 base_model（深度递归查找）"""
        if name in ['_base_model', '_integration', '_cl_context']:
            return object.__getattribute__(self, name)
        
        try:
            _base_model = object.__getattribute__(self, '_base_model')
            
            # 辅助函数：递归查找属性
            def find_attr(obj, attr_name, depth=0, max_depth=5):
                if depth > max_depth:
                    return None
                
                # 直接查找
                if hasattr(obj, attr_name):
                    return getattr(obj, attr_name)
                
                # 查找 model 属性
                if hasattr(obj, 'model'):
                    result = find_attr(obj.model, attr_name, depth + 1, max_depth)
                    if result is not None:
                        return result
                
                # 查找 base_model 属性（PEFT 嵌套）
                if hasattr(obj, 'base_model'):
                    result = find_attr(obj.base_model, attr_name, depth + 1, max_depth)
                    if result is not None:
                        return result
                
                return None
            
            # 使用递归查找
            result = find_attr(_base_model, name)
            if result is not None:
                return result
            
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """拦截属性设置"""
        if name in ['_base_model', '_integration', '_cl_context']:
            object.__setattr__(self, name, value)
        else:
            try:
                _base_model = object.__getattribute__(self, '_base_model')
                setattr(_base_model, name, value)
            except AttributeError:
                object.__setattr__(self, name, value)
    
    # ========== 委托 parameters() 到 _base_model ==========
    def parameters(self, recurse: bool = True):
        """返回所有参数（委托到 base_model）"""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            yield from _base_model.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """返回所有命名参数"""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            yield from _base_model.named_parameters(prefix=prefix, recurse=recurse)
    
    
    def state_dict(self, *args, **kwargs):
        """返回状态字典"""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            return _base_model.state_dict(*args, **kwargs)
        return {}
    
    def modules(self):
        """返回所有模块"""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            yield from _base_model.modules()
    def load_state_dict(self, *args, **kwargs):
        """加载状态字典"""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            return _base_model.load_state_dict(*args, **kwargs)
    
    # ========== 重写 to() 和 cuda() ==========
    def to(self, *args, **kwargs):
        """确保 to() 调用正确委托到 base_model"""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            _base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def cuda(self, device=None):
        """确保 cuda() 调用正确委托到 base_model"""
        _base_model = object.__getattribute__(self, '_base_model')
        if _base_model is not None:
            _base_model.cuda(device)
        return super().cuda(device)
    # =======================================================
    def _get_attr_recursive(self, name):
        """递归查找属性（处理多层嵌套）"""
        # Level 1: 直接在 self 上查找
        if hasattr(self, name):
            return getattr(self, name)
        
        # Level 2: 在 _base_model 上查找
        _base_model = object.__getattribute__(self, '_base_model', None)
        if _base_model is not None:
            if hasattr(_base_model, name):
                return getattr(_base_model, name)
            
            # Level 3: 在 _base_model.base_model 上查找（PEFT 嵌套）
            if hasattr(_base_model, 'base_model'):
                if hasattr(_base_model.base_model, name):
                    return getattr(_base_model.base_model, name)
                
                # Level 4: 在 _base_model.base_model.model 上查找（LLaVA 结构）
                if hasattr(_base_model.base_model, 'model'):
                    if hasattr(_base_model.base_model.model, name):
                        return getattr(_base_model.base_model.model, name)
        
        return None
    # ========== 重写 prepare_inputs_labels_for_multimodal ==========
    # method/base/cl_model.py
    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images,
        image_sizes=None  # ← 关键字参数，不是位置参数
    ):
        """拦截点 1: 输入准备阶段"""
        
        # ========== [调试] 检查 text_tower ==========
        _base_model = object.__getattribute__(self, '_base_model')
        text_tower = self._get_attr_recursive('text_tower')
        if text_tower is None:
            print(f"  ⚠️  text_tower 未找到，HiDe 文本路由将跳过")

        
        if text_tower is None:
            # 尝试从更深层获取
            if hasattr(_base_model, 'base_model'):
                text_tower = getattr(_base_model.base_model, 'text_tower', None)
                print(f"  text_tower (from base_model.base_model) is None = {text_tower is None}")
        # ===================================
        
        print(f"{'='*70}\n")
        
        # 执行 CL 逻辑
        if hasattr(self, '_integration'):
            self._integration.on_input_prep(
                self, 
                (input_ids, position_ids, attention_mask, past_key_values, labels, images),
                {'images': images, 'image_sizes': image_sizes},
                self._cl_context
            )
        
        # 调用 base_model 的原始方法
        # ========== 关键修复：只传递原始方法期望的参数 ==========
        return _base_model.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, 
            past_key_values, labels, images
            # 不传递 image_sizes，除非原始方法支持
        )
    
    # ========== 重写 forward，手动调用 prepare_inputs_labels_for_multimodal ==========
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """拦截点 2: 模型 Forward 阶段"""
        
        # CL 逻辑：Forward 开始
        self._cl_context.clear()
        self._cl_context.is_training = self.training
        self._integration.on_forward_start(self, self._cl_context)
        
        # 手动调用 prepare_inputs_labels_for_multimodal
        if inputs_embeds is None and images is not None:
            (input_ids, position_ids, attention_mask, 
             past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, 
                past_key_values, labels, images, kwargs.get('image_sizes')
            )
        
        # 调用 base_model 的 forward
        _base_model = object.__getattribute__(self, '_base_model')
        outputs = _base_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        # CL 逻辑：Forward 结束
        outputs = self._integration.on_forward_end(self, outputs, self._cl_context)
        
        return outputs
    
    def set_task_id(self, task_id: int) -> None:
        self._cl_context.task_id = task_id

    @property
    def integration(self):
        return self._integration

    @property
    def cl_context(self):
        return self._cl_context
    
    def get_context(self) -> CLContext:
        return self._cl_context
    
    def cleanup(self) -> None:
        if hasattr(self._integration, 'hook_manager'):
            self._integration.hook_manager.remove_all()
        print("✅ CLModel 资源已清理")

    # method/base/cl_model.py
    def pre_generate_hook(self, model, input_ids, images, context) -> bool:
        """
        在 generate 之前的钩子，子类可以重写
        
        Returns:
            bool: True 表示已处理，False 表示未处理
        """
        return False
    
    def generate(self, *args, **kwargs):

        """重写 generate，调用 integration 的 pre_generate_hook"""
        
        # ========== 调用 pre_generate_hook ==========
        if hasattr(self, '_integration'):
            input_ids = kwargs.get('input_ids', args[0] if args else None)
            images = kwargs.get('images', None)
            context = CLContext(task_id=getattr(self, 'current_task_id', 0))
            
            if hasattr(self._integration, 'pre_generate_hook'):
                self._integration.pre_generate_hook(self, input_ids, images, context)
        # ===========================================
        
        if hasattr(self, '_base_model'):
            return self._base_model.generate(*args, **kwargs)
        else:
            return super().generate(*args, **kwargs)