# method/base/hooks.py
"""
统一 Hook 管理器
支持带 Context 传递的 Hook，避免污染 Backbone 代码
"""
import torch.nn as nn
from typing import Callable, Dict, List, Optional
from .context import CLContext


class HookManager:
    """
    统一管理 Forward Hook 的注册与移除
    支持将 Context 传递给 Hook 函数
    """
    
    def __init__(self):
        self.handles: List[nn.modules.hooks.RemovableHandle] = []
        self.hook_registry: Dict[str, Callable] = {}
        
    def register_layer_hook(
        self, 
        model: nn.Module, 
        layer_path: str, 
        hook_fn: Callable,
        context: Optional[CLContext] = None
    ) -> None:
        """
        给指定层注册 Hook
        
        Args:
            model: 模型实例
            layer_path: 层路径，如 "model.layers.10"
            hook_fn: Hook 函数，签名 (module, input, output, context)
            context: 可选的 Context 对象，会在 Hook 中传递
        """
        try:
            layer = dict(model.named_modules())[layer_path]
            
            # 包装 Hook 函数以注入 Context
            def wrapped_hook(module, input, output):
                if context is not None:
                    return hook_fn(module, input, output, context)
                else:
                    return hook_fn(module, input, output)
            
            handle = layer.register_forward_hook(wrapped_hook)
            self.handles.append(handle)
            self.hook_registry[layer_path] = hook_fn
            
        except KeyError:
            print(f"⚠️ Warning: Layer {layer_path} not found in model.")
    
    def register_module_hook(
        self, 
        module: nn.Module, 
        hook_fn: Callable,
        context: Optional[CLContext] = None
    ) -> None:
        """
        给整个 Module 注册 Hook
        """
        def wrapped_hook(module, input, output):
            if context is not None:
                return hook_fn(module, input, output, context)
            else:
                return hook_fn(module, input, output)
        
        handle = module.register_forward_hook(wrapped_hook)
        self.handles.append(handle)
    
    def remove_all(self) -> None:
        """移除所有 Hook"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.hook_registry.clear()
    
    def remove_hook(self, layer_path: str) -> None:
        """移除指定层的 Hook"""
        if layer_path in self.hook_registry:
            # 需要重新遍历找到对应的 handle
            self.remove_all()  # 简单实现：全部移除后重新注册需要的
            del self.hook_registry[layer_path]
    
    def get_registered_hooks(self) -> Dict[str, Callable]:
        """获取已注册的 Hook 字典"""
        return self.hook_registry.copy()