# method/factory.py
"""
持续学习方法工厂
根据配置自动实例化不同的 CL 方法
支持自动发现 method/*/integration.py 中的方法实现。
"""
from typing import Any, Dict, Optional, Set
import importlib
from pathlib import Path

from .base.integration import CLIntegration


class CLMethodFactory:
    """
    CL 方法工厂类
    负责根据配置加载并实例化不同的 CL 方法
    """
    
    # 方法注册表（name/alias -> integration class）
    _method_registry: Dict[str, type] = {}
    _discovered_modules: Set[str] = set()
    _discovery_done: bool = False
    
    @classmethod
    def register(cls, *method_names: str):
        """
        装饰器：注册 CL 方法
        用法：@CLMethodFactory.register("sp", "some_alias")
        """
        def decorator(integration_class: type):
            if not method_names:
                raise ValueError("register 至少需要一个 method_name")
            for method_name in method_names:
                cls._method_registry[method_name.lower()] = integration_class
            return integration_class
        return decorator
