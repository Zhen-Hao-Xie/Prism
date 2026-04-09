# method/factory.py
"""
持续学习方法工厂
根据配置自动实例化不同的 CL 方法
"""
from typing import Any, Dict, Optional
import yaml
import os
from pathlib import Path

from .base.integration import CLIntegration


class CLMethodFactory:
    """
    CL 方法工厂类
    负责根据配置加载并实例化不同的 CL 方法
    """
    
    # 方法注册表
    _method_registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, method_name: str):
        """
        装饰器：注册 CL 方法
        用法：@CLMethodFactory.register("sp")
        """
        def decorator(integration_class: type):
            cls._method_registry[method_name.lower()] = integration_class
            return integration_class
        return decorator
    
    @classmethod
    def get_available_methods(cls) -> list:
        """获取所有已注册的方法名称"""
        return list(cls._method_registry.keys())
    
    @classmethod
    def create_integration(
        cls, 
        method_name: str, 
        config: Dict[str, Any]
    ) -> CLIntegration:
        """
        根据方法名和配置创建 Integration 实例
        
        Args:
            method_name: 方法名称 (sp, ranpac, hide, sefe 等)
            config: 方法配置字典
            
        Returns:
            CLIntegration 实例
        """
        method_name = method_name.lower()
        
        if method_name not in cls._method_registry:
            available = cls.get_available_methods()
            raise ValueError(
                f"未知方法：{method_name}\n"
                f"可用方法：{available}"
            )
        
        integration_class = cls._method_registry[method_name]
        return integration_class(config)
    
    @classmethod
    def load_method_config(cls, method_name: str) -> Dict[str, Any]:
        """
        从配置文件加载方法配置
        
        Args:
            method_name: 方法名称
            
        Returns:
            配置字典
        """
        config_dir = Path(__file__).parent.parent / "config" / "methods"
        config_file = config_dir / f"{method_name.lower()}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在：{config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @classmethod
    def create_from_config_file(
        cls, 
        method_name: str,
        override_config: Optional[Dict[str, Any]] = None
    ) -> CLIntegration:
        """
        从配置文件创建 Integration 实例（支持配置覆盖）
        
        Args:
            method_name: 方法名称
            override_config: 覆盖配置（命令行参数优先）
            
        Returns:
            CLIntegration 实例
        """
        # 加载配置文件
        config = cls.load_method_config(method_name)
        
        # 覆盖配置（命令行参数优先）
        if override_config:
            config.update(override_config)
        
        # 创建实例
        return cls.create_integration(method_name, config)


# ========== 注册所有 CL 方法 ==========

# # SP 方法
# try:
#     from .sp.integration import SPIntegration
#     CLMethodFactory.register("sp")(SPIntegration)
# except ImportError:
#     pass

# # RanPAC 方法
# try:
#     from .ranpac.integration import RanPACIntegration
#     CLMethodFactory.register("ranpac")(RanPACIntegration)
# except ImportError:
#     pass

# # HiDe-LLaVA 方法
# try:
#     from .hide.integration import HiDeIntegration
#     CLMethodFactory.register("hide")(HiDeIntegration)
# except ImportError:
#     pass

# # SEFE 方法
# try:
#     from .sefe.integration import SEFEIntegration
#     CLMethodFactory.register("sefe")(SEFEIntegration)
# except ImportError:
#     pass