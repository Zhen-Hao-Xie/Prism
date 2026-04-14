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

    @classmethod
    def _discover_method_modules(cls) -> None:
        """
        自动发现并导入 method/*/integration.py
        使每个方法在自身目录完成注册即可生效。
        """
        if cls._discovery_done:
            return

        method_root = Path(__file__).parent
        for child in method_root.iterdir():
            if not child.is_dir():
                continue
            if child.name in {"base", "__pycache__"}:
                continue
            integration_file = child / "integration.py"
            if not integration_file.exists():
                continue

            # 只导入 integration 模块，避免触发 method.<name>.__init__ 的副作用
            module_name = f"method.{child.name}.integration"
            if module_name in cls._discovered_modules:
                continue
            importlib.import_module(module_name)
            cls._discovered_modules.add(module_name)

        cls._discovery_done = True
    
    @classmethod
    def get_available_methods(cls) -> list:
        """获取所有已注册的方法名称"""
        cls._discover_method_modules()
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
        cls._discover_method_modules()
        
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
        # Prefer python module configs for consistency with run.py
        try:
            mod = importlib.import_module(f"config.methods.{method_name.lower()}")
            cfg = getattr(mod, "METHOD_CONFIG", None)
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass

        # Backward compatibility: allow legacy yaml if it exists
        config_dir = Path(__file__).parent.parent / "config" / "methods"
        config_file = config_dir / f"{method_name.lower()}.yaml"
        if config_file.exists():
            import yaml  # lazy import

            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config or {}

        raise FileNotFoundError(
            f"配置文件不存在：{config_dir / (method_name.lower() + '.py')} 或 {config_file}"
        )

    @classmethod
    def create_from_config_file(
        cls,
        method_name: str,
        override_config: Optional[Any] = None,
    ) -> CLIntegration:
        """
        从 config/methods/<method>.py 的 METHOD_CONFIG（或旧 yaml）创建 Integration 实例。

        override_config 支持：
        - dict：直接 merge 覆盖
        - Namespace/任意对象：读取其 __dict__ 覆盖（常用于 argparse）
        """
        config = cls.load_method_config(method_name)

        if override_config:
            if isinstance(override_config, dict):
                config.update(override_config)
            else:
                # Namespace or any object with attributes
                config.update({k: v for k, v in vars(override_config).items() if v is not None})

        return cls.create_integration(method_name, config)
    

