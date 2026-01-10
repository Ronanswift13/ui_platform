# -*- coding: utf-8 -*-
"""
插件兼容性修复模块
==================

解决插件不兼容或无法正常识别、加载的问题：
1. 统一插件初始化接口
2. 模型注册器注入
3. 错误捕获和降级处理
4. 配置加载增强

作者: AI巡检系统
版本: 1.0.0
"""

from __future__ import annotations
import hashlib
import logging
import traceback
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import importlib
import importlib.util
import sys

logger = logging.getLogger(__name__)


# =============================================================================
# 插件状态枚举
# =============================================================================

class PluginStatus(str, Enum):
    """插件状态"""
    UNLOADED = "unloaded"           # 未加载
    LOADING = "loading"             # 加载中
    LOADED = "loaded"               # 已加载
    INITIALIZED = "initialized"     # 已初始化
    RUNNING = "running"             # 运行中
    ERROR = "error"                 # 错误
    DISABLED = "disabled"           # 已禁用
    DEGRADED = "degraded"           # 降级运行


# =============================================================================
# 插件清单数据结构
# =============================================================================

@dataclass
class PluginManifest:
    """插件清单"""
    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    entry_point: str = "plugin.py"
    plugin_class: str = ""
    config_file: str = "configs/default.yaml"
    models: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    min_platform_version: str = "1.0.0"
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PluginManifest':
        """从字典创建清单"""
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            author=data.get('author', ''),
            dependencies=data.get('dependencies', []),
            modalities=data.get('modalities', []),
            entry_point=data.get('entry_point', 'plugin.py'),
            plugin_class=data.get('plugin_class', ''),
            config_file=data.get('config_file', 'configs/default.yaml'),
            models=data.get('models', {}),
            capabilities=data.get('capabilities', []),
            min_platform_version=data.get('min_platform_version', '1.0.0')
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'PluginManifest':
        """从YAML文件加载清单"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


# =============================================================================
# 增强型插件基类
# =============================================================================

class EnhancedPluginBase(ABC):
    """
    增强型插件基类
    
    提供统一的插件接口和兼容性支持：
    1. 多种初始化方式支持
    2. 自动代码哈希计算
    3. 配置自动加载
    4. 错误捕获和降级处理
    """
    
    def __init__(
        self,
        manifest: Optional[PluginManifest] = None,
        plugin_dir: Optional[Path] = None,
        config: Optional[Dict] = None,
        **kwargs
    ):
        """
        支持多种初始化方式的构造函数
        
        Args:
            manifest: 插件清单
            plugin_dir: 插件目录
            config: 配置字典
            **kwargs: 其他参数，用于兼容不同的初始化方式
        """
        # 处理参数
        self._manifest = manifest
        self._plugin_dir = Path(plugin_dir) if plugin_dir else None
        
        # 状态跟踪
        self._status = PluginStatus.UNLOADED
        self._initialized = False
        self._last_error: Optional[str] = None
        self._error_count = 0
        self._degraded_mode = False
        
        # 配置处理
        if isinstance(config, dict):
            self._config = config
        else:
            self._config = {}
        
        # 模型注册器引用
        self._model_registry = None
        self._models: Dict[str, Any] = {}
        
        # 计算代码哈希
        self._code_hash = self._calculate_code_hash()
        
        # 加载默认配置
        if not self._config and self._plugin_dir:
            self._load_default_config()
        
        # 处理额外参数
        for key, value in kwargs.items():
            if not hasattr(self, f'_{key}'):
                setattr(self, f'_{key}', value)
    
    # =========================================================================
    # 属性
    # =========================================================================
    
    @property
    def plugin_id(self) -> str:
        """插件ID"""
        if self._manifest:
            return self._manifest.id
        return self.__class__.__name__
    
    @property
    def plugin_name(self) -> str:
        """插件名称"""
        if self._manifest:
            return self._manifest.name
        return self.__class__.__name__
    
    @property
    def plugin_dir(self) -> Optional[Path]:
        """插件目录"""
        return self._plugin_dir
    
    @property
    def status(self) -> PluginStatus:
        """插件状态"""
        return self._status
    
    @property
    def code_hash(self) -> str:
        """代码哈希值"""
        return self._code_hash
    
    @property
    def last_error(self) -> Optional[str]:
        """最后错误信息"""
        return self._last_error
    
    @property
    def is_degraded(self) -> bool:
        """是否处于降级模式"""
        return self._degraded_mode
    
    # =========================================================================
    # 抽象方法
    # =========================================================================
    
    @abstractmethod
    def init(self, config: Optional[Dict] = None) -> bool:
        """
        初始化插件
        
        子类必须实现此方法。应该捕获所有异常并返回布尔值表示是否成功。
        
        Args:
            config: 配置字典
            
        Returns:
            初始化是否成功
        """
        pass
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        处理数据
        
        子类必须实现此方法。
        
        Args:
            data: 输入数据
            
        Returns:
            处理结果
        """
        pass
    
    # =========================================================================
    # 生命周期方法
    # =========================================================================
    
    def safe_init(self, config: Optional[Dict] = None) -> bool:
        """
        安全初始化（带错误捕获）
        
        Args:
            config: 配置字典
            
        Returns:
            初始化是否成功
        """
        self._status = PluginStatus.LOADING
        
        try:
            # 合并配置
            if config:
                self._config = self._merge_config(self._config, config)
            
            # 调用子类初始化
            success = self.init(self._config)
            
            if success:
                self._status = PluginStatus.INITIALIZED
                self._initialized = True
                self._last_error = None
                logger.info(f"Plugin {self.plugin_id} initialized successfully")
            else:
                self._status = PluginStatus.ERROR
                self._last_error = "Initialization returned False"
                logger.warning(f"Plugin {self.plugin_id} initialization failed")
            
            return success
            
        except Exception as e:
            self._status = PluginStatus.ERROR
            self._last_error = str(e)
            self._error_count += 1
            
            logger.error(f"Plugin {self.plugin_id} initialization error: {e}")
            logger.debug(traceback.format_exc())
            
            # 尝试降级初始化
            if self._try_degraded_init():
                self._status = PluginStatus.DEGRADED
                self._degraded_mode = True
                logger.warning(f"Plugin {self.plugin_id} running in degraded mode")
                return True
            
            return False
    
    def _try_degraded_init(self) -> bool:
        """
        尝试降级初始化
        
        在正常初始化失败时调用，尝试以最小功能运行
        
        Returns:
            是否成功进入降级模式
        """
        # 子类可以覆盖此方法实现自定义降级逻辑
        return False
    
    def shutdown(self) -> None:
        """关闭插件"""
        try:
            self._on_shutdown()
        except Exception as e:
            logger.error(f"Plugin {self.plugin_id} shutdown error: {e}")
        finally:
            self._status = PluginStatus.UNLOADED
            self._initialized = False
    
    def _on_shutdown(self) -> None:
        """关闭时的清理工作，子类可覆盖"""
        pass
    
    # =========================================================================
    # 模型注册
    # =========================================================================
    
    def set_model_registry(self, registry: Any) -> None:
        """
        设置模型注册器
        
        Args:
            registry: 模型注册器实例
        """
        self._model_registry = registry
        
        # 自动加载声明的模型
        if self._manifest and self._manifest.models:
            self._load_declared_models()
    
    def _load_declared_models(self) -> None:
        """加载清单中声明的模型"""
        if not self._model_registry:
            return
        
        for model_name, model_path in self._manifest.models.items():
            try:
                model = self._model_registry.get_model(model_path)
                if model:
                    self._models[model_name] = model
                    logger.debug(f"Loaded model {model_name} for plugin {self.plugin_id}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
    
    def get_model(self, name: str) -> Optional[Any]:
        """
        获取已加载的模型
        
        Args:
            name: 模型名称
            
        Returns:
            模型实例或None
        """
        return self._models.get(name)
    
    # =========================================================================
    # 配置管理
    # =========================================================================
    
    def _load_default_config(self) -> None:
        """加载默认配置"""
        if not self._plugin_dir:
            return
        
        # 尝试多个配置文件位置
        config_paths = [
            self._plugin_dir / "configs" / "default.yaml",
            self._plugin_dir / "config.yaml",
            self._plugin_dir / "configs" / "config.yaml",
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        loaded_config = yaml.safe_load(f) or {}
                    self._config = self._merge_config(loaded_config, self._config)
                    logger.debug(f"Loaded config from {config_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
    
    def _merge_config(self, base: Dict, override: Dict) -> Dict:
        """
        递归合并配置
        
        Args:
            base: 基础配置
            override: 覆盖配置
            
        Returns:
            合并后的配置
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键（支持点分隔路径，如 'model.input_size'）
            default: 默认值
            
        Returns:
            配置值
        """
        if key is None:
            return self._config
        
        # 支持点分隔路径
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    # =========================================================================
    # 代码哈希计算
    # =========================================================================
    
    def _calculate_code_hash(self) -> str:
        """
        计算代码版本哈希
        
        Returns:
            代码哈希字符串
        """
        if not self._plugin_dir:
            return "sha256:unknown"
        
        h = hashlib.sha256()
        
        # 要计算哈希的文件
        files_to_hash = [
            "plugin.py",
            "__init__.py",
            "detector.py",
            "detector_enhanced.py",
            "engine.py",
            "processor.py",
            "analyzer.py",
        ]
        
        for fname in files_to_hash:
            fpath = self._plugin_dir / fname
            if fpath.exists():
                try:
                    h.update(fpath.read_bytes())
                except Exception:
                    pass
        
        return f"sha256:{h.hexdigest()[:12]}"
    
    # =========================================================================
    # 结果格式化
    # =========================================================================
    
    def format_result(
        self,
        success: bool,
        detections: List[Dict] = None,
        status: str = "normal",
        confidence: float = 0.0,
        message: str = "",
        metadata: Dict = None
    ) -> Dict:
        """
        格式化处理结果
        
        Args:
            success: 处理是否成功
            detections: 检测结果列表
            status: 状态
            confidence: 置信度
            message: 消息
            metadata: 元数据
            
        Returns:
            标准化的结果字典
        """
        return {
            "success": success,
            "plugin_id": self.plugin_id,
            "plugin_version": self._manifest.version if self._manifest else "1.0.0",
            "status": status,
            "confidence": confidence,
            "detections": detections or [],
            "message": message,
            "metadata": metadata or {},
            "degraded_mode": self._degraded_mode,
            "code_hash": self._code_hash
        }
    
    # =========================================================================
    # 告警发送
    # =========================================================================
    
    def emit_alarm(
        self,
        level: str,
        title: str,
        description: str,
        device_id: str = "",
        detection: Dict = None
    ) -> None:
        """
        发送告警
        
        通过平台的告警系统发送告警
        
        Args:
            level: 告警级别 (info, warning, alarm, critical)
            title: 告警标题
            description: 告警描述
            device_id: 设备ID
            detection: 相关检测结果
        """
        # 获取插件管理器的emit_alarm方法
        if hasattr(self, '_plugin_manager') and self._plugin_manager:
            try:
                self._plugin_manager.emit_alarm(
                    plugin_id=self.plugin_id,
                    level=level,
                    title=title,
                    description=description,
                    device_id=device_id,
                    detection=detection
                )
            except Exception as e:
                logger.error(f"Failed to emit alarm: {e}")
        else:
            # 降级：只记录日志
            log_level = {
                'info': logging.INFO,
                'warning': logging.WARNING,
                'alarm': logging.ERROR,
                'critical': logging.CRITICAL
            }.get(level, logging.WARNING)
            
            logger.log(log_level, f"[ALARM] {title}: {description}")


# =============================================================================
# 插件加载器
# =============================================================================

class PluginLoader:
    """
    插件加载器
    
    负责加载、验证和实例化插件
    """
    
    def __init__(
        self,
        plugins_dir: Path,
        model_registry: Any = None
    ):
        """
        Args:
            plugins_dir: 插件目录
            model_registry: 模型注册器
        """
        self.plugins_dir = Path(plugins_dir)
        self.model_registry = model_registry
        
        self._loaded_plugins: Dict[str, EnhancedPluginBase] = {}
        self._plugin_classes: Dict[str, Type] = {}
        self._load_errors: Dict[str, str] = {}
    
    def discover_plugins(self) -> List[str]:
        """
        发现所有可用插件
        
        Returns:
            插件ID列表
        """
        plugin_ids = []
        
        if not self.plugins_dir.exists():
            logger.warning(f"Plugins directory not found: {self.plugins_dir}")
            return plugin_ids
        
        for item in self.plugins_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                manifest_path = item / "manifest.yaml"
                if manifest_path.exists():
                    try:
                        manifest = PluginManifest.from_yaml(manifest_path)
                        plugin_ids.append(manifest.id)
                    except Exception as e:
                        logger.warning(f"Failed to load manifest from {manifest_path}: {e}")
                        plugin_ids.append(item.name)  # 使用目录名作为ID
        
        return plugin_ids
    
    def load_plugin(self, plugin_id: str) -> Optional[EnhancedPluginBase]:
        """
        加载单个插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            插件实例或None
        """
        if plugin_id in self._loaded_plugins:
            return self._loaded_plugins[plugin_id]
        
        # 查找插件目录
        plugin_dir = self._find_plugin_dir(plugin_id)
        if not plugin_dir:
            self._load_errors[plugin_id] = "Plugin directory not found"
            return None
        
        try:
            # 加载清单
            manifest = self._load_manifest(plugin_dir)
            
            # 加载插件类
            plugin_class = self._load_plugin_class(plugin_dir, manifest)
            if not plugin_class:
                return None
            
            # 创建实例
            plugin = self._create_instance(plugin_class, manifest, plugin_dir)
            if not plugin:
                return None
            
            # 注入模型注册器
            if self.model_registry:
                plugin.set_model_registry(self.model_registry)
            
            self._loaded_plugins[plugin_id] = plugin
            self._plugin_classes[plugin_id] = plugin_class
            
            logger.info(f"Loaded plugin: {plugin_id}")
            return plugin
            
        except Exception as e:
            self._load_errors[plugin_id] = str(e)
            logger.error(f"Failed to load plugin {plugin_id}: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def _find_plugin_dir(self, plugin_id: str) -> Optional[Path]:
        """查找插件目录"""
        # 首先尝试精确匹配
        exact_match = self.plugins_dir / plugin_id
        if exact_match.exists():
            return exact_match
        
        # 然后搜索所有目录的manifest
        for item in self.plugins_dir.iterdir():
            if item.is_dir():
                manifest_path = item / "manifest.yaml"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r', encoding='utf-8') as f:
                            data = yaml.safe_load(f) or {}
                        if data.get('id') == plugin_id:
                            return item
                    except Exception:
                        pass
        
        return None
    
    def _load_manifest(self, plugin_dir: Path) -> PluginManifest:
        """加载插件清单"""
        manifest_path = plugin_dir / "manifest.yaml"
        
        if manifest_path.exists():
            return PluginManifest.from_yaml(manifest_path)
        
        # 没有清单时创建默认清单
        return PluginManifest(
            id=plugin_dir.name,
            name=plugin_dir.name.replace('_', ' ').title(),
            version='1.0.0'
        )
    
    def _load_plugin_class(
        self,
        plugin_dir: Path,
        manifest: PluginManifest
    ) -> Optional[Type]:
        """加载插件类"""
        entry_point = manifest.entry_point or "plugin.py"
        plugin_path = plugin_dir / entry_point
        
        if not plugin_path.exists():
            self._load_errors[manifest.id] = f"Entry point not found: {entry_point}"
            return None
        
        try:
            # 动态导入模块
            module_name = f"plugins.{plugin_dir.name}.plugin"
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # 查找插件类
            plugin_class_name = manifest.plugin_class
            if not plugin_class_name:
                # 尝试自动发现
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type) and 
                        name.endswith('Plugin') and
                        name != 'EnhancedPluginBase'):
                        plugin_class_name = name
                        break
            
            if not plugin_class_name:
                self._load_errors[manifest.id] = "Plugin class not found"
                return None
            
            return getattr(module, plugin_class_name)
            
        except Exception as e:
            self._load_errors[manifest.id] = f"Failed to load module: {e}"
            return None
    
    def _create_instance(
        self,
        plugin_class: Type,
        manifest: PluginManifest,
        plugin_dir: Path
    ) -> Optional[EnhancedPluginBase]:
        """创建插件实例"""
        try:
            # 尝试多种实例化方式
            
            # 方式1: manifest + plugin_dir
            try:
                return plugin_class(manifest=manifest, plugin_dir=plugin_dir)
            except TypeError:
                pass
            
            # 方式2: 只有plugin_dir
            try:
                return plugin_class(plugin_dir=plugin_dir)
            except TypeError:
                pass
            
            # 方式3: 无参数
            try:
                instance = plugin_class()
                # 手动设置属性
                if hasattr(instance, '_manifest'):
                    instance._manifest = manifest
                if hasattr(instance, '_plugin_dir'):
                    instance._plugin_dir = plugin_dir
                return instance
            except TypeError:
                pass
            
            # 方式4: 关键字参数
            try:
                return plugin_class(
                    manifest=manifest,
                    plugin_dir=plugin_dir,
                    config={}
                )
            except TypeError:
                pass
            
            self._load_errors[manifest.id] = "Failed to instantiate plugin class"
            return None
            
        except Exception as e:
            self._load_errors[manifest.id] = f"Instance creation error: {e}"
            return None
    
    def init_plugin(self, plugin_id: str, config: Dict = None) -> bool:
        """
        初始化插件
        
        Args:
            plugin_id: 插件ID
            config: 配置
            
        Returns:
            是否成功
        """
        plugin = self._loaded_plugins.get(plugin_id)
        if not plugin:
            return False
        
        return plugin.safe_init(config)
    
    def get_plugin(self, plugin_id: str) -> Optional[EnhancedPluginBase]:
        """获取已加载的插件"""
        return self._loaded_plugins.get(plugin_id)
    
    def get_load_errors(self) -> Dict[str, str]:
        """获取加载错误"""
        return self._load_errors.copy()
    
    def unload_plugin(self, plugin_id: str) -> None:
        """卸载插件"""
        if plugin_id in self._loaded_plugins:
            plugin = self._loaded_plugins[plugin_id]
            plugin.shutdown()
            del self._loaded_plugins[plugin_id]
            logger.info(f"Unloaded plugin: {plugin_id}")
    
    def reload_plugin(self, plugin_id: str) -> Optional[EnhancedPluginBase]:
        """重新加载插件"""
        self.unload_plugin(plugin_id)
        return self.load_plugin(plugin_id)


# =============================================================================
# 插件管理器增强
# =============================================================================

class PluginManagerEnhanced:
    """
    增强型插件管理器
    
    提供插件生命周期管理和告警集成
    """
    
    def __init__(
        self,
        plugins_dir: Path,
        model_registry: Any = None,
        config: Dict = None
    ):
        """
        Args:
            plugins_dir: 插件目录
            model_registry: 模型注册器
            config: 配置
        """
        self.plugins_dir = Path(plugins_dir)
        self.model_registry = model_registry
        self.config = config or {}
        
        self._loader = PluginLoader(plugins_dir, model_registry)
        self._alarm_handlers: List[Callable] = []
        self._running = False
    
    def start(self, plugin_ids: List[str] = None) -> Dict[str, bool]:
        """
        启动插件
        
        Args:
            plugin_ids: 要启动的插件ID列表，None表示全部
            
        Returns:
            插件ID -> 是否成功 的映射
        """
        results = {}
        
        if plugin_ids is None:
            plugin_ids = self._loader.discover_plugins()
        
        for plugin_id in plugin_ids:
            # 加载
            plugin = self._loader.load_plugin(plugin_id)
            if not plugin:
                results[plugin_id] = False
                continue
            
            # 设置插件管理器引用
            if hasattr(plugin, '_plugin_manager'):
                plugin._plugin_manager = self
            
            # 获取配置
            plugin_config = self.config.get('plugins', {}).get(plugin_id, {})
            
            # 初始化
            success = self._loader.init_plugin(plugin_id, plugin_config)
            results[plugin_id] = success
        
        self._running = True
        return results
    
    def stop(self) -> None:
        """停止所有插件"""
        for plugin_id in list(self._loader._loaded_plugins.keys()):
            self._loader.unload_plugin(plugin_id)
        self._running = False
    
    def get_plugin(self, plugin_id: str) -> Optional[EnhancedPluginBase]:
        """获取插件"""
        return self._loader.get_plugin(plugin_id)
    
    def get_status(self) -> Dict[str, Dict]:
        """获取所有插件状态"""
        status = {}
        
        for plugin_id, plugin in self._loader._loaded_plugins.items():
            status[plugin_id] = {
                'status': plugin.status.value,
                'initialized': plugin._initialized,
                'degraded': plugin.is_degraded,
                'last_error': plugin.last_error,
                'code_hash': plugin.code_hash
            }
        
        # 添加加载错误信息
        for plugin_id, error in self._loader.get_load_errors().items():
            if plugin_id not in status:
                status[plugin_id] = {
                    'status': 'error',
                    'initialized': False,
                    'degraded': False,
                    'last_error': error,
                    'code_hash': None
                }
        
        return status
    
    def register_alarm_handler(self, handler: Callable) -> None:
        """注册告警处理器"""
        self._alarm_handlers.append(handler)
    
    def emit_alarm(
        self,
        plugin_id: str,
        level: str,
        title: str,
        description: str,
        device_id: str = "",
        detection: Dict = None
    ) -> None:
        """
        发送告警
        
        Args:
            plugin_id: 发送告警的插件ID
            level: 告警级别
            title: 告警标题
            description: 告警描述
            device_id: 设备ID
            detection: 相关检测结果
        """
        import time
        
        alarm = {
            'plugin_id': plugin_id,
            'level': level,
            'title': title,
            'description': description,
            'device_id': device_id,
            'detection': detection,
            'timestamp': time.time()
        }
        
        for handler in self._alarm_handlers:
            try:
                handler(alarm)
            except Exception as e:
                logger.error(f"Alarm handler error: {e}")


# =============================================================================
# 工具函数
# =============================================================================

def patch_existing_plugin(plugin: Any) -> Any:
    """
    修补现有插件以添加兼容性支持
    
    Args:
        plugin: 现有插件实例
        
    Returns:
        修补后的插件实例
    """
    # 添加缺失的属性
    if not hasattr(plugin, '_status'):
        plugin._status = PluginStatus.UNLOADED
    
    if not hasattr(plugin, '_last_error'):
        plugin._last_error = None
    
    if not hasattr(plugin, '_code_hash'):
        if hasattr(plugin, 'plugin_dir') and plugin.plugin_dir:
            h = hashlib.sha256()
            plugin_path = Path(plugin.plugin_dir) / "plugin.py"
            if plugin_path.exists():
                h.update(plugin_path.read_bytes())
            plugin._code_hash = f"sha256:{h.hexdigest()[:12]}"
        else:
            plugin._code_hash = "sha256:unknown"
    
    if not hasattr(plugin, '_degraded_mode'):
        plugin._degraded_mode = False
    
    if not hasattr(plugin, 'code_hash'):
        plugin.code_hash = property(lambda self: self._code_hash)
    
    # 包装init方法添加错误处理
    if hasattr(plugin, 'init'):
        original_init = plugin.init
        
        def safe_init(config=None):
            try:
                result = original_init(config)
                plugin._status = PluginStatus.INITIALIZED if result else PluginStatus.ERROR
                return result
            except Exception as e:
                plugin._status = PluginStatus.ERROR
                plugin._last_error = str(e)
                logger.error(f"Plugin init error: {e}")
                return False
        
        plugin.init = safe_init
    
    return plugin


def create_plugin_manifest(
    plugin_dir: Path,
    plugin_id: str = None,
    name: str = None,
    **kwargs
) -> PluginManifest:
    """
    为现有插件创建清单
    
    Args:
        plugin_dir: 插件目录
        plugin_id: 插件ID
        name: 插件名称
        **kwargs: 其他清单属性
        
    Returns:
        PluginManifest实例
    """
    plugin_dir = Path(plugin_dir)
    
    return PluginManifest(
        id=plugin_id or plugin_dir.name,
        name=name or plugin_dir.name.replace('_', ' ').title(),
        version=kwargs.get('version', '1.0.0'),
        description=kwargs.get('description', ''),
        author=kwargs.get('author', ''),
        dependencies=kwargs.get('dependencies', []),
        modalities=kwargs.get('modalities', []),
        entry_point=kwargs.get('entry_point', 'plugin.py'),
        plugin_class=kwargs.get('plugin_class', ''),
        config_file=kwargs.get('config_file', 'configs/default.yaml'),
        models=kwargs.get('models', {}),
        capabilities=kwargs.get('capabilities', [])
    )
