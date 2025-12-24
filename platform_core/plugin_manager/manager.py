"""
插件管理器

负责:
- 扫描和发现插件
- 加载和卸载插件
- 管理插件生命周期
- 提供插件调用接口
"""


from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np

from platform_core.config import get_config
from platform_core.exceptions import PluginError, PluginLoadError, PluginValidationError
from platform_core.logging import get_logger
from platform_core.plugin_manager.base import (
    BasePlugin,
    HealthStatus,
    PluginCapability,
    PluginContext,
    PluginManifest,
    PluginStatus,
)
from platform_core.plugin_manager.registry import PluginRegistry
from platform_core.schema.models import Alarm, AlarmRule, PluginOutput, RecognitionResult, ROI
from platform_core.schema.validator import validate_plugin_output

logger = get_logger(__name__)


class PluginManager:
    """
    插件管理器

    单例模式,管理所有插件的生命周期
    """

    _instance: Optional["PluginManager"] = None

    def __new__(cls) -> "PluginManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.config = get_config()
        self.plugins_dir = self.config.get_plugins_path()
        self.registry = PluginRegistry()
        self._plugins: dict[str, BasePlugin] = {}
        self._initialized = True

        logger.info(f"插件管理器初始化完成, 插件目录: {self.plugins_dir}")

    def discover_plugins(self) -> list[PluginManifest]:
        """
        扫描插件目录,发现所有可用插件

        Returns:
            插件清单列表
        """
        manifests = []

        if not self.plugins_dir.exists():
            logger.warning(f"插件目录不存在: {self.plugins_dir}")
            return manifests

        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            manifest_path = plugin_dir / "manifest.json"
            if not manifest_path.exists():
                logger.debug(f"跳过无manifest的目录: {plugin_dir.name}")
                continue

            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                manifest = PluginManifest.from_dict(manifest_data)
                manifests.append(manifest)
                logger.info(f"发现插件: {manifest.id} v{manifest.version}")
            except Exception as e:
                logger.error(f"解析manifest失败 [{plugin_dir.name}]: {e}")

        return manifests

    def load_plugin(self, plugin_id: str) -> BasePlugin:
        """
        加载指定插件

        Args:
            plugin_id: 插件ID

        Returns:
            加载的插件实例

        Raises:
            PluginLoadError: 加载失败
        """
        if plugin_id in self._plugins:
            return self._plugins[plugin_id]

        plugin_dir = self.plugins_dir / plugin_id
        manifest_path = plugin_dir / "manifest.json"

        if not manifest_path.exists():
            raise PluginLoadError(plugin_id, f"manifest.json不存在: {manifest_path}")

        try:
            # 读取manifest
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            manifest = PluginManifest.from_dict(manifest_data)

            # 加载插件模块
            entrypoint_path = plugin_dir / manifest.entrypoint
            if not entrypoint_path.exists():
                raise PluginLoadError(plugin_id, f"入口文件不存在: {entrypoint_path}")

            # 动态加载模块
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_id}",
                entrypoint_path,
            )
            if spec is None or spec.loader is None:
                raise PluginLoadError(plugin_id, "无法创建模块spec")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"plugins.{plugin_id}"] = module
            spec.loader.exec_module(module)

            # 获取插件类
            _plugin_class = getattr(module, manifest.plugin_class, None)
            if _plugin_class is None:
                raise PluginLoadError(plugin_id, f"找不到插件类: {manifest.plugin_class}")
            plugin_class: Type[BasePlugin] = _plugin_class

            # 实例化插件
            plugin = plugin_class(manifest, plugin_dir)
            plugin.set_status(PluginStatus.LOADING)

            # 初始化插件
            plugin_config = self._load_plugin_config(plugin_id)
            if plugin.init(plugin_config):
                plugin.set_status(PluginStatus.READY)
            else:
                plugin.set_status(PluginStatus.ERROR, "初始化返回False")
                raise PluginLoadError(plugin_id, "初始化失败")

            self._plugins[plugin_id] = plugin
            self.registry.register(plugin)
            logger.info(f"插件加载成功: {plugin}")

            return plugin

        except PluginLoadError:
            raise
        except Exception as e:
            raise PluginLoadError(plugin_id, str(e)) from e

    def unload_plugin(self, plugin_id: str) -> bool:
        """卸载插件"""
        if plugin_id not in self._plugins:
            return False

        plugin = self._plugins[plugin_id]
        try:
            plugin.cleanup()
            plugin.set_status(PluginStatus.UNLOADED)
            self.registry.unregister(plugin_id)
            del self._plugins[plugin_id]

            # 清理sys.modules
            module_name = f"plugins.{plugin_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            logger.info(f"插件卸载成功: {plugin_id}")
            return True
        except Exception as e:
            logger.error(f"插件卸载失败 [{plugin_id}]: {e}")
            return False

    def reload_plugin(self, plugin_id: str) -> BasePlugin:
        """重新加载插件"""
        self.unload_plugin(plugin_id)
        return self.load_plugin(plugin_id)

    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """获取已加载的插件"""
        return self._plugins.get(plugin_id)

    def list_plugins(self) -> list[BasePlugin]:
        """列出所有已加载的插件"""
        return list(self._plugins.values())

    def get_plugins_by_capability(self, capability: PluginCapability) -> list[BasePlugin]:
        """按能力筛选插件"""
        return [
            p for p in self._plugins.values()
            if capability in p.manifest.capabilities
        ]

    def execute_plugin(
        self,
        plugin_id: str,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
        rules: list[AlarmRule] | None = None,
        validate_output: bool = True,
    ) -> PluginOutput:
        """
        执行插件推理

        Args:
            plugin_id: 插件ID
            frame: 输入图像帧
            rois: ROI列表
            context: 运行上下文
            rules: 告警规则 (可选)
            validate_output: 是否验证输出格式

        Returns:
            插件输出

        Raises:
            PluginError: 执行失败
            PluginValidationError: 输出格式验证失败
        """
        import time

        plugin = self.get_plugin(plugin_id)
        if plugin is None:
            plugin = self.load_plugin(plugin_id)

        if plugin.status != PluginStatus.READY:
            raise PluginError(plugin_id, f"插件状态异常: {plugin.status}")

        start_time = time.perf_counter()

        try:
            plugin.set_status(PluginStatus.RUNNING)

            # 执行推理
            results = plugin.infer(frame, rois, context)

            # 后处理
            alarms = []
            if rules:
                alarms = plugin.postprocess(results, rules)

            processing_time = (time.perf_counter() - start_time) * 1000

            # 创建输出
            output = plugin.create_output(
                task_id=context.task_id,
                results=results,
                alarms=alarms,
                processing_time_ms=processing_time,
            )

            # 验证输出格式
            if validate_output and self.config.plugin.strict_validation:
                validate_plugin_output(output.model_dump(), plugin_id)

            plugin.set_status(PluginStatus.READY)
            return output

        except Exception as e:
            plugin.set_status(PluginStatus.ERROR, str(e))
            logger.error(f"插件执行失败 [{plugin_id}]: {e}")
            raise PluginError(plugin_id, str(e)) from e

    def healthcheck_all(self) -> dict[str, HealthStatus]:
        """检查所有插件健康状态"""
        results = {}
        for plugin_id, plugin in self._plugins.items():
            try:
                results[plugin_id] = plugin.healthcheck()
            except Exception as e:
                results[plugin_id] = HealthStatus(
                    healthy=False,
                    message=str(e),
                )
        return results

    def _load_plugin_config(self, plugin_id: str) -> dict[str, Any]:
        """加载插件配置"""
        config_path = self.plugins_dir / plugin_id / "configs" / "default.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}


def get_plugin_manager() -> PluginManager:
    """获取插件管理器单例"""
    return PluginManager()
