"""
插件管理器

负责:
- 扫描和发现插件（支持按 indoor/outdoor 分类）
- 加载和卸载插件
- 管理插件生命周期
- 提供插件调用接口
- 支持单插件热重载（不影响其他插件）
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

# 插件分类常量
CATEGORY_INDOOR = "indoor"
CATEGORY_OUTDOOR = "outdoor"


class PluginManager:
    """
    插件管理器

    单例模式,管理所有插件的生命周期
    支持按 indoor/outdoor 分类发现和加载插件
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
        self._manifests: dict[str, PluginManifest] = {}
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
                self._manifests[manifest.id] = manifest
                logger.info(
                    f"发现插件: {manifest.id} v{manifest.version} "
                    f"[{manifest.category or 'uncategorized'}]"
                )
            except Exception as e:
                logger.error(f"解析manifest失败 [{plugin_dir.name}]: {e}")

        return manifests

    def discover_by_category(self, category: str) -> list[PluginManifest]:
        """
        按分类发现插件（室内/室外）

        Args:
            category: "indoor" 或 "outdoor"

        Returns:
            该分类下的插件清单列表
        """
        all_manifests = self.discover_plugins() if not self._manifests else list(self._manifests.values())
        return [m for m in all_manifests if m.category == category]

    def load_category(self, category: str) -> list[BasePlugin]:
        """
        按分类批量加载插件

        Args:
            category: "indoor" 或 "outdoor"

        Returns:
            成功加载的插件列表
        """
        manifests = self.discover_by_category(category)
        loaded = []
        for manifest in manifests:
            try:
                plugin = self.load_plugin(manifest.id)
                loaded.append(plugin)
            except Exception as e:
                logger.error(f"加载{category}插件失败 [{manifest.id}]: {e}")
        logger.info(f"[{category}] 加载完成: {len(loaded)}/{len(manifests)} 个插件")
        return loaded

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

            # 动态加载模块 - 使用包导入以支持相对导入
            module_name = f"plugins.{plugin_id}.{manifest.entrypoint.replace('.py', '')}"
            try:
                # 先尝试直接导入（如果已经在sys.path中）
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                # 如果失败，使用spec_from_file_location但设置正确的包结构
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    entrypoint_path,
                    submodule_search_locations=[str(plugin_dir)]
                )
                if spec is None or spec.loader is None:
                    raise PluginLoadError(plugin_id, "无法创建模块spec")

                module = importlib.util.module_from_spec(spec)
                # 确保父包存在
                parent_module_name = f"plugins.{plugin_id}"
                if parent_module_name not in sys.modules:
                    parent_spec = importlib.util.spec_from_file_location(
                        parent_module_name,
                        plugin_dir / "__init__.py",
                        submodule_search_locations=[str(plugin_dir)]
                    )
                    if parent_spec and parent_spec.loader:
                        parent_module = importlib.util.module_from_spec(parent_spec)
                        sys.modules[parent_module_name] = parent_module
                        parent_spec.loader.exec_module(parent_module)

                sys.modules[module_name] = module
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
            self._manifests[plugin_id] = manifest
            self.registry.register(plugin)
            logger.info(f"插件加载成功: {plugin} [{manifest.category}]")

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
            keys_to_remove = [k for k in sys.modules if k.startswith(module_name)]
            for key in keys_to_remove:
                del sys.modules[key]

            logger.info(f"插件卸载成功: {plugin_id}")
            return True
        except Exception as e:
            logger.error(f"插件卸载失败 [{plugin_id}]: {e}")
            return False

    def reload_plugin(self, plugin_id: str) -> BasePlugin:
        """热重载单个插件（不影响其他插件）"""
        self.unload_plugin(plugin_id)
        return self.load_plugin(plugin_id)

    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """获取已加载的插件"""
        return self._plugins.get(plugin_id)

    def list_plugins(self) -> list[BasePlugin]:
        """列出所有已加载的插件"""
        return list(self._plugins.values())

    def list_plugins_by_category(self, category: str) -> list[BasePlugin]:
        """按分类列出已加载的插件"""
        return [
            p for p in self._plugins.values()
            if p.manifest.category == category
        ]

    def get_plugins_by_capability(self, capability: PluginCapability) -> list[BasePlugin]:
        """按能力筛选插件"""
        return [
            p for p in self._plugins.values()
            if capability in p.manifest.capabilities
        ]

    def get_category_summary(self) -> dict[str, list[str]]:
        """
        获取插件分类摘要

        Returns:
            {"indoor": ["indoor_fence", ...], "outdoor": ["transformer_inspection", ...]}
        """
        summary: dict[str, list[str]] = {"indoor": [], "outdoor": [], "uncategorized": []}
        for manifest in self._manifests.values():
            key = manifest.category if manifest.category in ("indoor", "outdoor") else "uncategorized"
            summary[key].append(manifest.id)
        # 不返回空分类
        return {k: v for k, v in summary.items() if v}

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
