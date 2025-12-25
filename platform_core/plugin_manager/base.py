"""
插件基类定义

所有业务插件必须继承BasePlugin并实现规定的接口:
1. init(config) -> handler
2. infer(frame, rois, context) -> results
3. postprocess(results, rules) -> alarms
4. healthcheck() -> status
"""


from __future__ import annotations
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from platform_core.schema.models import (
    Alarm,
    AlarmRule,
    PluginOutput,
    RecognitionResult,
    ROI,
)


class PluginCapability(str, Enum):
    """插件能力枚举"""
    DEFECT_DETECTION = "defect_detection"  # 缺陷检测
    STATE_RECOGNITION = "state_recognition"  # 状态识别
    METER_READING = "meter_reading"  # 表计读数
    THERMAL_ANALYSIS = "thermal_analysis"  # 热成像分析
    INTRUSION_DETECTION = "intrusion_detection"  # 入侵检测
    IMAGE_QUALITY = "image_quality"  # 图像质量评估
    FOCUS_SUGGESTION = "focus_suggestion"  # 对焦建议


class PluginStatus(str, Enum):
    """插件状态"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginManifest:
    """插件清单 - 从manifest.json加载"""
    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    entrypoint: str = "plugin.py"  # 入口文件
    plugin_class: str = "Plugin"  # 插件类名
    capabilities: list[PluginCapability] = field(default_factory=list)
    device_types: list[str] = field(default_factory=list)  # 支持的设备类型
    dependencies: list[str] = field(default_factory=list)  # Python依赖
    min_platform_version: str = "1.0.0"
    config_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginManifest":
        """从字典创建清单"""
        capabilities = [
            PluginCapability(c) if isinstance(c, str) else c
            for c in data.get("capabilities", [])
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            entrypoint=data.get("entrypoint", "plugin.py"),
            plugin_class=data.get("plugin_class", "Plugin"),
            capabilities=capabilities,
            device_types=data.get("device_types", []),
            dependencies=data.get("dependencies", []),
            min_platform_version=data.get("min_platform_version", "1.0.0"),
            config_schema=data.get("config_schema", {}),
        )


@dataclass
class PluginContext:
    """插件运行上下文"""
    task_id: str
    site_id: str
    device_id: str
    component_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "site_id": self.site_id,
            "device_id": self.device_id,
            "component_id": self.component_id,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config,
            "metadata": self.metadata,
        }


@dataclass
class HealthStatus:
    """健康状态"""
    healthy: bool
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)


class BasePlugin(ABC):
    """
    插件基类

    所有业务模块插件必须继承此类并实现以下方法:
    - init(config): 初始化插件
    - infer(frame, rois, context): 执行推理
    - postprocess(results, rules): 后处理和告警生成
    - healthcheck(): 健康检查
    """

    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        self.manifest = manifest
        self.plugin_dir = plugin_dir
        self.status = PluginStatus.UNLOADED
        self._config: dict[str, Any] = {}
        self._code_hash: str = ""
        self._last_error: str = ""

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def code_hash(self) -> str:
        """计算插件代码hash (用于可追溯性)"""
        if not self._code_hash:
            self._code_hash = self._calculate_code_hash()
        return self._code_hash

    def _calculate_code_hash(self) -> str:
        """计算插件目录下所有Python文件的hash"""
        hasher = hashlib.sha256()
        for py_file in sorted(self.plugin_dir.rglob("*.py")):
            hasher.update(py_file.read_bytes())
        return hasher.hexdigest()[:12]

    # ============== 必须实现的接口 ==============

    @abstractmethod
    def init(self, config: dict[str, Any]) -> bool:
        """
        初始化插件

        Args:
            config: 插件配置字典

        Returns:
            初始化是否成功
        """
        pass

    @abstractmethod
    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """
        执行推理

        Args:
            frame: 输入图像帧 (BGR格式, numpy数组)
            rois: 识别区域列表
            context: 运行上下文

        Returns:
            识别结果列表
        """
        pass

    @abstractmethod
    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """
        后处理和告警生成

        Args:
            results: 推理结果列表
            rules: 告警规则列表

        Returns:
            告警列表
        """
        pass

    @abstractmethod
    def healthcheck(self) -> HealthStatus:
        """
        健康检查

        Returns:
            健康状态
        """
        pass

    # ============== 可选实现的接口 ==============

    def cleanup(self) -> None:
        """清理资源 (可选实现)"""
        pass

    def on_config_update(self, new_config: dict[str, Any]) -> None:
        """配置更新回调 (可选实现)"""
        self._config = new_config

    def analyze_thermal(self, frame: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """热成像分析 (可选实现)"""
        return None

    def get_model_version(self) -> str:
        """获取模型版本 (可选实现)"""
        return self.version

    # ============== 辅助方法 ==============

    def create_output(
        self,
        task_id: str,
        results: list[RecognitionResult],
        alarms: list[Alarm],
        processing_time_ms: float = 0,
        success: bool = True,
        error_message: str = "",
    ) -> PluginOutput:
        """创建标准插件输出"""
        return PluginOutput(
            task_id=task_id,
            plugin_id=self.id,
            plugin_version=self.version,
            code_hash=self.code_hash,
            success=success,
            results=results,
            alarms=alarms,
            error_message=error_message,
            processing_time_ms=processing_time_ms,
        )

    def set_status(self, status: PluginStatus, error: str = "") -> None:
        """设置插件状态"""
        self.status = status
        if error:
            self._last_error = error

    def __repr__(self) -> str:
        return f"<Plugin {self.id} v{self.version} [{self.status.value}]>"
