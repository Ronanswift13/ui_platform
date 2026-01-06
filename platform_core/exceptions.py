"""
平台统一异常定义

所有平台级别的异常都从PlatformError继承
便于统一处理和日志记录
"""


from __future__ import annotations
from typing import Any


class PlatformError(Exception):
    """平台基础异常"""

    def __init__(self, message: str, code: str = "PLATFORM_ERROR", details: dict[str, Any] | None = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class PluginError(PlatformError):
    """插件相关异常"""

    def __init__(self, plugin_id: str, message: str, details: dict[str, Any] | None = None):
        self.plugin_id = plugin_id
        super().__init__(
            message=f"[Plugin: {plugin_id}] {message}",
            code="PLUGIN_ERROR",
            details={"plugin_id": plugin_id, **(details or {})},
        )


class PluginLoadError(PluginError):
    """插件加载失败"""

    def __init__(self, plugin_id: str, reason: str):
        super().__init__(plugin_id, f"加载失败: {reason}")
        self.code = "PLUGIN_LOAD_ERROR"


class PluginValidationError(PluginError):
    """插件验证失败"""

    def __init__(self, plugin_id: str, violations: list[str]):
        self.violations = violations
        super().__init__(
            plugin_id,
            f"验证失败: {', '.join(violations)}",
            details={"violations": violations},
        )
        self.code = "PLUGIN_VALIDATION_ERROR"


class SchemaValidationError(PlatformError):
    """Schema校验失败"""

    def __init__(self, schema_name: str, errors: list[dict[str, Any]]):
        self.schema_name = schema_name
        self.errors = errors
        super().__init__(
            message=f"Schema校验失败 [{schema_name}]: {len(errors)} 个错误",
            code="SCHEMA_VALIDATION_ERROR",
            details={"schema": schema_name, "errors": errors},
        )


class TaskError(PlatformError):
    """任务相关异常"""

    def __init__(self, task_id: str, message: str, details: dict[str, Any] | None = None):
        self.task_id = task_id
        super().__init__(
            message=f"[Task: {task_id}] {message}",
            code="TASK_ERROR",
            details={"task_id": task_id, **(details or {})},
        )


class TaskTimeoutError(TaskError):
    """任务超时"""

    def __init__(self, task_id: str, timeout_seconds: float):
        super().__init__(task_id, f"任务超时 ({timeout_seconds}s)")
        self.code = "TASK_TIMEOUT"


class DeviceError(PlatformError):
    """设备相关异常"""

    def __init__(self, device_id: str, message: str, details: dict[str, Any] | None = None):
        self.device_id = device_id
        super().__init__(
            message=f"[Device: {device_id}] {message}",
            code="DEVICE_ERROR",
            details={"device_id": device_id, **(details or {})},
        )


class EvidenceError(PlatformError):
    """证据链相关异常"""

    def __init__(self, run_id: str, message: str):
        self.run_id = run_id
        super().__init__(
            message=f"[Evidence: {run_id}] {message}",
            code="EVIDENCE_ERROR",
            details={"run_id": run_id},
        )


class ConfigError(PlatformError):
    """配置相关异常"""

    def __init__(self, config_path: str, message: str):
        self.config_path = config_path
        super().__init__(
            message=f"配置错误 [{config_path}]: {message}",
            code="CONFIG_ERROR",
            details={"config_path": config_path},
        )
