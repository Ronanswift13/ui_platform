"""
统一Schema校验模块

定义所有数据模型和校验规则:
- 站点/点位/设备/部件模型
- 任务模型
- ROI模型
- 识别结果模型
- 告警模型
"""


from __future__ import annotations
from platform_core.schema.models import (
    Site,
    Position,
    Device,
    Component,
    ROI,
    Task,
    TaskTemplate,
    RecognitionResult,
    Alarm,
    Evidence,
    PluginOutput,
)
from platform_core.schema.validator import SchemaValidator, validate_plugin_output
from platform_core.schema.indoor_fence_events import (
    UnifiedEvent,
    EventType,
    AlertLevel,
    PersonState,
    IndoorFenceEventBuilder,
    EventValidator,
)

__all__ = [
    "Site",
    "Position",
    "Device",
    "Component",
    "ROI",
    "Task",
    "TaskTemplate",
    "RecognitionResult",
    "Alarm",
    "Evidence",
    "PluginOutput",
    "SchemaValidator",
    "validate_plugin_output",
    "UnifiedEvent",
    "EventType",
    "AlertLevel",
    "PersonState",
    "IndoorFenceEventBuilder",
    "EventValidator",
]
