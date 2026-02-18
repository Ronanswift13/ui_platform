"""
统一数据模型定义 - 兼容层

核心数据模型已迁移到 darkbreaker_sdk.schemas
本模块保留向后兼容的 re-export，并包含平台特有模型（Site, Device, Task 等）。

插件开发者应使用:
    from darkbreaker_sdk.schemas import BoundingBox, RecognitionResult, Alarm, ...
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ===== Re-exports from darkbreaker_sdk (backward compatibility) =====

from darkbreaker_sdk.schemas.common import (  # noqa: F401
    generate_id,
    BaseEntity,
    ROIType,
    EvidenceType,
    Evidence,
)

from darkbreaker_sdk.schemas.detection import (  # noqa: F401
    BoundingBox,
    RecognitionResult,
)

from darkbreaker_sdk.schemas.alarm import (  # noqa: F401
    Alarm,
    AlarmLevel,
    AlarmRule,
    AlarmStatus,
)

from darkbreaker_sdk.schemas.plugin_io import (  # noqa: F401
    PluginOutput,
)

from darkbreaker_sdk.schemas.common import (  # noqa: F401
    ROI,
)


# ===== 平台特有模型 (不在 SDK 中) =====

class Site(BaseEntity):
    """站点模型 - 变电站"""
    code: str  # 站点编码
    location: str = ""  # 地理位置
    voltage_level: str = ""  # 电压等级
    positions: list["Position"] = Field(default_factory=list)


class Position(BaseEntity):
    """点位模型 - 摄像头位置"""
    site_id: str
    camera_id: str = ""
    ptz_preset: dict[str, float] = Field(default_factory=dict)  # 云台预置位
    devices: list["Device"] = Field(default_factory=list)


class DeviceType(str, Enum):
    """设备类型枚举"""
    TRANSFORMER = "transformer"  # 主变
    SWITCH = "switch"  # 开关
    BUSBAR = "busbar"  # 母线
    CAPACITOR = "capacitor"  # 电容器
    METER = "meter"  # 表计
    OTHER = "other"


class Device(BaseEntity):
    """设备模型"""
    position_id: str
    device_type: DeviceType
    model: str = ""  # 设备型号
    components: list["Component"] = Field(default_factory=list)


class Component(BaseEntity):
    """部件模型"""
    device_id: str
    component_type: str  # 部件类型 (套管/散热器/油位计等)
    rois: list["ROI"] = Field(default_factory=list)


# ============== 任务模型 ==============

class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskTemplate(BaseEntity):
    """任务模板"""
    plugin_id: str  # 关联的插件
    device_type: DeviceType
    default_config: dict[str, Any] = Field(default_factory=dict)
    required_capabilities: list[str] = Field(default_factory=list)


class Task(BaseEntity):
    """任务实例"""
    template_id: str
    site_id: str
    position_id: str
    device_id: str
    plugin_id: str
    roi_ids: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    config: dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""
    result_id: Optional[str] = None


# 更新forward references
Site.model_rebuild()
Position.model_rebuild()
Device.model_rebuild()
Component.model_rebuild()
