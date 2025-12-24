"""
统一数据模型定义

所有数据模型使用Pydantic v2定义
确保类型安全和自动验证
"""


from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def generate_id() -> str:
    """生成唯一ID"""
    return str(uuid4())


class BaseEntity(BaseModel):
    """基础实体模型"""
    id: str = Field(default_factory=generate_id)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============== 站点/设备层级模型 ==============

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


# ============== ROI模型 ==============

class ROIType(str, Enum):
    """ROI识别类型"""
    DEFECT = "defect"  # 缺陷检测
    STATE = "state"  # 状态识别
    METER = "meter"  # 表计读数
    THERMAL = "thermal"  # 热成像
    INTRUSION = "intrusion"  # 入侵检测


class BoundingBox(BaseModel):
    """边界框"""
    x: float  # 左上角x (0-1归一化)
    y: float  # 左上角y
    width: float
    height: float

    @field_validator("x", "y", "width", "height")
    @classmethod
    def validate_normalized(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("坐标值必须在0-1之间")
        return v


class ROI(BaseEntity):
    """识别区域模型"""
    component_id: str
    roi_type: ROIType
    bbox: BoundingBox
    recognition_types: list[str] = Field(default_factory=list)  # 绑定的识别类型
    rules: list["AlarmRule"] = Field(default_factory=list)


class AlarmRule(BaseModel):
    """告警规则"""
    id: str = Field(default_factory=generate_id)
    name: str
    condition: str  # 条件表达式
    level: str = "warning"  # info, warning, error, critical
    message_template: str = ""
    enabled: bool = True


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


# ============== 识别结果模型 ==============

class RecognitionResult(BaseModel):
    """单个识别结果 - 插件输出的最小单位"""
    task_id: str
    site_id: str
    device_id: str
    component_id: str
    roi_id: str
    bbox: BoundingBox
    label: str  # 识别标签
    value: Optional[Any] = None  # 识别值 (表计读数/温度/状态等)
    confidence: float = Field(ge=0, le=1)
    evidence_path: str = ""  # 证据截图路径
    model_version: str = ""
    code_version: str = ""  # code hash
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    failure_reason: Optional[str] = None  # 失败原因码


class PluginOutput(BaseModel):
    """插件标准输出格式"""
    task_id: str
    plugin_id: str
    plugin_version: str
    code_hash: str
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    results: list[RecognitionResult] = Field(default_factory=list)
    alarms: list["Alarm"] = Field(default_factory=list)
    error_message: str = ""
    error_code: Optional[str] = None
    processing_time_ms: float = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============== 告警模型 ==============

class AlarmLevel(str, Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlarmStatus(str, Enum):
    """告警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class Alarm(BaseModel):
    """告警模型"""
    id: str = Field(default_factory=generate_id)
    task_id: str
    result_id: Optional[str] = None
    rule_id: Optional[str] = None
    level: AlarmLevel = AlarmLevel.WARNING
    status: AlarmStatus = AlarmStatus.ACTIVE
    title: str
    message: str
    site_id: str
    device_id: str
    component_id: str = ""
    evidence_path: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: str = ""
    resolved_by: str = ""
    notes: str = ""


# ============== 证据模型 ==============

class EvidenceType(str, Enum):
    """证据类型"""
    RAW_IMAGE = "raw_image"
    ANNOTATED_IMAGE = "annotated_image"
    VIDEO_CLIP = "video_clip"
    THERMAL_IMAGE = "thermal_image"
    LOG = "log"
    RESULT_JSON = "result_json"


class Evidence(BaseModel):
    """证据记录"""
    id: str = Field(default_factory=generate_id)
    run_id: str  # 任务运行ID
    task_id: str
    evidence_type: EvidenceType
    file_path: str
    file_size: int = 0
    checksum: str = ""  # MD5/SHA256
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# 更新forward references
Site.model_rebuild()
Position.model_rebuild()
Device.model_rebuild()
Component.model_rebuild()
ROI.model_rebuild()
