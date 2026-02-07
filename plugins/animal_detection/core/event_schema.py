#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - 统一事件契约
================================

遵循全平台统一事件契约:
timestamp / type / location / value / confidence
+ evidence(证据链) + suggestion(处置建议) + trace_id

作者: G组 | 版本: 1.0.0
"""

from __future__ import annotations
import uuid
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class AnimalClass(str, Enum):
    """动物分类"""
    MOUSE = "mouse"         # 鼠
    CAT = "cat"             # 猫
    SNAKE = "snake"         # 蛇
    BIRD = "bird"           # 鸟
    DOG = "dog"             # 狗
    POULTRY = "poultry"     # 家禽
    INSECT = "insect"       # 昆虫（大型）
    OTHER = "other"         # 其他


class EventType(str, Enum):
    """事件类型"""
    INTRUSION_DETECTED = "animal_intrusion_detected"
    INTRUSION_CONFIRMED = "animal_intrusion_confirmed"
    INTRUSION_CLEARED = "animal_intrusion_cleared"
    DETERRENT_TRIGGERED = "animal_deterrent_triggered"
    DETERRENT_RESULT = "animal_deterrent_result"
    THERMAL_VALIDATED = "animal_thermal_validated"
    THERMAL_REJECTED = "animal_thermal_rejected"
    TRACKING_UPDATE = "animal_tracking_update"
    STAY_EXCEEDED = "animal_stay_exceeded"
    STATISTICS_REPORT = "animal_statistics_report"


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# 动物风险映射
ANIMAL_RISK_MAP: Dict[str, RiskLevel] = {
    AnimalClass.MOUSE: RiskLevel.HIGH,      # 鼠：啃咬电缆，高风险
    AnimalClass.SNAKE: RiskLevel.CRITICAL,   # 蛇：可能导致短路
    AnimalClass.CAT: RiskLevel.MEDIUM,       # 猫：可能损坏设备
    AnimalClass.BIRD: RiskLevel.LOW,         # 鸟：低风险
    AnimalClass.DOG: RiskLevel.MEDIUM,
    AnimalClass.POULTRY: RiskLevel.LOW,
    AnimalClass.INSECT: RiskLevel.LOW,
    AnimalClass.OTHER: RiskLevel.MEDIUM,
}

# 处置建议映射
SUGGESTION_MAP: Dict[str, str] = {
    AnimalClass.MOUSE: "立即启动超声波驱离；检查电缆沟密封状态；建议投放物理捕鼠器",
    AnimalClass.SNAKE: "紧急告警，检测到蛇入侵；通知人员勿接近；启动声光驱离；联系专业人员处置",
    AnimalClass.CAT: "启动声光驱离；检查门窗密封；记录入侵路径",
    AnimalClass.BIRD: "启动低频声波驱离；检查通风口防护网完整性",
    AnimalClass.DOG: "启动声光驱离；检查门禁状态；通知安保人员",
    AnimalClass.POULTRY: "启动声光驱离；检查围栏完整性",
    AnimalClass.INSECT: "记录并监控；如为蜂群则通知专业人员",
    AnimalClass.OTHER: "启动声光驱离；人工确认动物类型后制定处置方案",
}


@dataclass
class BoundingBox:
    """检测边界框 (归一化坐标 0-1)"""
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float,
                  img_w: int, img_h: int) -> "BoundingBox":
        """从像素坐标xyxy转换为归一化xywh"""
        return cls(
            x=x1 / img_w,
            y=y1 / img_h,
            width=(x2 - x1) / img_w,
            height=(y2 - y1) / img_h,
        )


@dataclass
class WorldPosition:
    """世界坐标（米）"""
    x: float
    y: float
    z: float = 0.0
    coordinate_system: str = "room_local"  # room_local / utm / wgs84

    def to_dict(self) -> Dict[str, Any]:
        return {"x": self.x, "y": self.y, "z": self.z, "coordinate_system": self.coordinate_system}


@dataclass
class EvidenceItem:
    """证据链条目"""
    type: str            # raw_image / annotated_image / thermal_image / video_clip / log
    path: str            # 文件路径
    description: str     # 描述
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "path": self.path,
            "description": self.description,
            "timestamp": self.timestamp,
        }


@dataclass
class AnimalDetectionResult:
    """单次动物检测结果"""
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    animal_class: str = AnimalClass.OTHER
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    world_position: Optional[WorldPosition] = None
    track_id: Optional[str] = None
    is_thermal_validated: bool = False
    thermal_diff_c: Optional[float] = None
    stay_duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "detection_id": self.detection_id,
            "animal_class": self.animal_class,
            "confidence": round(self.confidence, 4),
            "track_id": self.track_id,
            "is_thermal_validated": self.is_thermal_validated,
            "thermal_diff_c": round(self.thermal_diff_c, 2) if self.thermal_diff_c is not None else None,
            "stay_duration_s": round(self.stay_duration_s, 1),
        }
        if self.bbox:
            d["bbox"] = self.bbox.to_dict()
        if self.world_position:
            d["world_position"] = self.world_position.to_dict()
        return d


@dataclass
class AnimalEvent:
    """
    统一事件契约
    遵循: timestamp / type / location / value / confidence
          + evidence + suggestion + trace_id
    """
    # 核心字段
    timestamp: float = field(default_factory=time.time)
    type: str = EventType.INTRUSION_DETECTED
    location: str = ""                    # 位置描述 (e.g. "GIS室-A区")
    value: Any = None                     # 事件值 (检测列表/统计数据)
    confidence: float = 0.0               # 综合置信度

    # 扩展字段
    evidence: List[EvidenceItem] = field(default_factory=list)
    suggestion: str = ""                  # 处置建议
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # 元数据
    plugin_id: str = "animal_detection"
    plugin_version: str = "1.0.0"
    risk_level: str = RiskLevel.LOW
    source_camera_id: str = ""
    site_id: str = ""
    device_id: str = ""

    # 检测详情
    detections: List[AnimalDetectionResult] = field(default_factory=list)
    total_detections: int = 0
    deterrent_active: bool = False
    deterrent_result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            # 统一契约核心
            "timestamp": self.timestamp,
            "type": self.type,
            "location": self.location,
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "evidence": [e.to_dict() for e in self.evidence],
            "suggestion": self.suggestion,
            "trace_id": self.trace_id,
            # 元数据
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "risk_level": self.risk_level,
            "source_camera_id": self.source_camera_id,
            "site_id": self.site_id,
            "device_id": self.device_id,
            # 检测详情
            "detections": [d.to_dict() for d in self.detections],
            "total_detections": self.total_detections,
            "deterrent_active": self.deterrent_active,
            "deterrent_result": self.deterrent_result,
        }


def build_intrusion_event(
    detections: List[AnimalDetectionResult],
    camera_id: str,
    location: str,
    site_id: str = "",
    evidence: Optional[List[EvidenceItem]] = None,
) -> AnimalEvent:
    """构建标准入侵检测事件"""
    if not detections:
        return AnimalEvent(
            type=EventType.INTRUSION_CLEARED,
            location=location,
            source_camera_id=camera_id,
            site_id=site_id,
            value={"status": "clear", "count": 0},
        )

    # 计算综合置信度 (取最高)
    max_conf = max(d.confidence for d in detections)

    # 确定最高风险
    risk_levels = [ANIMAL_RISK_MAP.get(d.animal_class, RiskLevel.MEDIUM) for d in detections]
    risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    highest_risk = max(risk_levels, key=lambda r: risk_order.index(r))

    # 构建处置建议（聚合所有种类）
    classes = set(d.animal_class for d in detections)
    suggestions = [SUGGESTION_MAP.get(c, "") for c in classes]
    combined_suggestion = "；".join(s for s in suggestions if s)

    return AnimalEvent(
        type=EventType.INTRUSION_DETECTED,
        location=location,
        value={
            "status": "intrusion",
            "count": len(detections),
            "classes": list(classes),
        },
        confidence=max_conf,
        evidence=evidence or [],
        suggestion=combined_suggestion,
        risk_level=highest_risk,
        source_camera_id=camera_id,
        site_id=site_id,
        detections=detections,
        total_detections=len(detections),
    )
