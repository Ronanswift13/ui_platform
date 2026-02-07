"""
室内电子围栏 - 统一事件契约 (Event Schema)
==========================================

平台统一事件契约定义:
- timestamp: 事件时间戳 (ISO8601)
- type: 事件类型
- location: 位置信息 (统一坐标系)
- value: 事件数值
- confidence: 置信度 (0-1)
- evidence: 证据链
- suggestion: 处置建议
- trace_id: 全链路追踪ID

版本: 3.0.0
作者: 变电站AI监控平台架构组
"""

from __future__ import annotations
import uuid
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import json
import hashlib


# =============================================================================
# 枚举定义
# =============================================================================

class EventType(str, Enum):
    """事件类型"""
    # 人员状态事件
    PERSON_DETECTED = "person_detected"         # 人员检测到
    PERSON_LOST = "person_lost"                 # 人员丢失
    PERSON_STATE_CHANGE = "person_state_change" # 人员状态变化
    
    # 越线告警事件
    YELLOW_LINE_WARNING = "yellow_line_warning" # 压线预警
    YELLOW_LINE_ALARM = "yellow_line_alarm"     # 越线告警
    ZONE_INTRUSION = "zone_intrusion"           # 区域入侵
    
    # 授权事件
    UNAUTHORIZED_ACCESS = "unauthorized_access" # 未授权访问
    AUTHORIZATION_GRANTED = "authorization_granted"  # 授权通过
    
    # 系统事件
    SENSOR_HEALTH = "sensor_health"             # 传感器健康
    SYSTEM_STATUS = "system_status"             # 系统状态
    
    # 跟踪事件
    TRACK_CREATED = "track_created"             # 轨迹创建
    TRACK_UPDATED = "track_updated"             # 轨迹更新
    TRACK_DELETED = "track_deleted"             # 轨迹删除
    TRACK_REIDENTIFIED = "track_reidentified"   # 轨迹重识别


class PersonState(str, Enum):
    """人员状态"""
    SAFE = "safe"               # 安全
    ON_LINE = "on_line"         # 压线
    CROSS_LINE = "cross_line"   # 越线
    UNAUTHORIZED = "unauthorized"  # 未授权
    HIGH_RISK = "high_risk"     # 高风险


class AlertLevel(str, Enum):
    """告警级别"""
    INFO = "info"           # 信息
    WARNING = "warning"     # 警告
    ALARM = "alarm"         # 告警
    CRITICAL = "critical"   # 紧急


class SuggestionType(str, Enum):
    """处置建议类型"""
    NONE = "none"                   # 无需处置
    MONITOR = "monitor"             # 持续监控
    VERBAL_WARNING = "verbal_warning"  # 语音提醒
    LIGHT_WARNING = "light_warning"    # 灯光警示
    SOUND_ALARM = "sound_alarm"        # 声音告警
    MANUAL_CHECK = "manual_check"      # 人工核查
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止


# =============================================================================
# 坐标数据结构
# =============================================================================

@dataclass
class Point2D:
    """2D坐标点 (统一坐标系)"""
    x: float  # X坐标 (米)
    y: float  # Y坐标 (米)
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Point2D':
        return cls(x=float(data.get('x', 0)), y=float(data.get('y', 0)))


@dataclass
class BoundingBox:
    """边界框 (归一化坐标 0-1)"""
    x: float      # 左上角X
    y: float      # 左上角Y
    width: float  # 宽度
    height: float # 高度
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BoundingBox':
        return cls(
            x=float(data.get('x', 0)),
            y=float(data.get('y', 0)),
            width=float(data.get('width', 0)),
            height=float(data.get('height', 0))
        )


@dataclass
class Location:
    """位置信息"""
    ground_position: Point2D        # 地面坐标 (统一坐标系)
    pixel_position: Optional[Point2D] = None  # 像素坐标 (原始图像)
    bbox: Optional[BoundingBox] = None        # 边界框
    zone_id: Optional[str] = None             # 所在区域ID
    cabinet_id: Optional[int] = None          # 所在机柜ID
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"ground_position": self.ground_position.to_dict()}
        if self.pixel_position:
            result["pixel_position"] = self.pixel_position.to_dict()
        if self.bbox:
            result["bbox"] = self.bbox.to_dict()
        if self.zone_id:
            result["zone_id"] = self.zone_id
        if self.cabinet_id is not None:
            result["cabinet_id"] = self.cabinet_id
        return result


# =============================================================================
# 证据链数据结构
# =============================================================================

@dataclass
class EvidenceItem:
    """单条证据"""
    evidence_type: str      # 证据类型: frame, track, sensor, model
    evidence_id: str        # 证据ID
    source: str            # 来源: camera_01, lidar_01, tracker
    timestamp: str         # 时间戳
    data_path: Optional[str] = None   # 数据文件路径 (图片/视频)
    data_inline: Optional[Dict] = None  # 内联数据 (小型JSON)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "evidence_type": self.evidence_type,
            "evidence_id": self.evidence_id,
            "source": self.source,
            "timestamp": self.timestamp
        }
        if self.data_path:
            result["data_path"] = self.data_path
        if self.data_inline:
            result["data_inline"] = self.data_inline
        return result


@dataclass
class Evidence:
    """证据链"""
    frame_raw: Optional[str] = None        # 原始帧路径
    frame_annotated: Optional[str] = None  # 标注帧路径
    video_clip: Optional[str] = None       # 视频片段路径
    track_history: Optional[List[Dict]] = None  # 轨迹历史
    sensor_data: Optional[Dict] = None     # 传感器原始数据
    model_output: Optional[Dict] = None    # 模型输出
    items: List[EvidenceItem] = field(default_factory=list)  # 证据项列表
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self.frame_raw:
            result["frame_raw"] = self.frame_raw
        if self.frame_annotated:
            result["frame_annotated"] = self.frame_annotated
        if self.video_clip:
            result["video_clip"] = self.video_clip
        if self.track_history:
            result["track_history"] = self.track_history
        if self.sensor_data:
            result["sensor_data"] = self.sensor_data
        if self.model_output:
            result["model_output"] = self.model_output
        if self.items:
            result["items"] = [item.to_dict() for item in self.items]
        return result


# =============================================================================
# 处置建议数据结构
# =============================================================================

@dataclass
class Suggestion:
    """处置建议"""
    suggestion_type: SuggestionType
    message: str                    # 建议内容
    priority: int = 1               # 优先级 (1-5, 5最高)
    auto_execute: bool = False      # 是否自动执行
    target_device: Optional[str] = None  # 目标设备 (灯光/喇叭等)
    parameters: Dict[str, Any] = field(default_factory=dict)  # 执行参数
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_type": self.suggestion_type.value,
            "message": self.message,
            "priority": self.priority,
            "auto_execute": self.auto_execute,
            "target_device": self.target_device,
            "parameters": self.parameters
        }


# =============================================================================
# 统一事件数据结构
# =============================================================================

def generate_trace_id() -> str:
    """生成追踪ID"""
    return f"trace_{uuid.uuid4().hex[:16]}_{int(time.time() * 1000)}"


def generate_event_id() -> str:
    """生成事件ID"""
    return f"evt_{uuid.uuid4().hex[:12]}"


@dataclass
class UnifiedEvent:
    """
    统一事件契约
    
    所有室内电子围栏事件必须符合此结构
    """
    # === 必填字段 ===
    event_id: str                   # 事件唯一ID
    timestamp: str                  # ISO8601时间戳
    type: EventType                 # 事件类型
    location: Location              # 位置信息
    value: Dict[str, Any]          # 事件数值 (结构化数据)
    confidence: float              # 置信度 (0-1)
    evidence: Evidence             # 证据链
    suggestion: Suggestion         # 处置建议
    trace_id: str                  # 全链路追踪ID
    
    # === 选填字段 ===
    alert_level: AlertLevel = AlertLevel.INFO  # 告警级别
    person_id: Optional[str] = None            # 人员/轨迹ID
    person_state: Optional[PersonState] = None # 人员状态
    source_plugin: str = "indoor_fence"        # 来源插件
    source_adapter: Optional[str] = None       # 来源适配器
    parent_event_id: Optional[str] = None      # 父事件ID (事件链)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 扩展元数据
    
    def __post_init__(self):
        """校验事件数据"""
        assert 0 <= self.confidence <= 1, "confidence必须在0-1之间"
        assert self.trace_id, "trace_id不能为空"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典 (用于JSON序列化)"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "type": self.type.value,
            "location": self.location.to_dict(),
            "value": self.value,
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict(),
            "suggestion": self.suggestion.to_dict(),
            "trace_id": self.trace_id,
            "alert_level": self.alert_level.value,
            "person_id": self.person_id,
            "person_state": self.person_state.value if self.person_state else None,
            "source_plugin": self.source_plugin,
            "source_adapter": self.source_adapter,
            "parent_event_id": self.parent_event_id,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UnifiedEvent':
        """从字典创建事件"""
        return cls(
            event_id=data.get('event_id', generate_event_id()),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            type=EventType(data['type']),
            location=Location(
                ground_position=Point2D.from_dict(data['location']['ground_position']),
                pixel_position=Point2D.from_dict(data['location']['pixel_position']) 
                    if data['location'].get('pixel_position') else None,
                bbox=BoundingBox.from_dict(data['location']['bbox']) 
                    if data['location'].get('bbox') else None,
                zone_id=data['location'].get('zone_id'),
                cabinet_id=data['location'].get('cabinet_id')
            ),
            value=data.get('value', {}),
            confidence=float(data.get('confidence', 0)),
            evidence=Evidence(
                frame_raw=data['evidence'].get('frame_raw'),
                frame_annotated=data['evidence'].get('frame_annotated'),
                video_clip=data['evidence'].get('video_clip'),
                track_history=data['evidence'].get('track_history'),
                sensor_data=data['evidence'].get('sensor_data'),
                model_output=data['evidence'].get('model_output')
            ),
            suggestion=Suggestion(
                suggestion_type=SuggestionType(data['suggestion']['suggestion_type']),
                message=data['suggestion'].get('message', ''),
                priority=data['suggestion'].get('priority', 1),
                auto_execute=data['suggestion'].get('auto_execute', False),
                target_device=data['suggestion'].get('target_device'),
                parameters=data['suggestion'].get('parameters', {})
            ),
            trace_id=data.get('trace_id', generate_trace_id()),
            alert_level=AlertLevel(data.get('alert_level', 'info')),
            person_id=data.get('person_id'),
            person_state=PersonState(data['person_state']) if data.get('person_state') else None,
            source_plugin=data.get('source_plugin', 'indoor_fence'),
            source_adapter=data.get('source_adapter'),
            parent_event_id=data.get('parent_event_id'),
            metadata=data.get('metadata', {})
        )


# =============================================================================
# 事件构建器 (工厂方法)
# =============================================================================

class IndoorFenceEventBuilder:
    """室内电子围栏事件构建器"""
    
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or generate_trace_id()
    
    def build_person_detected_event(
        self,
        person_id: str,
        ground_position: Point2D,
        bbox: BoundingBox,
        confidence: float,
        frame_path: Optional[str] = None,
        cabinet_id: Optional[int] = None
    ) -> UnifiedEvent:
        """构建人员检测事件"""
        return UnifiedEvent(
            event_id=generate_event_id(),
            timestamp=datetime.now().isoformat(),
            type=EventType.PERSON_DETECTED,
            location=Location(
                ground_position=ground_position,
                bbox=bbox,
                cabinet_id=cabinet_id
            ),
            value={
                "person_count": 1,
                "detection_method": "yolo_v8"
            },
            confidence=confidence,
            evidence=Evidence(
                frame_raw=frame_path,
                model_output={"detector": "yolo", "confidence": confidence}
            ),
            suggestion=Suggestion(
                suggestion_type=SuggestionType.MONITOR,
                message="检测到人员,持续监控中"
            ),
            trace_id=self.trace_id,
            alert_level=AlertLevel.INFO,
            person_id=person_id
        )
    
    def build_yellow_line_alarm_event(
        self,
        person_id: str,
        ground_position: Point2D,
        distance_to_line: float,
        confidence: float,
        frame_path: Optional[str] = None,
        is_cross: bool = False
    ) -> UnifiedEvent:
        """构建越线告警事件"""
        event_type = EventType.YELLOW_LINE_ALARM if is_cross else EventType.YELLOW_LINE_WARNING
        alert_level = AlertLevel.ALARM if is_cross else AlertLevel.WARNING
        
        return UnifiedEvent(
            event_id=generate_event_id(),
            timestamp=datetime.now().isoformat(),
            type=event_type,
            location=Location(ground_position=ground_position),
            value={
                "distance_to_line_m": distance_to_line,
                "is_crossed": is_cross,
                "threshold_m": 0.5 if is_cross else 0.3
            },
            confidence=confidence,
            evidence=Evidence(
                frame_annotated=frame_path,
                model_output={"distance": distance_to_line}
            ),
            suggestion=Suggestion(
                suggestion_type=SuggestionType.SOUND_ALARM if is_cross else SuggestionType.VERBAL_WARNING,
                message="请立即退回安全区域" if is_cross else "已接近黄线,请注意安全",
                priority=5 if is_cross else 3,
                auto_execute=True,
                target_device="speaker_01"
            ),
            trace_id=self.trace_id,
            alert_level=alert_level,
            person_id=person_id,
            person_state=PersonState.CROSS_LINE if is_cross else PersonState.ON_LINE
        )
    
    def build_unauthorized_access_event(
        self,
        person_id: str,
        ground_position: Point2D,
        cabinet_id: int,
        confidence: float,
        frame_path: Optional[str] = None
    ) -> UnifiedEvent:
        """构建未授权访问事件"""
        return UnifiedEvent(
            event_id=generate_event_id(),
            timestamp=datetime.now().isoformat(),
            type=EventType.UNAUTHORIZED_ACCESS,
            location=Location(
                ground_position=ground_position,
                cabinet_id=cabinet_id
            ),
            value={
                "cabinet_id": cabinet_id,
                "authorized": False,
                "required_authorization": f"cabinet_{cabinet_id}"
            },
            confidence=confidence,
            evidence=Evidence(
                frame_annotated=frame_path,
                model_output={"cabinet_detection": cabinet_id}
            ),
            suggestion=Suggestion(
                suggestion_type=SuggestionType.LIGHT_WARNING,
                message=f"机柜{cabinet_id}未授权,请联系值班人员",
                priority=4,
                auto_execute=True,
                target_device="light_controller",
                parameters={"color": "red", "mode": "blink"}
            ),
            trace_id=self.trace_id,
            alert_level=AlertLevel.ALARM,
            person_id=person_id,
            person_state=PersonState.UNAUTHORIZED
        )
    
    def build_sensor_health_event(
        self,
        sensor_id: str,
        sensor_type: str,
        health_score: float,
        status: str,
        details: Dict[str, Any]
    ) -> UnifiedEvent:
        """构建传感器健康状态事件"""
        alert_level = AlertLevel.INFO
        if health_score < 0.5:
            alert_level = AlertLevel.WARNING
        if health_score < 0.3:
            alert_level = AlertLevel.ALARM
        
        return UnifiedEvent(
            event_id=generate_event_id(),
            timestamp=datetime.now().isoformat(),
            type=EventType.SENSOR_HEALTH,
            location=Location(ground_position=Point2D(0, 0)),  # 传感器位置可后续扩展
            value={
                "sensor_id": sensor_id,
                "sensor_type": sensor_type,
                "health_score": health_score,
                "status": status,
                "details": details
            },
            confidence=1.0,  # 系统事件置信度为1
            evidence=Evidence(sensor_data=details),
            suggestion=Suggestion(
                suggestion_type=SuggestionType.MANUAL_CHECK if health_score < 0.5 else SuggestionType.NONE,
                message=f"传感器{sensor_id}健康度: {health_score:.2f}",
                priority=3 if health_score < 0.5 else 1
            ),
            trace_id=self.trace_id,
            alert_level=alert_level,
            source_adapter=sensor_id
        )


# =============================================================================
# 事件验证器
# =============================================================================

class EventValidator:
    """事件验证器"""
    
    @staticmethod
    def validate(event: UnifiedEvent) -> tuple[bool, List[str]]:
        """
        验证事件是否符合契约
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 必填字段检查
        if not event.event_id:
            errors.append("event_id不能为空")
        if not event.timestamp:
            errors.append("timestamp不能为空")
        if not event.trace_id:
            errors.append("trace_id不能为空")
        
        # 置信度范围检查
        if not (0 <= event.confidence <= 1):
            errors.append(f"confidence必须在0-1之间, 当前值: {event.confidence}")
        
        # 坐标有效性检查
        if event.location and event.location.ground_position:
            pos = event.location.ground_position
            if pos.x < -100 or pos.x > 100:
                errors.append(f"ground_position.x超出合理范围: {pos.x}")
            if pos.y < -100 or pos.y > 100:
                errors.append(f"ground_position.y超出合理范围: {pos.y}")
        
        # 证据完整性检查 (告警级别事件必须有证据)
        if event.alert_level in [AlertLevel.ALARM, AlertLevel.CRITICAL]:
            if not event.evidence.frame_raw and not event.evidence.frame_annotated:
                errors.append("告警级别事件必须包含帧图像证据")
        
        return len(errors) == 0, errors


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 枚举
    'EventType',
    'PersonState', 
    'AlertLevel',
    'SuggestionType',
    # 数据结构
    'Point2D',
    'BoundingBox',
    'Location',
    'EvidenceItem',
    'Evidence',
    'Suggestion',
    'UnifiedEvent',
    # 工具
    'IndoorFenceEventBuilder',
    'EventValidator',
    'generate_trace_id',
    'generate_event_id',
]
