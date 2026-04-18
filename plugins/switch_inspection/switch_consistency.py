# -*- coding: utf-8 -*-
"""
开关状态一致性校核模块
======================

将 switch_inspection 从"视频分合位识别"升级为"视频+传感+历史动作记录的一致性校核"。

一致性判定对象包含:
- 视频判别结果
- 传感/遥信输入(通过适配层接入)
- 历史动作事件引用(接 action_event_monitoring)
- 最终状态结论
- 一致性状态: consistent / inconsistent / insufficient / review_required
- 冲突原因
- 可信度
- 边界说明

【边界说明】
本模块结论仅用于远程巡视辅助与一致性校核，不得直接生成远方控制命令。
当视频不清晰、传感信号缺失、动作链不完整时，必须降级为 review_required 或 insufficient。
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举与常量
# =============================================================================

class ConsistencyStatus(str, Enum):
    """一致性状态"""
    CONSISTENT = "consistent"           # 一致: 视频+传感+动作记录三路吻合
    INCONSISTENT = "inconsistent"       # 不一致: 至少两路证据冲突
    INSUFFICIENT = "insufficient"       # 数据不足: 关键证据缺失，无法判定
    REVIEW_REQUIRED = "review_required" # 需人工复核: 可疑但不确定


class SwitchDeviceType(str, Enum):
    """开关设备类型"""
    BREAKER = "breaker"           # 断路器
    ISOLATOR = "isolator"         # 隔离开关
    GROUNDING = "grounding"       # 接地开关


class EvidenceSource(str, Enum):
    """证据来源"""
    VIDEO = "video"               # 视频识别
    SENSOR = "sensor"             # 传感/遥信
    ACTION_EVENT = "action_event" # 历史动作事件


# =============================================================================
# 传感/遥信适配层
# =============================================================================

class SensorAdapter:
    """
    传感/遥信信号适配层

    当前实现为接口适配层，支持以下接入方式:
    - OPC UA: 通过 platform_core.data_import.protocol_adapters 接入
    - IEC 104: 通过遥信变位帧解析
    - MQTT: 通过主题订阅
    - 手动录入: 通过 API 直接设置

    适配层负责将外部信号统一为 SensorReading 格式。
    """

    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}
        self._cache: Dict[str, "SensorReading"] = {}

    def get_reading(self, device_id: str) -> Optional["SensorReading"]:
        """获取设备的最新传感读数"""
        cached = self._cache.get(device_id)
        if cached:
            return cached
        # 无缓存数据时返回 None，由调用方处理 insufficient 状态
        return None

    def set_reading(self, device_id: str, reading: "SensorReading") -> None:
        """手动设置传感读数(供 API/协议适配器回调使用)"""
        self._cache[device_id] = reading

    def set_from_protocol(
        self,
        device_id: str,
        state: str,
        source_protocol: str = "manual",
        timestamp: str = "",
        quality: float = 1.0,
    ) -> "SensorReading":
        """从协议数据创建并缓存传感读数"""
        reading = SensorReading(
            device_id=device_id,
            state=state,
            source_protocol=source_protocol,
            timestamp=timestamp or datetime.now().isoformat(),
            quality=quality,
        )
        self._cache[device_id] = reading
        return reading

    def clear(self, device_id: Optional[str] = None) -> None:
        if device_id:
            self._cache.pop(device_id, None)
        else:
            self._cache.clear()


# =============================================================================
# 数据模型
# =============================================================================

@dataclass
class SensorReading:
    """传感/遥信读数"""
    device_id: str = ""
    state: str = ""                    # open / closed / unknown
    source_protocol: str = "manual"    # opc_ua / iec104 / mqtt / manual
    timestamp: str = ""
    quality: float = 1.0               # 0-1, 信号质量

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoJudgment:
    """视频判别结果"""
    device_id: str = ""
    device_type: str = ""              # breaker / isolator / grounding
    state: str = ""                    # open / closed / intermediate / unknown
    confidence: float = 0.0
    clarity_score: float = 0.0
    evidence_sources: List[str] = field(default_factory=list)  # ocr / color / angle / dl / yolov8_vit
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionEventReference:
    """历史动作事件引用"""
    device_id: str = ""
    latest_action_type: str = ""       # breaker_open / breaker_close / ...
    latest_action_time: str = ""
    implied_state: str = ""            # 由动作推断的状态: open / closed / unknown
    event_count: int = 0
    data_status: str = "no_data"       # sufficient / partial / no_data
    candidate_event_ids: List[str] = field(default_factory=list)
    source_detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConflictDetail:
    """冲突详情"""
    source_a: str = ""                 # 冲突方A(如 video)
    source_b: str = ""                 # 冲突方B(如 sensor)
    state_a: str = ""                  # A的状态判定
    state_b: str = ""                  # B的状态判定
    description: str = ""              # 人可读的冲突说明
    severity: str = "warning"          # warning / error

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsistencyResult:
    """
    一致性判定对象

    【边界说明】
    本结论仅用于远程巡视辅助与一致性校核，不得直接生成远方控制命令。
    当关键证据缺失或冲突时，consistency_status 会被设为 review_required 或 insufficient。
    """
    result_id: str = ""
    device_id: str = ""
    device_type: str = ""              # breaker / isolator / grounding
    station_id: str = ""
    bay_id: str = ""
    # 三路证据
    video_judgment: Optional[VideoJudgment] = None
    sensor_reading: Optional[SensorReading] = None
    action_event_ref: Optional[ActionEventReference] = None
    # 最终结论
    final_state: str = ""              # open / closed / intermediate / unknown
    consistency_status: str = ConsistencyStatus.INSUFFICIENT.value
    confidence: float = 0.0
    # 冲突
    conflicts: List[ConflictDetail] = field(default_factory=list)
    # 边界说明
    boundary_note: str = "本结论仅用于远程巡视辅助与一致性校核，不得直接生成远方控制命令"
    # 人工复核
    review_required: bool = False
    review_reason: str = ""
    # 时间
    timestamp: str = ""

    def __post_init__(self):
        if not self.result_id:
            self.result_id = f"CR-{uuid.uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "result_id": self.result_id,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "station_id": self.station_id,
            "bay_id": self.bay_id,
            "video_judgment": self.video_judgment.to_dict() if self.video_judgment else None,
            "sensor_reading": self.sensor_reading.to_dict() if self.sensor_reading else None,
            "action_event_ref": self.action_event_ref.to_dict() if self.action_event_ref else None,
            "final_state": self.final_state,
            "consistency_status": self.consistency_status,
            "confidence": self.confidence,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "boundary_note": self.boundary_note,
            "review_required": self.review_required,
            "review_reason": self.review_reason,
            "timestamp": self.timestamp,
        }
        return result


# =============================================================================
# 一致性校核引擎
# =============================================================================

# 规则阈值
VIDEO_CONFIDENCE_THRESHOLD = 0.6          # 视频识别可信度阈值
VIDEO_CLARITY_THRESHOLD = 0.5             # 视频清晰度阈值(低于此值视为不可信)
SENSOR_QUALITY_THRESHOLD = 0.3            # 传感信号质量阈值
ACTION_TIME_WINDOW_S = 300                # 动作事件时间窗(5分钟内的动作记录才参与校核)

# 传感信号优先级(当视频与传感冲突时)
# 断路器: 传感优先(遥信可靠性高于视频)
# 隔离开关: 视频优先(隔离开关无遥信或遥信不可靠)
# 接地开关: 传感优先(接地开关位置传感器可靠)
SENSOR_PRIORITY_BY_TYPE = {
    SwitchDeviceType.BREAKER.value: "sensor",
    SwitchDeviceType.ISOLATOR.value: "video",
    SwitchDeviceType.GROUNDING.value: "sensor",
}


class ConsistencyChecker:
    """
    开关状态一致性校核引擎

    判据规则:
    1. 视频识别可信度 >= VIDEO_CONFIDENCE_THRESHOLD 才作为有效证据
    2. 视频清晰度 < VIDEO_CLARITY_THRESHOLD → 视频证据不可信
    3. 传感信号质量 < SENSOR_QUALITY_THRESHOLD → 传感证据不可信
    4. 动作事件仅参考 ACTION_TIME_WINDOW_S 内的记录
    5. 三路一致 → consistent
    6. 两路一致一路缺失 → consistent(降低置信度)
    7. 两路冲突 → inconsistent + 冲突说明
    8. 仅一路证据 → insufficient 或 review_required
    9. 所有证据缺失 → insufficient
    """

    def __init__(
        self,
        sensor_adapter: Optional[SensorAdapter] = None,
        action_event_store=None,
    ):
        self.sensor_adapter = sensor_adapter or SensorAdapter()
        self.action_event_store = action_event_store

    def check_consistency(
        self,
        device_id: str,
        device_type: str,
        station_id: str = "",
        bay_id: str = "",
        video_judgment: Optional[VideoJudgment] = None,
        sensor_reading: Optional[SensorReading] = None,
        time_window_s: float = ACTION_TIME_WINDOW_S,
    ) -> ConsistencyResult:
        """
        执行一致性校核

        Args:
            device_id: 设备ID
            device_type: 设备类型(breaker/isolator/grounding)
            station_id: 厂站ID
            bay_id: 间隔ID
            video_judgment: 视频判别结果(可从 plugin.infer 获取)
            sensor_reading: 传感读数(可从适配层获取)
            time_window_s: 动作事件查询时间窗(秒)

        Returns:
            ConsistencyResult 一致性判定对象
        """
        # 1. 收集传感证据
        if sensor_reading is None:
            sensor_reading = self.sensor_adapter.get_reading(device_id)

        # 2. 收集历史动作事件
        action_ref = self._fetch_action_events(device_id, time_window_s)

        # 3. 校验视频证据有效性
        video_valid = self._is_video_valid(video_judgment)
        sensor_valid = self._is_sensor_valid(sensor_reading)
        action_valid = action_ref is not None and action_ref.data_status != "no_data"

        # 4. 提取各路状态
        video_state = video_judgment.state if video_valid and video_judgment else None
        sensor_state = sensor_reading.state if sensor_valid and sensor_reading else None
        action_state = action_ref.implied_state if action_valid and action_ref else None

        # 5. 一致性判定
        states = {
            EvidenceSource.VIDEO.value: video_state,
            EvidenceSource.SENSOR.value: sensor_state,
            EvidenceSource.ACTION_EVENT.value: action_state,
        }
        valid_states = {k: v for k, v in states.items() if v and v != "unknown"}

        conflicts: List[ConflictDetail] = []
        consistency_status = ConsistencyStatus.INSUFFICIENT.value
        final_state = "unknown"
        confidence = 0.0
        review_required = False
        review_reason = ""

        if len(valid_states) == 0:
            # 无有效证据
            consistency_status = ConsistencyStatus.INSUFFICIENT.value
            final_state = "unknown"
            confidence = 0.0
            review_required = True
            review_reason = "所有证据路径均缺失(视频/传感/动作记录)，无法判定设备状态"

        elif len(valid_states) == 1:
            # 仅一路证据
            source, state = next(iter(valid_states.items()))
            final_state = state
            # 单路证据不给高置信度
            if source == EvidenceSource.VIDEO.value and video_judgment:
                confidence = min(video_judgment.confidence * 0.7, 0.6)
            elif source == EvidenceSource.SENSOR.value and sensor_reading:
                confidence = min(sensor_reading.quality * 0.8, 0.7)
            else:
                confidence = 0.4
            consistency_status = ConsistencyStatus.REVIEW_REQUIRED.value
            review_required = True
            review_reason = f"仅有{source}一路证据，不足以独立确认状态，需人工复核"

        else:
            # 多路证据 → 比对一致性
            unique_states = set(valid_states.values())

            if len(unique_states) == 1:
                # 所有有效证据一致
                final_state = unique_states.pop()
                consistency_status = ConsistencyStatus.CONSISTENT.value
                confidence = self._calc_confidence_consistent(
                    video_judgment, sensor_reading, action_ref,
                    video_valid, sensor_valid, action_valid, device_type)

            else:
                # 存在冲突
                consistency_status = ConsistencyStatus.INCONSISTENT.value
                conflicts = self._build_conflicts(valid_states, device_type)

                # 冲突解决: 按设备类型选择优先证据
                final_state, confidence = self._resolve_conflict(
                    valid_states, device_type, video_judgment, sensor_reading)
                review_required = True
                conflict_desc = "; ".join(c.description for c in conflicts)
                review_reason = f"证据冲突: {conflict_desc}"

        # 6. 额外降级检查
        if video_judgment and not video_valid:
            if video_judgment.clarity_score < VIDEO_CLARITY_THRESHOLD:
                review_required = True
                if not review_reason:
                    review_reason = f"视频清晰度 {video_judgment.clarity_score:.2f} 低于阈值 {VIDEO_CLARITY_THRESHOLD}，视频证据不可信"

        # 视频显示 intermediate → 必须人工确认
        if video_judgment and video_judgment.state == "intermediate":
            review_required = True
            consistency_status = ConsistencyStatus.REVIEW_REQUIRED.value
            if not review_reason:
                review_reason = "视频识别到设备处于中间态，需人工确认是否为过渡状态或机构故障"

        # 有动作记录但视频滞后: 动作事件与视频状态不一致且动作时间近
        if action_valid and action_ref and video_valid and video_judgment:
            if action_ref.implied_state != video_judgment.state and action_ref.implied_state != "unknown":
                # 检查是否为视频滞后(动作时间在最近30秒内)
                try:
                    action_dt = datetime.fromisoformat(action_ref.latest_action_time)
                    now = datetime.now()
                    if (now - action_dt).total_seconds() < 30:
                        review_required = True
                        if consistency_status != ConsistencyStatus.INCONSISTENT.value:
                            consistency_status = ConsistencyStatus.REVIEW_REQUIRED.value
                        review_reason = (
                            f"最新动作事件({action_ref.latest_action_type})发生在 "
                            f"{action_ref.latest_action_time}，但视频显示 {video_judgment.state}，"
                            "可能为视频帧滞后，需确认实际状态"
                        )
                except (ValueError, TypeError):
                    pass

        # 无动作记录但状态突变检查(由调用方根据历史缓存判断)

        boundary_note = "本结论仅用于远程巡视辅助与一致性校核，不得直接生成远方控制命令"
        if review_required:
            boundary_note += f"。{review_reason}"

        return ConsistencyResult(
            device_id=device_id,
            device_type=device_type,
            station_id=station_id,
            bay_id=bay_id,
            video_judgment=video_judgment,
            sensor_reading=sensor_reading,
            action_event_ref=action_ref,
            final_state=final_state,
            consistency_status=consistency_status,
            confidence=round(confidence, 4),
            conflicts=conflicts,
            boundary_note=boundary_note,
            review_required=review_required,
            review_reason=review_reason,
        )

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _is_video_valid(self, vj: Optional[VideoJudgment]) -> bool:
        if vj is None:
            return False
        if vj.confidence < VIDEO_CONFIDENCE_THRESHOLD:
            return False
        if vj.clarity_score < VIDEO_CLARITY_THRESHOLD:
            return False
        if vj.state in ("", "unknown"):
            return False
        return True

    def _is_sensor_valid(self, sr: Optional[SensorReading]) -> bool:
        if sr is None:
            return False
        if sr.quality < SENSOR_QUALITY_THRESHOLD:
            return False
        if sr.state in ("", "unknown"):
            return False
        return True

    def _fetch_action_events(self, device_id: str, window_s: float) -> Optional[ActionEventReference]:
        """从 action_event_store 查询历史动作事件"""
        if self.action_event_store is None:
            return ActionEventReference(
                device_id=device_id,
                data_status="no_data",
                source_detail="动作事件存储未接入",
            )

        try:
            now = datetime.now()
            start_time = (now - timedelta(seconds=window_s)).isoformat()
            end_time = now.isoformat()

            result = self.action_event_store.query_events_for_device(
                device_id=device_id,
                start_time=start_time,
                end_time=end_time,
            )

            events = result.get("events", [])
            candidates = result.get("candidate_events", [])
            data_status = result.get("data_status", "no_data")

            if not events:
                return ActionEventReference(
                    device_id=device_id,
                    data_status="no_data",
                    source_detail=f"在 {window_s}s 内无动作事件记录",
                )

            # 取最新事件推断状态
            latest = events[0]  # query_events_for_device 按时间倒序
            action_type = latest.get("action_type", "")
            implied_state = self._action_to_state(action_type)

            candidate_ids = []
            for c in candidates:
                cid = c.get("event_id", "")
                if cid:
                    candidate_ids.append(cid)

            return ActionEventReference(
                device_id=device_id,
                latest_action_type=action_type,
                latest_action_time=latest.get("source_ts", ""),
                implied_state=implied_state,
                event_count=len(events),
                data_status=data_status,
                candidate_event_ids=candidate_ids,
                source_detail=f"{len(events)} 条事件在 {window_s}s 内",
            )

        except Exception as e:
            logger.error(f"查询动作事件失败: {e}")
            return ActionEventReference(
                device_id=device_id,
                data_status="no_data",
                source_detail=f"查询失败: {e}",
            )

    @staticmethod
    def _action_to_state(action_type: str) -> str:
        """从动作类型推断设备状态"""
        open_actions = {"breaker_open", "trip_open", "disconnect_open"}
        close_actions = {"breaker_close"}
        if action_type in open_actions:
            return "open"
        if action_type in close_actions:
            return "closed"
        return "unknown"

    def _calc_confidence_consistent(
        self,
        vj: Optional[VideoJudgment],
        sr: Optional[SensorReading],
        ar: Optional[ActionEventReference],
        v_valid: bool, s_valid: bool, a_valid: bool,
        device_type: str,
    ) -> float:
        """计算一致情况下的综合置信度"""
        scores = []
        if v_valid and vj:
            scores.append(vj.confidence * 0.9)
        if s_valid and sr:
            scores.append(sr.quality * 0.95)
        if a_valid and ar and ar.data_status == "sufficient":
            scores.append(0.8)
        elif a_valid and ar:
            scores.append(0.6)

        if not scores:
            return 0.0

        # 多路一致给予奖励
        base = max(scores)
        if len(scores) >= 3:
            base = min(base + 0.1, 0.98)
        elif len(scores) >= 2:
            base = min(base + 0.05, 0.95)

        return base

    def _build_conflicts(
        self, valid_states: Dict[str, str], device_type: str
    ) -> List[ConflictDetail]:
        """构建冲突详情"""
        conflicts = []
        sources = list(valid_states.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                sa, sb = sources[i], sources[j]
                state_a, state_b = valid_states[sa], valid_states[sb]
                if state_a != state_b:
                    source_labels = {
                        "video": "视频识别",
                        "sensor": "传感/遥信",
                        "action_event": "动作事件记录",
                    }
                    state_labels = {
                        "open": "分闸", "closed": "合闸",
                        "intermediate": "中间态",
                    }
                    desc = (
                        f"{source_labels.get(sa, sa)}判定为"
                        f"{state_labels.get(state_a, state_a)}，"
                        f"但{source_labels.get(sb, sb)}判定为"
                        f"{state_labels.get(state_b, state_b)}"
                    )
                    conflicts.append(ConflictDetail(
                        source_a=sa, source_b=sb,
                        state_a=state_a, state_b=state_b,
                        description=desc,
                        severity="error" if "sensor" in (sa, sb) else "warning",
                    ))
        return conflicts

    def _resolve_conflict(
        self,
        valid_states: Dict[str, str],
        device_type: str,
        vj: Optional[VideoJudgment],
        sr: Optional[SensorReading],
    ) -> tuple:
        """冲突解决: 按设备类型优先级选择"""
        priority = SENSOR_PRIORITY_BY_TYPE.get(device_type, "sensor")

        if priority == "sensor" and EvidenceSource.SENSOR.value in valid_states:
            final = valid_states[EvidenceSource.SENSOR.value]
            conf = sr.quality * 0.7 if sr else 0.5
        elif priority == "video" and EvidenceSource.VIDEO.value in valid_states:
            final = valid_states[EvidenceSource.VIDEO.value]
            conf = vj.confidence * 0.6 if vj else 0.4
        else:
            # 取任一有效状态
            final = next(iter(valid_states.values()))
            conf = 0.4

        # 冲突时置信度上限
        conf = min(conf, 0.65)
        return final, conf
