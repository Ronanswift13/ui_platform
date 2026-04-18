# -*- coding: utf-8 -*-
"""
二次设备动作事件数据模型
========================

定义动作事件(ActionEvent)、跳闸范围判定(TripScopeResult)、
根因分析结果(RootCauseResult)等核心数据结构。

复用: platform_core/schema/indoor_fence_events.py 的 Evidence/EvidenceItem 模式
对齐: 大理供电局二次智能运维平台业务字段

版本: 1.0.0
"""

from __future__ import annotations
import uuid
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum
import json


# =============================================================================
# 枚举定义
# =============================================================================

class ActionType(str, Enum):
    """动作类型"""
    PROTECTION_START = "protection_start"           # 保护启动
    PROTECTION_TRIP = "protection_trip"             # 保护出口(跳闸)
    BREAKER_OPEN = "breaker_open"                   # 断路器分闸
    BREAKER_CLOSE = "breaker_close"                 # 断路器合闸
    RECLOSER_CHARGE = "recloser_charge"             # 重合闸充电完成
    RECLOSER_ACTION = "recloser_action"             # 重合闸动作
    RECLOSER_SUCCESS = "recloser_success"           # 重合闸成功
    RECLOSER_FAIL = "recloser_fail"                 # 重合闸失败
    CONTROL_LOOP_BREAK = "control_loop_break"       # 控制回路断线
    SPRING_NOT_CHARGED = "spring_not_charged"       # 弹簧未储能
    SIGNAL_CHANGE = "signal_change"                 # 遥信变位
    SOE_EVENT = "soe_event"                         # SOE事件
    ALARM_SIGNAL = "alarm_signal"                   # 告知信号
    ABNORMAL_SIGNAL = "abnormal_signal"             # 异常信号
    SWITCH_ACTION = "switch_action"                 # 开关动作
    DISCONNECT_OPEN = "disconnect_open"             # 台闸(停电)
    TRIP_OPEN = "trip_open"                         # 分闸(停电)
    BACKUP_AUTO_SWITCH = "backup_auto_switch"       # 备自投动作
    BACKUP_AUTO_CHARGE = "backup_auto_charge"       # 备自投充电
    BACKUP_AUTO_LOCK = "backup_auto_lock"           # 备自投闭锁
    BUS_DIFF_TRIP = "bus_diff_trip"                 # 母差保护动作
    BREAKER_FAIL_START = "breaker_fail_start"       # 失灵保护启动
    BREAKER_FAIL_TRIP = "breaker_fail_trip"         # 失灵保护跳闸
    DC_GROUND = "dc_ground"                         # 直流接地
    DC_LOSS = "dc_loss"                             # 直流消失
    PLATE_IN = "plate_in"                           # 出口压板投入
    PLATE_OUT = "plate_out"                         # 出口压板退出
    UNKNOWN = "unknown"                             # 未知动作


class ProtectionType(str, Enum):
    """保护类型(保护装置的动作段别)"""
    MAIN_PILOT = "main_pilot"                      # 纵联保护(主保护)
    MAIN_DIFF = "main_diff"                        # 差动保护(主保护)
    BACKUP_DISTANCE_I = "backup_distance_1"        # 后备距离I段
    BACKUP_DISTANCE_II = "backup_distance_2"       # 后备距离II段
    BACKUP_DISTANCE_III = "backup_distance_3"      # 后备距离III段
    BACKUP_OVERCURRENT = "backup_overcurrent"      # 后备过流
    BACKUP_ZERO_SEQ_I = "backup_zero_seq_1"        # 零序I段
    BACKUP_ZERO_SEQ_II = "backup_zero_seq_2"       # 零序II段
    BACKUP_ZERO_SEQ_III = "backup_zero_seq_3"      # 零序III段
    BUS_DIFF = "bus_diff"                          # 母线差动保护
    BREAKER_FAIL = "breaker_fail"                  # 断路器失灵保护
    TRANSFORMER_DIFF = "transformer_diff"          # 变压器差动
    TRANSFORMER_GAS = "transformer_gas"            # 变压器瓦斯
    TRANSFORMER_OVERLOAD = "transformer_overload"  # 变压器过负荷
    AUTO_RECLOSER = "auto_recloser"                # 自动重合闸
    BACKUP_AUTO = "backup_auto"                    # 备自投
    UNKNOWN = "unknown"


class FaultType(str, Enum):
    """故障类型"""
    SINGLE_PHASE_GROUND = "single_phase_ground"    # 单相接地
    PHASE_TO_PHASE = "phase_to_phase"              # 相间短路
    TWO_PHASE_GROUND = "two_phase_ground"          # 两相接地短路
    THREE_PHASE = "three_phase"                    # 三相短路
    UNKNOWN = "unknown"


class SignalGroup(str, Enum):
    """信号组(告警组名)"""
    PROTECTION = "protection"                       # 保护动作
    BREAKER = "breaker"                             # 断路器
    RECLOSER = "recloser"                           # 重合闸
    CONTROL_LOOP = "control_loop"                   # 控制回路
    MECHANISM = "mechanism"                         # 操作机构
    TELEMETRY = "telemetry"                         # 遥信
    SOE = "soe"                                     # SOE
    ALARM = "alarm"                                 # 告知/告警
    ABNORMAL = "abnormal"                           # 异常
    BACKUP_AUTO = "backup_auto"                     # 备自投
    DC_SYSTEM = "dc_system"                         # 直流系统
    POWER_QUALITY = "power_quality"                 # 电能质量
    OTHER = "other"                                 # 其他


class SourceSystem(str, Enum):
    """数据来源系统"""
    OCS = "ocs"                                     # OCS系统
    SUBSTATION_MGMT = "substation_mgmt"             # 变电管理平台
    OMS = "oms"                                     # OMS系统
    GRID_MGMT = "grid_mgmt"                         # 电网管理平台
    PROTECTION_INFO = "protection_info"             # 保信子站
    EXTERNAL_INFO = "external_info"                 # 外信子站
    FAULT_RECORDER = "fault_recorder"               # 故障录波
    PROTECTION_DEVICE = "protection_device"         # 保护装置
    BACKUP_AUTO_DEVICE = "backup_auto_device"       # 备自投装置
    PLATE_MONITOR = "plate_monitor"                 # 出口压板监测
    TERMINAL_BOX = "terminal_box"                   # 端子箱
    DC_MONITOR = "dc_monitor"                       # 直流系统监测
    POWER_QUALITY = "power_quality"                 # 电能质量
    PT_PARALLEL = "pt_parallel"                     # PT并列装置
    WAVE_RANGING = "wave_ranging"                   # 行波测距
    MANUAL = "manual"                               # 人工录入
    SIMULATED = "simulated"                         # 模拟数据


class SeverityHint(str, Enum):
    """严重程度提示"""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


class RootCauseCategory(str, Enum):
    """根因分类"""
    PRIMARY_EQUIPMENT_FAULT = "primary_equipment_fault"       # 一次设备故障
    SECONDARY_EQUIPMENT_FAULT = "secondary_equipment_fault"   # 二次设备故障
    CONTROL_LOOP_ISSUE = "control_loop_issue"                 # 控制回路/机构问题
    SIGNAL_ANOMALY = "signal_anomaly"                         # 信号异常/监测点异常
    EXTERNAL_CAUSE = "external_cause"                         # 外部原因(雷击/鸟害等)
    UNKNOWN = "unknown"                                       # 未知


class ChainType(str, Enum):
    """动作链类型"""
    NORMAL_TRIP = "normal_trip"                     # 正常跳闸链
    RECLOSER_FAIL = "recloser_fail"                 # 重合闸失败链
    REFUSE_ACTION = "refuse_action"                 # 拒动链
    FALSE_ACTION = "false_action"                   # 误动链
    CONTROL_LOOP_ABNORMAL = "control_loop_abnormal" # 控制回路异常链
    MECHANISM_ABNORMAL = "mechanism_abnormal"        # 机构异常链
    SIGNAL_JITTER = "signal_jitter"                 # 信号抖动/重复动作链
    UNKNOWN = "unknown"


class ManualReviewStatus(str, Enum):
    """人工复核状态"""
    PENDING = "pending"                 # 待复核
    REQUIRED = "required"              # 需人工复核(系统判定证据不足时自动设置)
    CONFIRMED = "confirmed"            # 已确认
    REJECTED = "rejected"              # 已驳回
    FALSE_POSITIVE = "false_positive"  # 标记为误报
    HANDLED = "handled"                # 已处置
    FIELD_CHECK = "field_check"        # 需现场复核


class SequenceCompleteness(str, Enum):
    """动作序列完整性"""
    COMPLETE = "complete"              # 完整(保护启动→保护出口→断路器分闸 全链路具备)
    PARTIAL = "partial"                # 部分缺失(关键节点存在但中间环节有缺)
    INCOMPLETE = "incomplete"          # 不完整(关键节点缺失)
    UNKNOWN = "unknown"                # 无法判定


# =============================================================================
# 辅助函数
# =============================================================================

def generate_event_id() -> str:
    return f"AE-{uuid.uuid4().hex[:12]}"


def generate_trace_id() -> str:
    return f"trace_{uuid.uuid4().hex[:16]}_{int(time.time() * 1000)}"


# =============================================================================
# 核心数据模型
# =============================================================================

@dataclass
class ActionEvent:
    """
    二次设备动作事件模型

    对齐大理供电局独立监视/历史告警/动作描述等业务字段。
    """
    event_id: str = ""
    trace_id: str = ""
    # 厂站
    station_id: str = ""
    station_name: str = ""
    # 间隔
    bay_id: str = ""
    bay_name: str = ""
    # 一次设备
    primary_device_id: str = ""
    primary_device_type: str = ""           # transformer/breaker/busbar/capacitor/line...
    # 二次设备
    secondary_device_id: str = ""
    secondary_device_type: str = ""         # protection/recloser/backup_auto/monitor...
    # 信号
    signal_id: str = ""
    signal_name: str = ""
    signal_group: str = SignalGroup.OTHER.value
    # 动作
    action_type: str = ActionType.UNKNOWN.value
    action_desc: str = ""                   # 原始动作描述(如"装置重合闸充电完成")
    protection_type: str = ""               # 保护类型(ProtectionType枚举值)
    value_before: Optional[str] = None
    value_after: Optional[str] = None
    value_numeric: Optional[float] = None   # 遥测量值(故障电流/电压等数值型)
    # 电气属性
    voltage_level: str = ""                 # 电压等级(220kV/110kV/35kV...)
    phase: str = ""                         # 故障相别(A/B/C/AB/BC/CA/ABC/N)
    fault_current_ka: Optional[float] = None  # 故障电流(kA)
    wave_record_id: Optional[str] = None    # 关联录波文件ID
    cabinet_id: str = ""                    # 二次设备屏柜编号
    # 关联
    seq_group_id: str = ""                  # 同组序列标识(跨来源归并同一故障)
    # 时间
    source_ts: str = ""                     # 源端时间戳(ISO8601)
    receive_ts: str = ""                    # 接收时间戳(ISO8601)
    # 来源
    source_system: str = SourceSystem.SIMULATED.value
    seq_no: int = 0                         # 序列号(SOE序号)
    # 严重程度
    severity_hint: str = SeverityHint.INFO.value
    # 原始报文
    raw_payload: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.event_id:
            self.event_id = generate_event_id()
        if not self.trace_id:
            self.trace_id = generate_trace_id()
        if not self.receive_ts:
            self.receive_ts = datetime.now().isoformat()
        if self.raw_payload is None:
            self.raw_payload = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionEvent':
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class TripScopeResult:
    """
    跳闸范围判定结果

    区分一次/二次/影响范围，供告警展示、故障归档、报表统计使用。
    """
    trace_id: str = ""
    # 电气属性
    voltage_level: str = ""
    """故障电压等级"""
    fault_type: str = ""
    """故障类型(FaultType枚举值): single_phase_ground/phase_to_phase/..."""
    fault_phase: str = ""
    """故障相别(A/B/C/AB/BC/CA/ABC/N)"""
    # 范围
    primary_scope: List[Dict[str, str]] = field(default_factory=list)
    """一次设备检修范围: [{device_id, device_type, device_name, bay_id, bay_name}]"""
    secondary_scope: List[Dict[str, str]] = field(default_factory=list)
    """二次设备检修范围: [{device_id, device_type, device_name, bay_id, cabinet_id}]"""
    impact_scope: List[Dict[str, str]] = field(default_factory=list)
    """跳闸影响范围(事件直接涉及): [{station_id, bay_id, bay_name, voltage_level}]"""
    # 动作细节
    protection_acted: List[Dict[str, str]] = field(default_factory=list)
    """动作保护装置: [{device_id, device_name, protection_type, action_time}]"""
    trip_breakers: List[Dict[str, str]] = field(default_factory=list)
    """跳开断路器: [{device_id, device_name, bay_id, trip_time}]"""
    outage_scope: List[Dict[str, str]] = field(default_factory=list)
    """停电/失压范围(拓扑推导): [{bay_id, bay_name, voltage_level, outage_type}]"""
    # 判定依据
    reason: str = ""
    confidence: float = 0.0
    is_real_trip: bool = False
    """是否真正形成跳闸(有保护出口+断路器状态变化)"""
    # 故障归档字段
    fault_line: str = ""
    """故障报告(线路)"""
    recloser_enabled: Optional[bool] = None
    """重合闸是否投入"""
    recloser_success: Optional[bool] = None
    """重合闸是否成功"""
    fault_current: Optional[float] = None
    """故障电流(kA)"""
    wave_record_status: str = ""
    """录波情况"""
    # 时间
    timestamp: str = ""
    action_events: List[str] = field(default_factory=list)
    """关联的动作事件ID列表"""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceNode:
    """
    证据链节点

    支持类型: SOE, 遥信变位, 保护动作, 断路器动作,
             一次设备证据, 二次健康证据, 人工复核结论
    """
    node_type: str = ""     # soe/signal_change/protection/breaker/primary_evidence/secondary_health/manual_review
    source: str = ""        # 来源模块/插件
    timestamp: str = ""
    description: str = ""
    confidence: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RootCauseResult:
    """
    根因分析结果

    输出: 更偏一次设备故障 / 更偏二次设备故障 /
          更偏控制回路机构问题 / 更偏信号异常监测点异常
    """
    trace_id: str = ""
    # 根因
    root_cause_category: str = RootCauseCategory.UNKNOWN.value
    root_cause_reason: str = ""
    confidence: float = 0.0
    """最高概率值(用于告警阈值判断)"""
    # 概率分布(key=RootCauseCategory.value)
    probabilities: Dict[str, float] = field(default_factory=dict)
    """各根因类别后验概率"""
    # 向后兼容的独立概率字段
    primary_equipment_fault_probability: float = 0.0
    secondary_equipment_fault_probability: float = 0.0
    control_loop_fault_probability: float = 0.0
    signal_anomaly_probability: float = 0.0
    external_cause_probability: float = 0.0
    # 证据链
    evidence_chain: List[EvidenceNode] = field(default_factory=list)
    # 反证 / 证据不足说明
    counter_evidence: List[str] = field(default_factory=list)
    """反证列表: 为什么不是其他类别(如'无一次侧巡视缺陷佐证')"""
    evidence_gaps: List[str] = field(default_factory=list)
    """证据缺口列表: 当前缺少哪些输入(如'缺少录波数据')"""
    evidence_sufficiency: str = "unknown"
    """证据充分性: sufficient / partial / insufficient / unknown"""
    # 人工复核 & 下一步
    manual_review_items: List[str] = field(default_factory=list)
    """人工复核项: 针对证据缺口生成的具体核查要求"""
    next_actions: List[str] = field(default_factory=list)
    """推荐下一步动作: 比 recommendations 更具体, 与证据缺口挂钩"""
    # 建议(向后兼容)
    recommendations: List[str] = field(default_factory=list)
    # 关联的动作链类型
    chain_type: str = ChainType.UNKNOWN.value
    # 关联的跳闸范围
    trip_scope: Optional[TripScopeResult] = None
    # 时间
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.trip_scope:
            result['trip_scope'] = self.trip_scope.to_dict()
        result['evidence_chain'] = [e.to_dict() if isinstance(e, EvidenceNode) else e for e in self.evidence_chain]
        return result


@dataclass
class FaultArchive:
    """
    故障归档记录

    对齐大理供电局故障归档页面业务字段:
    故障报告(线路)、一次/二次设备检修范围、重合闸投入/成功、
    故障电流、录波情况、动作描述、告警组名、故障类型/相别。
    """
    archive_id: str = ""
    trace_id: str = ""
    # 基本信息
    station_id: str = ""
    station_name: str = ""
    fault_time: str = ""
    fault_line: str = ""
    voltage_level: str = ""
    # 故障电气属性
    fault_type: str = ""
    """故障类型(FaultType枚举值)"""
    fault_phase: str = ""
    """故障相别"""
    # 范围(结构化)
    primary_repair_scope: List[Dict[str, str]] = field(default_factory=list)
    """一次设备检修范围: [{device_id, device_name, device_type, bay_id, bay_name}]"""
    secondary_repair_scope: List[Dict[str, str]] = field(default_factory=list)
    """二次设备检修范围: [{device_id, device_name, device_type, bay_id, cabinet_id}]"""
    impact_description: str = ""
    outage_scope: List[Dict[str, str]] = field(default_factory=list)
    """停电范围: [{bay_id, bay_name, voltage_level, outage_type}]"""
    # 动作描述与告警
    action_description: str = ""
    """汇总动作描述文本(如"距离I段出口→甲线断路器分闸")"""
    alarm_group: str = ""
    """告警组名(SignalGroup枚举值)"""
    # 重合闸
    recloser_enabled: Optional[bool] = None
    recloser_success: Optional[bool] = None
    # 故障参数
    fault_current: Optional[float] = None
    """故障电流(kA)"""
    wave_record_status: str = ""
    """录波情况"""
    # 分析结果
    root_cause: Optional[RootCauseResult] = None
    trip_scope: Optional[TripScopeResult] = None
    # 状态
    status: str = "draft"       # draft/confirmed/archived
    created_by: str = "system"
    created_at: str = ""
    confirmed_by: str = ""
    confirmed_at: str = ""

    def __post_init__(self):
        if not self.archive_id:
            self.archive_id = f"FA-{uuid.uuid4().hex[:10]}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.root_cause and isinstance(self.root_cause, RootCauseResult):
            result['root_cause'] = self.root_cause.to_dict()
        if self.trip_scope and isinstance(self.trip_scope, TripScopeResult):
            result['trip_scope'] = self.trip_scope.to_dict()
        return result


@dataclass
class CandidateEvent:
    """
    候选事件对象

    将原始动作事件流聚合为值班人员可直接使用的候选事件。
    每个候选事件代表一次需要关注的动作序列(如一次跳闸、一次拒动等)。

    【边界说明】
    本对象输出仅用于辅助决策和人工复核，不得作为自动控制指令的依据。
    当关键字段缺失、时序不足或证据冲突时，manual_review_status 会被
    自动设置为 required，confidence 会降级，boundary_note 会说明原因。
    """
    event_id: str = ""
    trace_id: str = ""
    device_id: str = ""
    station_id: str = ""
    station_name: str = ""
    bay_id: str = ""
    bay_name: str = ""
    # 时间范围
    start_time: str = ""
    end_time: str = ""
    # 事件类型与严重程度
    event_type: str = ""
    """事件类型: normal_trip/recloser_fail/refuse_action/false_action/
    control_loop_abnormal/mechanism_abnormal/signal_jitter"""
    severity: str = SeverityHint.INFO.value
    confidence: float = 0.0
    # 证据
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    """证据列表: [{node_type, source, timestamp, description, confidence, data}]"""
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    """关键动作序列(按时间排序): [{event_id, action_type, action_desc, source_ts, signal_group}]"""
    sequence_completeness: str = SequenceCompleteness.UNKNOWN.value
    """动作序列完整性"""
    # 异常标记
    has_refuse_action: bool = False
    """是否存在拒动"""
    has_false_action: bool = False
    """是否存在误动"""
    has_jitter: bool = False
    """是否存在信号抖动"""
    has_repeated_action: bool = False
    """是否存在重复动作"""
    has_abnormal_sequence: bool = False
    """是否存在保护先后异常(如后备保护先于主保护动作)"""
    # 根因候选
    root_cause_candidates: List[Dict[str, Any]] = field(default_factory=list)
    """根因候选列表: [{category, probability, reason}]"""
    # 历史相似度
    historical_similarity: Optional[Dict[str, Any]] = None
    """与历史类似事件的相似度: {matched_archive_id, similarity_score, matched_event_type, match_basis}
    当暂无历史库时返回 {status: 'no_history_data', similarity_score: 0.0, match_basis: '历史故障库尚未接入'}"""
    # 推断依据
    inference_basis: str = ""
    """推断依据: 解释为什么做出当前判定的文字说明"""
    # 人工复核
    manual_review_status: str = ManualReviewStatus.PENDING.value
    review_comment: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""
    # 边界说明
    boundary_note: str = ""
    """边界说明: 当不能给出强结论时,说明原因(证据不足/冲突/超出分析能力等)"""
    # 关联
    related_event_ids: List[str] = field(default_factory=list)
    """关联的原始动作事件ID列表"""
    trip_scope: Optional[TripScopeResult] = None
    root_cause_result: Optional[RootCauseResult] = None
    # 时间戳
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"CE-{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at
        # 边界说明默认值
        if not self.boundary_note:
            self.boundary_note = "本结果仅为辅助决策建议，不构成自动控制指令，最终判定需人工确认"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.trip_scope and isinstance(self.trip_scope, TripScopeResult):
            result['trip_scope'] = self.trip_scope.to_dict()
        if self.root_cause_result and isinstance(self.root_cause_result, RootCauseResult):
            result['root_cause_result'] = self.root_cause_result.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CandidateEvent':
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


# =============================================================================
# 动作词映射(配置化)
# =============================================================================

# 原始动作描述 -> 标准化 ActionType 映射
# 支持配置化扩展
ACTION_KEYWORD_MAP: Dict[str, str] = {
    # 保护
    "保护启动": ActionType.PROTECTION_START.value,
    "保护出口": ActionType.PROTECTION_TRIP.value,
    "保护动作": ActionType.PROTECTION_TRIP.value,
    "跳闸": ActionType.PROTECTION_TRIP.value,
    # 差动/母差
    "差动动作": ActionType.PROTECTION_TRIP.value,
    "母差动作": ActionType.BUS_DIFF_TRIP.value,
    "母线差动": ActionType.BUS_DIFF_TRIP.value,
    "变差动作": ActionType.PROTECTION_TRIP.value,
    # 失灵保护
    "失灵保护动作": ActionType.BREAKER_FAIL_TRIP.value,
    "失灵保护启动": ActionType.BREAKER_FAIL_START.value,
    "失灵启动": ActionType.BREAKER_FAIL_START.value,
    "失灵跳闸": ActionType.BREAKER_FAIL_TRIP.value,
    # 断路器
    "断路器分位": ActionType.BREAKER_OPEN.value,
    "分闸": ActionType.BREAKER_OPEN.value,
    "断路器合位": ActionType.BREAKER_CLOSE.value,
    "合闸": ActionType.BREAKER_CLOSE.value,
    # 重合闸
    "重合闸充电完成": ActionType.RECLOSER_CHARGE.value,
    "重合闸充电": ActionType.RECLOSER_CHARGE.value,
    "重合闸动作": ActionType.RECLOSER_ACTION.value,
    "重合闸成功": ActionType.RECLOSER_SUCCESS.value,
    "重合闸失败": ActionType.RECLOSER_FAIL.value,
    # 控制回路/机构
    "控制回路断线": ActionType.CONTROL_LOOP_BREAK.value,
    "弹簧未储能": ActionType.SPRING_NOT_CHARGED.value,
    "断路器弹簧未储能": ActionType.SPRING_NOT_CHARGED.value,
    # 备自投
    "备自投": ActionType.BACKUP_AUTO_SWITCH.value,
    "备自投动作": ActionType.BACKUP_AUTO_SWITCH.value,
    "备自投充电": ActionType.BACKUP_AUTO_CHARGE.value,
    "备自投闭锁": ActionType.BACKUP_AUTO_LOCK.value,
    # 直流系统
    "直流接地": ActionType.DC_GROUND.value,
    "直流消失": ActionType.DC_LOSS.value,
    # 出口压板
    "出口压板投入": ActionType.PLATE_IN.value,
    "出口压板退出": ActionType.PLATE_OUT.value,
    "压板投入": ActionType.PLATE_IN.value,
    "压板退出": ActionType.PLATE_OUT.value,
    # 通用信号
    "遥信变位": ActionType.SIGNAL_CHANGE.value,
    "SOE": ActionType.SOE_EVENT.value,
    "告知信号": ActionType.ALARM_SIGNAL.value,
    "异常信号": ActionType.ABNORMAL_SIGNAL.value,
    "开关动作": ActionType.SWITCH_ACTION.value,
    "台闸": ActionType.DISCONNECT_OPEN.value,
    "台闸(停电)": ActionType.DISCONNECT_OPEN.value,
    "分闸(停电)": ActionType.TRIP_OPEN.value,
}

# 保护类型关键词映射(从动作描述推导保护段别)
PROTECTION_TYPE_KEYWORD_MAP: Dict[str, str] = {
    "纵联": ProtectionType.MAIN_PILOT.value,
    "差动": ProtectionType.MAIN_DIFF.value,
    "距离I段": ProtectionType.BACKUP_DISTANCE_I.value,
    "距离Ⅰ段": ProtectionType.BACKUP_DISTANCE_I.value,
    "距离II段": ProtectionType.BACKUP_DISTANCE_II.value,
    "距离Ⅱ段": ProtectionType.BACKUP_DISTANCE_II.value,
    "距离III段": ProtectionType.BACKUP_DISTANCE_III.value,
    "距离Ⅲ段": ProtectionType.BACKUP_DISTANCE_III.value,
    "过流": ProtectionType.BACKUP_OVERCURRENT.value,
    "零序I段": ProtectionType.BACKUP_ZERO_SEQ_I.value,
    "零序Ⅰ段": ProtectionType.BACKUP_ZERO_SEQ_I.value,
    "零序II段": ProtectionType.BACKUP_ZERO_SEQ_II.value,
    "零序Ⅱ段": ProtectionType.BACKUP_ZERO_SEQ_II.value,
    "零序III段": ProtectionType.BACKUP_ZERO_SEQ_III.value,
    "零序Ⅲ段": ProtectionType.BACKUP_ZERO_SEQ_III.value,
    "母差": ProtectionType.BUS_DIFF.value,
    "母线差动": ProtectionType.BUS_DIFF.value,
    "失灵": ProtectionType.BREAKER_FAIL.value,
    "变差": ProtectionType.TRANSFORMER_DIFF.value,
    "变压器差动": ProtectionType.TRANSFORMER_DIFF.value,
    "瓦斯": ProtectionType.TRANSFORMER_GAS.value,
    "过负荷": ProtectionType.TRANSFORMER_OVERLOAD.value,
    "重合闸": ProtectionType.AUTO_RECLOSER.value,
    "备自投": ProtectionType.BACKUP_AUTO.value,
}


def normalize_protection_type(action_desc: str) -> str:
    """根据动作描述推导保护类型"""
    if not action_desc:
        return ProtectionType.UNKNOWN.value
    for keyword, ptype in PROTECTION_TYPE_KEYWORD_MAP.items():
        if keyword in action_desc:
            return ptype
    return ProtectionType.UNKNOWN.value


def normalize_action_type(action_desc: str) -> str:
    """
    根据原始动作描述归一化为标准ActionType

    支持模糊匹配: 动作描述中包含关键词即匹配。
    """
    if not action_desc:
        return ActionType.UNKNOWN.value
    for keyword, action_type in ACTION_KEYWORD_MAP.items():
        if keyword in action_desc:
            return action_type
    return ActionType.UNKNOWN.value


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'ActionType', 'ProtectionType', 'FaultType',
    'SignalGroup', 'SourceSystem', 'SeverityHint',
    'RootCauseCategory', 'ChainType',
    'ManualReviewStatus', 'SequenceCompleteness',
    'ActionEvent', 'TripScopeResult', 'RootCauseResult',
    'EvidenceNode', 'FaultArchive', 'CandidateEvent',
    'generate_event_id', 'generate_trace_id',
    'normalize_action_type', 'normalize_protection_type',
    'ACTION_KEYWORD_MAP', 'PROTECTION_TYPE_KEYWORD_MAP',
]
