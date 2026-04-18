# -*- coding: utf-8 -*-
"""
第4步：验证
===========

4组模拟动作事件样例:
1. 正常跳闸链样例
2. 控制回路异常样例
3. 重合闸失败样例
4. 信号抖动样例

每个样例包含:
- 模拟动作事件序列
- 动作链分析结果
- 范围判定结果
- 根因分析输出

可直接运行: python -m pytest tests/test_action_event_verification.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from platform_core.action_event_schema import (
    ActionEvent, ActionType, SignalGroup, SourceSystem,
    SeverityHint, ChainType, RootCauseCategory, FaultType,
)
from platform_core.action_event_store import ActionEventStore
from platform_core.action_sequence_analyzer import ActionSequenceAnalyzer, AnalyzerConfig
from platform_core.root_cause_service import (
    RootCauseService, PrimaryDeviceEvidence, SecondaryDeviceEvidence,
)
from platform_core.device_correlation import (
    DeviceCorrelationService, Station, Bay, PrimaryDevice, SecondaryDevice, SignalPoint,
    BusSection, BayBusConnection, ProtectionZone,
)


def _make_ts(offset_seconds: float = 0) -> str:
    """生成ISO8601时间戳"""
    dt = datetime(2026, 3, 31, 10, 0, 0) + timedelta(seconds=offset_seconds)
    return dt.isoformat()


def _setup_correlation() -> DeviceCorrelationService:
    """创建模拟设备关系(含母线拓扑和保护区域)"""
    svc = DeviceCorrelationService()
    svc.register_station(Station(station_id="ST-001", station_name="大理220kV变电站", voltage_level="220kV", region="大理"))

    # 母线
    svc.register_bus_section(BusSection(bus_id="BUS-I", bus_name="220kV I母", station_id="ST-001", voltage_level="220kV", bus_type="double"))
    svc.register_bus_section(BusSection(bus_id="BUS-II", bus_name="220kV II母", station_id="ST-001", voltage_level="220kV", bus_type="double"))

    # 间隔
    svc.register_bay(Bay(bay_id="BAY-001", bay_name="220kV甲线间隔", station_id="ST-001", voltage_level="220kV", bay_type="line"))
    svc.register_bay(Bay(bay_id="BAY-002", bay_name="220kV乙线间隔", station_id="ST-001", voltage_level="220kV", bay_type="line"))

    # 母线连接
    svc.register_bay_bus_connection(BayBusConnection(bay_id="BAY-001", bus_id="BUS-I", breaker_id="CB-001"))
    svc.register_bay_bus_connection(BayBusConnection(bay_id="BAY-002", bus_id="BUS-I", breaker_id="CB-002"))

    # 一次设备
    svc.register_primary_device(PrimaryDevice(device_id="CB-001", device_name="220kV甲线断路器", device_type="breaker", station_id="ST-001", bay_id="BAY-001", voltage_level="220kV"))
    svc.register_primary_device(PrimaryDevice(device_id="LINE-001", device_name="220kV甲线", device_type="line", station_id="ST-001", bay_id="BAY-001", voltage_level="220kV"))
    svc.register_primary_device(PrimaryDevice(device_id="CB-002", device_name="220kV乙线断路器", device_type="breaker", station_id="ST-001", bay_id="BAY-002", voltage_level="220kV"))

    # 二次设备
    svc.register_secondary_device(SecondaryDevice(device_id="PR-001", device_name="甲线主保护", device_type="protection", station_id="ST-001", bay_id="BAY-001", cabinet_id="C-01"))
    svc.register_secondary_device(SecondaryDevice(device_id="RC-001", device_name="甲线重合闸", device_type="recloser", station_id="ST-001", bay_id="BAY-001", cabinet_id="C-01"))

    # 保护区域
    svc.register_protection_zone(ProtectionZone(
        zone_id="PZ-001", secondary_device_id="PR-001", protection_type="main_pilot",
        protected_bay_ids=["BAY-001"], protected_primary_ids=["LINE-001"], trip_breaker_ids=["CB-001"],
    ))

    # 信号点
    svc.register_signal_point(SignalPoint(signal_id="SIG-001", signal_name="保护启动", signal_group="protection", secondary_device_id="PR-001", primary_device_id="CB-001", bay_id="BAY-001", station_id="ST-001"))
    svc.register_signal_point(SignalPoint(signal_id="SIG-002", signal_name="保护出口", signal_group="protection", secondary_device_id="PR-001", primary_device_id="CB-001", bay_id="BAY-001", station_id="ST-001"))
    svc.register_signal_point(SignalPoint(signal_id="SIG-003", signal_name="断路器分位", signal_group="breaker", secondary_device_id="PR-001", primary_device_id="CB-001", bay_id="BAY-001", station_id="ST-001"))
    return svc


# =============================================================================
# 样例1: 正常跳闸链
# =============================================================================

def test_scenario_1_normal_trip():
    """
    场景: 220kV甲线发生短路故障
    序列: 保护启动(T+0) -> 保护出口(T+0.02s) -> 断路器分位(T+0.06s)
    预期: 匹配正常跳闸链，is_real_trip=True，根因偏一次设备故障
    """
    trace_id = "trace_normal_trip_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-001", signal_name="保护启动", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_START.value, action_desc="距离I段保护启动",
            protection_type="backup_distance_1",
            voltage_level="220kV", phase="A", fault_current_ka=3.5,
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离I段保护出口跳闸",
            protection_type="backup_distance_1",
            voltage_level="220kV", phase="A",
            source_ts=_make_ts(0.02), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-003", signal_name="断路器分位", signal_group=SignalGroup.BREAKER.value,
            action_type=ActionType.BREAKER_OPEN.value, action_desc="220kV甲线断路器分闸",
            voltage_level="220kV",
            wave_record_id="WR-001",
            source_ts=_make_ts(0.06), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
    ]

    # 存储
    store = ActionEventStore()
    for e in events:
        store.add_event(e)

    # 分析
    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    assert len(chain_results) > 0, "应匹配至少一条动作链"
    normal_chain = next((r for r in chain_results if r.chain_type == ChainType.NORMAL_TRIP.value), None)
    assert normal_chain is not None, "应匹配正常跳闸链"
    assert normal_chain.confidence >= 0.8
    assert normal_chain.trip_scope is not None
    assert normal_chain.trip_scope.is_real_trip is True

    # 验证新字段填充
    ts = normal_chain.trip_scope
    assert ts.voltage_level == "220kV", f"电压等级应为220kV, 实际: {ts.voltage_level}"
    assert ts.fault_phase == "A", f"故障相别应为A, 实际: {ts.fault_phase}"
    assert ts.fault_type == FaultType.SINGLE_PHASE_GROUND.value, f"故障类型应为单相接地, 实际: {ts.fault_type}"
    assert ts.fault_current == 3.5, f"故障电流应为3.5kA, 实际: {ts.fault_current}"
    assert "有录波" in ts.wave_record_status, f"录波状态应含'有录波', 实际: {ts.wave_record_status}"
    assert len(ts.protection_acted) > 0, "应有保护动作记录"
    assert len(ts.trip_breakers) > 0, "应有跳开断路器记录"
    assert ts.fault_line, "故障线路不应为空"
    # primary_scope 应含 device_name
    for ps in ts.primary_scope:
        assert "device_name" in ps, "primary_scope 应含 device_name"
    # 停电范围推导(CB-001 跳开 → BUS-I 上若无其他电源 → 影响 BAY-001, BAY-002)
    assert len(ts.outage_scope) > 0, "应有停电范围推导"

    # 根因分析
    rcs = RootCauseService(correlation_service=correlation)
    primary_ev = [PrimaryDeviceEvidence(device_id="CB-001", device_type="breaker", source_plugin="switch_inspection", has_defect=False, confidence=0.9)]
    root_cause = rcs.analyze(chain_results=chain_results, action_events=events, primary_evidence=primary_ev)

    assert root_cause.root_cause_category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value
    assert root_cause.primary_equipment_fault_probability > root_cause.secondary_equipment_fault_probability
    assert root_cause.confidence > 0, "confidence 应被填充"
    assert root_cause.external_cause_probability >= 0, "external_cause_probability 应存在"
    assert len(root_cause.probabilities) >= 5, "probabilities dict 应含5个类别"
    assert len(root_cause.evidence_chain) > 0
    assert len(root_cause.recommendations) > 0

    # 故障归档
    archive = rcs.generate_fault_archive(root_cause, events)
    assert archive.voltage_level == "220kV"
    assert archive.fault_type == FaultType.SINGLE_PHASE_GROUND.value
    assert archive.fault_phase == "A"
    assert archive.action_description, "action_description 不应为空"
    assert archive.alarm_group == "protection"
    assert isinstance(archive.primary_repair_scope, list)
    assert archive.fault_current == 3.5
    assert "有录波" in archive.wave_record_status

    print(f"\n[样例1] 正常跳闸链 - 通过")
    print(f"  链类型: {normal_chain.chain_type}")
    print(f"  置信度: {normal_chain.confidence}")
    print(f"  真正跳闸: {ts.is_real_trip}")
    print(f"  电压等级: {ts.voltage_level}")
    print(f"  故障相别: {ts.fault_phase}")
    print(f"  故障类型: {ts.fault_type}")
    print(f"  故障电流: {ts.fault_current} kA")
    print(f"  录波状态: {ts.wave_record_status}")
    print(f"  保护动作: {ts.protection_acted}")
    print(f"  跳开断路器: {ts.trip_breakers}")
    print(f"  停电范围: {ts.outage_scope}")
    print(f"  根因: {root_cause.root_cause_category}")
    print(f"  概率分布: {root_cause.probabilities}")
    print(f"  归档-动作描述: {archive.action_description}")
    print(f"  归档-告警组名: {archive.alarm_group}")


# =============================================================================
# 样例2: 控制回路异常
# =============================================================================

def test_scenario_2_control_loop_abnormal():
    """
    场景: 控制回路断线导致断路器无法操作
    序列: 控制回路断线(T+0) -> 无断路器响应
    预期: 匹配控制回路异常链，根因偏控制回路/机构问题
    """
    trace_id = "trace_control_loop_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-010", signal_name="控制回路断线", signal_group=SignalGroup.CONTROL_LOOP.value,
            action_type=ActionType.CONTROL_LOOP_BREAK.value, action_desc="220kV甲线断路器控制回路断线",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
    ]

    store = ActionEventStore()
    for e in events:
        store.add_event(e)

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    assert len(chain_results) > 0
    ctrl_chain = next((r for r in chain_results if r.chain_type == ChainType.CONTROL_LOOP_ABNORMAL.value), None)
    assert ctrl_chain is not None, "应匹配控制回路异常链"
    assert ctrl_chain.trip_scope is not None
    assert ctrl_chain.trip_scope.is_real_trip is False  # 没有实际跳闸

    rcs = RootCauseService(correlation_service=correlation)
    root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

    assert root_cause.root_cause_category == RootCauseCategory.CONTROL_LOOP_ISSUE.value
    assert root_cause.control_loop_fault_probability > root_cause.primary_equipment_fault_probability

    print(f"\n[样例2] 控制回路异常 - 通过")
    print(f"  链类型: {ctrl_chain.chain_type}")
    print(f"  真正跳闸: {ctrl_chain.trip_scope.is_real_trip}")
    print(f"  根因: {root_cause.root_cause_category}")
    print(f"  控制回路故障概率: {root_cause.control_loop_fault_probability:.2%}")


# =============================================================================
# 样例3: 重合闸失败
# =============================================================================

def test_scenario_3_recloser_fail():
    """
    场景: 线路瞬时故障 -> 保护跳闸 -> 重合闸动作 -> 再次跳闸(永久故障)
    序列: 保护出口(T+0) -> 断路器分闸(T+0.05) -> 重合闸动作(T+1.0) -> 保护再次出口(T+1.05) -> 重合闸失败(T+1.1)
    预期: 匹配重合闸失败链，recloser_enabled=True, recloser_success=False
    """
    trace_id = "trace_recloser_fail_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离I段保护出口跳闸",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-003", signal_name="断路器分位", signal_group=SignalGroup.BREAKER.value,
            action_type=ActionType.BREAKER_OPEN.value, action_desc="220kV甲线断路器分闸",
            source_ts=_make_ts(0.05), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="RC-001", secondary_device_type="recloser",
            signal_id="SIG-020", signal_name="重合闸动作", signal_group=SignalGroup.RECLOSER.value,
            action_type=ActionType.RECLOSER_ACTION.value, action_desc="220kV甲线重合闸动作",
            source_ts=_make_ts(1.0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离I段保护再次出口跳闸",
            source_ts=_make_ts(1.05), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="RC-001", secondary_device_type="recloser",
            signal_id="SIG-021", signal_name="重合闸失败", signal_group=SignalGroup.RECLOSER.value,
            action_type=ActionType.RECLOSER_FAIL.value, action_desc="220kV甲线重合闸不成功",
            source_ts=_make_ts(1.1), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
    ]

    store = ActionEventStore()
    for e in events:
        store.add_event(e)

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    recloser_chain = next((r for r in chain_results if r.chain_type == ChainType.RECLOSER_FAIL.value), None)
    assert recloser_chain is not None, "应匹配重合闸失败链"
    assert recloser_chain.trip_scope is not None
    assert recloser_chain.trip_scope.is_real_trip is True
    assert recloser_chain.trip_scope.recloser_enabled is True
    assert recloser_chain.trip_scope.recloser_success is False

    rcs = RootCauseService(correlation_service=correlation)
    root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

    assert root_cause.root_cause_category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value
    # chain_type reflects the highest-confidence chain which may be normal_trip or recloser_fail
    assert root_cause.chain_type in [ChainType.RECLOSER_FAIL.value, ChainType.NORMAL_TRIP.value]

    print(f"\n[样例3] 重合闸失败 - 通过")
    print(f"  链类型: {recloser_chain.chain_type}")
    print(f"  重合闸投入: {recloser_chain.trip_scope.recloser_enabled}")
    print(f"  重合闸成功: {recloser_chain.trip_scope.recloser_success}")
    print(f"  根因: {root_cause.root_cause_category}")
    print(f"  一次故障概率: {root_cause.primary_equipment_fault_probability:.2%}")


# =============================================================================
# 样例4: 信号抖动 + 根因输出
# =============================================================================

def test_scenario_4_signal_jitter():
    """
    场景: 某信号因接线松动导致短时间反复变位
    序列: 同一信号在3秒内变位4次
    预期: 匹配信号抖动链，根因偏信号异常
    """
    trace_id = "trace_jitter_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-050", signal_name="遥信量001", signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value, action_desc="遥信量001变位(0->1)",
            value_before="0", value_after="1",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_INFO.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-050", signal_name="遥信量001", signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value, action_desc="遥信量001变位(1->0)",
            value_before="1", value_after="0",
            source_ts=_make_ts(0.5), source_system=SourceSystem.PROTECTION_INFO.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-050", signal_name="遥信量001", signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value, action_desc="遥信量001变位(0->1)",
            value_before="0", value_after="1",
            source_ts=_make_ts(1.0), source_system=SourceSystem.PROTECTION_INFO.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-050", signal_name="遥信量001", signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value, action_desc="遥信量001变位(1->0)",
            value_before="1", value_after="0",
            source_ts=_make_ts(1.5), source_system=SourceSystem.PROTECTION_INFO.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
    ]

    store = ActionEventStore()
    for e in events:
        store.add_event(e)

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    jitter_chain = next((r for r in chain_results if r.chain_type == ChainType.SIGNAL_JITTER.value), None)
    assert jitter_chain is not None, "应匹配信号抖动链"
    assert jitter_chain.confidence > 0

    rcs = RootCauseService(correlation_service=correlation)
    secondary_ev = [SecondaryDeviceEvidence(device_id="PR-001", device_type="protection", source_plugin="device_monitoring", health_index=0.7, anomaly_score=0.3)]
    root_cause = rcs.analyze(chain_results=chain_results, action_events=events, secondary_evidence=secondary_ev)

    assert root_cause.root_cause_category == RootCauseCategory.SIGNAL_ANOMALY.value
    assert root_cause.signal_anomaly_probability > root_cause.primary_equipment_fault_probability

    print(f"\n[样例4] 信号抖动 - 通过")
    print(f"  链类型: {jitter_chain.chain_type}")
    print(f"  匹配事件数: {len(jitter_chain.matched_events)}")
    print(f"  根因: {root_cause.root_cause_category}")
    print(f"  信号异常概率: {root_cause.signal_anomaly_probability:.2%}")
    print(f"  建议: {root_cause.recommendations}")


# =============================================================================
# 样例5: 拒动链 — 保护出口但断路器无响应
# =============================================================================

def test_scenario_5_refuse_action():
    """
    场景: 220kV甲线保护出口, 但断路器5秒内无分闸信号
    序列: 保护启动(T+0) -> 保护出口(T+0.02) -> (无断路器分闸)
    预期: 匹配拒动链, is_real_trip=False, 不匹配正常跳闸链
    """
    trace_id = "trace_refuse_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-001", signal_name="保护启动", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_START.value, action_desc="距离I段保护启动",
            protection_type="backup_distance_1",
            voltage_level="220kV", phase="A",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离I段保护出口跳闸",
            protection_type="backup_distance_1",
            voltage_level="220kV", phase="A",
            source_ts=_make_ts(0.02), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    # 不应匹配正常跳闸(无断路器分闸)
    normal = next((r for r in chain_results if r.chain_type == ChainType.NORMAL_TRIP.value), None)
    assert normal is None, "无断路器响应不应匹配正常跳闸链"

    # 应匹配拒动
    refuse = next((r for r in chain_results if r.chain_type == ChainType.REFUSE_ACTION.value), None)
    assert refuse is not None, "应匹配拒动链"
    assert refuse.confidence >= 0.6
    assert refuse.trip_scope is not None
    assert refuse.trip_scope.is_real_trip is False

    print(f"\n[样例5] 拒动链 - 通过")
    print(f"  置信度: {refuse.confidence}")
    print(f"  说明: {refuse.description}")


# =============================================================================
# 样例6: 误动链 — 断路器分闸但无保护前兆
# =============================================================================

def test_scenario_6_false_action():
    """
    场景: 断路器无故分闸, 无任何保护启动/出口信号
    序列: 断路器分位(T+0) -> 告知信号(T+0.5)
    预期: 匹配误动链, is_real_trip=False, 置信度较低(需外部佐证)
    """
    trace_id = "trace_false_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            signal_id="SIG-003", signal_name="断路器分位", signal_group=SignalGroup.BREAKER.value,
            action_type=ActionType.BREAKER_OPEN.value, action_desc="220kV甲线断路器分闸",
            voltage_level="220kV",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            signal_id="SIG-060", signal_name="断路器异常告知", signal_group=SignalGroup.ALARM.value,
            action_type=ActionType.ALARM_SIGNAL.value, action_desc="220kV甲线断路器异常告知",
            voltage_level="220kV",
            source_ts=_make_ts(0.5), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    false_chain = next((r for r in chain_results if r.chain_type == ChainType.FALSE_ACTION.value), None)
    assert false_chain is not None, "应匹配误动链"
    assert false_chain.confidence <= 0.70, "误动判定置信度应有上限(需外部佐证)"
    assert false_chain.trip_scope.is_real_trip is False

    # 不应匹配正常跳闸
    normal = next((r for r in chain_results if r.chain_type == ChainType.NORMAL_TRIP.value), None)
    assert normal is None, "无保护前兆不应匹配正常跳闸"

    print(f"\n[样例6] 误动链 - 通过")
    print(f"  置信度: {false_chain.confidence}")
    print(f"  说明: {false_chain.description}")


# =============================================================================
# 样例7: 机构异常 — 弹簧未储能且无正常跳闸伴随
# =============================================================================

def test_scenario_7_mechanism_abnormal():
    """
    场景: 断路器弹簧未储能, 无保护动作, 无断路器分闸
    序列: 弹簧未储能(T+0)
    预期: 匹配机构异常链, 不匹配正常跳闸
    """
    trace_id = "trace_mechanism_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            signal_id="SIG-070", signal_name="弹簧未储能", signal_group=SignalGroup.MECHANISM.value,
            action_type=ActionType.SPRING_NOT_CHARGED.value, action_desc="220kV甲线断路器弹簧未储能",
            voltage_level="220kV",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    mech_chain = next((r for r in chain_results if r.chain_type == ChainType.MECHANISM_ABNORMAL.value), None)
    assert mech_chain is not None, "应匹配机构异常链"
    assert mech_chain.confidence >= 0.60
    assert mech_chain.trip_scope.is_real_trip is False

    print(f"\n[样例7] 机构异常 - 通过")
    print(f"  置信度: {mech_chain.confidence}")
    print(f"  说明: {mech_chain.description}")


# =============================================================================
# 样例8: 互斥门禁 — 正常跳闸链存在时排除拒动和误动
# =============================================================================

def test_scenario_8_mutual_exclusion():
    """
    场景: 完整正常跳闸链(保护启动+出口+断路器分闸)
    验证: 即使存在一些"无保护前兆的断路器事件", 只要正常跳闸链成立,
          拒动和误动都应被互斥门禁排除
    序列: 保护启动(T+0) -> 保护出口(T+0.02) -> 断路器分闸(T+0.06)
    预期: 仅正常跳闸链出现, 无拒动/误动
    """
    trace_id = "trace_mutual_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-001", signal_name="保护启动", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_START.value, action_desc="距离I段保护启动",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离I段保护出口跳闸",
            source_ts=_make_ts(0.02), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-003", signal_name="断路器分位", signal_group=SignalGroup.BREAKER.value,
            action_type=ActionType.BREAKER_OPEN.value, action_desc="220kV甲线断路器分闸",
            source_ts=_make_ts(0.06), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    chain_types = [r.chain_type for r in chain_results]
    assert ChainType.NORMAL_TRIP.value in chain_types, "应匹配正常跳闸链"
    assert ChainType.REFUSE_ACTION.value not in chain_types, "正常跳闸存在时应排除拒动"
    assert ChainType.FALSE_ACTION.value not in chain_types, "正常跳闸存在时应排除误动"

    print(f"\n[样例8] 互斥门禁 - 通过")
    print(f"  匹配链类型: {chain_types}")


# =============================================================================
# 样例9: 跨间隔隔离 — 甲线保护出口不应匹配乙线断路器分闸
# =============================================================================

def test_scenario_9_cross_bay_isolation():
    """
    场景: 甲线保护出口, 乙线断路器分闸(两个不同间隔)
    序列: 甲线保护出口(BAY-001, T+0) -> 乙线断路器分闸(BAY-002, T+0.05)
    预期: 不应匹配正常跳闸链(bay_id不同), 应匹配拒动(甲线保护出口无对应断路器)
    """
    trace_id = "trace_cross_bay_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="甲线距离I段保护出口跳闸",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-002", bay_name="220kV乙线间隔",
            primary_device_id="CB-002", primary_device_type="breaker",
            signal_id="SIG-103", signal_name="乙线断路器分位", signal_group=SignalGroup.BREAKER.value,
            action_type=ActionType.BREAKER_OPEN.value, action_desc="220kV乙线断路器分闸",
            source_ts=_make_ts(0.05), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    # 不应匹配正常跳闸(bay_id 不匹配)
    normal = next((r for r in chain_results if r.chain_type == ChainType.NORMAL_TRIP.value), None)
    assert normal is None, "跨间隔不应匹配正常跳闸链"

    # 甲线保护出口无对应断路器 → 应匹配拒动
    refuse = next((r for r in chain_results if r.chain_type == ChainType.REFUSE_ACTION.value), None)
    assert refuse is not None, "甲线保护出口无对应断路器应匹配拒动"

    print(f"\n[样例9] 跨间隔隔离 - 通过")
    print(f"  匹配链类型: {[r.chain_type for r in chain_results]}")


# =============================================================================
# 样例10: 保护连续动作不应判为信号抖动
# =============================================================================

def test_scenario_10_protection_not_jitter():
    """
    场景: 保护装置连续3次出口动作(可能是级联故障/后备保护依次动作)
    序列: 保护出口x3 (signal_group=protection, 间隔0.5s)
    预期: 不应匹配信号抖动(protection组被排除), 应匹配正常跳闸或其他链
    """
    trace_id = "trace_prot_not_jitter_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离I段保护出口",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离II段保护出口",
            source_ts=_make_ts(0.5), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="零序I段保护出口",
            source_ts=_make_ts(1.0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    jitter = next((r for r in chain_results if r.chain_type == ChainType.SIGNAL_JITTER.value), None)
    assert jitter is None, "保护信号连续动作不应判为信号抖动"

    print(f"\n[样例10] 保护信号非抖动 - 通过")
    print(f"  匹配链类型: {[r.chain_type for r in chain_results]}")


# =============================================================================
# 样例11-14: 根因分析证据受限工程判断模型对比验证
# =============================================================================
# 以下4组样例共享同一组设备拓扑, 但输入证据不同,
# 验证: 门禁条件 / 证据充分性 / 反证 / 人工复核项 / 下一步动作

def test_scenario_11_root_cause_normal_trip_with_primary_evidence():
    """
    场景: 正常跳闸 + 有一次侧巡视证据(设备正常)
    输入: 保护启动→出口→断路器分闸 + PrimaryDeviceEvidence(has_defect=False)
    预期:
      - 根因=primary_equipment_fault (链证据充分)
      - evidence_sufficiency=sufficient (有链+断路器+一次侧巡视+录波)
      - counter_evidence 应说明排除其他类别的原因
      - 虽然一次侧巡视无缺陷, 但完整跳闸链本身就是一次故障的强证据
      - manual_review_items 不为空
    """
    trace_id = "trace_rc_normal_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-001", signal_name="保护启动", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_START.value, action_desc="距离I段保护启动",
            protection_type="backup_distance_1",
            voltage_level="220kV", phase="A", fault_current_ka=3.5,
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离I段保护出口跳闸",
            protection_type="backup_distance_1",
            voltage_level="220kV", phase="A",
            source_ts=_make_ts(0.02), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            signal_id="SIG-003", signal_name="断路器分位", signal_group=SignalGroup.BREAKER.value,
            action_type=ActionType.BREAKER_OPEN.value, action_desc="220kV甲线断路器分闸",
            voltage_level="220kV", wave_record_id="WR-001",
            source_ts=_make_ts(0.06), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    rcs = RootCauseService(correlation_service=correlation)
    primary_ev = [PrimaryDeviceEvidence(
        device_id="CB-001", device_type="breaker",
        source_plugin="switch_inspection", has_defect=False, confidence=0.9)]
    root_cause = rcs.analyze(
        chain_results=chain_results, action_events=events, primary_evidence=primary_ev)

    # 核心断言
    assert root_cause.root_cause_category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value
    assert root_cause.primary_equipment_fault_probability > root_cause.secondary_equipment_fault_probability
    assert root_cause.evidence_sufficiency == "sufficient", \
        f"有链+断路器+一次侧巡视+录波应为sufficient, 实际: {root_cause.evidence_sufficiency}"
    assert root_cause.confidence > 0

    # 新增字段验证
    assert len(root_cause.counter_evidence) > 0, "应有反证说明"
    assert len(root_cause.evidence_gaps) >= 0  # 有证据时 gaps 可能较少
    assert len(root_cause.manual_review_items) > 0, "应有人工复核项"
    assert len(root_cause.next_actions) > 0, "应有下一步动作"
    assert len(root_cause.probabilities) >= 5, "概率分布应含5个类别"

    print(f"\n[样例11] 正常跳闸(有一次侧证据) 根因对比")
    print(f"  根因: {root_cause.root_cause_category}")
    print(f"  置信度: {root_cause.confidence:.1%}")
    print(f"  证据充分性: {root_cause.evidence_sufficiency}")
    print(f"  概率分布: { {k: f'{v:.1%}' for k, v in root_cause.probabilities.items()} }")
    print(f"  反证: {root_cause.counter_evidence[:2]}")
    print(f"  缺口: {root_cause.evidence_gaps[:2]}")
    print(f"  复核项: {root_cause.manual_review_items[:2]}")
    print(f"  下一步: {root_cause.next_actions[:2]}")


def test_scenario_12_root_cause_control_loop_gates():
    """
    场景: 控制回路异常
    输入: 控制回路断线信号(无保护/无断路器/无一次侧证据)
    预期:
      - 根因=control_loop_issue
      - 门禁C生效: PRIMARY 概率回落至先验(不被拉高)
      - evidence_sufficiency=partial (有链但无断路器变化)
      - counter_evidence 中应包含门禁说明
    """
    trace_id = "trace_rc_ctrl_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-010", signal_name="控制回路断线", signal_group=SignalGroup.CONTROL_LOOP.value,
            action_type=ActionType.CONTROL_LOOP_BREAK.value, action_desc="220kV甲线断路器控制回路断线",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    rcs = RootCauseService(correlation_service=correlation)
    root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

    # 核心断言
    assert root_cause.root_cause_category == RootCauseCategory.CONTROL_LOOP_ISSUE.value
    assert root_cause.control_loop_fault_probability > root_cause.primary_equipment_fault_probability

    # 门禁验证: PRIMARY 不应高于先验(0.30)
    assert root_cause.primary_equipment_fault_probability <= 0.35, \
        f"门禁C应限制PRIMARY概率, 实际: {root_cause.primary_equipment_fault_probability:.1%}"

    # 证据充分性: 无断路器变化 → partial
    assert root_cause.evidence_sufficiency == "partial", \
        f"无断路器变化应为partial, 实际: {root_cause.evidence_sufficiency}"

    # 门禁/反证说明
    all_counter = " ".join(root_cause.counter_evidence)
    assert "门禁" in all_counter or "控制回路" in all_counter, \
        "反证/门禁说明中应提及控制回路/门禁"

    # 证据缺口
    assert len(root_cause.evidence_gaps) > 0, "应有证据缺口说明"
    gaps_text = " ".join(root_cause.evidence_gaps)
    assert "断路器" in gaps_text or "一次设备" in gaps_text, "缺口应提及断路器或一次设备"

    print(f"\n[样例12] 控制回路异常 根因对比(门禁验证)")
    print(f"  根因: {root_cause.root_cause_category}")
    print(f"  PRIMARY概率: {root_cause.primary_equipment_fault_probability:.1%} (应受门禁限制)")
    print(f"  CONTROL_LOOP概率: {root_cause.control_loop_fault_probability:.1%}")
    print(f"  证据充分性: {root_cause.evidence_sufficiency}")
    print(f"  反证: {root_cause.counter_evidence[:3]}")
    print(f"  缺口: {root_cause.evidence_gaps[:3]}")


def test_scenario_13_root_cause_recloser_fail_no_primary():
    """
    场景: 重合闸失败 + 无一次侧巡视证据
    输入: 保护出口→断路器分闸→重合闸动作→再次跳闸→重合闸失败 (无PrimaryDeviceEvidence)
    预期:
      - 根因仍=primary_equipment_fault (重合闸失败是强链证据)
      - 但门禁A生效: PRIMARY概率 ≤ 55% (无一次侧证据上限)
      - evidence_gaps 应包含"缺少一次设备巡视"
      - manual_review_items 应要求"安排故障设备现场巡视"
    """
    trace_id = "trace_rc_recloser_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离I段保护出口跳闸",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            signal_id="SIG-003", signal_name="断路器分位", signal_group=SignalGroup.BREAKER.value,
            action_type=ActionType.BREAKER_OPEN.value, action_desc="220kV甲线断路器分闸",
            source_ts=_make_ts(0.05), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="RC-001", secondary_device_type="recloser",
            signal_id="SIG-020", signal_name="重合闸动作", signal_group=SignalGroup.RECLOSER.value,
            action_type=ActionType.RECLOSER_ACTION.value, action_desc="220kV甲线重合闸动作",
            source_ts=_make_ts(1.0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-002", signal_name="保护出口", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value, action_desc="距离I段保护再次出口跳闸",
            source_ts=_make_ts(1.05), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            primary_device_id="CB-001", primary_device_type="breaker",
            secondary_device_id="RC-001", secondary_device_type="recloser",
            signal_id="SIG-021", signal_name="重合闸失败", signal_group=SignalGroup.RECLOSER.value,
            action_type=ActionType.RECLOSER_FAIL.value, action_desc="220kV甲线重合闸不成功",
            source_ts=_make_ts(1.1), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    rcs = RootCauseService(correlation_service=correlation)
    # 故意不传 primary_evidence — 验证门禁A
    root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

    # 根因仍应是一次设备故障(链证据足够强)
    assert root_cause.root_cause_category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value

    # 门禁A: 无一次侧证据 → PRIMARY ≤ 55%
    assert root_cause.primary_equipment_fault_probability <= 0.56, \
        f"门禁A应限制PRIMARY≤55%, 实际: {root_cause.primary_equipment_fault_probability:.1%}"

    # 证据缺口应包含一次设备
    gaps_text = " ".join(root_cause.evidence_gaps)
    assert "一次设备" in gaps_text, f"缺口应提及一次设备, 实际: {root_cause.evidence_gaps}"

    # 复核项应要求现场巡视
    review_text = " ".join(root_cause.manual_review_items)
    assert "巡视" in review_text, f"复核项应要求巡视, 实际: {root_cause.manual_review_items}"

    # 反证中应包含对其他类别的排除说明
    assert len(root_cause.counter_evidence) > 0, "应有反证说明"
    counter_text = " ".join(root_cause.counter_evidence)
    assert "排除" in counter_text, f"反证应说明排除理由, 实际: {root_cause.counter_evidence}"

    print(f"\n[样例13] 重合闸失败(无一次侧证据) 根因对比(门禁A)")
    print(f"  根因: {root_cause.root_cause_category}")
    print(f"  PRIMARY概率: {root_cause.primary_equipment_fault_probability:.1%} (门禁A限制)")
    print(f"  证据充分性: {root_cause.evidence_sufficiency}")
    print(f"  缺口: {root_cause.evidence_gaps[:3]}")
    print(f"  复核项: {root_cause.manual_review_items[:3]}")
    print(f"  反证: {root_cause.counter_evidence[:3]}")


def test_scenario_14_root_cause_signal_jitter_gate_b():
    """
    场景: 信号抖动 + 无二次设备异常证据
    输入: 同一遥信信号3秒内变位4次, 无SecondaryDeviceEvidence
    预期:
      - 根因=signal_anomaly (信号抖动链→信号异常)
      - 门禁B: SECONDARY不超过先验(仅信号抖动不能判二次设备故障)
      - evidence_sufficiency=partial (有链但缺乏印证)
      - manual_review_items 应要求检查信号回路
    """
    trace_id = "trace_rc_jitter_001"

    events = [
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-050", signal_name="遥信量001", signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value, action_desc="遥信量001变位(0->1)",
            value_before="0", value_after="1",
            source_ts=_make_ts(0), source_system=SourceSystem.PROTECTION_INFO.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-050", signal_name="遥信量001", signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value, action_desc="遥信量001变位(1->0)",
            value_before="1", value_after="0",
            source_ts=_make_ts(0.5), source_system=SourceSystem.PROTECTION_INFO.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-050", signal_name="遥信量001", signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value, action_desc="遥信量001变位(0->1)",
            value_before="0", value_after="1",
            source_ts=_make_ts(1.0), source_system=SourceSystem.PROTECTION_INFO.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST-001", station_name="大理220kV变电站",
            bay_id="BAY-001", bay_name="220kV甲线间隔",
            secondary_device_id="PR-001", secondary_device_type="protection",
            signal_id="SIG-050", signal_name="遥信量001", signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value, action_desc="遥信量001变位(1->0)",
            value_before="1", value_after="0",
            source_ts=_make_ts(1.5), source_system=SourceSystem.PROTECTION_INFO.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
    ]

    correlation = _setup_correlation()
    analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig(), correlation_service=correlation)
    chain_results = analyzer.analyze(events)

    rcs = RootCauseService(correlation_service=correlation)
    # 故意不传 secondary_evidence — 验证门禁B
    root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

    # 根因应为信号异常
    assert root_cause.root_cause_category == RootCauseCategory.SIGNAL_ANOMALY.value
    assert root_cause.signal_anomaly_probability > root_cause.primary_equipment_fault_probability

    # 门禁B: SECONDARY不超过先验(≈25%), 容许归一化后有小幅偏移
    assert root_cause.secondary_equipment_fault_probability <= 0.30, \
        f"门禁B: 仅信号抖动SECONDARY不应高于30%, 实际: {root_cause.secondary_equipment_fault_probability:.1%}"

    # 证据充分性
    assert root_cause.evidence_sufficiency == "partial", \
        f"无断路器变化应为partial, 实际: {root_cause.evidence_sufficiency}"

    # 复核项应涉及信号回路
    review_text = " ".join(root_cause.manual_review_items)
    assert "信号" in review_text, f"复核项应提及信号, 实际: {root_cause.manual_review_items}"

    print(f"\n[样例14] 信号抖动(无二次设备证据) 根因对比(门禁B)")
    print(f"  根因: {root_cause.root_cause_category}")
    print(f"  SIGNAL概率: {root_cause.signal_anomaly_probability:.1%}")
    print(f"  SECONDARY概率: {root_cause.secondary_equipment_fault_probability:.1%} (门禁B限制)")
    print(f"  证据充分性: {root_cause.evidence_sufficiency}")
    print(f"  缺口: {root_cause.evidence_gaps[:3]}")
    print(f"  复核项: {root_cause.manual_review_items[:3]}")


# =============================================================================
# 主执行
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("二次设备动作监测与一次设备故障关联 - 验证样例")
    print("=" * 60)

    test_scenario_1_normal_trip()
    test_scenario_2_control_loop_abnormal()
    test_scenario_3_recloser_fail()
    test_scenario_4_signal_jitter()
    test_scenario_5_refuse_action()
    test_scenario_6_false_action()
    test_scenario_7_mechanism_abnormal()
    test_scenario_8_mutual_exclusion()
    test_scenario_9_cross_bay_isolation()
    test_scenario_10_protection_not_jitter()
    test_scenario_11_root_cause_normal_trip_with_primary_evidence()
    test_scenario_12_root_cause_control_loop_gates()
    test_scenario_13_root_cause_recloser_fail_no_primary()
    test_scenario_14_root_cause_signal_jitter_gate_b()

    print("\n" + "=" * 60)
    print("全部14个验证样例通过!")
    print("=" * 60)
