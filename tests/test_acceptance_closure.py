# -*- coding: utf-8 -*-
"""
工程化收口验收测试
==================

覆盖5类核心业务场景，每个场景验证:
- 动作链分析 → chain_type + confidence
- 跳闸范围判定 → is_real_trip + primary_scope/secondary_scope
- 根因分析 → root_cause_category + evidence_sufficiency

场景清单:
1. 正常跳闸 (normal_trip)
2. 重合闸失败 (recloser_fail)
3. 控制回路异常 (control_loop_abnormal)
4. 信号抖动 (signal_jitter)
5. 疑似误动 (false_action)

运行: python3 -m pytest tests/test_acceptance_closure.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from platform_core.action_event_schema import (
    ActionEvent, ActionType, SignalGroup, SourceSystem,
    SeverityHint, ChainType, RootCauseCategory,
)
from platform_core.action_event_store import ActionEventStore
from platform_core.action_sequence_analyzer import ActionSequenceAnalyzer, AnalyzerConfig
from platform_core.root_cause_service import (
    RootCauseService, PrimaryDeviceEvidence, SecondaryDeviceEvidence,
)
from platform_core.device_correlation import (
    DeviceCorrelationService, Station, Bay, PrimaryDevice, SecondaryDevice,
    SignalPoint, BusSection, BayBusConnection, ProtectionZone,
)


# =============================================================================
# 公共 fixture
# =============================================================================

T0 = datetime(2026, 4, 6, 10, 0, 0)


def ts(offset_s: float = 0) -> str:
    return (T0 + timedelta(seconds=offset_s)).isoformat()


def make_correlation() -> DeviceCorrelationService:
    svc = DeviceCorrelationService()
    svc.register_station(Station(
        station_id="ST-DL", station_name="大理220kV变电站",
        voltage_level="220kV", region="大理",
    ))
    svc.register_bus_section(BusSection(
        bus_id="BUS-I", bus_name="220kV I母",
        station_id="ST-DL", voltage_level="220kV", bus_type="double",
    ))
    svc.register_bay(Bay(
        bay_id="BAY-JIA", bay_name="220kV甲线间隔",
        station_id="ST-DL", voltage_level="220kV", bay_type="line",
    ))
    svc.register_bay_bus_connection(BayBusConnection(
        bay_id="BAY-JIA", bus_id="BUS-I", breaker_id="CB-JIA",
    ))
    svc.register_primary_device(PrimaryDevice(
        device_id="CB-JIA", device_name="220kV甲线断路器",
        device_type="breaker", station_id="ST-DL",
        bay_id="BAY-JIA", voltage_level="220kV",
    ))
    svc.register_primary_device(PrimaryDevice(
        device_id="LINE-JIA", device_name="220kV甲线",
        device_type="line", station_id="ST-DL",
        bay_id="BAY-JIA", voltage_level="220kV",
    ))
    svc.register_secondary_device(SecondaryDevice(
        device_id="PR-JIA", device_name="甲线主保护",
        device_type="protection", station_id="ST-DL",
        bay_id="BAY-JIA", cabinet_id="C-01",
    ))
    svc.register_secondary_device(SecondaryDevice(
        device_id="RC-JIA", device_name="甲线重合闸",
        device_type="recloser", station_id="ST-DL",
        bay_id="BAY-JIA", cabinet_id="C-01",
    ))
    svc.register_signal_point(SignalPoint(
        signal_id="SIG-PROT-TRIP", signal_name="甲线保护出口",
        signal_group=SignalGroup.PROTECTION.value,
        secondary_device_id="PR-JIA", primary_device_id="LINE-JIA",
        bay_id="BAY-JIA", station_id="ST-DL",
    ))
    svc.register_signal_point(SignalPoint(
        signal_id="SIG-CB-POS", signal_name="甲线断路器分位",
        signal_group=SignalGroup.BREAKER.value,
        secondary_device_id="PR-JIA", primary_device_id="CB-JIA",
        bay_id="BAY-JIA", station_id="ST-DL",
    ))
    svc.register_protection_zone(ProtectionZone(
        zone_id="PZ-JIA", secondary_device_id="PR-JIA",
        protection_type="main_pilot",
        protected_bay_ids=["BAY-JIA"],
        protected_primary_ids=["LINE-JIA"],
        trip_breaker_ids=["CB-JIA"],
    ))
    return svc


def make_stack():
    """创建完整分析栈"""
    corr = make_correlation()
    cfg = AnalyzerConfig()
    analyzer = ActionSequenceAnalyzer(config=cfg, correlation_service=corr)
    rcs = RootCauseService(correlation_service=corr)
    return analyzer, rcs


def ev(trace_id, action_type, signal_group, offset_s, desc="", **kw):
    """快速构造事件"""
    return ActionEvent(
        trace_id=trace_id,
        station_id="ST-DL", station_name="大理220kV变电站",
        bay_id="BAY-JIA", bay_name="220kV甲线间隔",
        primary_device_id=kw.get("primary_device_id", "CB-JIA"),
        primary_device_type=kw.get("primary_device_type", "breaker"),
        secondary_device_id=kw.get("secondary_device_id", "PR-JIA"),
        signal_id=kw.get("signal_id", ""),
        signal_name=kw.get("signal_name", desc),
        signal_group=signal_group,
        action_type=action_type,
        action_desc=desc or action_type,
        voltage_level="220kV",
        source_ts=ts(offset_s),
        source_system=kw.get("source_system", SourceSystem.PROTECTION_DEVICE.value),
        severity_hint=kw.get("severity_hint", SeverityHint.ALARM.value),
    )


# =============================================================================
# 场景1: 正常跳闸
# 预期: chain=NORMAL_TRIP, is_real_trip=True, root_cause→PRIMARY(有巡视佐证时)
# =============================================================================

def test_acceptance_1_normal_trip():
    """正常跳闸: 保护启动→保护出口→断路器分闸"""
    analyzer, rcs = make_stack()
    tid = "acc_normal_001"
    events = [
        ev(tid, ActionType.PROTECTION_START.value, SignalGroup.PROTECTION.value, 0,
           "220kV甲线距离I段启动"),
        ev(tid, ActionType.PROTECTION_TRIP.value, SignalGroup.PROTECTION.value, 0.02,
           "220kV甲线距离I段出口"),
        ev(tid, ActionType.BREAKER_OPEN.value, SignalGroup.BREAKER.value, 0.06,
           "220kV甲线断路器分闸"),
    ]

    chains = analyzer.analyze(events)
    normal = next((c for c in chains if c.chain_type == ChainType.NORMAL_TRIP.value), None)
    assert normal is not None, "应匹配正常跳闸链"
    assert normal.confidence >= 0.70
    assert normal.trip_scope is not None
    assert normal.trip_scope.is_real_trip is True
    assert normal.trip_scope.voltage_level == "220kV"

    # 根因分析(附带一次侧巡视正常证据)
    rc = rcs.analyze(
        chain_results=chains, action_events=events,
        primary_evidence=[PrimaryDeviceEvidence(
            device_id="LINE-JIA", has_defect=False, confidence=0.9,
            source_plugin="transformer_inspection",
        )],
    )
    # 有完整保护链 → PRIMARY 仍为最高概率(巡视正常会降低但不会反转)
    assert rc.root_cause_category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value
    assert rc.evidence_sufficiency in ("sufficient", "partial")
    assert len(rc.evidence_chain) > 0

    print(f"\n[收口1] 正常跳闸 PASS")
    print(f"  chain_type={normal.chain_type}, confidence={normal.confidence:.2f}")
    print(f"  is_real_trip={normal.trip_scope.is_real_trip}")
    print(f"  root_cause={rc.root_cause_category}, sufficiency={rc.evidence_sufficiency}")


# =============================================================================
# 场景2: 重合闸失败
# 预期: chain=RECLOSER_FAIL, root_cause→PRIMARY(无一次证据时受Gate A限制)
# =============================================================================

def test_acceptance_2_recloser_fail():
    """重合闸失败: 跳闸→重合闸动作→再次跳闸→重合闸失败"""
    analyzer, rcs = make_stack()
    tid = "acc_recl_001"
    events = [
        ev(tid, ActionType.PROTECTION_TRIP.value, SignalGroup.PROTECTION.value, 0,
           "220kV甲线保护出口"),
        ev(tid, ActionType.BREAKER_OPEN.value, SignalGroup.BREAKER.value, 0.05,
           "220kV甲线断路器分闸"),
        ev(tid, ActionType.RECLOSER_ACTION.value, SignalGroup.RECLOSER.value, 1.0,
           "220kV甲线重合闸动作", secondary_device_id="RC-JIA"),
        ev(tid, ActionType.PROTECTION_TRIP.value, SignalGroup.PROTECTION.value, 1.05,
           "220kV甲线保护再次出口"),
        ev(tid, ActionType.RECLOSER_FAIL.value, SignalGroup.RECLOSER.value, 1.1,
           "220kV甲线重合闸失败", secondary_device_id="RC-JIA"),
    ]

    chains = analyzer.analyze(events)
    recl = next((c for c in chains if c.chain_type == ChainType.RECLOSER_FAIL.value), None)
    assert recl is not None, "应匹配重合闸失败链"
    assert recl.confidence >= 0.75

    # 根因分析(无一次侧证据)
    rc = rcs.analyze(chain_results=chains, action_events=events)
    # 重合闸失败 → 强烈指向PRIMARY, 但Gate A无一次证据 → 上限55%
    assert rc.root_cause_category == RootCauseCategory.PRIMARY_EQUIPMENT_FAULT.value
    assert rc.primary_equipment_fault_probability <= 0.60
    assert "一次" in " ".join(rc.evidence_gaps) or len(rc.evidence_gaps) > 0
    assert rc.evidence_sufficiency in ("partial", "insufficient")

    print(f"\n[收口2] 重合闸失败 PASS")
    print(f"  chain_type={recl.chain_type}, confidence={recl.confidence:.2f}")
    print(f"  root_cause={rc.root_cause_category}, P(primary)={rc.primary_equipment_fault_probability:.2f}")
    print(f"  sufficiency={rc.evidence_sufficiency}, gaps={rc.evidence_gaps}")


# =============================================================================
# 场景3: 控制回路异常
# 预期: chain=CONTROL_LOOP_ABNORMAL, root_cause→CONTROL_LOOP_ISSUE
# =============================================================================

def test_acceptance_3_control_loop_abnormal():
    """控制回路断线: 无断路器响应"""
    analyzer, rcs = make_stack()
    tid = "acc_ctrl_001"
    events = [
        ev(tid, ActionType.CONTROL_LOOP_BREAK.value, SignalGroup.CONTROL_LOOP.value, 0,
           "220kV甲线控制回路断线",
           severity_hint=SeverityHint.ALARM.value),
    ]

    chains = analyzer.analyze(events)
    ctrl = next((c for c in chains if c.chain_type == ChainType.CONTROL_LOOP_ABNORMAL.value), None)
    assert ctrl is not None, "应匹配控制回路异常链"
    assert ctrl.confidence >= 0.70

    rc = rcs.analyze(chain_results=chains, action_events=events)
    assert rc.root_cause_category == RootCauseCategory.CONTROL_LOOP_ISSUE.value
    # Gate C: 控制回路链 → PRIMARY ≤ prior
    assert rc.control_loop_fault_probability > rc.primary_equipment_fault_probability
    assert len(rc.manual_review_items) > 0  # 应要求人工核查

    print(f"\n[收口3] 控制回路异常 PASS")
    print(f"  chain_type={ctrl.chain_type}, confidence={ctrl.confidence:.2f}")
    print(f"  root_cause={rc.root_cause_category}")
    print(f"  review_items={rc.manual_review_items}")


# =============================================================================
# 场景4: 信号抖动
# 预期: chain=SIGNAL_JITTER, root_cause→SIGNAL_ANOMALY
# =============================================================================

def test_acceptance_4_signal_jitter():
    """信号抖动: 同一信号3秒内4次交替变位"""
    analyzer, rcs = make_stack()
    tid = "acc_jitter_001"
    # 构造交替变位序列: 0→1, 1→0, 0→1, 1→0
    events = []
    for i in range(4):
        vb, va = ("0", "1") if i % 2 == 0 else ("1", "0")
        events.append(ActionEvent(
            trace_id=tid,
            station_id="ST-DL", station_name="大理220kV变电站",
            bay_id="BAY-JIA", bay_name="220kV甲线间隔",
            secondary_device_id="PR-JIA", secondary_device_type="protection",
            signal_id="SIG-TELE-001", signal_name="遥信量001",
            signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value,
            action_desc=f"遥信量001变位({vb}->{va})",
            value_before=vb, value_after=va,
            voltage_level="220kV",
            source_ts=ts(i * 0.8),
            source_system=SourceSystem.PROTECTION_INFO.value,
            severity_hint=SeverityHint.WARNING.value,
        ))

    chains = analyzer.analyze(events)
    jitter = next((c for c in chains if c.chain_type == ChainType.SIGNAL_JITTER.value), None)
    assert jitter is not None, "应匹配信号抖动链"
    assert jitter.confidence >= 0.50

    rc = rcs.analyze(chain_results=chains, action_events=events)
    assert rc.root_cause_category == RootCauseCategory.SIGNAL_ANOMALY.value
    # Gate B: 信号抖动 + 无二次异常 → SECONDARY ≤ prior
    assert rc.signal_anomaly_probability > rc.secondary_equipment_fault_probability

    print(f"\n[收口4] 信号抖动 PASS")
    print(f"  chain_type={jitter.chain_type}, confidence={jitter.confidence:.2f}")
    print(f"  root_cause={rc.root_cause_category}, P(signal)={rc.signal_anomaly_probability:.2f}")


# =============================================================================
# 场景5: 疑似误动
# 预期: chain=FALSE_ACTION, is_real_trip=False, root_cause→SECONDARY(最可能)
# =============================================================================

def test_acceptance_5_false_action():
    """疑似误动: 断路器分闸但无保护前兆"""
    analyzer, rcs = make_stack()
    tid = "acc_false_001"
    events = [
        ev(tid, ActionType.BREAKER_OPEN.value, SignalGroup.BREAKER.value, 0,
           "220kV甲线断路器分闸"),
        ev(tid, ActionType.ALARM_SIGNAL.value, SignalGroup.ALARM.value, 0.5,
           "220kV甲线断路器异常告知",
           severity_hint=SeverityHint.WARNING.value),
    ]

    chains = analyzer.analyze(events)
    false_chain = next((c for c in chains if c.chain_type == ChainType.FALSE_ACTION.value), None)
    assert false_chain is not None, "应匹配误动链"
    assert false_chain.confidence <= 0.70
    assert false_chain.trip_scope is not None
    assert false_chain.trip_scope.is_real_trip is False

    # 根因分析
    rc = rcs.analyze(chain_results=chains, action_events=events)
    # 误动链 → SECONDARY偏高, 但置信度低 → 结论仅供参考
    assert rc.root_cause_category in (
        RootCauseCategory.SECONDARY_EQUIPMENT_FAULT.value,
        RootCauseCategory.SIGNAL_ANOMALY.value,
    )
    assert rc.confidence < 0.60  # 低置信度
    assert rc.evidence_sufficiency in ("partial", "insufficient")
    assert len(rc.manual_review_items) > 0

    print(f"\n[收口5] 疑似误动 PASS")
    print(f"  chain_type={false_chain.chain_type}, confidence={false_chain.confidence:.2f}")
    print(f"  is_real_trip={false_chain.trip_scope.is_real_trip}")
    print(f"  root_cause={rc.root_cause_category}, confidence={rc.confidence:.2f}")
    print(f"  sufficiency={rc.evidence_sufficiency}")
    print(f"  review_items={rc.manual_review_items}")
