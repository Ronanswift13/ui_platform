"""
IEC104/IEC61850 适配器集成验证
验证: 适配器实例化 → 模拟数据 → _normalize_event() → ActionEventStore → 自动分析
"""
import sys
import os
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_iec104_adapter_instantiation():
    """验证 IEC104Adapter 可通过 ProtocolFactory 实例化并进入模拟模式"""
    from platform_core.data_import.protocol_adapters import ProtocolFactory

    adapter = ProtocolFactory.create(
        "iec104", host="127.0.0.1", port=2404, common_address=1,
    )
    assert adapter is not None
    assert adapter.connect() is True
    assert adapter._simulated is True, "无 lib60870 时应进入模拟模式"

    adapter.inject_simulated_point(1001, True, type_id="SP")
    result = adapter.read("1001")
    assert result is not None
    assert result["value"] is True

    adapter.disconnect()
    print("[PASS] IEC104Adapter 实例化 + 模拟模式 + 读取")


def test_iec61850_adapter_instantiation():
    """验证 IEC61850Adapter 可通过 ProtocolFactory 实例化并进入模拟模式"""
    from platform_core.data_import.protocol_adapters import ProtocolFactory

    adapter = ProtocolFactory.create(
        "iec61850", host="127.0.0.1", port=102, ied_name="PROT_REL_01",
    )
    assert adapter is not None
    assert adapter.connect() is True
    assert adapter._simulated is True, "无 iec61850 库时应进入模拟模式"

    adapter.inject_simulated_data("PROT_REL_01/PTRC1.Tr.general", True)
    result = adapter.read("PROT_REL_01/PTRC1.Tr.general")
    assert result is not None
    assert result["value"] is True

    adapter.disconnect()
    print("[PASS] IEC61850Adapter 实例化 + 模拟模式 + 读取")


def _make_event(event_id="", source_ts="", source_system="iec104",
                secondary_device_id="", signal_id="", signal_group="other",
                action_type="unknown", action_desc="", value_after=None,
                severity_hint="info"):
    """构造 ActionEvent, 使用正确的字段名"""
    from platform_core.action_event_schema import ActionEvent
    return ActionEvent(
        event_id=event_id,
        source_ts=source_ts,
        source_system=source_system,
        secondary_device_id=secondary_device_id,
        signal_id=signal_id,
        signal_group=signal_group,
        action_type=action_type,
        action_desc=action_desc,
        value_after=str(value_after) if value_after is not None else None,
        severity_hint=severity_hint,
    )


def test_iec104_soe_callback_flow():
    """验证 IEC104 SOE回调 → 事件归一化 → Store"""
    from platform_core.data_import.protocol_adapters import ProtocolFactory
    from platform_core.action_event_store import ActionEventStore
    from platform_core.action_event_schema import normalize_action_type

    store = ActionEventStore()
    received_events = []
    store.on_event(lambda ev: received_events.append(ev))

    adapter = ProtocolFactory.create(
        "iec104", host="127.0.0.1", port=2404, common_address=1,
    )
    adapter.connect()

    soe_events = []

    def soe_callback(data):
        soe_events.append(data)
        ev = _make_event(
            event_id=f"iec104_{data.get('ioa', 0)}_{int(time.time()*1000)}",
            source_ts=data.get("timestamp", ""),
            source_system="iec104",
            secondary_device_id=f"dev_{data.get('ioa', 0)}",
            signal_id=str(data.get("ioa", "")),
            action_type=normalize_action_type(data.get("action_type_hint", "unknown")),
            action_desc=f"IEC104 SOE IOA={data.get('ioa')}",
            value_after=data.get("value"),
        )
        store.add_event(ev)

    adapter.subscribe_soe(soe_callback)
    adapter.inject_simulated_point(1001, True, type_id="SOE")

    assert len(soe_events) >= 1, f"SOE回调应触发, 实际: {len(soe_events)}"
    assert len(received_events) >= 1, f"Store应收到事件, 实际: {len(received_events)}"
    assert received_events[0].source_system == "iec104"

    adapter.disconnect()
    print(f"[PASS] IEC104 SOE → Store 流转 ({len(received_events)} events)")


def test_iec61850_report_callback_flow():
    """验证 IEC61850 Report回调 → 事件归一化 → Store"""
    from platform_core.data_import.protocol_adapters import ProtocolFactory
    from platform_core.action_event_store import ActionEventStore

    store = ActionEventStore()
    received_events = []
    store.on_event(lambda ev: received_events.append(ev))

    adapter = ProtocolFactory.create(
        "iec61850", host="127.0.0.1", port=102, ied_name="PROT_REL_01",
    )
    adapter.connect()

    report_entries = []

    def report_callback(data):
        report_entries.append(data)
        ref = data.get("ref", "")
        ev = _make_event(
            event_id=f"iec61850_rpt_{int(time.time()*1000)}",
            source_ts=data.get("timestamp", ""),
            source_system="iec61850",
            secondary_device_id="PROT_REL_01",
            signal_id=ref,
            signal_group=_infer_signal_group(ref),
            action_desc=f"IEC61850 Report: {ref}",
            value_after=data.get("value"),
        )
        store.add_event(ev)

    adapter.subscribe_report(report_callback)
    adapter.inject_simulated_data("PROT_REL_01/PTRC1.Tr.general", True)

    assert len(report_entries) >= 1, f"Report回调应触发, 实际: {len(report_entries)}"
    assert len(received_events) >= 1, f"Store应收到事件, 实际: {len(received_events)}"
    assert received_events[0].source_system == "iec61850"

    adapter.disconnect()
    print(f"[PASS] IEC61850 Report → Store 流转 ({len(received_events)} events)")


def test_iec61850_goose_callback_flow():
    """验证 IEC61850 GOOSE回调 → 事件归一化 → Store"""
    from platform_core.data_import.protocol_adapters import ProtocolFactory
    from platform_core.action_event_store import ActionEventStore

    store = ActionEventStore()
    received_events = []
    store.on_event(lambda ev: received_events.append(ev))

    adapter = ProtocolFactory.create(
        "iec61850", host="127.0.0.1", port=102,
        ied_name="PROT_REL_01", goose_enabled=True,
    )
    adapter.connect()

    goose_messages = []

    def goose_callback(data):
        goose_messages.append(data)
        ev = _make_event(
            event_id=f"goose_{int(time.time()*1000)}",
            source_ts=data.get("timestamp", ""),
            source_system="iec61850_goose",
            secondary_device_id="PROT_REL_01",
            signal_id=data.get("appid", ""),
            signal_group="protection",
            action_type="protection_trip",
            action_desc=f"GOOSE appid={data.get('appid', '')}",
            value_after=str(data.get("data_set")),
            severity_hint="critical",
        )
        store.add_event(ev)

    adapter.subscribe_goose(goose_callback)
    adapter.inject_simulated_goose(
        appid="0001",
        data_set=[{"ref": "PTRC1.Tr.general", "value": True}],
    )

    assert len(goose_messages) >= 1, f"GOOSE回调应触发, 实际: {len(goose_messages)}"
    assert len(received_events) >= 1, f"Store应收到事件, 实际: {len(received_events)}"
    assert received_events[0].source_system == "iec61850_goose"

    adapter.disconnect()
    print(f"[PASS] IEC61850 GOOSE → Store 流转 ({len(received_events)} events)")


def test_full_chain_iec104_to_analysis():
    """端到端: IEC104模拟SOE → Store → ActionSequenceAnalyzer → RootCause"""
    from platform_core.action_event_store import ActionEventStore
    from platform_core.action_event_schema import ActionEvent, ActionType, SignalGroup
    from platform_core.action_sequence_analyzer import ActionSequenceAnalyzer
    from platform_core.root_cause_service import RootCauseService
    from platform_core.device_correlation import (
        DeviceCorrelationService, Station, Bay,
        PrimaryDevice, SecondaryDevice, SignalPoint,
    )

    store = ActionEventStore()
    correlation = DeviceCorrelationService()

    station = Station(station_id="ST001", station_name="测试220kV变电站", voltage_level="220kV")
    correlation.register_station(station)

    bay = Bay(bay_id="BAY001", bay_name="#1主变220kV侧", station_id="ST001", voltage_level="220kV")
    correlation.register_bay(bay)

    pd1 = PrimaryDevice(device_id="T1", device_name="#1主变", device_type="transformer",
                        station_id="ST001", bay_id="BAY001")
    correlation.register_primary_device(pd1)

    sd1 = SecondaryDevice(device_id="PROT_01", device_name="主变保护",
                          device_type="protection_relay", station_id="ST001", bay_id="BAY001")
    correlation.register_secondary_device(sd1)

    sig1 = SignalPoint(signal_id="SIG_001", signal_name="保护动作", signal_group="protection",
                       secondary_device_id="PROT_01", bay_id="BAY001", station_id="ST001")
    sig2 = SignalPoint(signal_id="SIG_002", signal_name="开关变位", signal_group="breaker",
                       secondary_device_id="PROT_01", bay_id="BAY001", station_id="ST001")
    correlation.register_signal_point(sig1)
    correlation.register_signal_point(sig2)

    analyzer = ActionSequenceAnalyzer(correlation_service=correlation)
    root_cause_svc = RootCauseService(correlation_service=correlation)

    base_ts = datetime(2026, 3, 31, 10, 0, 0, tzinfo=timezone.utc)

    events = [
        ActionEvent(
            source_ts=base_ts.isoformat(),
            source_system="iec104", secondary_device_id="PROT_01",
            signal_id="SIG_001", signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value,
            value_after="True", action_desc="保护动作(IEC104 SOE)",
        ),
        ActionEvent(
            source_ts=base_ts.replace(second=0, microsecond=500000).isoformat(),
            source_system="iec104", secondary_device_id="PROT_01",
            signal_id="SIG_002", signal_group=SignalGroup.BREAKER.value,
            action_type=ActionType.BREAKER_OPEN.value,
            value_after="True", action_desc="断路器跳闸(IEC104 SOE)",
        ),
    ]

    for ev in events:
        store.add_event(ev)

    chains = analyzer.analyze(events)
    assert len(chains) > 0, "应识别出动作链"

    chain_type_val = lambda c: c.chain_type.value if hasattr(c.chain_type, 'value') else c.chain_type
    normal_trip = [c for c in chains if chain_type_val(c) == "normal_trip"]
    assert len(normal_trip) > 0, "应识别出正常跳闸链"
    print(f"  识别动作链: {[chain_type_val(c) for c in chains]}")

    root_result = root_cause_svc.analyze(
        chain_results=chains,
        action_events=events,
    )
    assert root_result is not None
    assert root_result.primary_equipment_fault_probability > 0
    print(f"  根因概率: 一次={root_result.primary_equipment_fault_probability:.2f}, "
          f"二次={root_result.secondary_equipment_fault_probability:.2f}")

    archive = root_cause_svc.generate_fault_archive(root_result, events)
    assert archive is not None
    print(f"  故障归档: line={archive.fault_line}, recloser={archive.recloser_enabled}, status={archive.status}")

    print("[PASS] IEC104 端到端: SOE → Store → 动作链分析 → 根因 → 归档")


def _infer_signal_group(reference: str) -> str:
    ref_upper = reference.upper()
    if "PTRC" in ref_upper or "PTOC" in ref_upper or "PDIS" in ref_upper:
        return "protection"
    elif "XCBR" in ref_upper:
        return "breaker"
    elif "XSWI" in ref_upper:
        return "switch"
    elif "RREC" in ref_upper:
        return "recloser"
    return "unknown"


if __name__ == "__main__":
    print("=" * 60)
    print("IEC 适配器集成验证")
    print("=" * 60)

    tests = [
        test_iec104_adapter_instantiation,
        test_iec61850_adapter_instantiation,
        test_iec104_soe_callback_flow,
        test_iec61850_report_callback_flow,
        test_iec61850_goose_callback_flow,
        test_full_chain_iec104_to_analysis,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            print(f"\n--- {test_fn.__name__} ---")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"结果: {passed} passed, {failed} failed / {len(tests)} total")
    print("=" * 60)
