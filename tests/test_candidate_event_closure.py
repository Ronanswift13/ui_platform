# -*- coding: utf-8 -*-
"""
候选事件闭环测试
================

覆盖P0要求的6项测试场景:
1. 正常动作链 → 候选事件生成
2. 拒动/误动/抖动/乱序/重复动作 → 异常标记与降级
3. 关键证据缺失时的降级输出
4. 人工复核状态流转
5. API返回结构稳定性
6. 跨插件查询接口契约

可直接运行: python -m pytest tests/test_candidate_event_closure.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from platform_core.action_event_schema import (
    ActionEvent, ActionType, SignalGroup, SourceSystem,
    SeverityHint, ChainType, RootCauseCategory,
    CandidateEvent, ManualReviewStatus, SequenceCompleteness,
)
from platform_core.action_event_store import ActionEventStore
from platform_core.action_sequence_analyzer import ActionSequenceAnalyzer, AnalyzerConfig
from platform_core.root_cause_service import RootCauseService


def _ts(offset_s: float = 0) -> str:
    dt = datetime(2026, 4, 7, 14, 0, 0) + timedelta(seconds=offset_s)
    return dt.isoformat()


def _make_normal_trip_events(trace_id: str = "trace_test_001") -> list:
    """正常跳闸链: 保护启动 → 保护出口 → 断路器分闸"""
    return [
        ActionEvent(
            trace_id=trace_id, station_id="ST001", station_name="大理变",
            bay_id="BAY01", bay_name="甲线间隔",
            secondary_device_id="PD001", secondary_device_type="protection",
            signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_START.value,
            action_desc="距离I段保护启动",
            source_ts=_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.WARNING.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST001", station_name="大理变",
            bay_id="BAY01", bay_name="甲线间隔",
            secondary_device_id="PD001", secondary_device_type="protection",
            signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value,
            action_desc="距离I段保护出口",
            source_ts=_ts(0.02), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST001", station_name="大理变",
            bay_id="BAY01", bay_name="甲线间隔",
            primary_device_id="CB001", primary_device_type="breaker",
            signal_group=SignalGroup.BREAKER.value,
            action_type=ActionType.BREAKER_OPEN.value,
            action_desc="甲线断路器分闸",
            source_ts=_ts(0.06), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.ALARM.value,
        ),
    ]


def _make_refuse_action_events(trace_id: str = "trace_refuse_001") -> list:
    """拒动链: 保护出口但无断路器响应"""
    return [
        ActionEvent(
            trace_id=trace_id, station_id="ST001",
            bay_id="BAY02", bay_name="乙线间隔",
            secondary_device_id="PD002",
            signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_START.value,
            action_desc="纵联保护启动",
            source_ts=_ts(0), source_system=SourceSystem.PROTECTION_DEVICE.value,
        ),
        ActionEvent(
            trace_id=trace_id, station_id="ST001",
            bay_id="BAY02", bay_name="乙线间隔",
            secondary_device_id="PD002",
            signal_group=SignalGroup.PROTECTION.value,
            action_type=ActionType.PROTECTION_TRIP.value,
            action_desc="纵联保护出口",
            source_ts=_ts(0.03), source_system=SourceSystem.PROTECTION_DEVICE.value,
            severity_hint=SeverityHint.CRITICAL.value,
        ),
        # 无断路器分闸事件 → 拒动
    ]


def _make_jitter_events(trace_id: str = "trace_jitter_001") -> list:
    """信号抖动链: 同一信号在短时间内反复变位"""
    events = []
    for i in range(5):
        events.append(ActionEvent(
            trace_id=trace_id, station_id="ST001",
            bay_id="BAY03",
            signal_id="SIG_TEMP_001",
            signal_group=SignalGroup.TELEMETRY.value,
            action_type=ActionType.SIGNAL_CHANGE.value,
            action_desc="温度越限信号",
            value_before=str(i % 2), value_after=str((i + 1) % 2),
            source_ts=_ts(i * 0.5),
            source_system=SourceSystem.TERMINAL_BOX.value,
        ))
    return events


# =========================================================================
# 1. 正常动作链 → 候选事件生成
# =========================================================================

class TestNormalTripCandidate:
    def test_normal_trip_generates_candidate(self):
        """正常跳闸链应生成候选事件"""
        store = ActionEventStore()
        analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig())
        rcs = RootCauseService()

        events = _make_normal_trip_events()
        for e in events:
            store.add_event(e)

        chain_results = analyzer.analyze(events)
        assert len(chain_results) > 0
        assert chain_results[0].chain_type == ChainType.NORMAL_TRIP.value

        root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

        # 构建候选事件(模拟API逻辑)
        from apps.action_event_api import _build_candidate_from_analysis
        candidate = _build_candidate_from_analysis(
            trace_id=events[0].trace_id,
            events=events,
            chain_results=chain_results,
            root_cause_result=root_cause,
        )

        assert candidate.event_id.startswith("CE-")
        assert candidate.trace_id == events[0].trace_id
        assert candidate.event_type == ChainType.NORMAL_TRIP.value
        assert candidate.severity in (SeverityHint.ALARM.value, SeverityHint.CRITICAL.value)
        # 证据为partial(缺少一次设备巡视/录波等)时置信度会被降级到0.5
        assert candidate.confidence >= 0.5
        assert len(candidate.action_sequence) == 3
        assert candidate.sequence_completeness == SequenceCompleteness.COMPLETE.value
        assert candidate.has_refuse_action is False
        assert candidate.has_false_action is False
        assert len(candidate.root_cause_candidates) > 0
        assert candidate.historical_similarity is not None
        assert candidate.historical_similarity["status"] == "no_history_data"
        assert candidate.inference_basis != ""
        assert candidate.boundary_note != ""
        assert len(candidate.related_event_ids) == 3

    def test_candidate_stored_and_queryable(self):
        """候选事件可存储和查询"""
        store = ActionEventStore()
        candidate = CandidateEvent(
            trace_id="trace_store_001",
            station_id="ST001",
            event_type=ChainType.NORMAL_TRIP.value,
            severity=SeverityHint.ALARM.value,
            confidence=0.85,
        )
        eid = store.save_candidate_event(candidate)
        assert eid == candidate.event_id

        retrieved = store.get_candidate_event(eid)
        assert retrieved is not None
        assert retrieved.trace_id == "trace_store_001"

        by_trace = store.get_candidates_by_trace("trace_store_001")
        assert len(by_trace) == 1

        queried = store.query_candidate_events(station_id="ST001")
        assert len(queried) == 1


# =========================================================================
# 2. 拒动/误动/抖动/乱序/重复动作 → 异常标记
# =========================================================================

class TestAnomalyDetection:
    def test_refuse_action_detected(self):
        """拒动应被正确标记"""
        analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig())
        rcs = RootCauseService()
        events = _make_refuse_action_events()

        chain_results = analyzer.analyze(events)
        refuse_chains = [r for r in chain_results if r.chain_type == ChainType.REFUSE_ACTION.value]
        assert len(refuse_chains) > 0

        root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

        from apps.action_event_api import _build_candidate_from_analysis
        candidate = _build_candidate_from_analysis(
            trace_id=events[0].trace_id, events=events,
            chain_results=chain_results, root_cause_result=root_cause,
        )
        assert candidate.has_refuse_action is True
        assert candidate.event_type == ChainType.REFUSE_ACTION.value

    def test_jitter_detected(self):
        """信号抖动应被正确标记"""
        analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig())
        rcs = RootCauseService()
        events = _make_jitter_events()

        chain_results = analyzer.analyze(events)
        jitter_chains = [r for r in chain_results if r.chain_type == ChainType.SIGNAL_JITTER.value]
        assert len(jitter_chains) > 0

        root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

        from apps.action_event_api import _build_candidate_from_analysis
        candidate = _build_candidate_from_analysis(
            trace_id=events[0].trace_id, events=events,
            chain_results=chain_results, root_cause_result=root_cause,
        )
        assert candidate.has_jitter is True

    def test_repeated_action_flagged(self):
        """重复动作应被标记"""
        events = []
        for i in range(4):
            events.append(ActionEvent(
                trace_id="trace_repeat_001", station_id="ST001",
                bay_id="BAY01",
                signal_group=SignalGroup.BREAKER.value,
                action_type=ActionType.BREAKER_OPEN.value,
                action_desc="断路器分闸",
                source_ts=_ts(i * 0.1),
            ))

        analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig())
        rcs = RootCauseService()
        chain_results = analyzer.analyze(events)
        root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

        from apps.action_event_api import _build_candidate_from_analysis
        candidate = _build_candidate_from_analysis(
            trace_id="trace_repeat_001", events=events,
            chain_results=chain_results, root_cause_result=root_cause,
        )
        assert candidate.has_repeated_action is True


# =========================================================================
# 3. 关键证据缺失时的降级输出
# =========================================================================

class TestDegradedOutput:
    def test_insufficient_evidence_degrades(self):
        """证据不足时应降级并要求人工复核"""
        # 仅有一条告知信号，不足以形成任何链
        events = [
            ActionEvent(
                trace_id="trace_degrade_001",
                signal_group=SignalGroup.ALARM.value,
                action_type=ActionType.ALARM_SIGNAL.value,
                action_desc="告知信号",
                source_ts=_ts(0),
                source_system=SourceSystem.SIMULATED.value,
            ),
        ]
        analyzer = ActionSequenceAnalyzer(config=AnalyzerConfig())
        rcs = RootCauseService()
        chain_results = analyzer.analyze(events)
        root_cause = rcs.analyze(chain_results=chain_results, action_events=events)

        from apps.action_event_api import _build_candidate_from_analysis
        candidate = _build_candidate_from_analysis(
            trace_id="trace_degrade_001", events=events,
            chain_results=chain_results, root_cause_result=root_cause,
        )
        assert candidate.manual_review_status == ManualReviewStatus.REQUIRED.value
        assert candidate.confidence <= 0.5
        assert "证据" in candidate.boundary_note or "置信度" in candidate.boundary_note
        assert candidate.sequence_completeness in (
            SequenceCompleteness.INCOMPLETE.value,
            SequenceCompleteness.UNKNOWN.value,
        )

    def test_low_confidence_requires_review(self):
        """低置信度应要求人工复核"""
        candidate = CandidateEvent(
            confidence=0.3,
            manual_review_status=ManualReviewStatus.REQUIRED.value,
            boundary_note="置信度不足",
        )
        assert candidate.manual_review_status == ManualReviewStatus.REQUIRED.value


# =========================================================================
# 4. 人工复核状态流转
# =========================================================================

class TestReviewStatusFlow:
    def test_review_status_transitions(self):
        """复核状态应正确流转"""
        store = ActionEventStore()
        candidate = CandidateEvent(
            trace_id="trace_review_001",
            manual_review_status=ManualReviewStatus.PENDING.value,
        )
        store.save_candidate_event(candidate)

        # pending → confirmed
        updated = store.update_review_status(
            candidate.event_id, ManualReviewStatus.CONFIRMED.value,
            comment="确认为正常跳闸", reviewed_by="张三",
        )
        assert updated is not None
        assert updated.manual_review_status == ManualReviewStatus.CONFIRMED.value
        assert updated.review_comment == "确认为正常跳闸"
        assert updated.reviewed_by == "张三"
        assert updated.reviewed_at != ""

        # confirmed → rejected (允许修正)
        updated2 = store.update_review_status(
            candidate.event_id, ManualReviewStatus.REJECTED.value,
            comment="复核后驳回", reviewed_by="李四",
        )
        assert updated2.manual_review_status == ManualReviewStatus.REJECTED.value
        assert updated2.reviewed_by == "李四"

    def test_review_false_positive(self):
        """标记为误报"""
        store = ActionEventStore()
        candidate = CandidateEvent(
            trace_id="trace_fp_001",
            manual_review_status=ManualReviewStatus.REQUIRED.value,
        )
        store.save_candidate_event(candidate)

        updated = store.update_review_status(
            candidate.event_id, ManualReviewStatus.FALSE_POSITIVE.value,
            comment="误报，为测试信号", reviewed_by="operator",
        )
        assert updated.manual_review_status == ManualReviewStatus.FALSE_POSITIVE.value

    def test_review_nonexistent_event(self):
        """复核不存在的事件应返回None"""
        store = ActionEventStore()
        result = store.update_review_status("nonexistent_id", "confirmed")
        assert result is None

    def test_field_check_status(self):
        """标记为需现场复核"""
        store = ActionEventStore()
        candidate = CandidateEvent(trace_id="trace_fc_001")
        store.save_candidate_event(candidate)

        updated = store.update_review_status(
            candidate.event_id, ManualReviewStatus.FIELD_CHECK.value,
            comment="需派员现场检查断路器机构",
        )
        assert updated.manual_review_status == ManualReviewStatus.FIELD_CHECK.value


# =========================================================================
# 5. API返回结构稳定性(候选事件对象字段完整性)
# =========================================================================

class TestCandidateEventSchema:
    def test_candidate_event_required_fields(self):
        """候选事件必须包含所有P0要求字段"""
        candidate = CandidateEvent(
            trace_id="t1", device_id="d1", station_id="s1",
            start_time=_ts(0), end_time=_ts(1),
            event_type="normal_trip", severity="alarm",
            confidence=0.85,
        )
        d = candidate.to_dict()
        required_fields = [
            "event_id", "trace_id", "device_id", "station_id",
            "start_time", "end_time", "event_type", "severity",
            "confidence", "evidence", "root_cause_candidates",
            "manual_review_status", "review_comment", "boundary_note",
            "action_sequence", "sequence_completeness",
            "has_refuse_action", "has_false_action", "has_jitter",
            "has_repeated_action", "has_abnormal_sequence",
            "historical_similarity", "inference_basis",
        ]
        for field in required_fields:
            assert field in d, f"缺少必要字段: {field}"

    def test_candidate_to_dict_roundtrip(self):
        """to_dict应可序列化且字段值正确"""
        candidate = CandidateEvent(
            trace_id="t1", event_type="refuse_action",
            confidence=0.65, manual_review_status="required",
            boundary_note="证据不足",
            root_cause_candidates=[{"category": "primary_equipment_fault", "probability": 0.4, "reason": "一次设备故障"}],
        )
        d = candidate.to_dict()
        assert d["trace_id"] == "t1"
        assert d["event_type"] == "refuse_action"
        assert d["confidence"] == 0.65
        assert d["manual_review_status"] == "required"
        assert len(d["root_cause_candidates"]) == 1

    def test_statistics_includes_candidate_counts(self):
        """统计信息应包含候选事件和复核状态分布"""
        store = ActionEventStore()
        store.save_candidate_event(CandidateEvent(
            trace_id="t1", manual_review_status=ManualReviewStatus.PENDING.value))
        store.save_candidate_event(CandidateEvent(
            trace_id="t2", manual_review_status=ManualReviewStatus.CONFIRMED.value))
        store.save_candidate_event(CandidateEvent(
            trace_id="t3", manual_review_status=ManualReviewStatus.PENDING.value))

        stats = store.get_statistics()
        assert stats["total_candidates"] == 3
        assert stats["review_status_distribution"]["pending"] == 2
        assert stats["review_status_distribution"]["confirmed"] == 1


# =========================================================================
# 6. 跨插件查询接口契约
# =========================================================================

class TestCrossPluginInterface:
    def test_device_events_query_structure(self):
        """query_events_for_device 返回结构应符合契约"""
        store = ActionEventStore()
        events = _make_normal_trip_events()
        for e in events:
            store.add_event(e)

        result = store.query_events_for_device("PD001")
        assert "device_id" in result
        assert "events" in result
        assert "candidate_events" in result
        assert "data_status" in result
        assert "event_count" in result
        assert "candidate_count" in result
        assert "time_range" in result
        assert result["device_id"] == "PD001"
        assert result["data_status"] in ("sufficient", "partial", "no_data")

    def test_device_events_no_data(self):
        """无数据时应返回 no_data 状态"""
        store = ActionEventStore()
        result = store.query_events_for_device("NONEXISTENT")
        assert result["data_status"] == "no_data"
        assert result["event_count"] == 0

    def test_device_events_with_time_window(self):
        """时间窗口过滤应正常工作"""
        store = ActionEventStore()
        events = _make_normal_trip_events()
        for e in events:
            store.add_event(e)

        result = store.query_events_for_device(
            "PD001",
            start_time=_ts(-10),
            end_time=_ts(10),
        )
        assert result["event_count"] > 0

        result_empty = store.query_events_for_device(
            "PD001",
            start_time=_ts(100),
            end_time=_ts(200),
        )
        assert result_empty["event_count"] == 0

    def test_timeline_includes_candidates(self):
        """get_timeline 返回应包含候选事件"""
        store = ActionEventStore()
        events = _make_normal_trip_events()
        for e in events:
            store.add_event(e)

        # 保存候选事件
        candidate = CandidateEvent(
            trace_id=events[0].trace_id,
            event_type="normal_trip",
        )
        store.save_candidate_event(candidate)

        timeline = store.get_timeline(trace_id=events[0].trace_id)
        assert "candidate_events" in timeline
        assert events[0].trace_id in timeline["candidate_events"]
        assert "candidate_count" in timeline["summary"]
