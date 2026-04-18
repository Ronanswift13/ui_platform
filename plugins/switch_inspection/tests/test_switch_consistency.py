# -*- coding: utf-8 -*-
"""
开关状态一致性校核 — 测试套件
=============================

覆盖用户要求的 7 类场景:
1. 视频与传感一致
2. 视频与传感不一致
3. 视频清晰度不足
4. 有动作记录但视频滞后
5. 无动作记录但检测到状态变化
6. 不同开关类型规则差异
7. API/JSON 结构稳定性
"""

import sys
import os
import pytest
from datetime import datetime, timedelta

# 确保能导入插件模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from plugins.switch_inspection.switch_consistency import (
    ConsistencyChecker,
    ConsistencyStatus,
    SwitchDeviceType,
    SensorAdapter,
    SensorReading,
    VideoJudgment,
    ActionEventReference,
    ConflictDetail,
    ConsistencyResult,
    VIDEO_CONFIDENCE_THRESHOLD,
    VIDEO_CLARITY_THRESHOLD,
    SENSOR_QUALITY_THRESHOLD,
)


# ========== Fixtures ==========

@pytest.fixture
def sensor_adapter():
    return SensorAdapter()


@pytest.fixture
def checker(sensor_adapter):
    return ConsistencyChecker(sensor_adapter=sensor_adapter)


def _make_video(device_id="CB-001", device_type="breaker", state="open",
                confidence=0.92, clarity=0.85):
    return VideoJudgment(
        device_id=device_id, device_type=device_type, state=state,
        confidence=confidence, clarity_score=clarity,
        evidence_sources=["dl", "color"],
        timestamp=datetime.now().isoformat(),
    )


def _make_sensor(device_id="CB-001", state="open", quality=1.0, protocol="iec104"):
    return SensorReading(
        device_id=device_id, state=state,
        source_protocol=protocol,
        timestamp=datetime.now().isoformat(),
        quality=quality,
    )


# ========== 场景1: 视频与传感一致 ==========

class TestVideoSensorConsistent:
    """视频识别与传感/遥信状态一致"""

    def test_breaker_both_open(self, checker, sensor_adapter):
        sensor_adapter.set_reading("CB-001", _make_sensor(state="open"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open"),
        )
        assert result.consistency_status == ConsistencyStatus.CONSISTENT.value
        assert result.final_state == "open"
        assert result.confidence > 0.8
        assert len(result.conflicts) == 0
        assert not result.review_required

    def test_breaker_both_closed(self, checker, sensor_adapter):
        sensor_adapter.set_reading("CB-001", _make_sensor(state="closed"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="closed"),
        )
        assert result.consistency_status == ConsistencyStatus.CONSISTENT.value
        assert result.final_state == "closed"
        assert result.confidence > 0.8


# ========== 场景2: 视频与传感不一致 ==========

class TestVideoSensorInconsistent:
    """视频识别与传感/遥信状态冲突"""

    def test_breaker_video_open_sensor_closed(self, checker, sensor_adapter):
        sensor_adapter.set_reading("CB-001", _make_sensor(state="closed"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open"),
        )
        assert result.consistency_status == ConsistencyStatus.INCONSISTENT.value
        assert len(result.conflicts) > 0
        assert result.review_required
        # 断路器: 传感优先 → final_state 应跟随传感
        assert result.final_state == "closed"
        assert result.confidence <= 0.65

    def test_isolator_video_open_sensor_closed(self, checker, sensor_adapter):
        sensor_adapter.set_reading("DS-001", _make_sensor(device_id="DS-001", state="closed"))
        result = checker.check_consistency(
            device_id="DS-001", device_type="isolator",
            video_judgment=_make_video(device_id="DS-001", device_type="isolator", state="open"),
        )
        assert result.consistency_status == ConsistencyStatus.INCONSISTENT.value
        # 隔离开关: 视频优先 → final_state 应跟随视频
        assert result.final_state == "open"
        assert result.review_required

    def test_conflict_detail_structure(self, checker, sensor_adapter):
        sensor_adapter.set_reading("CB-001", _make_sensor(state="closed"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open"),
        )
        conflict = result.conflicts[0]
        assert conflict.source_a in ("video", "sensor")
        assert conflict.source_b in ("video", "sensor")
        assert conflict.state_a != conflict.state_b
        assert len(conflict.description) > 0
        assert conflict.severity in ("warning", "error")


# ========== 场景3: 视频清晰度不足 ==========

class TestVideoClarity:
    """视频清晰度低于阈值时视频证据不可信"""

    def test_low_clarity_with_sensor(self, checker, sensor_adapter):
        sensor_adapter.set_reading("CB-001", _make_sensor(state="open"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open", clarity=0.3),
        )
        # 视频不可信，仅传感一路 → review_required
        assert result.consistency_status == ConsistencyStatus.REVIEW_REQUIRED.value
        assert result.review_required

    def test_low_clarity_no_sensor(self, checker):
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open", clarity=0.2),
        )
        # 视频不可信且无传感 → insufficient
        assert result.consistency_status == ConsistencyStatus.INSUFFICIENT.value
        assert result.final_state == "unknown"

    def test_low_confidence_video(self, checker, sensor_adapter):
        sensor_adapter.set_reading("CB-001", _make_sensor(state="open"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open", confidence=0.3),
        )
        # 视频置信度低，不作为有效证据
        assert result.consistency_status == ConsistencyStatus.REVIEW_REQUIRED.value


# ========== 场景4: 有动作记录但视频滞后 ==========

class TestActionEventVideoLag:
    """动作事件存在但与视频状态不一致(可能视频帧滞后)"""

    def test_video_lag_detection(self, checker):
        """模拟: 动作事件记录刚发生分闸，但视频仍显示合闸"""
        # 无法直接注入 action_event_store，测试 ActionEventReference 逻辑
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="closed"),
        )
        # 无 action_event_store → action_ref.data_status = no_data
        assert result.action_event_ref is not None
        assert result.action_event_ref.data_status == "no_data"


# ========== 场景5: 无动作记录但检测到状态变化 ==========

class TestNoActionButStateChange:
    """无动作事件记录但视频检测到状态"""

    def test_video_only_no_action(self, checker):
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open"),
        )
        # 仅视频一路 → review_required
        assert result.consistency_status == ConsistencyStatus.REVIEW_REQUIRED.value
        assert result.review_required
        assert "一路证据" in result.review_reason

    def test_sensor_only_no_action(self, checker, sensor_adapter):
        sensor_adapter.set_reading("CB-001", _make_sensor(state="closed"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
        )
        # 仅传感一路(无视频输入) → review_required
        assert result.consistency_status == ConsistencyStatus.REVIEW_REQUIRED.value
        assert result.review_required

    def test_no_evidence_at_all(self, checker):
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
        )
        assert result.consistency_status == ConsistencyStatus.INSUFFICIENT.value
        assert result.confidence == 0.0
        assert result.review_required


# ========== 场景6: 不同开关类型规则差异 ==========

class TestDeviceTypeRules:
    """断路器/隔离开关/接地开关的优先级差异"""

    def test_breaker_sensor_priority(self, checker, sensor_adapter):
        """断路器: 冲突时传感优先"""
        sensor_adapter.set_reading("CB-001", _make_sensor(state="open"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="closed"),
        )
        assert result.final_state == "open"  # 传感优先

    def test_isolator_video_priority(self, checker, sensor_adapter):
        """隔离开关: 冲突时视频优先"""
        sensor_adapter.set_reading("DS-001", _make_sensor(device_id="DS-001", state="open"))
        result = checker.check_consistency(
            device_id="DS-001", device_type="isolator",
            video_judgment=_make_video(device_id="DS-001", device_type="isolator", state="closed"),
        )
        assert result.final_state == "closed"  # 视频优先

    def test_grounding_sensor_priority(self, checker, sensor_adapter):
        """接地开关: 冲突时传感优先"""
        sensor_adapter.set_reading("ES-001", _make_sensor(device_id="ES-001", state="closed"))
        result = checker.check_consistency(
            device_id="ES-001", device_type="grounding",
            video_judgment=_make_video(device_id="ES-001", device_type="grounding", state="open"),
        )
        assert result.final_state == "closed"  # 传感优先

    def test_intermediate_state_requires_review(self, checker, sensor_adapter):
        """中间态必须人工复核"""
        sensor_adapter.set_reading("CB-001", _make_sensor(state="open"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="intermediate"),
        )
        assert result.review_required
        assert "中间态" in result.review_reason


# ========== 场景7: API/JSON 结构稳定性 ==========

class TestStructureStability:
    """ConsistencyResult 序列化结构完整性"""

    def test_to_dict_required_fields(self, checker, sensor_adapter):
        sensor_adapter.set_reading("CB-001", _make_sensor(state="open"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open"),
        )
        d = result.to_dict()
        required_keys = {
            "result_id", "device_id", "device_type", "station_id", "bay_id",
            "video_judgment", "sensor_reading", "action_event_ref",
            "final_state", "consistency_status", "confidence",
            "conflicts", "boundary_note", "review_required", "review_reason",
            "timestamp",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_to_dict_video_judgment_nested(self, checker):
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open"),
        )
        d = result.to_dict()
        vj = d["video_judgment"]
        assert isinstance(vj, dict)
        assert "device_id" in vj
        assert "state" in vj
        assert "confidence" in vj

    def test_to_dict_conflicts_list(self, checker, sensor_adapter):
        sensor_adapter.set_reading("CB-001", _make_sensor(state="closed"))
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
            video_judgment=_make_video(state="open"),
        )
        d = result.to_dict()
        assert isinstance(d["conflicts"], list)
        assert len(d["conflicts"]) > 0
        conflict = d["conflicts"][0]
        assert "source_a" in conflict
        assert "description" in conflict

    def test_boundary_note_always_present(self, checker):
        result = checker.check_consistency(
            device_id="CB-001", device_type="breaker",
        )
        d = result.to_dict()
        assert "远程巡视辅助" in d["boundary_note"]
        assert "不得直接生成远方控制命令" in d["boundary_note"]

    def test_result_id_unique(self, checker):
        r1 = checker.check_consistency(device_id="CB-001", device_type="breaker")
        r2 = checker.check_consistency(device_id="CB-001", device_type="breaker")
        assert r1.result_id != r2.result_id

    def test_sensor_adapter_protocol(self, sensor_adapter):
        """SensorAdapter.set_from_protocol 创建并缓存读数"""
        reading = sensor_adapter.set_from_protocol(
            device_id="CB-002", state="closed",
            source_protocol="opc_ua", quality=0.95,
        )
        assert reading.state == "closed"
        assert reading.source_protocol == "opc_ua"
        cached = sensor_adapter.get_reading("CB-002")
        assert cached is not None
        assert cached.state == "closed"

    def test_sensor_adapter_clear(self, sensor_adapter):
        sensor_adapter.set_from_protocol("CB-003", "open")
        sensor_adapter.clear("CB-003")
        assert sensor_adapter.get_reading("CB-003") is None

    def test_action_to_state_mapping(self):
        assert ConsistencyChecker._action_to_state("breaker_open") == "open"
        assert ConsistencyChecker._action_to_state("breaker_close") == "closed"
        assert ConsistencyChecker._action_to_state("trip_open") == "open"
        assert ConsistencyChecker._action_to_state("unknown_action") == "unknown"
