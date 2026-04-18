#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - 单元测试 & 集成测试
=========================================

覆盖:
- 事件契约校验
- 检测器输出格式
- 热成像校验逻辑
- 跟踪器关联与停留统计
- 驱离控制器策略
- 统计模块聚合
- 插件完整流水线

运行: pytest tests/test_plugin.py -v

作者: G组 | 版本: 1.0.0
"""

import json
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock  # noqa: F401


import numpy as np
import pytest

# 确保路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from plugins.animal_detection.core.event_schema import (
    AnimalClass, EventType, RiskLevel, ANIMAL_RISK_MAP, SUGGESTION_MAP,
    BoundingBox, AnimalDetectionResult, AnimalEvent, build_intrusion_event,
    WorldPosition,
)
from plugins.animal_detection.core.tracker import AnimalTracker, AnimalTrack
from plugins.animal_detection.core.thermal_validator import ThermalValidator
from plugins.animal_detection.core.deterrent import (
    DeterrentController, DeterrentResult, DETERRENT_STRATEGY,
)
from plugins.animal_detection.core.statistics import StatisticsCollector


# =============================================================================
# 1. 事件契约测试
# =============================================================================

class TestEventSchema:
    """测试统一事件契约"""

    def test_animal_event_has_required_fields(self):
        """事件必须包含统一契约核心字段"""
        event = AnimalEvent()
        d = event.to_dict()
        required = [
            "timestamp", "type", "location", "value", "confidence",
            "evidence", "suggestion", "trace_id",
        ]
        for field in required:
            assert field in d, f"缺少必要字段: {field}"

    def test_trace_id_is_uuid(self):
        """trace_id 必须是有效UUID"""
        event = AnimalEvent()
        uuid.UUID(event.trace_id)  # 如果无效会抛异常

    def test_detection_result_serialization(self):
        """检测结果可正确序列化"""
        det = AnimalDetectionResult(
            animal_class=AnimalClass.MOUSE,
            confidence=0.85,
            bbox=BoundingBox(0.1, 0.2, 0.3, 0.4),
            track_id="AT-000001",
            is_thermal_validated=True,
            thermal_diff_c=3.5,
            stay_duration_s=15.0,
        )
        d = det.to_dict()
        assert d["animal_class"] == "mouse"
        assert d["confidence"] == 0.85
        assert d["bbox"]["x"] == 0.1
        assert d["track_id"] == "AT-000001"
        assert d["is_thermal_validated"] is True
        assert d["thermal_diff_c"] == 3.5

    def test_build_intrusion_event(self):
        """构建入侵事件应正确聚合信息"""
        dets = [
            AnimalDetectionResult(animal_class=AnimalClass.MOUSE, confidence=0.9,
                                  bbox=BoundingBox(0.1, 0.2, 0.3, 0.4)),
            AnimalDetectionResult(animal_class=AnimalClass.SNAKE, confidence=0.7,
                                  bbox=BoundingBox(0.5, 0.6, 0.1, 0.1)),
        ]
        event = build_intrusion_event(dets, "cam-01", "GIS室-A区", "site-001")

        assert event.type == EventType.INTRUSION_DETECTED
        assert event.total_detections == 2
        assert event.risk_level == RiskLevel.CRITICAL  # 蛇是CRITICAL
        assert event.confidence == 0.9  # 取最高
        suggestion = event.suggestion.lower()
        assert "蛇" in event.suggestion or "snake" in suggestion or "短路" in event.suggestion

    def test_build_empty_event(self):
        """无检测时应返回清除事件"""
        event = build_intrusion_event([], "cam-01", "GIS室")
        assert event.type == EventType.INTRUSION_CLEARED
        assert event.total_detections == 0

    def test_risk_mapping_complete(self):
        """所有动物类别都有风险映射"""
        for cls in AnimalClass:
            assert cls.value in ANIMAL_RISK_MAP

    def test_suggestion_mapping_complete(self):
        """所有动物类别都有处置建议"""
        for cls in AnimalClass:
            assert cls.value in SUGGESTION_MAP


# =============================================================================
# 2. 热成像校验测试
# =============================================================================

class TestThermalValidator:
    """测试热成像校验器"""

    def test_no_thermal_data_passes_all(self):
        """无热成像数据时应全部通过"""
        validator = ThermalValidator(enabled=True)
        dets = [
            AnimalDetectionResult(confidence=0.8, bbox=BoundingBox(0.1, 0.1, 0.2, 0.2)),
        ]
        result = validator.validate_detections(dets, thermal_frame=None)
        assert len(result) == 1
        assert result[0].is_thermal_validated is False
        assert result[0].thermal_diff_c is None

    def test_warm_object_passes(self):
        """温差显著的目标应通过校验"""
        validator = ThermalValidator(enabled=True, heat_diff_threshold_c=2.0)

        # 创建温度图: 背景25°C, 目标区域30°C
        thermal = np.full((100, 100), 25.0, dtype=np.float32)
        thermal[10:30, 10:30] = 30.0  # 目标区域+5°C

        dets = [
            AnimalDetectionResult(
                confidence=0.8,
                bbox=BoundingBox(0.10, 0.10, 0.20, 0.20),  # 对应 [10:30, 10:30]
            ),
        ]
        result = validator.validate_detections(dets, thermal)
        assert result[0].is_thermal_validated is True
        assert result[0].thermal_diff_c > 2.0

    def test_cold_object_rejected(self):
        """温差不显著的目标应被标记"""
        validator = ThermalValidator(enabled=True, heat_diff_threshold_c=2.0)

        # 均匀温度图
        thermal = np.full((100, 100), 25.0, dtype=np.float32)

        dets = [
            AnimalDetectionResult(
                confidence=0.8,
                bbox=BoundingBox(0.10, 0.10, 0.20, 0.20),
            ),
        ]
        result = validator.validate_detections(dets, thermal)
        assert result[0].is_thermal_validated is False

    def test_filter_removes_non_thermal(self):
        """过滤器应移除未通过热成像校验的检测"""
        validator = ThermalValidator(enabled=True, heat_diff_threshold_c=2.0)

        thermal = np.full((100, 100), 25.0, dtype=np.float32)
        thermal[10:30, 10:30] = 30.0  # 一个热目标

        dets = [
            AnimalDetectionResult(confidence=0.8, bbox=BoundingBox(0.10, 0.10, 0.20, 0.20)),
            AnimalDetectionResult(confidence=0.6, bbox=BoundingBox(0.50, 0.50, 0.10, 0.10)),
        ]
        validated = validator.validate_detections(dets, thermal)
        filtered = validator.filter_non_thermal(validated)
        assert len(filtered) == 1

    def test_disabled_validator(self):
        """禁用时应不做任何过滤"""
        validator = ThermalValidator(enabled=False)
        dets = [AnimalDetectionResult(confidence=0.5, bbox=BoundingBox(0.1, 0.1, 0.2, 0.2))]
        result = validator.validate_detections(dets, np.zeros((100, 100), dtype=np.float32))
        assert len(result) == 1


# =============================================================================
# 3. 跟踪器测试
# =============================================================================

class TestAnimalTracker:
    """测试多目标跟踪器"""

    def test_create_new_tracks(self):
        """新检测应创建新轨迹"""
        tracker = AnimalTracker(min_hits=1)
        dets = [
            AnimalDetectionResult(
                animal_class=AnimalClass.MOUSE,
                confidence=0.8,
                bbox=BoundingBox(0.1, 0.2, 0.1, 0.1),
            ),
        ]
        results = tracker.update(dets)
        assert len(results) == 1
        assert results[0].track_id is not None
        assert results[0].track_id.startswith("AT-")

    def test_track_continuity(self):
        """重叠检测应关联到同一轨迹"""
        tracker = AnimalTracker(min_hits=1, iou_threshold=0.2)

        # 第1帧
        det1 = [AnimalDetectionResult(
            animal_class=AnimalClass.CAT,
            confidence=0.9,
            bbox=BoundingBox(0.5, 0.5, 0.1, 0.1),
        )]
        r1 = tracker.update(det1)
        tid1 = r1[0].track_id

        # 第2帧 - 稍微移动
        det2 = [AnimalDetectionResult(
            animal_class=AnimalClass.CAT,
            confidence=0.85,
            bbox=BoundingBox(0.52, 0.52, 0.1, 0.1),
        )]
        r2 = tracker.update(det2)
        tid2 = r2[0].track_id

        assert tid1 == tid2, "相同目标应保持同一track_id"

    def test_stay_duration_accumulates(self):
        """停留时间应随帧累积"""
        tracker = AnimalTracker(min_hits=1)

        det = AnimalDetectionResult(
            animal_class=AnimalClass.BIRD,
            confidence=0.7,
            bbox=BoundingBox(0.3, 0.3, 0.1, 0.1),
        )
        tracker.update([det])
        time.sleep(0.1)
        results = tracker.update([det])
        assert results[0].stay_duration_s > 0

    def test_overstayed_detection(self):
        """超时轨迹应被检测到"""
        tracker = AnimalTracker(min_hits=1)

        det = AnimalDetectionResult(
            confidence=0.8, bbox=BoundingBox(0.5, 0.5, 0.1, 0.1),
            animal_class=AnimalClass.MOUSE,
        )
        tracker.update([det])

        # 手动设置first_seen以模拟超时
        for t in tracker.get_all_tracks():
            t.first_seen = time.time() - 60
            t.update_stay()

        overstayed = tracker.get_overstayed_tracks(30.0)
        assert len(overstayed) >= 1

    def test_track_deletion(self):
        """长时间未检测到的轨迹应被删除"""
        tracker = AnimalTracker(max_age=3, min_hits=1)

        det = AnimalDetectionResult(
            confidence=0.8, bbox=BoundingBox(0.5, 0.5, 0.1, 0.1),
        )
        tracker.update([det])

        # 空帧更新超过max_age
        for _ in range(5):
            tracker.update([])

        assert len(tracker.get_all_tracks()) == 0


# =============================================================================
# 4. 驱离控制器测试
# =============================================================================

class TestDeterrentController:
    """测试驱离控制器"""

    def _make_track(self, animal_class=AnimalClass.MOUSE, stay=60.0) -> AnimalTrack:
        track = AnimalTrack(
            track_id="AT-000001",
            animal_class=animal_class,
            bbox=BoundingBox(0.5, 0.5, 0.1, 0.1),
            confidence=0.8,
        )
        track.first_seen = time.time() - stay
        track.update_stay()
        track.state = "confirmed"
        return track

    def test_trigger_on_overstayed(self):
        """超时轨迹应触发驱离"""
        controller = DeterrentController(
            enabled=True,
            stay_time_threshold_s=30.0,
            cooldown_s=5.0,
        )
        track = self._make_track(stay=60.0)
        actions = controller.evaluate_and_trigger([track])
        assert len(actions) == 1
        assert actions[0].track_id == "AT-000001"

    def test_cooldown_prevents_retrigger(self):
        """冷却时间内不应重复触发"""
        controller = DeterrentController(
            enabled=True,
            stay_time_threshold_s=30.0,
            cooldown_s=60.0,
        )
        track = self._make_track()
        controller.evaluate_and_trigger([track])
        actions2 = controller.evaluate_and_trigger([track])
        assert len(actions2) == 0

    def test_disabled_does_nothing(self):
        """禁用时不触发任何动作"""
        controller = DeterrentController(enabled=False)
        track = self._make_track()
        actions = controller.evaluate_and_trigger([track])
        assert len(actions) == 0

    def test_escalation_after_max_retries(self):
        """超过最大重试次数应升级"""
        controller = DeterrentController(
            enabled=True,
            cooldown_s=0,
            escalation_threshold=2,
        )
        track = self._make_track()

        controller.evaluate_and_trigger([track])
        controller.evaluate_and_trigger([track])
        controller.evaluate_and_trigger([track])  # 第3次应升级

        escalated = [a for a in controller.get_recent_actions()
                     if a["result"] == DeterrentResult.ESCALATED]
        assert len(escalated) >= 1

    def test_strategy_by_animal_class(self):
        """不同动物应使用不同驱离策略"""
        assert AnimalClass.MOUSE in DETERRENT_STRATEGY
        assert AnimalClass.SNAKE in DETERRENT_STRATEGY
        assert DETERRENT_STRATEGY[AnimalClass.MOUSE]["frequency_hz"] == 20000
        assert DETERRENT_STRATEGY[AnimalClass.SNAKE]["light_flash"] is True


# =============================================================================
# 5. 统计模块测试
# =============================================================================

class TestStatisticsCollector:
    """测试统计收集器"""

    def test_record_and_count(self):
        """记录入侵并统计次数"""
        stats = StatisticsCollector()
        det = AnimalDetectionResult(
            animal_class=AnimalClass.MOUSE, confidence=0.8,
        )
        stats.record_invasion(det, camera_id="cam-01", location="GIS室")
        assert stats.get_today_count() == 1

    def test_daily_summary(self):
        """日报聚合应正确"""
        stats = StatisticsCollector()
        for cls in [AnimalClass.MOUSE, AnimalClass.MOUSE, AnimalClass.CAT]:
            det = AnimalDetectionResult(animal_class=cls, confidence=0.8)
            stats.record_invasion(det)

        summary = stats.get_daily_summary()
        assert summary["total_invasions"] == 3
        assert summary["class_distribution"]["mouse"] == 2
        assert summary["class_distribution"]["cat"] == 1

    def test_export_csv(self):
        """CSV导出应包含表头和数据"""
        stats = StatisticsCollector()
        det = AnimalDetectionResult(animal_class=AnimalClass.BIRD, confidence=0.7)
        stats.record_invasion(det)

        csv_data = stats.export_csv()
        assert "record_id" in csv_data
        assert "bird" in csv_data

    def test_export_json(self):
        """JSON导出应可解析"""
        stats = StatisticsCollector()
        det = AnimalDetectionResult(animal_class=AnimalClass.SNAKE, confidence=0.6)
        stats.record_invasion(det)

        json_data = stats.export_json()
        parsed = json.loads(json_data)
        assert parsed["count"] == 1
        assert parsed["records"][0]["animal_class"] == "snake"

    def test_trend_data(self):
        """趋势数据应包含日期序列"""
        stats = StatisticsCollector()
        det = AnimalDetectionResult(animal_class=AnimalClass.MOUSE, confidence=0.8)
        stats.record_invasion(det)

        trend = stats.get_trend_data(days=7)
        assert len(trend["daily_counts"]) == 7
        assert trend["total"] >= 1


# =============================================================================
# 6. 插件集成测试
# =============================================================================

class TestPluginIntegration:
    """插件完整流水线集成测试"""

    def test_plugin_init_simulation_mode(self):
        """仿真模式初始化应成功"""
        from plugins.animal_detection.plugin import AnimalDetectionPlugin

        plugin = AnimalDetectionPlugin()
        success = plugin.init({
            "simulation": {"enabled": True},
            "tracking": {"enabled": True},
            "deterrent": {"enabled": False},
        })
        assert success is True
        assert plugin._simulation_mode is True
        plugin.cleanup()

    def test_plugin_infer_simulation_returns_clear(self):
        """仿真模式推理应返回清除事件（不产生假数据）"""
        from plugins.animal_detection.plugin import AnimalDetectionPlugin

        plugin = AnimalDetectionPlugin()
        plugin.init({"simulation": {"enabled": True}})

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        event = plugin.infer(frame, camera_id="test-cam", location="测试室")

        assert event.type == EventType.INTRUSION_CLEARED
        assert event.total_detections == 0
        plugin.cleanup()

    def test_plugin_healthcheck(self):
        """健康检查应返回完整状态"""
        from plugins.animal_detection.plugin import AnimalDetectionPlugin

        plugin = AnimalDetectionPlugin()
        plugin.init({"simulation": {"enabled": True}})

        health = plugin.healthcheck()
        assert "status" in health
        assert "detector" in health
        assert "tracker" in health
        assert "simulation_mode" in health
        assert health["simulation_mode"] is True
        plugin.cleanup()

    def test_plugin_missing_model_degraded_without_simulation(self, tmp_path):
        """非仿真且模型缺失时应降级运行，不生成假检测"""
        from plugins.animal_detection.plugin import AnimalDetectionPlugin

        missing_model = tmp_path / "missing.onnx"
        plugin = AnimalDetectionPlugin(
            config={
                "model": {"path": str(missing_model)},
                "simulation": {"enabled": False},
                "deterrent": {"enabled": False},
            }
        )
        success = plugin.init({})
        assert success is True

        health = plugin.healthcheck()
        assert health["status"] == "degraded"
        assert health["model_ready"] is False
        assert health["simulation_mode"] is False

        payload = plugin.get_detections()
        assert payload["status"] == "degraded"
        assert payload["detections"] == []
        assert payload["simulation_mode"] is False
        plugin.cleanup()

    def test_plugin_statistics_api(self):
        """统计API应正常工作"""
        from plugins.animal_detection.plugin import AnimalDetectionPlugin

        plugin = AnimalDetectionPlugin()
        plugin.init({"simulation": {"enabled": True}})

        stats = plugin.get_statistics("daily")
        assert "total_invasions" in stats

        stats_weekly = plugin.get_statistics("weekly")
        assert "total_invasions" in stats_weekly

        plugin.cleanup()

    def test_plugin_ui_config(self):
        """UI配置应包含必要信息"""
        from plugins.animal_detection.plugin import AnimalDetectionPlugin

        plugin = AnimalDetectionPlugin()
        plugin.init({"simulation": {"enabled": True}})

        ui = plugin.get_ui_config()
        assert "detection_types" in ui
        assert "animal_classes" in ui
        assert len(ui["animal_classes"]) == len(AnimalClass)
        plugin.cleanup()


# =============================================================================
# 7. 边界条件与异常测试
# =============================================================================

class TestEdgeCases:
    """边界条件测试"""

    def test_empty_frame(self):
        """空帧不应崩溃"""
        from plugins.animal_detection.plugin import AnimalDetectionPlugin
        plugin = AnimalDetectionPlugin()
        plugin.init({"simulation": {"enabled": True}})
        frame = np.zeros((0, 0, 3), dtype=np.uint8)
        event = plugin.infer(frame)
        assert event is not None
        plugin.cleanup()

    def test_bbox_from_xyxy(self):
        """边界框坐标转换"""
        bb = BoundingBox.from_xyxy(100, 200, 300, 400, 640, 480)
        assert 0 <= bb.x <= 1
        assert 0 <= bb.y <= 1
        assert 0 < bb.width <= 1
        assert 0 < bb.height <= 1

    def test_world_position(self):
        """世界坐标序列化"""
        wp = WorldPosition(1.5, 2.3, 0.0, "room_local")
        d = wp.to_dict()
        assert d["x"] == 1.5
        assert d["coordinate_system"] == "room_local"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
