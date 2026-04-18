#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件契约测试
============

验证 AnimalEvent 和相关数据结构符合统一事件契约。
"""

from core.event_schema import (
    AnimalClass,
    EventType,
    RiskLevel,
    BoundingBox,
    WorldPosition,
    EvidenceItem,
    AnimalDetectionResult,
    AnimalEvent,
    ANIMAL_RISK_MAP,
    SUGGESTION_MAP,
    build_intrusion_event,
)


class TestBoundingBox:
    """边界框测试"""

    def test_to_dict(self):
        bbox = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        d = bbox.to_dict()
        assert d == {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}

    def test_from_xyxy(self):
        bbox = BoundingBox.from_xyxy(100, 200, 300, 400, img_w=1000, img_h=1000)
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.width == 0.2
        assert bbox.height == 0.2


class TestWorldPosition:
    """世界坐标测试"""

    def test_to_dict(self):
        pos = WorldPosition(x=1.0, y=2.0, z=0.5)
        d = pos.to_dict()
        assert d["x"] == 1.0
        assert d["y"] == 2.0
        assert d["z"] == 0.5
        assert d["coordinate_system"] == "room_local"


class TestAnimalDetectionResult:
    """检测结果测试"""

    def test_default_values(self):
        result = AnimalDetectionResult()
        assert result.animal_class == AnimalClass.OTHER
        assert result.confidence == 0.0
        assert result.is_thermal_validated is False

    def test_to_dict_minimal(self):
        result = AnimalDetectionResult(
            animal_class="mouse",
            confidence=0.85,
        )
        d = result.to_dict()
        assert d["animal_class"] == "mouse"
        assert d["confidence"] == 0.85
        assert "bbox" not in d  # None 时不包含

    def test_to_dict_full(self):
        result = AnimalDetectionResult(
            animal_class="mouse",
            confidence=0.85,
            bbox=BoundingBox(x=0.1, y=0.2, width=0.1, height=0.1),
            world_position=WorldPosition(x=1.0, y=2.0),
            track_id="track_001",
            is_thermal_validated=True,
            thermal_diff_c=5.5,
            stay_duration_s=10.5,
        )
        d = result.to_dict()
        assert d["animal_class"] == "mouse"
        assert d["confidence"] == 0.85
        assert d["bbox"]["x"] == 0.1
        assert d["world_position"]["x"] == 1.0
        assert d["track_id"] == "track_001"
        assert d["is_thermal_validated"] is True
        assert d["thermal_diff_c"] == 5.5
        assert d["stay_duration_s"] == 10.5


class TestAnimalEvent:
    """事件契约测试"""

    def test_required_fields(self):
        """验证事件包含所有必需字段"""
        event = AnimalEvent()
        d = event.to_dict()

        # 统一契约核心字段
        assert "timestamp" in d
        assert "type" in d
        assert "location" in d
        assert "value" in d
        assert "confidence" in d
        assert "evidence" in d
        assert "suggestion" in d
        assert "trace_id" in d

        # 元数据字段
        assert "plugin_id" in d
        assert "plugin_version" in d
        assert "risk_level" in d

    def test_timestamp_is_float(self):
        """时间戳必须是浮点数"""
        event = AnimalEvent()
        assert isinstance(event.timestamp, float)
        assert event.timestamp > 0

    def test_trace_id_is_uuid(self):
        """trace_id 必须是有效 UUID 格式"""
        event = AnimalEvent()
        assert len(event.trace_id) == 36  # UUID 格式
        assert event.trace_id.count("-") == 4

    def test_plugin_id_correct(self):
        """插件 ID 必须正确"""
        event = AnimalEvent()
        assert event.plugin_id == "animal_detection"


class TestRiskMapping:
    """风险映射测试"""

    def test_mouse_is_high_risk(self):
        assert ANIMAL_RISK_MAP[AnimalClass.MOUSE] == RiskLevel.HIGH

    def test_snake_is_critical(self):
        assert ANIMAL_RISK_MAP[AnimalClass.SNAKE] == RiskLevel.CRITICAL

    def test_bird_is_low_risk(self):
        assert ANIMAL_RISK_MAP[AnimalClass.BIRD] == RiskLevel.LOW

    def test_all_classes_have_risk(self):
        """所有动物类型都必须有风险映射"""
        for animal_class in AnimalClass:
            assert animal_class in ANIMAL_RISK_MAP


class TestSuggestionMapping:
    """处置建议映射测试"""

    def test_all_classes_have_suggestion(self):
        """所有动物类型都必须有处置建议"""
        for animal_class in AnimalClass:
            assert animal_class in SUGGESTION_MAP
            assert len(SUGGESTION_MAP[animal_class]) > 0


class TestBuildIntrusionEvent:
    """事件构建函数测试"""

    def test_empty_detections_returns_cleared(self):
        """空检测列表返回 CLEARED 事件"""
        event = build_intrusion_event(
            detections=[],
            camera_id="cam_001",
            location="GIS室-A区",
        )
        assert event.type == EventType.INTRUSION_CLEARED
        assert event.value["status"] == "clear"
        assert event.value["count"] == 0

    def test_single_detection_returns_detected(self):
        """单个检测返回 DETECTED 事件"""
        detection = AnimalDetectionResult(
            animal_class="mouse",
            confidence=0.85,
        )
        event = build_intrusion_event(
            detections=[detection],
            camera_id="cam_001",
            location="GIS室-A区",
        )
        assert event.type == EventType.INTRUSION_DETECTED
        assert event.value["status"] == "intrusion"
        assert event.value["count"] == 1
        assert "mouse" in event.value["classes"]

    def test_highest_confidence_used(self):
        """使用最高置信度"""
        detections = [
            AnimalDetectionResult(animal_class="mouse", confidence=0.7),
            AnimalDetectionResult(animal_class="mouse", confidence=0.9),
            AnimalDetectionResult(animal_class="mouse", confidence=0.8),
        ]
        event = build_intrusion_event(
            detections=detections,
            camera_id="cam_001",
            location="GIS室-A区",
        )
        assert event.confidence == 0.9

    def test_highest_risk_used(self):
        """使用最高风险等级"""
        detections = [
            AnimalDetectionResult(animal_class="bird", confidence=0.9),  # LOW
            AnimalDetectionResult(animal_class="snake", confidence=0.7),  # CRITICAL
        ]
        event = build_intrusion_event(
            detections=detections,
            camera_id="cam_001",
            location="GIS室-A区",
        )
        assert event.risk_level == RiskLevel.CRITICAL

    def test_suggestions_aggregated(self):
        """处置建议聚合"""
        detections = [
            AnimalDetectionResult(animal_class="mouse", confidence=0.8),
            AnimalDetectionResult(animal_class="snake", confidence=0.7),
        ]
        event = build_intrusion_event(
            detections=detections,
            camera_id="cam_001",
            location="GIS室-A区",
        )
        # 建议应包含两种动物的处置方案
        assert "超声波" in event.suggestion or "声光" in event.suggestion

    def test_evidence_attached(self):
        """证据链附加"""
        evidence = [
            EvidenceItem(
                type="raw_image",
                path="/evidence/img_001.jpg",
                description="原始检测图像",
            )
        ]
        event = build_intrusion_event(
            detections=[AnimalDetectionResult(animal_class="mouse", confidence=0.8)],
            camera_id="cam_001",
            location="GIS室-A区",
            evidence=evidence,
        )
        assert len(event.evidence) == 1
        assert event.evidence[0].type == "raw_image"


class TestEventSerialization:
    """事件序列化测试"""

    def test_to_dict_is_json_serializable(self):
        """to_dict 结果必须可 JSON 序列化"""
        import json

        detection = AnimalDetectionResult(
            animal_class="mouse",
            confidence=0.85,
            bbox=BoundingBox(x=0.1, y=0.2, width=0.1, height=0.1),
        )
        event = build_intrusion_event(
            detections=[detection],
            camera_id="cam_001",
            location="GIS室-A区",
        )

        # 不应抛出异常
        json_str = json.dumps(event.to_dict())
        assert len(json_str) > 0

        # 反序列化后数据完整
        parsed = json.loads(json_str)
        assert parsed["type"] == EventType.INTRUSION_DETECTED
        assert parsed["detections"][0]["animal_class"] == "mouse"
