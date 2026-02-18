"""
Tests for darkbreaker_sdk.schemas

Validates all Pydantic models: BoundingBox, RecognitionResult, Alarm,
AlarmLevel, AlarmRule, AlarmStatus, BaseEntity, ROI, ROIType, Evidence,
EvidenceType, generate_id, PluginOutput.
"""

import pytest
from datetime import datetime

from darkbreaker_sdk.schemas.common import (
    BaseEntity,
    Evidence,
    EvidenceType,
    ROI,
    ROIType,
    generate_id,
)
from darkbreaker_sdk.schemas.detection import BoundingBox, RecognitionResult
from darkbreaker_sdk.schemas.alarm import Alarm, AlarmLevel, AlarmRule, AlarmStatus
from darkbreaker_sdk.schemas.plugin_io import PluginOutput


class TestGenerateId:
    """Tests for generate_id function."""

    def test_returns_string(self):
        assert isinstance(generate_id(), str)

    def test_unique(self):
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_valid_bbox(self):
        bb = BoundingBox(x=0.1, y=0.2, width=0.5, height=0.3)
        assert bb.x == 0.1
        assert bb.y == 0.2
        assert bb.width == 0.5
        assert bb.height == 0.3

    def test_boundary_values(self):
        bb = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        assert bb.x == 0.0
        assert bb.height == 1.0

    def test_x_out_of_range(self):
        with pytest.raises(Exception):
            BoundingBox(x=-0.1, y=0.0, width=0.5, height=0.5)

    def test_y_out_of_range(self):
        with pytest.raises(Exception):
            BoundingBox(x=0.0, y=1.1, width=0.5, height=0.5)

    def test_width_out_of_range(self):
        with pytest.raises(Exception):
            BoundingBox(x=0.0, y=0.0, width=2.0, height=0.5)

    def test_height_negative(self):
        with pytest.raises(Exception):
            BoundingBox(x=0.0, y=0.0, width=0.5, height=-0.1)

    def test_serialization(self):
        bb = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        data = bb.model_dump()
        assert data == {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}

    def test_deserialization(self):
        data = {"x": 0.5, "y": 0.5, "width": 0.25, "height": 0.25}
        bb = BoundingBox.model_validate(data)
        assert bb.x == 0.5


class TestRecognitionResult:
    """Tests for RecognitionResult model."""

    def _make_result(self, **overrides):
        defaults = {
            "task_id": "task-1",
            "site_id": "site-1",
            "device_id": "dev-1",
            "component_id": "comp-1",
            "roi_id": "roi-1",
            "bbox": BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
            "label": "defect",
            "confidence": 0.95,
        }
        defaults.update(overrides)
        return RecognitionResult(**defaults)

    def test_valid_result(self):
        r = self._make_result()
        assert r.label == "defect"
        assert r.confidence == 0.95
        assert isinstance(r.timestamp, datetime)

    def test_confidence_bounds(self):
        # Valid at boundaries
        self._make_result(confidence=0.0)
        self._make_result(confidence=1.0)

        # Invalid
        with pytest.raises(Exception):
            self._make_result(confidence=1.1)

        with pytest.raises(Exception):
            self._make_result(confidence=-0.1)

    def test_optional_fields(self):
        r = self._make_result(value=42.5, failure_reason="low_contrast")
        assert r.value == 42.5
        assert r.failure_reason == "low_contrast"

    def test_serialization_roundtrip(self):
        r = self._make_result()
        data = r.model_dump(mode="json")
        r2 = RecognitionResult.model_validate(data)
        assert r2.label == r.label
        assert r2.confidence == r.confidence


class TestAlarmModels:
    """Tests for Alarm, AlarmLevel, AlarmRule, AlarmStatus."""

    def test_alarm_level_values(self):
        assert AlarmLevel.INFO == "info"
        assert AlarmLevel.WARNING == "warning"
        assert AlarmLevel.ERROR == "error"
        assert AlarmLevel.CRITICAL == "critical"

    def test_alarm_status_values(self):
        expected = {"active", "acknowledged", "resolved", "false_positive"}
        actual = {s.value for s in AlarmStatus}
        assert actual == expected

    def test_alarm_rule(self):
        rule = AlarmRule(name="temp_high", condition="value > 80")
        assert rule.name == "temp_high"
        assert rule.condition == "value > 80"
        assert rule.level == "warning"
        assert rule.enabled is True
        assert isinstance(rule.id, str)
        assert len(rule.id) > 0

    def test_alarm(self):
        alarm = Alarm(
            task_id="task-1",
            title="High Temperature",
            message="Temperature exceeded threshold",
            site_id="site-1",
            device_id="dev-1",
            level=AlarmLevel.CRITICAL,
        )
        assert alarm.title == "High Temperature"
        assert alarm.level == AlarmLevel.CRITICAL
        assert alarm.status == AlarmStatus.ACTIVE
        assert isinstance(alarm.id, str)
        assert isinstance(alarm.created_at, datetime)

    def test_alarm_serialization(self):
        alarm = Alarm(
            task_id="task-1",
            title="Test",
            message="Test alarm",
            site_id="site-1",
            device_id="dev-1",
        )
        data = alarm.model_dump(mode="json")
        assert data["level"] == "warning"
        assert data["status"] == "active"

        alarm2 = Alarm.model_validate(data)
        assert alarm2.title == "Test"


class TestBaseEntity:
    """Tests for BaseEntity model."""

    def test_defaults(self):
        e = BaseEntity(name="test")
        assert isinstance(e.id, str)
        assert len(e.id) > 0
        assert e.name == "test"
        assert e.description == ""
        assert isinstance(e.created_at, datetime)
        assert isinstance(e.updated_at, datetime)
        assert e.metadata == {}

    def test_with_all_fields(self):
        e = BaseEntity(
            id="custom-id",
            name="entity",
            description="A test entity",
            metadata={"key": "value"},
        )
        assert e.id == "custom-id"
        assert e.description == "A test entity"


class TestROI:
    """Tests for ROI and ROIType."""

    def test_roi_type_values(self):
        expected = {"defect", "state", "meter", "thermal", "intrusion"}
        actual = {t.value for t in ROIType}
        assert actual == expected

    def test_roi_creation(self):
        roi = ROI(
            name="Test ROI",
            component_id="comp-1",
            roi_type=ROIType.DEFECT,
            bbox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
        )
        assert roi.name == "Test ROI"
        assert roi.roi_type == ROIType.DEFECT
        assert roi.bbox.x == 0.1
        assert roi.rules == []
        assert roi.recognition_types == []

    def test_roi_with_rules(self):
        rule = AlarmRule(name="temp", condition="v > 80")
        roi = ROI(
            name="ROI with rules",
            component_id="comp-1",
            roi_type=ROIType.THERMAL,
            bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            rules=[rule],
        )
        assert len(roi.rules) == 1
        assert roi.rules[0].name == "temp"


class TestEvidence:
    """Tests for Evidence and EvidenceType."""

    def test_evidence_type_values(self):
        expected = {
            "raw_image",
            "annotated_image",
            "video_clip",
            "thermal_image",
            "log",
            "result_json",
        }
        actual = {t.value for t in EvidenceType}
        assert actual == expected

    def test_evidence_creation(self):
        ev = Evidence(
            run_id="run-1",
            task_id="task-1",
            evidence_type=EvidenceType.RAW_IMAGE,
            file_path="/tmp/img.png",
        )
        assert ev.run_id == "run-1"
        assert ev.evidence_type == EvidenceType.RAW_IMAGE
        assert ev.file_size == 0
        assert isinstance(ev.id, str)


class TestPluginOutput:
    """Tests for PluginOutput model."""

    def test_minimal_output(self):
        out = PluginOutput(
            task_id="task-1",
            plugin_id="plugin-test",
            plugin_version="1.0.0",
            code_hash="abc123",
        )
        assert out.success is True
        assert out.results == []
        assert out.alarms == []
        assert out.processing_time_ms == 0

    def test_output_with_results(self):
        result = RecognitionResult(
            task_id="task-1",
            site_id="site-1",
            device_id="dev-1",
            component_id="comp-1",
            roi_id="roi-1",
            bbox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
            label="crack",
            confidence=0.88,
        )
        alarm = Alarm(
            task_id="task-1",
            title="Crack detected",
            message="Crack found on insulator",
            site_id="site-1",
            device_id="dev-1",
            level=AlarmLevel.WARNING,
        )
        out = PluginOutput(
            task_id="task-1",
            plugin_id="defect-plugin",
            plugin_version="2.1.0",
            code_hash="def456",
            results=[result],
            alarms=[alarm],
            processing_time_ms=45.2,
        )
        assert len(out.results) == 1
        assert len(out.alarms) == 1
        assert out.processing_time_ms == 45.2

    def test_serialization_roundtrip(self):
        out = PluginOutput(
            task_id="task-1",
            plugin_id="test",
            plugin_version="1.0.0",
            code_hash="abc",
            success=False,
            error_message="Model failed",
        )
        data = out.model_dump(mode="json")
        out2 = PluginOutput.model_validate(data)
        assert out2.success is False
        assert out2.error_message == "Model failed"
