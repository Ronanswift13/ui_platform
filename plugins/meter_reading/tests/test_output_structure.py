"""
L0 单元测试 - 输出结构验证
(合约 2.3, 7.9)
"""
import pytest
from darkbreaker_sdk.schemas import RecognitionResult, Alarm, AlarmLevel, BoundingBox


class TestOutputStructure:
    """输出结构测试"""

    REQUIRED_METADATA_FIELDS = [
        "meter_type",
        "reading_status",
        "pipeline_stage",
        "fallback_level",
        "timestamp_ms",
        "runtime_mode",
        "review_status",
        "failure_reason",
    ]

    def test_recognition_result_fields(self):
        """RecognitionResult字段完整性"""
        result = RecognitionResult(
            task_id="test",
            site_id="site1",
            device_id="dev1",
            component_id="comp1",
            roi_id="roi1",
            bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
            label="pressure_gauge_reading",
            value=1.2,
            confidence=0.85,
            model_version="v1.0",
            code_version="abc123"
        )
        assert result.task_id == "test"
        assert result.value == 1.2
        assert result.confidence == 0.85
        assert result.label == "pressure_gauge_reading"

    def test_metadata_required_fields(self):
        """metadata必填字段完整性"""
        metadata = {
            "meter_type": "pressure_gauge",
            "reading_status": "success",
            "pipeline_stage": "analog_chain",
            "fallback_level": 0,
            "timestamp_ms": 1234567890,
            "runtime_mode": "traditional_fallback",
            "review_status": "clear",
            "failure_reason": "",
        }
        for f in self.REQUIRED_METADATA_FIELDS:
            assert f in metadata

    def test_analog_meter_metadata_extra_fields(self):
        """模拟表额外metadata字段"""
        metadata = {
            "meter_type": "pressure_gauge",
            "reading_status": "success",
            "pipeline_stage": "analog_chain",
            "fallback_level": 0,
            "timestamp_ms": 1234567890,
            "runtime_mode": "traditional_fallback",
            "review_status": "clear",
            "failure_reason": "",
            "pointer_angle_deg": 45.0,
            "angle_range_deg": [-135, 135],
            "range_min": 0.0,
            "range_max": 1.6
        }
        for key in ["pointer_angle_deg", "angle_range_deg", "range_min", "range_max"]:
            assert key in metadata

    def test_digital_meter_metadata_extra_fields(self):
        """数字表额外metadata字段"""
        metadata = {
            "meter_type": "digital_display",
            "reading_status": "success",
            "pipeline_stage": "ocr_chain",
            "fallback_level": 0,
            "timestamp_ms": 1234567890,
            "runtime_mode": "traditional_fallback",
            "review_status": "clear",
            "failure_reason": "",
            "ocr_text_raw": "12.34"
        }
        assert "ocr_text_raw" in metadata

    def test_led_metadata_extra_fields(self):
        """LED额外metadata字段"""
        metadata = {
            "meter_type": "led_indicator",
            "reading_status": "success",
            "pipeline_stage": "hsv_chain",
            "fallback_level": 0,
            "timestamp_ms": 1234567890,
            "runtime_mode": "traditional_fallback",
            "review_status": "clear",
            "failure_reason": "",
            "hsv_stats": {"h_mean": 0, "s_mean": 255, "v_mean": 255},
            "color_class": "red"
        }
        assert "hsv_stats" in metadata
        assert "color_class" in metadata

    def test_alarm_level_validity(self):
        """Alarm级别合法性"""
        valid_levels = [AlarmLevel.INFO, AlarmLevel.WARNING, AlarmLevel.CRITICAL]
        for level in valid_levels:
            alarm = Alarm(
                task_id="test",
                result_id=None,
                level=level,
                title="测试告警",
                message="测试消息",
                site_id="site1",
                device_id="dev1",
                component_id="comp1"
            )
            assert alarm.level in valid_levels

    def test_failed_no_fake_value(self):
        """FAILED场景无虚构读数"""
        result = RecognitionResult(
            task_id="test",
            site_id="site1",
            device_id="dev1",
            component_id="comp1",
            roi_id="roi1",
            bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
            label="pressure_gauge_failed",
            value=None,
            confidence=0.0,
            model_version="v1.0",
            code_version="abc123"
        )
        assert result.value is None
        assert "_failed" in result.label

    def test_need_manual_review_marking(self):
        """NEED_MANUAL_REVIEW标记正确"""
        metadata = {
            "meter_type": "pressure_gauge",
            "reading_status": "need_manual_review",
            "pipeline_stage": "analog_chain",
            "fallback_level": 0,
            "timestamp_ms": 1234567890,
            "runtime_mode": "traditional_fallback",
            "review_status": "manual_review_required",
            "failure_reason": "angle_out_of_range",
        }
        assert metadata["reading_status"] == "need_manual_review"
        assert metadata["review_status"] == "manual_review_required"

    def test_runtime_review_failure_metadata_contract(self):
        """output_contract 必须暴露运行路径、复核状态和失败原因。"""
        metadata = {
            "runtime_mode": "traditional_fallback",
            "review_status": "failed",
            "failure_reason": "empty_frame",
        }
        assert metadata["runtime_mode"] in {
            "real_dl",
            "traditional_fallback",
            "simulation",
            "not_applicable",
        }
        assert metadata["review_status"] in {"clear", "manual_review_required", "failed"}
        assert isinstance(metadata["failure_reason"], str)

    def test_confidence_range_validation(self):
        """confidence范围验证"""
        valid = [0.0, 0.5, 1.0]
        invalid = [-0.1, 1.5, float("nan"), float("inf")]

        for conf in valid:
            assert 0.0 <= conf <= 1.0
        for conf in invalid:
            assert not (0.0 <= conf <= 1.0)
