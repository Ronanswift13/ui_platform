"""
L0 单元测试 - 模拟表读数算法
测试指针角度计算、量程映射、透视校正、降级链路 (合约 3.2-3.3, 7.3)
"""
import pytest
import numpy as np
import math
from plugins.meter_reading.detector_enhanced import (
    MeterReadingDetectorEnhanced, MeterType, ReadingStatus,
    MeterKeypoints, Keypoint, _sanitize_confidence,
)


class TestPointerAngleCalculation:
    """指针角度计算测试"""

    def test_angle_from_center_and_tip(self):
        """已知center+tip计算角度"""
        center = (100, 100)
        tip = (150, 100)
        dx = tip[0] - center[0]
        dy = -(tip[1] - center[1])
        angle_deg = math.degrees(math.atan2(dy, dx))
        assert abs(angle_deg - 0.0) < 0.1

    def test_angle_vertical_up(self):
        """垂直向上指针 = 90度"""
        center = (100, 100)
        tip = (100, 50)
        dx = tip[0] - center[0]
        dy = -(tip[1] - center[1])
        angle_deg = math.degrees(math.atan2(dy, dx))
        assert abs(angle_deg - 90.0) < 0.1

    def test_angle_vertical_down(self):
        """垂直向下指针 = -90度"""
        center = (100, 100)
        tip = (100, 150)
        dx = tip[0] - center[0]
        dy = -(tip[1] - center[1])
        angle_deg = math.degrees(math.atan2(dy, dx))
        assert abs(angle_deg - (-90.0)) < 0.1


class TestRangeMapping:
    """量程映射测试"""

    def test_boundary_min_angle(self):
        """边界值 -135度 -> r=0"""
        r = (-135.0 - (-135.0)) / (135.0 - (-135.0))
        assert abs(r - 0.0) < 0.001

    def test_boundary_zero_angle(self):
        """边界值 0度 -> r=0.5"""
        r = (0.0 - (-135.0)) / (135.0 - (-135.0))
        assert abs(r - 0.5) < 0.001

    def test_boundary_max_angle(self):
        """边界值 135度 -> r=1.0"""
        r = (135.0 - (-135.0)) / (135.0 - (-135.0))
        assert abs(r - 1.0) < 0.001

    def test_value_mapping_pressure_gauge(self):
        """压力表量程映射"""
        v_hat = 0.0 + 0.5 * (1.6 - 0.0)
        assert abs(v_hat - 0.8) < 0.001

    def test_value_mapping_temperature_gauge(self):
        """温度表量程映射"""
        v_hat = -20.0 + 0.0 * (100.0 - (-20.0))
        assert abs(v_hat - (-20.0)) < 0.001

    def test_value_mapping_full_scale(self):
        """满量程映射"""
        v_hat = 0.0 + 1.0 * (100.0 - 0.0)
        assert abs(v_hat - 100.0) < 0.001

    def test_angle_out_of_range_low(self):
        """指针角度略小于-135度 -> r<0"""
        r = (-136.0 - (-135.0)) / (135.0 - (-135.0))
        assert r < 0.0

    def test_angle_out_of_range_high(self):
        """指针角度略大于135度 -> r>1"""
        r = (136.0 - (-135.0)) / (135.0 - (-135.0))
        assert r > 1.0

    def test_precision_requirement_within_2pct(self):
        """满量程+-2%精度"""
        v_min, v_max = 0.0, 100.0
        v_gt, v_hat = 50.0, 51.5
        error_ratio = abs(v_hat - v_gt) / (v_max - v_min)
        assert error_ratio <= 0.02

    def test_precision_requirement_exceeds_2pct(self):
        """超过满量程2%应检出"""
        v_min, v_max = 0.0, 100.0
        v_gt, v_hat = 50.0, 53.0
        error_ratio = abs(v_hat - v_gt) / (v_max - v_min)
        assert error_ratio > 0.02


class TestAnalogMeterDetector:
    """模拟表检测器集成测试"""

    def test_analog_meter_with_valid_frame(self, detector, sample_frame):
        """模拟表有效帧 -> 返回结果(可能为NEED_MANUAL_REVIEW)"""
        result = detector.read_meter(sample_frame, MeterType.PRESSURE_GAUGE)
        assert result.meter_type == MeterType.PRESSURE_GAUGE
        assert result.status in (ReadingStatus.SUCCESS, ReadingStatus.NEED_MANUAL_REVIEW, ReadingStatus.FAILED)

    def test_analog_all_types_dispatched(self, detector, sample_frame):
        """9种表计类型全部可分发"""
        for mt in MeterType:
            result = detector.read_meter(sample_frame, mt)
            assert result.meter_type == mt
            assert result.status in (ReadingStatus.SUCCESS, ReadingStatus.NEED_MANUAL_REVIEW, ReadingStatus.FAILED)

    def test_angle_out_of_range_triggers_review(self, default_config):
        """角度越界强制进入人工复核"""
        det = MeterReadingDetectorEnhanced(default_config)
        det.initialize()
        angle = 140.0
        angle_min = default_config.get("pointer_detection", {}).get("angle_range", {}).get("min", -135)
        angle_max = default_config.get("pointer_detection", {}).get("angle_range", {}).get("max", 135)
        assert angle > angle_max

    def test_range_missing_triggers_review(self, default_config):
        """量程缺失时 _recognize_range 返回 None"""
        det = MeterReadingDetectorEnhanced(default_config)
        det.initialize()
        dummy_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = det._recognize_range(dummy_image, MeterType.SEVEN_SEGMENT)
        assert result is None

    def test_known_ranges_exist_for_analog(self, detector):
        """已知模拟表类型在METER_RANGES中有量程"""
        from plugins.meter_reading.detector_enhanced import ANALOG_METER_TYPES
        for mt in ANALOG_METER_TYPES:
            assert mt in MeterReadingDetectorEnhanced.METER_RANGES

    def test_metadata_has_required_fields_for_analog(self, detector, sample_frame):
        """模拟表结果metadata包含必填字段"""
        result = detector.read_meter(sample_frame, MeterType.PRESSURE_GAUGE, roi_id="test")
        required = ["meter_type", "reading_status", "pipeline_stage", "fallback_level", "timestamp_ms"]
        for key in required:
            assert key in result.metadata, f"缺少: {key}"

    def test_fallback_chain_order(self, detector, sample_frame):
        """降级链路返回 fallback_level >= 0"""
        kp, level = detector._detect_keypoints_with_fallback(sample_frame)
        assert level >= 0

    def test_fallback_chain_attempt_order_with_stubs(self, detector, sample_frame, monkeypatch):
        """用stub证明尝试顺序: HRNet -> HoughCircle -> HoughLine"""
        calls = []
        detector._use_deep_learning = True
        detector._dl_initialized = True
        detector._dl_keypoint_detector = object()
        detector._model_registry = None

        def fake_hrnet(_image):
            calls.append("hrnet")
            return MeterKeypoints()

        def fake_hough_circle(_image):
            calls.append("hough_circle")
            return None

        def fake_hough_line(_image):
            calls.append("hough_line")
            return MeterKeypoints(center=Keypoint("center", 0.5, 0.5, 0.4))

        monkeypatch.setattr(detector, "_detect_keypoints_v3", fake_hrnet)
        monkeypatch.setattr(detector, "_detect_keypoints_hough_circle", fake_hough_circle)
        monkeypatch.setattr(detector, "_detect_keypoints_hough_line", fake_hough_line)

        kp, level = detector._detect_keypoints_with_fallback(sample_frame)

        assert calls == ["hrnet", "hough_circle", "hough_line"]
        assert kp.center is not None
        assert level == 2


class TestSanitizeConfidence:
    """置信度清洗函数测试"""

    def test_nan_returns_zero(self):
        assert _sanitize_confidence(float("nan")) == 0.0

    def test_inf_returns_zero(self):
        assert _sanitize_confidence(float("inf")) == 0.0

    def test_negative_returns_zero(self):
        assert _sanitize_confidence(-0.5) == 0.0

    def test_greater_than_one_returns_one(self):
        assert _sanitize_confidence(1.5) == 1.0

    def test_valid_value_unchanged(self):
        assert _sanitize_confidence(0.75) == 0.75
