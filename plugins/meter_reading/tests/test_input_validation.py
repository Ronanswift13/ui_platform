"""
L0 单元测试 - 输入校验
测试输入非法场景的兜底逻辑 (合约 5.1)
"""
import pytest
import numpy as np
from plugins.meter_reading.detector_enhanced import (
    MeterReadingDetectorEnhanced,
    MeterType,
    ReadingStatus,
)


class TestInputValidation:
    """输入校验测试 (合约 7.1)"""

    def test_empty_frame(self, detector):
        """空帧输入 -> FAILED"""
        empty_frame = np.array([])
        result = detector.read_meter(empty_frame, MeterType.PRESSURE_GAUGE)

        assert result.value is None
        assert result.confidence == 0.0
        assert result.status == ReadingStatus.FAILED
        assert result.metadata.get("error_code") == "empty_frame"

    def test_none_frame(self, detector):
        """None帧输入 -> FAILED"""
        result = detector.read_meter(None, MeterType.PRESSURE_GAUGE)

        assert result.status == ReadingStatus.FAILED
        assert result.confidence == 0.0

    def test_grayscale_frame(self, detector):
        """灰度帧输入 -> FAILED"""
        gray_frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        result = detector.read_meter(gray_frame, MeterType.PRESSURE_GAUGE)

        assert result.value is None
        assert result.confidence == 0.0
        assert result.status == ReadingStatus.FAILED

    def test_four_channel_frame(self, detector):
        """四通道帧输入 -> FAILED"""
        rgba_frame = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
        result = detector.read_meter(rgba_frame, MeterType.PRESSURE_GAUGE)

        assert result.value is None
        assert result.confidence == 0.0
        assert result.status == ReadingStatus.FAILED

    def test_roi_negative_values(self, detector, sample_frame):
        """ROI为负值 -> FAILED"""
        invalid_roi = {"x": -0.1, "y": 0.2, "w": 0.4, "h": 0.4}
        result = detector.read_meter(sample_frame, MeterType.PRESSURE_GAUGE, invalid_roi)

        assert result.value is None
        assert result.status == ReadingStatus.FAILED

    def test_roi_out_of_bounds(self, detector, sample_frame):
        """ROI超出[0,1] -> FAILED"""
        invalid_roi = {"x": 0.8, "y": 0.8, "w": 0.5, "h": 0.5}
        result = detector.read_meter(sample_frame, MeterType.PRESSURE_GAUGE, invalid_roi)

        assert result.value is None
        assert result.status == ReadingStatus.FAILED

    def test_roi_zero_width(self, detector, sample_frame):
        """ROI宽度为0 -> FAILED"""
        invalid_roi = {"x": 0.2, "y": 0.2, "w": 0.0, "h": 0.4}
        result = detector.read_meter(sample_frame, MeterType.PRESSURE_GAUGE, invalid_roi)

        assert result.value is None
        assert result.status == ReadingStatus.FAILED

    def test_roi_zero_height(self, detector, sample_frame):
        """ROI高度为0 -> FAILED"""
        invalid_roi = {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.0}
        result = detector.read_meter(sample_frame, MeterType.PRESSURE_GAUGE, invalid_roi)

        assert result.value is None
        assert result.status == ReadingStatus.FAILED

    def test_roi_too_small_pixels(self, detector):
        """ROI裁剪后像素宽高<4 -> FAILED"""
        tiny_frame = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        tiny_roi = {"x": 0.0, "y": 0.0, "w": 0.1, "h": 0.1}
        result = detector.read_meter(tiny_frame, MeterType.PRESSURE_GAUGE, tiny_roi)

        assert result.value is None
        assert result.status == ReadingStatus.FAILED

    def test_failed_result_has_metadata(self, detector):
        """FAILED结果必须包含metadata必填字段"""
        empty_frame = np.array([])
        result = detector.read_meter(empty_frame, MeterType.PRESSURE_GAUGE, roi_id="test_roi")

        required_keys = ["meter_type", "reading_status", "pipeline_stage", "fallback_level", "timestamp_ms"]
        for key in required_keys:
            assert key in result.metadata, f"缺少metadata字段: {key}"

    def test_context_empty_strings_allowed(self, detector, sample_frame):
        """上下文字段允许空字符串"""
        result = detector.read_meter(sample_frame, MeterType.PRESSURE_GAUGE, roi_id="")
        assert result.metadata.get("roi_id") == "" or "roi_id" in result.metadata
