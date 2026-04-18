"""
L0 单元测试 - LED指示灯状态识别
测试HSV分类、可分离性校验、发光面积 (合约 3.5, 7.5)
"""
import pytest
import numpy as np
import cv2
from plugins.meter_reading.detector_enhanced import (
    MeterReadingDetectorEnhanced,
    MeterType,
    ReadingStatus,
)


class TestLEDHSVClassification:
    """LED HSV分类基础测试"""

    def test_hsv_red_classification(self):
        """红灯HSV: H<10 or H>170, S>100, V>100"""
        bgr_red = np.uint8([[[0, 0, 255]]])
        hsv = cv2.cvtColor(bgr_red, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0, 0]
        assert (h < 10 or h > 170) and s > 100 and v > 100

    def test_hsv_green_classification(self):
        """绿灯HSV: 40<H<80"""
        bgr_green = np.uint8([[[0, 255, 0]]])
        hsv = cv2.cvtColor(bgr_green, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0, 0]
        assert 40 < h < 80 and s > 100 and v > 100

    def test_hsv_yellow_classification(self):
        """黄灯HSV: 20<H<40"""
        bgr_yellow = np.uint8([[[0, 255, 255]]])
        hsv = cv2.cvtColor(bgr_yellow, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0, 0]
        assert 20 < h < 40 and s > 100 and v > 100

    def test_hsv_off_classification(self):
        """灭灯: V<50"""
        bgr_off = np.uint8([[[10, 10, 10]]])
        hsv = cv2.cvtColor(bgr_off, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[0, 0]
        assert v < 50

    def test_hsv_separability(self):
        """四种颜色在H通道可分离"""
        colors = {
            "red": np.uint8([[[0, 0, 255]]]),
            "green": np.uint8([[[0, 255, 0]]]),
            "yellow": np.uint8([[[0, 255, 255]]]),
            "off": np.uint8([[[10, 10, 10]]])
        }
        hsv_h = {}
        for name, bgr in colors.items():
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hsv_h[name] = hsv[0, 0, 0]

        assert hsv_h["red"] != hsv_h["green"]
        assert hsv_h["green"] != hsv_h["yellow"]

    def test_low_saturation_detection(self):
        """低饱和度应被检出"""
        bgr_low_sat = np.uint8([[[100, 100, 100]]])
        hsv = cv2.cvtColor(bgr_low_sat, cv2.COLOR_BGR2HSV)
        assert hsv[0, 0, 1] < 50

    def test_brightness_threshold(self):
        """亮暗区分"""
        bgr_bright = np.uint8([[[200, 200, 200]]])
        bgr_dark = np.uint8([[[20, 20, 20]]])
        v_bright = cv2.cvtColor(bgr_bright, cv2.COLOR_BGR2HSV)[0, 0, 2]
        v_dark = cv2.cvtColor(bgr_dark, cv2.COLOR_BGR2HSV)[0, 0, 2]
        assert v_bright > v_dark


class TestLEDDetector:
    """LED检测器集成测试"""

    def test_red_led_detection(self, detector, red_led_image):
        """红灯识别"""
        result = detector.read_meter(red_led_image, MeterType.LED_INDICATOR, roi_id="led_r")
        assert result.unit == "red"
        assert result.value == 1.0
        assert result.status == ReadingStatus.SUCCESS

    def test_green_led_detection(self, detector, green_led_image):
        """绿灯识别"""
        result = detector.read_meter(green_led_image, MeterType.LED_INDICATOR, roi_id="led_g")
        assert result.unit == "green"
        assert result.value == 2.0
        assert result.status == ReadingStatus.SUCCESS

    def test_yellow_led_detection(self, detector, yellow_led_image):
        """黄灯识别"""
        result = detector.read_meter(yellow_led_image, MeterType.LED_INDICATOR, roi_id="led_y")
        assert result.unit == "yellow"
        assert result.value == 3.0
        assert result.status == ReadingStatus.SUCCESS

    def test_off_led_detection(self, detector, off_led_image):
        """灭灯识别"""
        result = detector.read_meter(off_led_image, MeterType.LED_INDICATOR, roi_id="led_off")
        assert result.unit == "off"
        assert result.value == 0.0
        assert result.status == ReadingStatus.SUCCESS

    def test_low_saturation_triggers_review(self, detector):
        """低饱和度灰色 -> NEED_MANUAL_REVIEW"""
        gray_image = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = detector.read_meter(gray_image, MeterType.LED_INDICATOR)
        assert result.status == ReadingStatus.NEED_MANUAL_REVIEW
        assert result.need_manual_review is True

    def test_led_metadata_has_hsv_stats(self, detector, red_led_image):
        """LED metadata包含hsv_stats"""
        result = detector.read_meter(red_led_image, MeterType.LED_INDICATOR)
        assert "hsv_stats" in result.metadata

    def test_led_metadata_has_color_class(self, detector, red_led_image):
        """LED metadata包含color_class"""
        result = detector.read_meter(red_led_image, MeterType.LED_INDICATOR)
        assert "color_class" in result.metadata
        assert result.metadata["color_class"] == "red"

    def test_led_metadata_required_fields(self, detector, red_led_image):
        """LED metadata包含所有必填字段"""
        result = detector.read_meter(red_led_image, MeterType.LED_INDICATOR, roi_id="test")
        required = ["meter_type", "reading_status", "pipeline_stage", "fallback_level", "timestamp_ms"]
        for key in required:
            assert key in result.metadata, f"缺少: {key}"
