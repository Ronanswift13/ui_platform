"""
L0 单元测试 - 数字表OCR解析
测试清洗规则和数值解析 (合约 3.4, 7.4)
"""
import pytest
import numpy as np
from plugins.meter_reading.detector_enhanced import (
    MeterReadingDetectorEnhanced,
    MeterType,
    ReadingStatus,
    _clean_ocr_text,
)


class TestOCRTextCleaning:
    """OCR文本清洗测试 (合约 3.4.1)"""

    def test_parse_integer(self):
        """纯整数读数"""
        cleaned, reason = _clean_ocr_text("123")
        assert cleaned == "123"
        assert float(cleaned) == 123.0

    def test_parse_decimal(self):
        """小数读数"""
        cleaned, reason = _clean_ocr_text("12.34")
        assert cleaned == "12.34"
        assert abs(float(cleaned) - 12.34) < 0.001

    def test_parse_negative(self):
        """负数读数"""
        cleaned, reason = _clean_ocr_text("-5.6")
        assert cleaned == "-5.6"
        assert abs(float(cleaned) - (-5.6)) < 0.001

    def test_parse_leading_zero(self):
        """前导零读数"""
        cleaned, reason = _clean_ocr_text("007")
        assert cleaned == "007"
        assert float(cleaned) == 7.0

    def test_multiple_decimal_points_rejected(self):
        """多小数点非法串 -> 拒绝"""
        cleaned, reason = _clean_ocr_text("12.34.56")
        assert cleaned is None
        assert reason == "multiple_decimal_points"

    def test_negative_in_middle_rejected(self):
        """负号在中间位置 -> 拒绝"""
        cleaned, reason = _clean_ocr_text("12-34")
        assert cleaned is None
        assert reason == "negative_sign_not_at_start"

    def test_empty_string_rejected(self):
        """OCR空串 -> 拒绝"""
        cleaned, reason = _clean_ocr_text("")
        assert cleaned is None
        assert reason == "empty_string"

    def test_whitespace_only_rejected(self):
        """纯空白 -> 拒绝"""
        cleaned, reason = _clean_ocr_text("   ")
        assert cleaned is None
        assert reason == "empty_string"

    def test_non_numeric_chars_stripped(self):
        """非法字符被剥离"""
        cleaned, reason = _clean_ocr_text("12.3abc")
        assert cleaned == "12.3"

    def test_all_non_numeric_rejected(self):
        """全部非法字符 -> 拒绝"""
        cleaned, reason = _clean_ocr_text("abc")
        assert cleaned is None
        assert reason == "no_valid_chars"

    def test_valid_float_syntax_batch(self):
        """批量浮点语法验证"""
        valid_cases = ["123", "12.34", "-5.6", "0", "0.0", "-0.5"]
        for s in valid_cases:
            cleaned, reason = _clean_ocr_text(s)
            assert cleaned is not None, f"'{s}' should be valid"
            assert reason == ""

    def test_invalid_float_syntax_batch(self):
        """批量非法语法"""
        invalid_cases = ["12.34.56", "", "abc", "12-34"]
        for s in invalid_cases:
            cleaned, reason = _clean_ocr_text(s)
            assert cleaned is None, f"'{s}' should be invalid"

    def test_just_minus_sign(self):
        """仅负号"""
        cleaned, reason = _clean_ocr_text("-")
        assert cleaned is None


class TestDigitalMeterDetector:
    """数字表检测器测试"""

    def test_digital_meter_returns_review_on_empty_ocr(self, detector, sample_frame):
        """OCR空串 -> NEED_MANUAL_REVIEW"""
        result = detector.read_meter(sample_frame, MeterType.DIGITAL_DISPLAY)
        assert result.status in (ReadingStatus.NEED_MANUAL_REVIEW, ReadingStatus.FAILED)

    def test_seven_segment_returns_review_on_empty_ocr(self, detector, sample_frame):
        """七段码空串 -> NEED_MANUAL_REVIEW"""
        result = detector.read_meter(sample_frame, MeterType.SEVEN_SEGMENT)
        assert result.status in (ReadingStatus.NEED_MANUAL_REVIEW, ReadingStatus.FAILED)

    def test_digital_metadata_has_ocr_text_raw(self, detector, sample_frame):
        """数字表metadata包含ocr_text_raw"""
        result = detector.read_meter(sample_frame, MeterType.DIGITAL_DISPLAY, roi_id="test")
        assert "ocr_text_raw" in result.metadata

    def test_digital_metadata_has_required_fields(self, detector, sample_frame):
        """数字表metadata包含必填字段"""
        result = detector.read_meter(sample_frame, MeterType.DIGITAL_DISPLAY, roi_id="test")
        required = ["meter_type", "reading_status", "pipeline_stage", "fallback_level", "timestamp_ms"]
        for key in required:
            assert key in result.metadata, f"缺少: {key}"
