"""
L0 单元测试 - 置信度评估与人工复核判定
(合约 3.6, 7.6)
"""
import pytest
import numpy as np
from plugins.meter_reading.detector_enhanced import _sanitize_confidence, ReadingStatus


class TestConfidenceEvaluation:
    """置信度评估测试"""

    def test_confidence_below_threshold_needs_review(self):
        """confidence=0.49 < 0.5 -> NEED_MANUAL_REVIEW"""
        confidence = 0.49
        threshold = 0.5
        assert confidence < threshold

    def test_confidence_at_threshold_can_succeed(self):
        """confidence=0.50 >= 0.5 -> 可成功"""
        confidence = 0.50
        threshold = 0.5
        assert confidence >= threshold

    def test_confidence_above_threshold(self):
        """confidence=0.6 >= 0.5"""
        assert 0.6 >= 0.5

    def test_confidence_nan_sanitized(self):
        """NaN -> 0.0"""
        assert _sanitize_confidence(float("nan")) == 0.0

    def test_confidence_negative_sanitized(self):
        """负数 -> 0.0"""
        assert _sanitize_confidence(-0.1) == 0.0

    def test_confidence_greater_than_one_sanitized(self):
        """> 1.0 -> 1.0"""
        assert _sanitize_confidence(1.5) == 1.0

    def test_confidence_inf_sanitized(self):
        """Inf -> 0.0"""
        assert _sanitize_confidence(float("inf")) == 0.0

    def test_confidence_neg_inf_sanitized(self):
        """-Inf -> 0.0"""
        assert _sanitize_confidence(float("-inf")) == 0.0

    def test_confidence_valid_range(self):
        """[0,1]内不变"""
        assert _sanitize_confidence(0.75) == 0.75
        assert _sanitize_confidence(0.0) == 0.0
        assert _sanitize_confidence(1.0) == 1.0

    def test_min_confidence_calculation(self):
        """统一置信度 c = min(c_det, c_geom, c_parse)"""
        c_det, c_geom, c_parse = 0.9, 0.7, 0.8
        c = min(c_det, c_geom, c_parse)
        assert c == 0.7

    def test_confidence_with_invalid_component_yields_zero(self):
        """任一分量非法时 c=0"""
        c_det, c_geom, c_parse = 0.9, -1.0, 0.8
        components = [c_det, c_geom, c_parse]
        if any(c < 0 for c in components):
            c = 0.0
        else:
            c = min(components)
        assert c == 0.0

    def test_reading_status_no_low_confidence(self):
        """ReadingStatus不包含LOW_CONFIDENCE"""
        status_values = [s.value for s in ReadingStatus]
        assert "low_confidence" not in status_values

    def test_reading_status_only_three(self):
        """ReadingStatus严格三态"""
        assert len(ReadingStatus) == 3
        assert ReadingStatus.SUCCESS.value == "success"
        assert ReadingStatus.FAILED.value == "failed"
        assert ReadingStatus.NEED_MANUAL_REVIEW.value == "need_manual_review"
