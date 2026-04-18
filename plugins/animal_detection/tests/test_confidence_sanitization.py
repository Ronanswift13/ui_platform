#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QR-11 置信度消毒测试
====================
验证 detector 和 event_schema 对异常 confidence 值的防御。
"""

import math

from core.event_schema import (
    _sanitize_confidence,
    AnimalDetectionResult,
    build_intrusion_event,
)


class TestSanitizeConfidence:
    """_sanitize_confidence 纯函数测试"""

    def test_nan_returns_zero(self):
        assert _sanitize_confidence(float("nan")) == 0.0

    def test_inf_returns_zero(self):
        assert _sanitize_confidence(float("inf")) == 0.0

    def test_neg_inf_returns_zero(self):
        assert _sanitize_confidence(float("-inf")) == 0.0

    def test_negative_returns_zero(self):
        assert _sanitize_confidence(-0.5) == 0.0

    def test_greater_than_one_returns_one(self):
        assert _sanitize_confidence(1.5) == 1.0

    def test_valid_unchanged(self):
        assert _sanitize_confidence(0.0) == 0.0
        assert _sanitize_confidence(0.75) == 0.75
        assert _sanitize_confidence(1.0) == 1.0

    def test_zero_boundary(self):
        assert _sanitize_confidence(0.0) == 0.0

    def test_one_boundary(self):
        assert _sanitize_confidence(1.0) == 1.0


class TestDetectionResultSanitization:
    """AnimalDetectionResult.to_dict() 消毒测试"""

    def test_nan_confidence_in_to_dict(self):
        """to_dict 中 NaN confidence 被消毒为 0.0"""
        result = AnimalDetectionResult(
            animal_class="mouse",
            confidence=float("nan"),
        )
        d = result.to_dict()
        assert d["confidence"] == 0.0
        assert not math.isnan(d["confidence"])

    def test_inf_confidence_in_to_dict(self):
        """to_dict 中 Inf confidence 被消毒为 0.0"""
        result = AnimalDetectionResult(
            animal_class="cat",
            confidence=float("inf"),
        )
        d = result.to_dict()
        assert d["confidence"] == 0.0

    def test_negative_confidence_in_to_dict(self):
        """to_dict 中负数 confidence 被消毒为 0.0"""
        result = AnimalDetectionResult(
            animal_class="bird",
            confidence=-0.3,
        )
        d = result.to_dict()
        assert d["confidence"] == 0.0


class TestBuildIntrusionEventSanitization:
    """build_intrusion_event 中 confidence 消毒测试"""

    def test_max_conf_with_nan_input(self):
        """含 NaN confidence 的检测列表不应传播 NaN"""
        detections = [
            AnimalDetectionResult(animal_class="mouse", confidence=float("nan")),
            AnimalDetectionResult(animal_class="cat", confidence=0.8),
        ]
        event = build_intrusion_event(
            detections=detections,
            camera_id="cam_test",
            location="test_room",
        )
        assert not math.isnan(event.confidence)
        assert event.confidence == 0.8

    def test_max_conf_all_nan(self):
        """所有 confidence 都是 NaN 时，事件 confidence 应为 0.0"""
        detections = [
            AnimalDetectionResult(animal_class="mouse", confidence=float("nan")),
            AnimalDetectionResult(animal_class="snake", confidence=float("nan")),
        ]
        event = build_intrusion_event(
            detections=detections,
            camera_id="cam_test",
            location="test_room",
        )
        assert event.confidence == 0.0

    def test_max_conf_with_inf(self):
        """Inf confidence 不应传播"""
        detections = [
            AnimalDetectionResult(animal_class="mouse", confidence=float("inf")),
            AnimalDetectionResult(animal_class="cat", confidence=0.7),
        ]
        event = build_intrusion_event(
            detections=detections,
            camera_id="cam_test",
            location="test_room",
        )
        assert not math.isinf(event.confidence)
        assert event.confidence == 0.7

    def test_normal_confidence_unaffected(self):
        """正常 confidence 不受影响"""
        detections = [
            AnimalDetectionResult(animal_class="mouse", confidence=0.9),
            AnimalDetectionResult(animal_class="cat", confidence=0.7),
        ]
        event = build_intrusion_event(
            detections=detections,
            camera_id="cam_test",
            location="test_room",
        )
        assert event.confidence == 0.9
