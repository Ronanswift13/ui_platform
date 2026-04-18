#!/usr/bin/env python3
"""Detector contract tests for switch_inspection."""

import numpy as np

from plugins.switch_inspection.detector_enhanced import (
    ClarityLevel,
    GaugeReadingResult,
    SwitchDetectorEnhanced,
)


def test_evaluate_clarity_returns_normalized_score(default_config):
    detector = SwitchDetectorEnhanced(default_config)
    image = np.random.randint(0, 255, size=(96, 96, 3), dtype=np.uint8)
    result = detector.evaluate_clarity(image)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.level, ClarityLevel)


def test_recognize_indicator_state_uses_color_hint(default_config):
    detector = SwitchDetectorEnhanced(default_config)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :] = (0, 255, 0)
    result = detector.recognize_indicator_state(image, "breaker")
    assert result.state == "closed"
    assert 0.0 <= result.confidence <= 1.0
    assert result.evidence.green_ratio > 0


def test_validate_interlock_uses_switch_rules(default_config):
    detector = SwitchDetectorEnhanced(default_config)
    alarms = detector.validate_interlock(
        {"breaker": "closed", "isolator": "open", "grounding": "open"}
    )
    assert alarms
    assert alarms[0]["severity"] == "warning"
    assert alarms[0]["reason_code"] == 2002


def test_read_gauge_returns_standard_result(default_config):
    detector = SwitchDetectorEnhanced(default_config)
    result = detector.read_gauge(np.zeros((80, 80, 3), dtype=np.uint8))
    assert isinstance(result, GaugeReadingResult)
    assert result.value is None
    assert result.confidence == 0.0
