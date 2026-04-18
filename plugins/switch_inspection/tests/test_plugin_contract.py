#!/usr/bin/env python3
"""Plugin contract tests for switch_inspection."""

from types import SimpleNamespace

import numpy as np

from plugins.switch_inspection.detector_enhanced import (
    ClarityLevel,
    ClarityResult,
    GaugeReadingResult,
    RecognitionEvidenceSummary,
    SwitchRecognitionResult,
)
from plugins.switch_inspection.plugin import Plugin


class FakeDetector:
    gauge_reader = SimpleNamespace(enabled=False)

    def evaluate_clarity(self, image):
        return ClarityResult(score=0.95, level=ClarityLevel.GOOD, method="test")

    def recognize_indicator_state(self, image, device_type, clarity_score=None):
        return SwitchRecognitionResult(
            state="closed",
            confidence=0.91,
            evidence=RecognitionEvidenceSummary(
                green_ratio=0.7,
                clarity_score=clarity_score,
            ),
        )

    def recognize_linkage_state(self, image, device_type, clarity_score=None):
        return SwitchRecognitionResult(
            state="open",
            confidence=0.88,
            evidence=RecognitionEvidenceSummary(angle_deg=12.5, clarity_score=clarity_score),
        )

    def validate_interlock(self, bay_states):
        return []

    def read_gauge(self, image):
        return GaugeReadingResult()


class LowClarityDetector(FakeDetector):
    def evaluate_clarity(self, image):
        return ClarityResult(score=0.1, level=ClarityLevel.RESHOOT_REQUIRED, method="test")


def test_infer_accepts_none_context(default_config):
    plugin = Plugin.create_standalone(default_config)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    results = plugin.infer(frame, [], None)
    assert results == []


def test_infer_skips_invalid_roi(default_config, plugin_context, roi_factory):
    plugin = Plugin.create_standalone(default_config)
    plugin._detector = FakeDetector()
    bad_roi = roi_factory("breaker_indicator")
    bad_roi.bbox.width = 0.0
    bad_roi.bbox.height = 0.0
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert plugin.infer(frame, [bad_roi], plugin_context) == []


def test_infer_returns_state_result(default_config, plugin_context, roi_factory):
    plugin = Plugin.create_standalone(default_config)
    plugin._detector = FakeDetector()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    roi = roi_factory("breaker_indicator")
    results = plugin.infer(frame, [roi], plugin_context)
    assert len(results) == 1
    result = results[0]
    assert result.label == "breaker_closed"
    assert result.metadata["state"] == "closed"
    assert result.metadata["evidence"]["green_ratio"] == 0.7


def test_infer_returns_clarity_low_when_gate_fails(default_config, plugin_context, roi_factory):
    plugin = Plugin.create_standalone(default_config)
    plugin._detector = LowClarityDetector()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    roi = roi_factory("breaker_indicator")
    results = plugin.infer(frame, [roi], plugin_context)
    assert len(results) == 1
    assert results[0].label == "clarity_low"
