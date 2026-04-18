#!/usr/bin/env python3
"""Configuration contract tests for switch_inspection."""

from plugins.switch_inspection.detector_enhanced import SwitchDetectorEnhanced
from plugins.switch_inspection.plugin import Plugin


def test_plugin_reads_runtime_thresholds(default_config):
    plugin = Plugin.create_standalone(default_config)
    assert plugin.confidence_threshold == default_config["inference"]["confidence_threshold"]
    assert plugin.min_clarity_score == default_config["image_quality"]["min_clarity_score"]


def test_detector_reads_state_and_clarity_config(default_config):
    detector = SwitchDetectorEnhanced(default_config)
    assert detector.min_state_score == default_config["state_recognition"]["min_state_score"]
    assert detector.weights["ocr"] == default_config["state_recognition"]["fusion_weights"]["text"]
    assert detector.weights["color"] == default_config["state_recognition"]["fusion_weights"]["color"]
    assert detector.weights["angle"] == default_config["state_recognition"]["fusion_weights"]["angle"]
    assert detector.clarity_config["min_clarity_score"] == default_config["image_quality"]["min_clarity_score"]


def test_detector_reads_gauge_enablement(default_config):
    detector = SwitchDetectorEnhanced(default_config)
    assert detector.gauge_reader.enabled is False
