#!/usr/bin/env python3
"""Fallback-chain tests for switch_inspection detector."""

import numpy as np

from plugins.switch_inspection.detector_enhanced import SwitchDetectorEnhanced


class FailingRegistry:
    def infer(self, model_id, image):
        raise RuntimeError(f"forced failure for {model_id}")


def test_detector_falls_back_to_color_when_models_fail(default_config):
    detector = SwitchDetectorEnhanced(default_config, model_registry=FailingRegistry())
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 255)
    result = detector.recognize_switch_state(image)
    assert result["state"] == "open"
    assert result["confidence"] > 0
    assert any(item["source"] == "color" for item in result["evidences"])
