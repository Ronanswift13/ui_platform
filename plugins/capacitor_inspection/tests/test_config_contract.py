"""Configuration contract tests for capacitor_inspection."""

from __future__ import annotations

import copy


def test_plugin_and_detector_read_inference_config(loaded_config):
    """Plugin and detector should share the same nested inference thresholds."""
    from plugins.capacitor_inspection.detector_enhanced import CapacitorDetectorEnhanced
    from plugins.capacitor_inspection.plugin import CapacitorInspectionPlugin

    plugin = CapacitorInspectionPlugin.create_standalone(config=copy.deepcopy(loaded_config))
    detector = CapacitorDetectorEnhanced(copy.deepcopy(loaded_config))

    assert plugin.confidence_threshold == loaded_config["inference"]["confidence_threshold"]
    assert detector._confidence_threshold == loaded_config["inference"]["confidence_threshold"]
    assert detector._nms_threshold == loaded_config["inference"]["nms_threshold"]


def test_default_config_has_required_sections(loaded_config):
    """YAML config should expose the runtime sections used by plugin and detector."""
    for key in ("model", "inference", "structural_integrity", "intrusion_detection", "capacitor_bank"):
        assert key in loaded_config
