"""Configuration contract tests for transformer_inspection."""

from __future__ import annotations

import copy


def test_plugin_and_detector_read_inference_config(loaded_config):
    """Plugin and detector should share the same nested inference thresholds."""
    from plugins.transformer_inspection.detector_enhanced import TransformerDetectorEnhanced
    from plugins.transformer_inspection.plugin import TransformerInspectionPlugin

    plugin = TransformerInspectionPlugin.create_standalone(config=copy.deepcopy(loaded_config))
    detector = TransformerDetectorEnhanced(copy.deepcopy(loaded_config))

    assert plugin.confidence_threshold == loaded_config["inference"]["confidence_threshold"]
    assert plugin.nms_threshold == loaded_config["inference"]["nms_threshold"]
    assert detector._confidence_threshold == loaded_config["inference"]["confidence_threshold"]
    assert detector._nms_threshold == loaded_config["inference"]["nms_threshold"]


def test_default_config_declares_real_runtime_sections(loaded_config):
    """The YAML config should keep the sections used by plugin and detector."""
    for key in ("model", "inference", "thermal", "alarm"):
        assert key in loaded_config
