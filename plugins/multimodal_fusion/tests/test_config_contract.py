import json
from pathlib import Path

import yaml

from darkbreaker_sdk.utils import load_plugin_config
from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def test_default_config_declares_first_batch_contract():
    config_path = PLUGIN_DIR / "configs" / "default.yaml"
    assert config_path.exists()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in (
        "modalities",
        "modality_weights",
        "fusion_strategy",
        "thresholds",
        "evidence",
        "fallback",
        "upgrade_placeholders",
    ):
        assert key in config

    assert config["modalities"] == ["visual", "thermal", "acoustic", "gas", "hyperspectral"]
    assert set(config["modality_weights"]) == set(config["modalities"])
    assert config["fallback"]["missing_modality_policy"] == "degrade"
    assert "bayesian_fusion" in config["upgrade_placeholders"]


def test_manifest_declares_fusion_capability_and_input_modalities():
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert "multimodal_data_fusion" in manifest["capabilities"]
    assert manifest["input_modalities"] == ["visual", "thermal", "acoustic", "gas", "hyperspectral"]
    assert "plugin_outputs" in manifest["input_schema"]["properties"]
    assert "fused_status" in manifest["output_schema"]["properties"]


def test_plugin_parses_default_config_without_hardcoded_thresholds():
    config = load_plugin_config(PLUGIN_DIR / "configs" / "default.yaml")
    plugin = MultimodalFusionPlugin(config=config)

    assert plugin.config.fusion_strategy == config["fusion_strategy"]
    assert plugin.config.modalities == config["modalities"]
    assert plugin.config.modality_weights == config["modality_weights"]
    assert plugin.config.thresholds["consensus_min_modalities"] == 2
    assert plugin.config.upgrade_placeholders["temporal_fusion"]["status"] == "placeholder"
