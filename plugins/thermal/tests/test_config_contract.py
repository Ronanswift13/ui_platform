"""Configuration contract tests for thermal."""
from __future__ import annotations


def test_default_config_has_required_sections(loaded_config):
    for key in ("model", "thresholds", "quality", "trend", "labels"):
        assert key in loaded_config, f"config missing section: {key}"


def test_thresholds_are_valid(loaded_config):
    t = loaded_config["thresholds"]
    assert t["temp_warning"] < t["temp_serious"] < t["temp_critical"]
    assert t["gradient_threshold"] > 0


def test_upgrade_placeholders_present(loaded_config):
    up = loaded_config.get("upgrade_placeholders", {})
    assert "model_registry_id" in up
    assert "model_asset_status" in up
