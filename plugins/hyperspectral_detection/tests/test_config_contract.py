"""Configuration contract tests for hyperspectral_detection."""
from __future__ import annotations

def test_default_config_has_required_sections(loaded_config):
    for key in ("model", "spectral", "thresholds", "quality", "labels"):
        assert key in loaded_config, f"config missing section: {key}"

def test_spectral_config_valid(loaded_config):
    s = loaded_config["spectral"]
    assert s["bands"] > 0
    assert s["wavelength_start_nm"] < s["wavelength_end_nm"]

def test_upgrade_placeholders_present(loaded_config):
    up = loaded_config.get("upgrade_placeholders", {})
    assert "model_registry_id" in up
    assert "model_asset_status" in up
