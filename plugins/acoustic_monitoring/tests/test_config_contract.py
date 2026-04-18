import json
from pathlib import Path

import yaml


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def test_manifest_declares_sensor_timeseries_contract():
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["id"] == "acoustic_monitoring"
    domain_capabilities = manifest["domain_capabilities"]
    assert "sensor_input" in domain_capabilities
    assert "time_series_monitoring" in domain_capabilities
    assert "input_schema" in manifest
    assert "output_schema" in manifest


def test_default_config_has_b_class_sections():
    config = yaml.safe_load((PLUGIN_DIR / "configs" / "default.yaml").read_text(encoding="utf-8"))

    for key in (
        "sampling",
        "window",
        "thresholds",
        "runtime",
        "model",
        "alarm_rules",
        "upgrade_placeholders",
    ):
        assert key in config

    assert config["runtime"]["require_hardware"] is False
