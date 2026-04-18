import json
from pathlib import Path

import yaml


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def test_manifest_declares_sensor_timeseries_contract():
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["id"] == "action_event_monitoring"
    domain_capabilities = manifest["domain_capabilities"]
    assert "sensor_input" in domain_capabilities
    assert "time_series_monitoring" in domain_capabilities
    assert "input_schema" in manifest
    assert "output_schema" in manifest
    assert manifest["config_file"] == "configs/default.yaml"


def test_default_config_has_b_class_sections_and_no_prod_protocol():
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
    assert config["protocol"]["type"] == ""


def test_local_platform_unification_files_are_declared():
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert (PLUGIN_DIR / "requirements.txt").exists()
    assert (PLUGIN_DIR / "demo" / "run_demo.py").exists()
    assert (PLUGIN_DIR / "__main__.py").exists()
    assert (PLUGIN_DIR / "run_standalone.py").exists()
    assert manifest["standalone"]["port"] == 8097
