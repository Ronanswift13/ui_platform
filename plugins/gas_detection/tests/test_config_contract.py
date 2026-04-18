import json
from pathlib import Path

import yaml


PLUGIN_DIR = Path(__file__).resolve().parents[1]


def test_manifest_declares_sensor_timeseries_contract():
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["id"] == "gas_detection"
    domain_capabilities = manifest["domain_capabilities"]
    assert "sensor_input" in domain_capabilities
    assert "time_series_monitoring" in domain_capabilities
    assert "input_schema" in manifest
    assert "output_schema" in manifest
    assert manifest["config_file"] == "configs/default.yaml"


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

    assert "SF6" in config["thresholds"]
    assert config["runtime"]["require_hardware"] is False
    assert config["window"]["history_length"] == 24
    assert config["model"]["prediction_horizon"] == 24
    assert config["model"]["ids"]["lstm"] == config["model"]["ids"]["sf6_forecast"]
    assert config["model"]["ids"]["transformer"] == config["model"]["ids"]["multi_gas_forecast"]


def test_output_schema_declares_trend_contract_fields():
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))
    output_schema = manifest["output_schema"]

    assert "predictions" in output_schema["properties"]
    assert "trend_analysis" in output_schema["properties"]
    assert "trend_diagnosis" in output_schema["required"]
    assert "predictions" in output_schema["required"]
    assert "trend_analysis" in output_schema["required"]
