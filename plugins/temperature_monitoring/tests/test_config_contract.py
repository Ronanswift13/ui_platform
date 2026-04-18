"""Config contract tests for temperature_monitoring.

Validates that configs/default.yaml contains all required sections
for correct plugin operation.
"""
from __future__ import annotations

import yaml
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    config_path = PLUGIN_DIR / "configs" / "default.yaml"
    assert config_path.exists(), "configs/default.yaml missing"
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    return data


def test_config_has_sensor_section():
    cfg = _load_config()
    assert "sensor" in cfg, "missing 'sensor' section"
    assert "type" in cfg["sensor"]
    assert "resolution" in cfg["sensor"]


def test_config_has_thresholds_section():
    cfg = _load_config()
    assert "thresholds" in cfg, "missing 'thresholds' section"
    th = cfg["thresholds"]
    for field in ("normal_max", "warning", "alarm", "critical"):
        assert field in th, f"thresholds missing field: {field}"
    # 阈值应递增
    assert th["normal_max"] < th["warning"] < th["alarm"] < th["critical"]


def test_config_has_hotspot_section():
    cfg = _load_config()
    assert "hotspot" in cfg, "missing 'hotspot' section"
    assert "min_area_ratio" in cfg["hotspot"]
    assert "z_score_threshold" in cfg["hotspot"]


def test_config_has_prediction_section():
    cfg = _load_config()
    assert "prediction" in cfg, "missing 'prediction' section"
    pred = cfg["prediction"]
    assert "enabled" in pred
    assert "method" in pred


def test_config_has_zones_section():
    cfg = _load_config()
    assert "zones" in cfg, "missing 'zones' section"
    assert isinstance(cfg["zones"], list)
    assert len(cfg["zones"]) > 0


def test_config_has_linkage_section():
    cfg = _load_config()
    assert "linkage" in cfg, "missing 'linkage' section"


def test_config_has_history_section():
    cfg = _load_config()
    assert "history" in cfg, "missing 'history' section"
    assert "buffer_size" in cfg["history"]


def test_config_has_performance_section():
    cfg = _load_config()
    assert "performance" in cfg, "missing 'performance' section"
    assert "max_fps" in cfg["performance"]
