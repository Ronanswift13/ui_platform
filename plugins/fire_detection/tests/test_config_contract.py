"""Config contract tests for fire_detection.

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


def test_config_has_model_section():
    cfg = _load_config()
    assert "model" in cfg, "missing 'model' section"
    model = cfg["model"]
    assert "path" in model
    assert "device" in model
    assert "input_size" in model


def test_config_has_detection_section():
    cfg = _load_config()
    assert "detection" in cfg, "missing 'detection' section"
    det = cfg["detection"]
    assert "confidence_threshold" in det
    assert "nms_threshold" in det
    assert "min_confirm_frames" in det


def test_config_has_sensor_fusion_section():
    cfg = _load_config()
    assert "sensor_fusion" in cfg, "missing 'sensor_fusion' section"
    sf = cfg["sensor_fusion"]
    assert "enabled" in sf
    assert "fusion_method" in sf


def test_config_has_fire_level_section():
    cfg = _load_config()
    assert "fire_level" in cfg, "missing 'fire_level' section"
    fl = cfg["fire_level"]
    for level in ("smoldering", "small", "medium", "large", "critical"):
        assert level in fl, f"fire_level missing sub-level: {level}"


def test_config_has_suppression_section():
    cfg = _load_config()
    assert "suppression" in cfg, "missing 'suppression' section"
    assert "trigger_level" in cfg["suppression"]


def test_config_has_zones_section():
    cfg = _load_config()
    assert "zones" in cfg, "missing 'zones' section"
    assert isinstance(cfg["zones"], list)
    assert len(cfg["zones"]) > 0


def test_config_has_drill_section():
    cfg = _load_config()
    assert "drill" in cfg, "missing 'drill' section"
    assert "enabled" in cfg["drill"]


def test_config_has_history_section():
    cfg = _load_config()
    assert "history" in cfg, "missing 'history' section"
    assert "retention_days" in cfg["history"]


def test_config_has_performance_section():
    cfg = _load_config()
    assert "performance" in cfg, "missing 'performance' section"
    assert "max_fps" in cfg["performance"]
