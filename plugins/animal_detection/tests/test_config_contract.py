"""Config contract tests for animal_detection.

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


def test_config_has_inference_section():
    cfg = _load_config()
    assert "inference" in cfg, "missing 'inference' section"
    inf = cfg["inference"]
    assert "confidence_threshold" in inf
    assert "nms_threshold" in inf


def test_config_has_thermal_section():
    cfg = _load_config()
    assert "thermal" in cfg, "missing 'thermal' section"
    assert "enabled" in cfg["thermal"]


def test_config_has_tracking_section():
    cfg = _load_config()
    assert "tracking" in cfg, "missing 'tracking' section"
    trk = cfg["tracking"]
    assert "enabled" in trk
    assert "max_age" in trk
    assert "iou_threshold" in trk


def test_config_has_deterrent_section():
    cfg = _load_config()
    assert "deterrent" in cfg, "missing 'deterrent' section"
    det = cfg["deterrent"]
    assert "enabled" in det
    assert "cooldown_s" in det


def test_config_has_statistics_section():
    cfg = _load_config()
    assert "statistics" in cfg, "missing 'statistics' section"


def test_config_has_zones_section():
    cfg = _load_config()
    assert "zones" in cfg, "missing 'zones' section"
    assert isinstance(cfg["zones"], list)
    assert len(cfg["zones"]) > 0


def test_config_has_simulation_section():
    cfg = _load_config()
    assert "simulation" in cfg, "missing 'simulation' section"
    assert "enabled" in cfg["simulation"]
