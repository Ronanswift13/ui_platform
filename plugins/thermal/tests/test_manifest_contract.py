"""Manifest contract tests for thermal."""
from __future__ import annotations

import json
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def test_manifest_exists_and_is_valid_json():
    manifest_path = PLUGIN_DIR / "manifest.json"
    assert manifest_path.exists(), "manifest.json missing"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_manifest_required_fields():
    data = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))
    for field in ("id", "name", "version", "entrypoint", "plugin_class"):
        assert field in data, f"manifest missing required field: {field}"
    assert data["id"] == "thermal"
    assert data["plugin_class"] == "ThermalPlugin"


def test_manifest_capabilities_are_list():
    data = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(data.get("capabilities", []), list)


def test_manifest_device_types():
    data = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(data.get("device_types", []), list)
    assert len(data["device_types"]) > 0
