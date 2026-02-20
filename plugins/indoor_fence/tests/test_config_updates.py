#!/usr/bin/env python3
"""Tests for indoor fence config and zone update workflows."""

import copy
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@pytest.fixture()
def plugin_instance():
    from plugins.indoor_fence.plugin import Plugin

    plugin = Plugin.create_standalone()
    yield plugin
    plugin.cleanup()


def test_runtime_config_has_zone_config(plugin_instance):
    config = plugin_instance.get_runtime_config()
    assert "zone_config" in config
    assert config["zone_config"].get("path")

    zone_config = plugin_instance.get_zone_config()
    assert "zones" in zone_config
    assert "cabinets" in zone_config


def test_update_config_applies_threshold(plugin_instance):
    updated = plugin_instance.update_config(
        {"safety_zone": {"warning_distance_m": 0.25}},
        merge=True,
    )

    assert updated["safety_zone"]["warning_distance_m"] == 0.25
    assert plugin_instance._state_machine.on_line_threshold == pytest.approx(0.25)


def test_invalid_config_update_rolls_back(plugin_instance):
    before = copy.deepcopy(plugin_instance.get_runtime_config())

    with pytest.raises(ValueError):
        plugin_instance.update_config(
            {"safety_zone": {"warning_distance_m": -1.0}},
            merge=True,
        )

    after = plugin_instance.get_runtime_config()
    assert after["safety_zone"]["warning_distance_m"] == before["safety_zone"]["warning_distance_m"]


def test_update_zone_config_in_memory(plugin_instance):
    zone_data = {
        "id": "test-zone-layout",
        "name": "测试布局",
        "version": "1.0.0",
        "zones": [
            {
                "id": "passage_main",
                "name": "主通道",
                "type": "passage",
                "vertices": [[0.0, 0.0], [6.0, 0.0], [6.0, 2.0], [0.0, 2.0]],
            },
            {
                "id": "cabinet_1",
                "name": "机柜-01",
                "type": "cabinet",
                "vertices": [[0.0, 2.0], [3.0, 2.0], [3.0, 6.0], [0.0, 6.0]],
                "cabinet_id": 1,
            },
        ],
        "cabinets": [
            {
                "id": 1,
                "name": "机柜-01",
                "zone_id": "cabinet_1",
                "x_start": 0.0,
                "x_end": 0.5,
                "yellow_line": {"start": [0.0, 2.5], "end": [3.0, 2.5]},
            }
        ],
        "yellow_lines": [
            {
                "id": "yl_1",
                "name": "黄线1",
                "start": [0.0, 2.5],
                "end": [3.0, 2.5],
                "cabinets": [1],
            }
        ],
    }

    updated_zone = plugin_instance.update_zone_config(zone_data, persist=False)
    assert updated_zone["id"] == "test-zone-layout"
    assert len(updated_zone["cabinets"]) == 1
