"""fire_detection drill simulation 契约测试。"""

from __future__ import annotations


def test_start_drill_contract(plugin):
    result = plugin.start_drill("electrical_fire")

    assert result["success"] is True
    assert result["semantic_type"] == "drill_simulation"
    assert result["is_real_detection"] is False
    assert result["drill_active"] is True
    assert result["scenario"] == "electrical_fire"
    assert result["review_status"] == "simulation_only"
    assert "treat_as_simulation_only" in result["recommended_action"]
    assert "do_not_trigger_hardware_automatically" in result["recommended_action"]
    assert result["capability_states"]["drill_simulation"] == "experimental"


def test_stop_drill_contract(plugin):
    plugin.start_drill("cable_fire")
    result = plugin.stop_drill()

    assert result["success"] is True
    assert result["semantic_type"] == "drill_simulation"
    assert result["is_real_detection"] is False
    assert result["drill_active"] is False
    assert result["scenario"] == "cable_fire"
    assert "restore_normal_monitoring" in result["recommended_action"]
    assert result["runtime_mode"]["drill"] == "stopped"


def test_start_drill_does_not_require_detector():
    from plugins.fire_detection.plugin import FireDetectionPlugin

    plugin = FireDetectionPlugin(config={"drill": {"auto_reset_seconds": 42}})
    result = plugin.start_drill("general")

    assert result["success"] is True
    assert result["semantic_type"] == "drill_simulation"
    assert result["drill_active"] is True
    assert result["auto_reset_seconds"] == 42
