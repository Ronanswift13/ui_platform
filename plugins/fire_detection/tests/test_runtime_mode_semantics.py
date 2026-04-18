"""fire_detection runtime mode / blocked 语义测试。"""

from __future__ import annotations


def test_no_onnx_model_reports_simulation_mode(plugin, blank_frame):
    result = plugin.detect(blank_frame)

    assert result["success"] is True
    assert result["runtime_mode"]["analysis_mode"] == "simulation_only"
    assert result["runtime_mode"]["visual_detection"] == "simulation"
    assert result["metadata"]["model_loaded"] is False
    assert result["capability_states"]["real_dl_onnx_inference"] == "blocked"


def test_thermal_and_sensor_inputs_are_marked_blocked_not_verified(
    plugin,
    blank_frame,
    thermal_hotspot_frame,
    sensor_alarm_data,
):
    result = plugin.detect(
        blank_frame,
        thermal_frame=thermal_hotspot_frame,
        sensor_data=sensor_alarm_data,
    )

    assert result["runtime_mode"]["thermal_input"] == "provided"
    assert result["runtime_mode"]["sensor_input"] == "provided"
    assert result["capability_states"]["thermal_anomaly_detection"] == "blocked"
    assert result["capability_states"]["multi_sensor_fusion"] == "blocked"
    assert result["capability_states"]["active_suppression_control"] == "blocked"
    assert result["capability_states"]["evacuation_guidance"] == "blocked"
    assert result["review_status"] == "manual_review_required"
    assert "multi_sensor_fusion_blocked" in result["reason"]
    assert "thermal_anomaly_detection_blocked" in result["reason"]
    assert "verified" not in {
        result["capability_states"]["thermal_anomaly_detection"],
        result["capability_states"]["multi_sensor_fusion"],
        result["capability_states"]["active_suppression_control"],
        result["capability_states"]["evacuation_guidance"],
    }


def test_blocked_capabilities_are_not_surfaced_as_verified(plugin, blank_frame):
    result = plugin.detect(blank_frame)

    blocked = set(result["blocked_capabilities"])
    assert "thermal_anomaly_detection" in blocked
    assert "multi_sensor_fusion" in blocked
    assert "active_suppression_control" in blocked
    assert "evacuation_guidance" in blocked
    assert all(result["capability_states"][cap] == "blocked" for cap in blocked)
