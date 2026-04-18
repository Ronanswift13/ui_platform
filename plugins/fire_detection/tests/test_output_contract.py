"""fire_detection 最小输出契约测试。"""

from __future__ import annotations


def _run_until_confirmed(plugin, frame, **kwargs):
    result = None
    for _ in range(3):
        result = plugin.detect(frame, **kwargs)
    assert result is not None
    return result


def _assert_contract_fields(result):
    required = {
        "plugin_id",
        "plugin_version",
        "success",
        "status",
        "severity",
        "confidence",
        "reason",
        "recommended_action",
        "review_status",
        "runtime_mode",
        "capability_states",
        "blocked_capabilities",
        "semantic_type",
        "is_real_detection",
        "metadata",
    }
    missing = required - result.keys()
    assert not missing, f"missing keys: {sorted(missing)}"
    assert isinstance(result["confidence"], float)
    assert isinstance(result["reason"], str) and result["reason"]
    assert isinstance(result["recommended_action"], list) and result["recommended_action"]
    assert isinstance(result["runtime_mode"], dict)
    assert isinstance(result["capability_states"], dict)


def test_fire_output_contract(plugin, fire_frame):
    result = _run_until_confirmed(plugin, fire_frame)

    _assert_contract_fields(result)
    assert result["success"] is True
    assert result["semantic_type"] == "visual_detection"
    assert result["is_real_detection"] is True
    assert result["runtime_mode"]["analysis_mode"] == "simulation_only"
    assert result["detections"], "expected simulated fire detection after confirmation frames"
    assert result["detections"][0]["type"] == "fire"
    assert result["severity"] == result["fire_level"]
    assert "simulation_mode_only" in result["reason"]


def test_smoke_output_contract(plugin, smoke_frame):
    result = _run_until_confirmed(plugin, smoke_frame)

    _assert_contract_fields(result)
    assert result["success"] is True
    assert result["detections"], "expected simulated smoke detection after confirmation frames"
    assert result["detections"][0]["type"] == "smoke"
    assert result["severity"] == result["fire_level"]
    assert result["confidence"] >= 0.0
    assert result["reason"]


def test_failure_is_explainable(plugin):
    result = plugin.detect(None)

    _assert_contract_fields(result)
    assert result["success"] is False
    assert result["status"] == "error"
    assert result["reason"] == "invalid_frame_input"
    assert result["metadata"]["failure_reason"] == "无效输入帧"
    assert "provide_non_empty_bgr_frame" in result["recommended_action"]
