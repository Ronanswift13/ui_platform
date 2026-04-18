"""质量门禁三态与 simulator 对齐测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from darkbreaker_sdk.interfaces import PluginContext
from darkbreaker_sdk.utils import load_plugin_config

from plugins.busbar_inspection.detector_enhanced import (
    BusbarDetection,
    BusbarDefectType,
    BusbarDetectorEnhanced,
    DetectROIResult,
    QualityGateDecision,
    QualityGateResult,
    QualityGateStatus,
    ZoomSuggestion,
)
from plugins.busbar_inspection.plugin import Plugin
from plugins.busbar_inspection.reason_code_mapper import ReasonCodeMapper
from plugins.busbar_inspection.standalone.busbar_simulator import BusbarSimulator


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_default_config():
    return load_plugin_config(CONFIG_PATH)


def make_context() -> PluginContext:
    return PluginContext(task_id="t", site_id="s", device_id="d")


def test_simulator_quality_scenarios_align_with_real_gate():
    """simulator 场景期望值必须与真实 check_quality_gate() 一致。"""
    detector = BusbarDetectorEnhanced(load_default_config())
    simulator = BusbarSimulator(seed=42)

    for scenario_id in [
        "quality_blur",
        "quality_occlude",
        "quality_overexpose",
        "rainy_inspection",
        "normal_clear",
    ]:
        assert simulator.load_scenario(scenario_id) is True
        step = simulator.step()
        frame = simulator.render_frame(defects=step.defects)
        real_quality = detector.check_quality_gate(frame)
        expected = simulator.get_current_scenario()
        assert expected is not None

        real_gate_status = getattr(
            real_quality.quality_gate_status,
            "value",
            real_quality.quality_gate_status,
        )
        real_reason_code = (
            ReasonCodeMapper.map_to_external(int(real_quality.reason_code))
            if real_quality.reason_code
            else None
        )

        assert step.quality.quality_gate_status == expected["expected_quality_gate_status"]
        assert step.quality.status == expected["expected_quality_status"]
        assert step.quality.reason_code == expected["expected_reason_code"]

        assert real_gate_status == expected["expected_quality_gate_status"]
        assert real_quality.status.value == expected["expected_quality_status"]
        assert real_reason_code == expected["expected_reason_code"]


def test_soft_fail_detector_path_keeps_runtime_chain_alive():
    """SOFT_FAIL 不得提前阻断 detector 主链。"""
    detector = BusbarDetectorEnhanced(load_default_config())
    detector.initialize()

    simulator = BusbarSimulator(seed=42)
    assert simulator.load_scenario("rainy_inspection") is True
    step = simulator.step()
    frame = simulator.render_frame(defects=step.defects)

    result = detector.detect_roi(frame)

    assert result.quality is not None
    assert result.debug_info is not None
    assert result.debug_info["quality_gate_status"] == "soft_fail"
    assert result.debug_info["quality_gate"] == "soft_failed"
    assert result.debug_info["runtime_mode"] == "traditional_fallback"
    assert result.debug_info["runtime_mode"] != "quality_blocked"
    assert "detection_count" in result.debug_info


def test_soft_fail_detection_results_require_review():
    """SOFT_FAIL + detection 时必须输出缺陷并标记 review_required。"""
    plugin = Plugin.create_standalone()
    frame = np.random.default_rng(7).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

    quality = QualityGateResult(
        status=QualityGateStatus.FAIL_LOW_CONTRAST,
        clarity_score=0.42,
        brightness_score=0.48,
        contrast_score=0.18,
        occlusion_ratio=0.24,
        quality_gate_status=QualityGateDecision.SOFT_FAIL,
        reason_code="1005",
        failure_reason="对比度过低",
        metadata={
            "quality_gate_status": "soft_fail",
            "failure_reason": "对比度过低",
            "suggested_action": "RECAPTURE",
        },
    )
    detection = BusbarDetection(
        defect_type=BusbarDefectType.CRACK,
        bbox={"x": 0.15, "y": 0.20, "width": 0.12, "height": 0.04},
        confidence=0.91,
        class_name="crack",
        metadata={"source": "traditional"},
    )

    plugin._detector.detect_roi = lambda *_args, **_kwargs: DetectROIResult(
        detections=[detection],
        quality=quality,
        zoom_suggestion=ZoomSuggestion(
            current_zoom=1.0,
            recommended_zoom=1.0,
            reason="none",
        ),
        reason_code=1005,
        debug_info={
            "quality_gate": "soft_failed",
            "quality_gate_status": "soft_fail",
            "quality_failure_reason": "对比度过低",
            "quality_reason_code": "1005",
            "quality_suggested_action": "RECAPTURE",
            "runtime_mode": "traditional_fallback",
        },
    )

    results = plugin.infer(frame, [], make_context())

    assert len(results) == 1
    result = results[0]
    assert result.label == "crack"
    assert result.metadata["review_status"] == "review_required"
    assert result.metadata["quality_gate_status"] == "soft_fail"
    assert result.metadata["runtime_mode"] == "traditional_fallback"
    assert result.metadata["suggested_action"] == "RECAPTURE"
    assert result.metadata["quality"]["quality_gate_status"] == "soft_fail"


def test_soft_fail_without_detection_emits_reviewable_quality_failed():
    """SOFT_FAIL 无稳定检测时允许输出 quality_failed，但必须和 HARD_FAIL 区分。"""
    plugin = Plugin.create_standalone()
    frame = np.random.default_rng(8).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

    quality = QualityGateResult(
        status=QualityGateStatus.FAIL_LOW_CONTRAST,
        clarity_score=0.40,
        brightness_score=0.50,
        contrast_score=0.16,
        occlusion_ratio=0.22,
        quality_gate_status=QualityGateDecision.SOFT_FAIL,
        reason_code="1005",
        failure_reason="对比度过低",
        metadata={
            "quality_gate_status": "soft_fail",
            "failure_reason": "对比度过低",
            "suggested_action": "RECAPTURE",
        },
    )

    plugin._detector.detect_roi = lambda *_args, **_kwargs: DetectROIResult(
        detections=[],
        quality=quality,
        zoom_suggestion=None,
        reason_code=1005,
        debug_info={
            "quality_gate": "soft_failed",
            "quality_gate_status": "soft_fail",
            "quality_failure_reason": "对比度过低",
            "quality_reason_code": "1005",
            "quality_suggested_action": "RECAPTURE",
            "runtime_mode": "traditional_fallback",
        },
    )

    results = plugin.infer(frame, [], make_context())

    assert len(results) == 1
    result = results[0]
    assert result.label == "quality_failed"
    assert result.metadata["quality_gate_status"] == "soft_fail"
    assert result.metadata["review_status"] == "review_required"
    assert result.metadata["runtime_mode"] == "traditional_fallback"
    assert result.metadata["suggested_action"] == "RECAPTURE"


def test_hard_fail_outputs_quality_failed_and_quality_blocked_runtime_mode():
    """HARD_FAIL 时必须阻断并输出 quality_blocked。"""
    plugin = Plugin.create_standalone()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = plugin.infer(frame, [], make_context())

    assert len(results) >= 1
    result = results[0]
    assert result.label == "quality_failed"
    assert result.metadata["quality_gate_status"] == "hard_fail"
    assert result.metadata["review_status"] == "blocked"
    assert result.metadata["runtime_mode"] == "quality_blocked"
