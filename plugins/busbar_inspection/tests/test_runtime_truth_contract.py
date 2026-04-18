"""运行真实性契约测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plugins.busbar_inspection.detector_enhanced import BusbarDetectorEnhanced
from plugins.busbar_inspection.plugin import Plugin


def test_missing_model_does_not_claim_real_dl_or_initialized():
    """缺失模型文件时，禁止把 simulation/空跑标记成已加载 DL。"""
    detector = BusbarDetectorEnhanced(
        {
            "model": {"model_path": "models/busbar_det.onnx"},
            "runtime": {"providers": ["CPUExecutionProvider"]},
            "thresholds": {"conf_thr": 0.25, "nms_iou": 0.50},
            "quality": {"blur_thr": 0.35},
        }
    )

    initialized = detector.initialize()
    runtime_status = detector.get_runtime_status()

    assert initialized is True
    assert detector._dl_initialized is False
    assert runtime_status["runtime_mode"] != "real_dl"
    assert runtime_status["runtime_mode"] == "traditional_fallback"
    assert runtime_status["model_path_configured"] == "models/busbar_det.onnx"
    assert runtime_status["model_file_exists"] is False
    assert runtime_status["real_model_loaded"] is False
    assert runtime_status["onnx_session_ready"] is False
    assert runtime_status["dl_preflight_checked"] is True
    assert runtime_status["dl_preflight_passed"] is False
    assert runtime_status["dl_failure_reason"] == "model_missing"


def test_quality_blocked_runtime_mode_is_exposed_by_detector():
    """quality gate 阻断时，debug_info 必须明确暴露 quality_blocked。"""
    detector = BusbarDetectorEnhanced(
        {
            "model": {"model_path": "models/busbar_det.onnx"},
            "runtime": {"providers": ["CPUExecutionProvider"]},
            "thresholds": {"conf_thr": 0.25, "nms_iou": 0.50},
            "quality": {"blur_thr": 0.35},
        }
    )
    detector.initialize()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect_roi(frame)

    assert result.quality is not None
    assert result.debug_info is not None
    assert result.debug_info["quality_gate"] == "failed"
    assert result.debug_info["quality_gate_status"] == "hard_fail"
    assert result.debug_info["runtime_mode"] == "quality_blocked"


def test_healthcheck_exposes_runtime_truth_fields():
    """plugin.healthcheck() 必须暴露真实运行模式与模型状态。"""
    plugin = Plugin.create_standalone()

    health = plugin.healthcheck()

    assert "runtime_mode" in health.details
    assert "model_path_configured" in health.details
    assert "model_path_resolved" in health.details
    assert "model_file_exists" in health.details
    assert "real_model_loaded" in health.details
    assert "onnx_session_ready" in health.details
    assert "fallback_enabled" in health.details
    assert health.details["runtime_mode"] == "traditional_fallback"
    assert health.details["model_path_configured"] == "models/busbar_det.onnx"
    assert health.details["model_path_resolved"] == str(
        Path(plugin.plugin_dir) / "models" / "busbar_det.onnx"
    )
    assert health.details["model_file_exists"] is False
    assert health.details["real_model_loaded"] is False
    assert health.details["onnx_session_ready"] is False
    assert health.details["dl_preflight_checked"] is True
    assert health.details["dl_preflight_passed"] is False
    assert health.details["dl_failure_reason"] == "model_missing"
