"""真实 ONNX preflight 与 real_dl 接入验证测试。"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from darkbreaker_sdk.interfaces import PluginContext
from darkbreaker_sdk.utils import load_plugin_config

from plugins.busbar_inspection.detector_enhanced import BusbarDetectorEnhanced
from plugins.busbar_inspection.plugin import Plugin


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
REAL_DETECTION_ONNX_B64 = (
    "CAcSFWJ1c2Jhcl9wcmVmbGlnaHRfdGVzdDr7AgqeAhIGb3V0cHV0IghDb25zdGFudCqJAgoFdmFsdWUq/AEIAQgHCAgQAUIPY29uc3RhbnRfb3V0cHV0SuABAACgQwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKBDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8EIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACgQgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAzM3M/CtcjPArXIzwK1yM8CtcjPArXIzwK1yM8CtcjPM3MTD0K1yM8CtcjPArXIzwK1yM8CtcjPArXIzwK1yM8CtejPArXIzwK1yM8CtcjPArXIzwK1yM8CtcjPArXIzygAQQSFmJ1c2Jhcl9yZWFsX2RsX2ZpeHR1cmVaIgoGaW1hZ2VzEhgKFggBEhIKAggBCgIIAwoDCIAFCgMIgAViHAoGb3V0cHV0EhIKEAgBEgwKAggBCgIIBwoCCAhCBAoAEA0="
)


def _load_default_config() -> dict:
    return load_plugin_config(CONFIG_PATH)


def _write_manifest(
    manifest_path: Path,
    *,
    class_names: list[str] | None,
    input_size: list[int] | None = None,
    output_format: str = "yolov8_detect",
    model_version: str = "busbar-test@p3",
) -> None:
    payload = {
        "model_version": model_version,
        "input_size": input_size or [640, 640],
        "output_format": output_format,
    }
    if class_names is not None:
        payload["class_names"] = class_names
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_invalid_onnx(model_path: Path) -> None:
    model_path.write_bytes(b"not-a-valid-onnx")


def _write_real_detection_onnx(
    model_path: Path,
) -> None:
    model_path.write_bytes(base64.b64decode(REAL_DETECTION_ONNX_B64))


def _make_context() -> PluginContext:
    return PluginContext(task_id="task-p3", site_id="site", device_id="device")


def _textured_frame(seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0,
        255,
        size=(480, 640, 3),
        dtype=np.uint8,
    )


def test_invalid_onnx_keeps_traditional_fallback(tmp_path: Path):
    """模型文件存在但 session 无法建立时，必须 fail fast。"""
    model_path = tmp_path / "invalid_busbar.onnx"
    manifest_path = tmp_path / "invalid_busbar.manifest.json"
    _write_invalid_onnx(model_path)
    _write_manifest(
        manifest_path,
        class_names=["pin_missing", "crack", "foreign_object"],
    )

    config = _load_default_config()
    config["model"]["model_path"] = str(model_path)

    detector = BusbarDetectorEnhanced(config)
    assert detector.initialize() is True

    runtime_status = detector.get_runtime_status()
    assert detector._dl_initialized is False
    assert runtime_status["runtime_mode"] == "traditional_fallback"
    assert runtime_status["model_file_exists"] is True
    assert runtime_status["real_model_loaded"] is False
    assert runtime_status["onnx_session_ready"] is False
    assert runtime_status["dl_preflight_checked"] is True
    assert runtime_status["dl_preflight_passed"] is False
    assert runtime_status["dl_failure_reason"] == "onnx_session_failed"
    assert runtime_status["manifest_exists"] is True
    assert runtime_status["class_map_exists"] is True
    assert runtime_status["class_map_compatible"] is True


def test_label_contract_mismatch_does_not_enter_real_dl(tmp_path: Path):
    """class map 与 label_contract 不兼容时，session_ready 也不得冒充 real_dl。"""
    model_path = tmp_path / "mismatch_busbar.onnx"
    manifest_path = tmp_path / "mismatch_busbar.manifest.json"
    _write_real_detection_onnx(model_path)
    _write_manifest(
        manifest_path,
        class_names=["pin_missing", "fitting_loose", "foreign_object"],
    )

    config = _load_default_config()
    config["model"]["model_path"] = str(model_path)

    detector = BusbarDetectorEnhanced(config)
    assert detector.initialize() is True

    runtime_status = detector.get_runtime_status()
    assert detector._dl_initialized is False
    assert runtime_status["runtime_mode"] == "traditional_fallback"
    assert runtime_status["real_model_loaded"] is False
    assert runtime_status["onnx_session_ready"] is True
    assert runtime_status["dl_preflight_passed"] is False
    assert runtime_status["dl_failure_reason"] == "label_contract_mismatch"
    assert runtime_status["class_map_compatible"] is False
    assert "fitting_loose" in runtime_status["validated_class_names"]
    assert "fitting_loose" not in runtime_status["runtime_supported_defect_labels"]


def test_real_dl_preflight_success_with_real_openable_onnx_fixture(tmp_path: Path):
    """只有真实可打开且契约兼容的 ONNX fixture 才允许切到 real_dl。"""
    model_path = tmp_path / "real_busbar.onnx"
    manifest_path = tmp_path / "real_busbar.manifest.json"
    _write_real_detection_onnx(model_path)
    _write_manifest(
        manifest_path,
        class_names=["pin_missing", "crack", "foreign_object"],
    )

    config = _load_default_config()
    config["model"]["model_path"] = str(model_path)
    config["runtime"]["providers"] = ["CPUExecutionProvider"]

    plugin = Plugin.create_standalone(config=config)
    health = plugin.healthcheck()

    assert health.healthy is True
    assert health.details["runtime_mode"] == "real_dl"
    assert health.details["real_model_loaded"] is True
    assert health.details["onnx_session_ready"] is True
    assert health.details["dl_preflight_checked"] is True
    assert health.details["dl_preflight_passed"] is True
    assert health.details["dl_failure_reason"] is None
    assert health.details["class_map_compatible"] is True
    assert health.details["output_structure_compatible"] is True

    results = plugin.infer(_textured_frame(), [], _make_context())

    assert len(results) >= 1
    assert any(result.label == "pin_missing" for result in results)
    first = results[0]
    assert first.metadata["runtime_mode"] == "real_dl"
    assert first.metadata["source"] == "dl"
    assert first.metadata["quality_gate_status"] in {"pass", "soft_fail"}
