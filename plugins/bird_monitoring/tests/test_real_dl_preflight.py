"""Real-DL preflight 契约测试。

当前阶段没有真实 ONNX 模型，所以 preflight 期望的事实是：
- `performed=False`（未执行，因为 session 不可用）
- `passed=False`
- `get_runtime_status()` 暴露 `preflight` 字段
- 当提供虚假模型路径时，不会 crash，只会落入 simulation

同时通过 mock session 验证 preflight 判定逻辑（input/output shape 和 class map）。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class _FakeIO:
    def __init__(self, shape):
        self.shape = shape


class _FakeSession:
    def __init__(self, input_shape, output_shape):
        self._inputs = [_FakeIO(input_shape)]
        self._outputs = [_FakeIO(output_shape)]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


class TestPreflightSimulationBaseline:
    def test_healthcheck_exposes_preflight(self, plugin_instance):
        status = plugin_instance.healthcheck().details
        assert "preflight" in status
        preflight = status["preflight"]
        # 无模型 → preflight 未执行
        assert preflight["performed"] is False
        assert preflight["passed"] is False

    def test_detector_runtime_status_has_preflight_block(self, default_config):
        from plugins.bird_monitoring.detector import BirdDetector
        det = BirdDetector(default_config)
        rs = det.get_runtime_status()
        assert "preflight" in rs
        assert rs["preflight"]["performed"] is False

    def test_nonexistent_model_does_not_crash(self, default_config):
        from plugins.bird_monitoring.detector import BirdDetector
        cfg = dict(default_config)
        cfg["model"] = {"path": "nonexistent/path/to.onnx", "device": "cpu",
                        "input_size": [640, 640]}
        det = BirdDetector(cfg)
        assert det.real_model_loaded is False
        assert det.onnx_session_ready is False
        assert det.runtime_mode == "simulation"


class TestPreflightLogic:
    """直接校验 _preflight_onnx 的判定逻辑。"""

    def test_preflight_passes_on_valid_shapes(self, default_config):
        from plugins.bird_monitoring.detector import BirdDetector
        det = BirdDetector(default_config)
        # 伪造一个合法 session：input [1,3,640,640], output [1, 14, 8400]
        # CLASSES 有 10 项 → expected_channels = 4+10 = 14
        det.session = _FakeSession(
            input_shape=[1, 3, 640, 640],
            output_shape=[1, 14, 8400],
        )
        report = det._preflight_onnx()
        assert report["performed"] is True
        assert report["passed"] is True, report
        assert report["checks"]["channels_match"] is True

    def test_preflight_flags_input_shape_mismatch(self, default_config):
        from plugins.bird_monitoring.detector import BirdDetector
        det = BirdDetector(default_config)
        det.session = _FakeSession(
            input_shape=[1, 3, 320, 320],
            output_shape=[1, 14, 8400],
        )
        report = det._preflight_onnx()
        assert report["passed"] is False
        assert any("input_h_mismatch" in issue or "input_w_mismatch" in issue
                   for issue in report["issues"])

    def test_preflight_flags_class_channel_mismatch(self, default_config):
        from plugins.bird_monitoring.detector import BirdDetector
        det = BirdDetector(default_config)
        det.session = _FakeSession(
            input_shape=[1, 3, 640, 640],
            output_shape=[1, 9, 8400],  # only 5 classes implied
        )
        report = det._preflight_onnx()
        assert report["passed"] is False
        assert any("class_channel_mismatch" in issue for issue in report["issues"])

    def test_preflight_handles_dynamic_dims(self, default_config):
        """'batch' 或 -1 的动态维度不应被误判为 mismatch。"""
        from plugins.bird_monitoring.detector import BirdDetector
        det = BirdDetector(default_config)
        det.session = _FakeSession(
            input_shape=["batch", 3, 640, 640],
            output_shape=["batch", 14, 8400],
        )
        report = det._preflight_onnx()
        assert report["passed"] is True, report
