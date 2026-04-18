"""bird_monitoring 输入质量评估测试"""
import pytest
import numpy as np


class TestInputQuality:
    """输入质量评估"""

    def test_valid_frame_passes(self, plugin_instance, sample_frame, make_roi):
        roi = make_roi()
        quality = plugin_instance._assess_input_quality(sample_frame, roi)
        assert quality["is_valid"] is True

    def test_none_frame_fails(self, plugin_instance, make_roi):
        quality = plugin_instance._assess_input_quality(None, make_roi())
        assert quality["is_valid"] is False
        assert any("为空" in issue for issue in quality["issues"])

    def test_empty_frame_fails(self, plugin_instance, make_roi):
        empty = np.array([], dtype=np.uint8)
        quality = plugin_instance._assess_input_quality(empty, make_roi())
        assert quality["is_valid"] is False

    def test_tiny_frame_has_issues(self, plugin_instance, tiny_frame, make_roi):
        quality = plugin_instance._assess_input_quality(tiny_frame, make_roi())
        assert quality["is_valid"] is False
        assert len(quality["issues"]) > 0
        assert any("尺寸" in issue for issue in quality["issues"])

    def test_quality_fail_result_structure(
        self, plugin_instance, tiny_frame, make_context, make_roi
    ):
        """质量失败的 infer 结果结构"""
        # 使用极小帧但设置 min_dimension 高于帧尺寸
        plugin_instance._config["quality"] = {"min_dimension": 100}
        ctx = make_context()
        results = plugin_instance.infer(tiny_frame, [make_roi()], ctx)
        # 应包含质量信息
        assert len(results) >= 1
        assert results[0].label == "quality_failed"
        assert "training_placeholders" in results[0].metadata

    def test_dark_frame_returns_quality_failed(
        self, plugin_instance, dark_frame, make_context, make_roi
    ):
        results = plugin_instance.infer(dark_frame, [make_roi()], make_context())
        assert len(results) == 1
        assert results[0].label == "quality_failed"
        assert any("暗" in issue for issue in results[0].metadata["quality_issues"])


class TestDetectorQuality:
    """检测器层质量评估"""

    def test_detector_assess_image_quality(self, default_config, sample_frame):
        from plugins.bird_monitoring.detector import BirdDetector
        det = BirdDetector(default_config)
        quality = det.assess_image_quality(sample_frame)
        assert "is_valid" in quality
        assert "overall_score" in quality
        assert isinstance(quality["issues"], list)

    def test_dark_frame_detected(self, default_config, dark_frame):
        from plugins.bird_monitoring.detector import BirdDetector
        det = BirdDetector(default_config)
        quality = det.assess_image_quality(dark_frame)
        assert any("暗" in issue for issue in quality["issues"])

    def test_runtime_mode_property(self, default_config):
        from plugins.bird_monitoring.detector import BirdDetector
        det = BirdDetector(default_config)
        assert det.runtime_mode == "simulation"  # 无模型
