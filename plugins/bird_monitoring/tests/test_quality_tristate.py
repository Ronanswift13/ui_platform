"""Quality gate 三态契约测试 (pass / soft_fail / hard_fail)。

约束:
- hard_fail 必须阻断推理，输出 label=`quality_failed`
- soft_fail 必须继续推理，但结果被强制进入 `review_required`
- pass 不改变现有行为
"""
from __future__ import annotations

import numpy as np
import pytest


class TestQualityGateTriState:
    def test_pass_on_clean_frame(self, plugin_instance, sample_frame, make_roi):
        quality = plugin_instance._assess_input_quality(sample_frame, make_roi())
        assert quality["status"] in ("pass", "soft_fail")
        # sample_frame 是 0-255 均匀随机 → overall_score 合理；不应 hard_fail
        assert quality["is_valid"] is True

    def test_hard_fail_on_dark_frame(
        self, plugin_instance, dark_frame, make_context, make_roi
    ):
        results = plugin_instance.infer(dark_frame, [make_roi()], make_context())
        assert len(results) == 1
        assert results[0].label == "quality_failed"
        q = results[0].metadata["input_quality"]
        assert q["status"] == "hard_fail"
        assert q["is_valid"] is False
        assert q.get("hard_issues"), "hard_fail 必须填充 hard_issues"

    def test_hard_fail_on_tiny_frame(self, plugin_instance, make_roi):
        tiny = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        plugin_instance._config["quality"]["min_dimension"] = 100
        quality = plugin_instance._assess_input_quality(tiny, make_roi())
        assert quality["status"] == "hard_fail"
        assert quality["is_valid"] is False

    def test_soft_fail_forces_review_required(
        self, plugin_instance, sample_frame, make_context, make_roi
    ):
        """把 soft 阈值抬到 1.0 即可把 sample_frame 推入 soft_fail。"""
        plugin_instance._config["quality"]["soft_overall_threshold"] = 1.0
        plugin_instance._config["quality"]["soft_clarity_threshold"] = 1.0
        plugin_instance._config["quality"]["soft_brightness_threshold"] = 1.0

        quality = plugin_instance._assess_input_quality(sample_frame, make_roi())
        assert quality["status"] == "soft_fail"
        assert quality["is_valid"] is True
        assert quality["soft_issues"], "soft_fail 必须填充 soft_issues"

        # infer 阶段: simulation 默认没有 detections，因此只有 no_bird；
        # no_bird 结果 metadata 也要带 review_required 标记，便于 UI 提示
        results = plugin_instance.infer(
            sample_frame, [make_roi()], make_context()
        )
        assert len(results) == 1
        result = results[0]
        assert result.label == "no_bird"
        assert result.metadata["quality_status"] == "soft_fail"
        assert result.metadata.get("review_required") is True

    def test_quality_failed_alarm_level_is_warning(
        self, plugin_instance, dark_frame, make_context, make_roi
    ):
        results = plugin_instance.infer(dark_frame, [make_roi()], make_context())
        alarms = plugin_instance.postprocess(results, [])
        assert len(alarms) == 1
        assert alarms[0].level.name == "WARNING"

    def test_soft_fail_does_not_emit_quality_failed(
        self, plugin_instance, sample_frame, make_context, make_roi
    ):
        plugin_instance._config["quality"]["soft_overall_threshold"] = 1.0
        plugin_instance._config["quality"]["soft_clarity_threshold"] = 1.0
        plugin_instance._config["quality"]["soft_brightness_threshold"] = 1.0
        results = plugin_instance.infer(
            sample_frame, [make_roi()], make_context()
        )
        labels = [r.label for r in results]
        assert "quality_failed" not in labels
