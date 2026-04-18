"""bird_monitoring 插件契约测试 — 验证结果结构和输出规范"""
import pytest
import numpy as np


class TestResultStructure:
    """结果结构契约"""

    def test_no_bird_result_has_required_fields(
        self, plugin_instance, sample_frame, make_context, make_roi
    ):
        results = plugin_instance.infer(sample_frame, [make_roi()], make_context())
        for r in results:
            assert r.task_id is not None
            assert r.roi_id is not None
            assert r.label is not None
            assert r.model_version is not None
            assert r.code_version is not None
            assert isinstance(r.metadata, dict)

    def test_training_placeholders_present(
        self, plugin_instance, sample_frame, make_context, make_roi
    ):
        results = plugin_instance.infer(sample_frame, [make_roi()], make_context())
        for r in results:
            tp = r.metadata.get("training_placeholders", {})
            assert "hard_negative_candidate" in tp
            assert "hard_positive_candidate" in tp
            assert "suggested_label_for_dataset" in tp
            assert "annotation_status" in tp
            assert "model_placeholder" in tp

    def test_failure_results_keep_training_placeholders(
        self, plugin_instance, tiny_frame, make_context, make_roi
    ):
        results = plugin_instance.infer(tiny_frame, [make_roi()], make_context())
        assert results[0].label == "quality_failed"
        assert "training_placeholders" in results[0].metadata

    def test_runtime_mode_in_metadata(
        self, plugin_instance, sample_frame, make_context, make_roi
    ):
        results = plugin_instance.infer(sample_frame, [make_roi()], make_context())
        for r in results:
            assert "runtime_mode" in r.metadata

    def test_multiple_rois_produce_multiple_results(
        self, plugin_instance, sample_frame, make_context, make_roi
    ):
        rois = [make_roi("roi_1"), make_roi("roi_2"), make_roi("roi_3")]
        results = plugin_instance.infer(sample_frame, rois, make_context())
        assert len(results) == 3

    def test_empty_rois_returns_empty(
        self, plugin_instance, sample_frame, make_context
    ):
        results = plugin_instance.infer(sample_frame, [], make_context())
        assert results == []


class TestAlarmContract:
    """告警生成契约"""

    def test_review_required_generates_warning(self, plugin_instance):
        from darkbreaker_sdk.schemas import RecognitionResult, BoundingBox, AlarmLevel
        result = RecognitionResult(
            task_id="t1", site_id="s1", device_id="d1", component_id="c1",
            roi_id="r1", bbox=BoundingBox(x=0, y=0, width=1, height=1),
            label="review_required", confidence=0.3,
            value="需人工复核: 检测置信度低", model_version="1.0.0",
            code_version="test", metadata={},
        )
        alarms = plugin_instance.postprocess([result], [])
        assert len(alarms) == 1
        assert alarms[0].level == AlarmLevel.WARNING

    def test_unknown_bird_generates_warning(self, plugin_instance):
        from darkbreaker_sdk.schemas import RecognitionResult, BoundingBox, AlarmLevel
        result = RecognitionResult(
            task_id="t1", site_id="s1", device_id="d1", component_id="c1",
            roi_id="r1", bbox=BoundingBox(x=0, y=0, width=1, height=1),
            label="unknown_bird", confidence=0.6,
            value="未识别鸟类", model_version="1.0.0",
            code_version="test", metadata={},
        )
        alarms = plugin_instance.postprocess([result], [])
        assert len(alarms) == 1
        assert "未识别" in alarms[0].title

    def test_quality_failed_generates_warning(self, plugin_instance):
        from darkbreaker_sdk.schemas import RecognitionResult, BoundingBox, AlarmLevel
        result = RecognitionResult(
            task_id="t1", site_id="s1", device_id="d1", component_id="c1",
            roi_id="r1", bbox=BoundingBox(x=0, y=0, width=1, height=1),
            label="quality_failed", confidence=0.0,
            value="输入质量不合格", model_version="1.0.0",
            code_version="test", metadata={"quality_issues": ["图像过暗"]},
        )
        alarms = plugin_instance.postprocess([result], [])
        assert len(alarms) == 1
        assert alarms[0].level == AlarmLevel.WARNING

    def test_bird_danger_generates_error_alarm(self, plugin_instance):
        from darkbreaker_sdk.schemas import RecognitionResult, BoundingBox, AlarmLevel
        result = RecognitionResult(
            task_id="t1", site_id="s1", device_id="d1", component_id="c1",
            roi_id="r1", bbox=BoundingBox(x=0, y=0, width=1, height=1),
            label="bird_danger", confidence=0.8,
            value="风险评分: 70.0", model_version="1.0.0",
            code_version="test",
            metadata={"risk_level": "danger", "species_name": "老鹰",
                       "distance_m": 4.0, "risk_score": 70.0,
                       "deterrent_suggestion": {}},
        )
        alarms = plugin_instance.postprocess([result], [])
        assert len(alarms) == 1
        assert alarms[0].level == AlarmLevel.ERROR

    def test_bird_safe_no_alarm(self, plugin_instance):
        from darkbreaker_sdk.schemas import RecognitionResult, BoundingBox
        result = RecognitionResult(
            task_id="t1", site_id="s1", device_id="d1", component_id="c1",
            roi_id="r1", bbox=BoundingBox(x=0, y=0, width=1, height=1),
            label="bird_safe", confidence=0.9,
            value="风险评分: 10.0", model_version="1.0.0",
            code_version="test",
            metadata={"risk_level": "safe"},
        )
        alarms = plugin_instance.postprocess([result], [])
        assert alarms == []


class TestDeterrentSuggestion:
    """驱离建议输出"""

    def test_safe_bird_no_deterrent(self, plugin_instance):
        from plugins.bird_monitoring.plugin import BirdDetection
        det = BirdDetection(
            track_id=1, species_id="sparrow", species_name="麻雀",
            confidence=0.9, bbox={}, risk_level="safe", risk_score=10.0,
            distance_to_line_m=20.0,
        )
        suggestion = plugin_instance._make_deterrent_suggestion(det)
        assert suggestion["action"] == "none"

    def test_danger_bird_suggests_deterrent(self, plugin_instance):
        from plugins.bird_monitoring.plugin import BirdDetection
        det = BirdDetection(
            track_id=2, species_id="eagle", species_name="老鹰",
            confidence=0.9, bbox={}, risk_level="danger", risk_score=70.0,
            distance_to_line_m=4.0,
        )
        suggestion = plugin_instance._make_deterrent_suggestion(det)
        assert suggestion["action"] == "suggest_deterrent"
        assert "acoustic" in suggestion.get("methods", [])
