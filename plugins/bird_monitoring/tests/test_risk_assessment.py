"""bird_monitoring 风险评估测试"""
import pytest
from plugins.bird_monitoring.plugin import BirdDetection, BirdMonitoringPlugin


class TestRiskAssessment:
    """风险评估逻辑"""

    def test_safe_distance_returns_safe(self, plugin_instance, make_context):
        det = BirdDetection(
            track_id=1, species_id="sparrow", species_name="麻雀",
            confidence=0.9, bbox={}, distance_to_line_m=20.0, status="flying",
        )
        ctx = make_context()
        score, level = plugin_instance._assess_risk(det, ctx)
        assert level == "safe"
        assert score < 40

    def test_close_eagle_returns_critical_or_danger(self, plugin_instance, make_context):
        det = BirdDetection(
            track_id=2, species_id="eagle", species_name="老鹰",
            confidence=0.9, bbox={}, distance_to_line_m=3.0, status="perched",
        )
        ctx = make_context()
        score, level = plugin_instance._assess_risk(det, ctx)
        assert level in ("danger", "critical")
        assert score >= 60

    def test_unknown_bird_close_returns_danger(self, plugin_instance, make_context):
        det = BirdDetection(
            track_id=3, species_id="unknown_bird", species_name="未知鸟类",
            confidence=0.5, bbox={}, distance_to_line_m=3.0, status="flying",
        )
        ctx = make_context()
        score, level = plugin_instance._assess_risk(det, ctx)
        assert level == "danger"

    def test_unknown_bird_far_returns_safe(self, plugin_instance, make_context):
        det = BirdDetection(
            track_id=4, species_id="unknown_bird", species_name="未知鸟类",
            confidence=0.5, bbox={}, distance_to_line_m=20.0, status="flying",
        )
        ctx = make_context()
        score, level = plugin_instance._assess_risk(det, ctx)
        assert level == "safe"

    def test_score_to_level_boundaries(self):
        assert BirdMonitoringPlugin._score_to_level(0) == "safe"
        assert BirdMonitoringPlugin._score_to_level(39) == "safe"
        assert BirdMonitoringPlugin._score_to_level(40) == "warning"
        assert BirdMonitoringPlugin._score_to_level(59) == "warning"
        assert BirdMonitoringPlugin._score_to_level(60) == "danger"
        assert BirdMonitoringPlugin._score_to_level(79) == "danger"
        assert BirdMonitoringPlugin._score_to_level(80) == "critical"
        assert BirdMonitoringPlugin._score_to_level(100) == "critical"


class TestSpeciesIdentification:
    """物种识别逻辑"""

    def test_known_species_returns_correct_id(self, plugin_instance):
        det = {"class_name": "eagle", "confidence": 0.9}
        species_id, conf = plugin_instance._identify_species(det)
        assert species_id == "eagle"
        assert conf == 0.9

    def test_unknown_class_returns_unknown_bird(self, plugin_instance):
        det = {"class_name": "penguin", "confidence": 0.8}
        species_id, conf = plugin_instance._identify_species(det)
        assert species_id == "unknown_bird"
        assert conf == 0.0

    def test_empty_class_returns_unknown_bird(self, plugin_instance):
        det = {"class_name": "", "confidence": 0.7}
        species_id, conf = plugin_instance._identify_species(det)
        assert species_id == "unknown_bird"

    def test_bird_generic_maps_to_unknown(self, plugin_instance):
        det = {"class_name": "bird_generic", "confidence": 0.6}
        species_id, conf = plugin_instance._identify_species(det)
        assert species_id == "unknown_bird"

    def test_case_insensitive(self, plugin_instance):
        det = {"class_name": "Eagle", "confidence": 0.85}
        species_id, _ = plugin_instance._identify_species(det)
        assert species_id == "eagle"
