from pathlib import Path

from darkbreaker_sdk.interfaces import PluginContext
from darkbreaker_sdk.utils import load_plugin_config

from detector_enhanced import (
    BusbarDefectType,
    BusbarDetection,
    BusbarDetectorEnhanced,
)
from plugins.busbar_inspection.plugin import Plugin
from plugins.busbar_inspection.standalone.busbar_simulator import BusbarSimulator


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "standalone"
    / "templates"
    / "busbar_inspection.html"
)


def load_default_config():
    return load_plugin_config(CONFIG_PATH)


def test_crack_foreign_conflict_marks_review_required():
    detector = BusbarDetectorEnhanced(load_default_config())
    crack = BusbarDetection(
        defect_type=BusbarDefectType.CRACK,
        bbox={"x": 0.20, "y": 0.20, "width": 0.12, "height": 0.04},
        confidence=0.64,
        class_name="crack",
        metadata={"source": "traditional", "axis_aligned": False},
    )
    foreign = BusbarDetection(
        defect_type=BusbarDefectType.FOREIGN_OBJECT,
        bbox={"x": 0.21, "y": 0.21, "width": 0.13, "height": 0.05},
        confidence=0.62,
        class_name="foreign_object",
        metadata={"source": "traditional"},
    )

    processed = detector._apply_decision_postprocess(
        [crack, foreign],
        roi_type="insulator_string",
    )

    assert len(processed) == 1
    winner = processed[0]
    assert winner.metadata["review_status"] == "review_required"
    assert winner.metadata["suggested_action"] == "MANUAL_REVIEW"
    assert set(winner.metadata["candidate_labels"]) == {"crack", "foreign_object"}
    assert winner.reason_code == "2002"


def test_foreign_object_scene_does_not_emit_crack_flood():
    plugin = Plugin.create_standalone()
    ctx = PluginContext(task_id="t", site_id="s", device_id="d")
    sim = BusbarSimulator(seed=42)
    assert sim.load_scenario("foreign_object") is True
    step = sim.step()
    frame = sim.render_frame(defects=step.defects)

    results = plugin.infer(frame, [], ctx)

    crack_count = sum(result.label == "crack" for result in results)
    foreign_count = sum(result.label == "foreign_object" for result in results)
    assert crack_count == 0
    assert foreign_count >= 1


def test_mixed_defects_results_expose_detection_detail_fields():
    plugin = Plugin.create_standalone()
    ctx = PluginContext(task_id="t", site_id="s", device_id="d")
    sim = BusbarSimulator(seed=42)
    assert sim.load_scenario("mixed_defects") is True
    step = sim.step()
    frame = sim.render_frame(defects=step.defects)

    results = plugin.infer(frame, [], ctx)

    assert results
    detection_ids = [result.metadata["detection_id"] for result in results]
    assert len(detection_ids) == len(set(detection_ids))
    for result in results:
        assert "detection_id" in result.metadata
        assert "candidate_labels" in result.metadata
        assert "review_status" in result.metadata
        assert "source" in result.metadata
        assert isinstance(result.metadata["candidate_labels"], list)


def test_simulator_api_exposes_scene_comparison():
    sim = BusbarSimulator(seed=42)
    assert sim.load_scenario("mixed_defects") is True
    step = sim.step()

    payload = sim.result_to_api_format(step)
    comparison = payload["scene_comparison"]

    assert payload["summary"]["scenario_name"] == step.scenario_name
    assert comparison["scenario_name"] == step.scenario_name
    assert "expected_defects" in comparison
    assert "expected_quality_gate_status" in comparison
    assert "actual_quality_gate_status" in comparison
    assert "quality_alignment" in comparison
    assert "actual_detections" in comparison
    assert "false_positives" in comparison
    assert "false_negatives" in comparison
    assert "conflicts" in comparison


def test_live_template_uses_image_summary_not_results0():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "data.results?.[0]" not in template
    assert "buildImageSummary" in template
    assert "detection_id" in template
    assert "candidate_labels" in template
    assert "review_status" in template
