import time

from darkbreaker_sdk.schemas import Alarm, RecognitionResult
from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin


def abnormal_output(modality: str, confidence: float) -> dict:
    return {
        "modality": modality,
        "plugin_id": f"{modality}_fixture_simulated",
        "task_id": "fusion-output-contract",
        "timestamp": time.time(),
        "results": [{
            "task_id": "fusion-output-contract",
            "site_id": "fixture-site",
            "device_id": "transformer-01",
            "component_id": f"{modality}_component",
            "roi_id": f"{modality}_roi",
            "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
            "label": "overheating",
            "value": {"simulated": True, "source": modality},
            "confidence": confidence,
            "evidence_path": f"fixtures/simulated/{modality}_overheating.json",
            "model_version": "fixture",
            "code_version": "fixture",
            "timestamp": time.time(),
            "metadata": {"simulated": True, "severity": "abnormal"},
        }],
        "alarms": [],
        "metadata": {"simulated": True, "severity": "abnormal"},
    }


def test_consistent_multimodal_abnormal_input_escalates_to_critical():
    plugin = MultimodalFusionPlugin.create_standalone()
    plugin._use_enhanced_engine = False

    result = plugin.process([
        abnormal_output("visual", 0.92),
        abnormal_output("gas", 0.89),
    ])

    assert result["success"] is True
    assert result["fused_label"] == "overheating"
    assert result["fused_status"] == "critical"
    assert result["fusion_confidence"] > 0.8
    assert result["conflict_status"] == "none"


def test_output_contains_required_standard_fields_and_sdk_mappings():
    plugin = MultimodalFusionPlugin.create_standalone()
    plugin._use_enhanced_engine = False

    result = plugin.process([
        abnormal_output("visual", 0.91),
        abnormal_output("gas", 0.88),
    ])

    for key in (
        "fused_label",
        "fused_status",
        "fusion_confidence",
        "contributing_modalities",
        "missing_modalities",
        "conflict_status",
        "evidence_chain",
        "recommended_actions",
        "metadata",
    ):
        assert key in result

    metadata = result["metadata"]
    assert metadata["runtime_mode"] == "rule_fusion"
    assert metadata["fusion_strategy"] == "hybrid"
    assert metadata["modality_weights"]["visual"] == 0.25
    assert metadata["algorithm_stage"] == "stage_1_rule_fusion_contract"
    assert "bayesian_fusion" in metadata["upgrade_placeholders"]

    RecognitionResult.model_validate(result["results"][0])
    assert result["alarms"]
    Alarm.model_validate(result["alarms"][0])
