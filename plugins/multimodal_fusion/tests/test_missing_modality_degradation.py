import time

from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin


def output(
    modality: str,
    label: str = "normal",
    severity: str = "normal",
    confidence: float = 0.9,
    evidence_path: str = "fixtures/simulated/evidence.json",
) -> dict:
    return {
        "modality": modality,
        "plugin_id": f"{modality}_fixture_simulated",
        "task_id": "fusion-degradation-contract",
        "timestamp": time.time(),
        "results": [{
            "label": label,
            "severity": severity,
            "confidence": confidence,
            "value": {"simulated": True},
            "evidence_path": evidence_path,
            "component_id": f"{modality}_component",
            "metadata": {"simulated": True},
        }],
        "alarms": [],
        "metadata": {"simulated": True},
    }


def test_single_modality_input_degrades_missing_modalities_without_failure():
    plugin = MultimodalFusionPlugin.create_standalone()
    plugin._use_enhanced_engine = False

    result = plugin.process([output("visual")])

    assert result["success"] is True
    assert result["fused_status"] == "normal"
    assert result["contributing_modalities"] == ["visual"]
    assert set(result["missing_modalities"]) == {"thermal", "acoustic", "gas", "hyperspectral"}
    assert not result["alarms"]


def test_conflicting_modalities_report_conflict_status():
    plugin = MultimodalFusionPlugin.create_standalone()
    plugin._use_enhanced_engine = False

    result = plugin.process([
        output("visual", label="normal", severity="normal", confidence=0.95),
        output("gas", label="gas_alarm", severity="critical", confidence=0.91),
    ])

    assert result["success"] is True
    assert result["conflict_status"] == "conflict_detected"
    assert result["fused_status"] == "critical"


def test_missing_upstream_fields_are_defaulted_with_degradation_reason():
    plugin = MultimodalFusionPlugin.create_standalone()
    plugin._use_enhanced_engine = False

    result = plugin.process([{
        "modality": "acoustic",
        "plugin_id": "acoustic_fixture_simulated",
        "task_id": "missing-fields",
        "timestamp": time.time(),
        "results": [{
            "value": {"simulated": True},
            "component_id": "audio_channel",
            "metadata": {"simulated": True},
        }],
        "metadata": {"simulated": True},
    }])

    assert result["success"] is True
    reasons = result["metadata"]["degradation_reasons"][0]["reasons"]
    assert "missing_label_defaulted" in reasons
    assert "missing_confidence_defaulted" in reasons
    assert "missing_evidence_path" in reasons


def test_enhanced_engine_disabled_still_smokes_with_rule_fusion():
    plugin = MultimodalFusionPlugin.create_standalone()
    plugin._use_enhanced_engine = False
    plugin.enhanced_engine = None

    result = plugin.process([output("gas", label="gas_warning", severity="warning", confidence=0.78)])

    assert result["success"] is True
    assert result["metadata"]["runtime_mode"] == "rule_fusion"
    assert result["fused_status"] == "warning"
