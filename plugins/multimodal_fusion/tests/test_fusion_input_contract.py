import time

from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin


def simulated_output(
    modality: str,
    label: str = "normal",
    severity: str = "normal",
    confidence: float = 0.9,
) -> dict:
    return {
        "modality": modality,
        "plugin_id": f"{modality}_fixture_simulated",
        "task_id": "fusion-input-contract",
        "timestamp": time.time(),
        "results": [
            {
                "task_id": "fusion-input-contract",
                "site_id": "fixture-site",
                "device_id": "fixture-device",
                "component_id": f"{modality}_component",
                "roi_id": f"{modality}_roi",
                "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
                "label": label,
                "value": {"simulated": True},
                "confidence": confidence,
                "evidence_path": f"fixtures/simulated/{modality}.json",
                "model_version": "fixture",
                "code_version": "fixture",
                "timestamp": time.time(),
                "metadata": {"simulated": True, "modality": modality},
            }
        ],
        "alarms": [],
        "metadata": {"simulated": True, "severity": severity},
    }


def test_accepts_list_of_plugin_output_equivalent_dicts():
    plugin = MultimodalFusionPlugin.create_standalone()
    plugin._use_enhanced_engine = False

    result = plugin.process([
        simulated_output("visual"),
        simulated_output("gas", label="gas_threshold_warning", severity="warning", confidence=0.74),
    ])

    assert result["success"] is True
    assert set(result["contributing_modalities"]) == {"visual", "gas"}
    assert "thermal" in result["missing_modalities"]
    assert result["evidence_chain"][0]["source_plugin_id"] == "visual_fixture_simulated"
    assert result["evidence_chain"][0]["evidence_path"] == "fixtures/simulated/visual.json"
    assert result["evidence_chain"][0]["simulated"] is True


def test_accepts_wrapped_plugin_outputs_payload():
    plugin = MultimodalFusionPlugin.create_standalone()
    plugin._use_enhanced_engine = False

    result = plugin.process({
        "device_id": "wrapped-device",
        "task_id": "wrapped-task",
        "plugin_outputs": [
            simulated_output("thermal", label="hotspot", severity="warning", confidence=0.8)
        ],
    })

    assert result["success"] is True
    assert result["device_id"] == "wrapped-device"
    assert result["task_id"] == "wrapped-task"
    assert result["contributing_modalities"] == ["thermal"]
