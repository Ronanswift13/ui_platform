import numpy as np


REQUIRED_METADATA = {
    "modality",
    "sensor_type",
    "sampling_rate",
    "sample_interval",
    "window_size",
    "threshold_snapshot",
    "runtime_mode",
    "algorithm_stage",
    "model_status",
    "fallback_level",
    "trend_prediction_available",
    "upgrade_placeholders",
}
REQUIRED_TEMPORAL_FIELDS = {
    "plugin_name",
    "task_type",
    "summary",
    "anomaly_events",
    "abnormal_intervals",
    "severity",
    "reason_codes",
    "recommended_actions",
    "trend_diagnosis",
    "evidence",
    "review_required",
    "model_info",
    "placeholders",
    "input_protocol",
}
REQUIRED_PLACEHOLDERS = {
    "model_features_placeholder",
    "sequence_embedding_placeholder",
    "temporal_pattern_placeholder",
    "anomaly_score_trace_placeholder",
    "root_cause_feature_placeholder",
}


def create_plugin(config=None):
    from plugins.acoustic_monitoring.plugin import Plugin

    return Plugin.create_standalone(config)


def assert_standard_output(result):
    assert "success" in result
    assert "status" in result
    assert "label" in result
    assert "value" in result
    assert 0 <= result["confidence"] <= 1
    assert REQUIRED_METADATA.issubset(result["metadata"])
    assert REQUIRED_TEMPORAL_FIELDS.issubset(result)
    assert REQUIRED_PLACEHOLDERS.issubset(result["placeholders"])
    assert result["metadata"]["modality"] == "acoustic"
    assert result["results"][0]["bbox"] == {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    assert result["results"][0]["metadata"]["virtual_roi"] is True


def test_single_audio_buffer_returns_standard_shell():
    plugin = create_plugin({"sample_rate": 8000, "audio_duration": 0.5})
    sample_rate = 8000
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 50 * t)).astype(float).tolist()

    result = plugin.process({
        "device_id": "acoustic_channel_01",
        "audio_buffer": audio,
        "sample_rate": sample_rate,
        "context": {"task_id": "acoustic-contract", "site_id": "site-a"},
    })

    assert result["success"] is True
    assert result["metadata"]["trend_prediction_available"] is False
    assert_standard_output(result)


def test_sampled_sequence_input_and_bad_timestamp_are_tolerated():
    plugin = create_plugin({"sample_rate": 8000, "audio_duration": 0.25})
    result = plugin.process({
        "device_id": "acoustic_channel_01",
        "timestamp": "not-a-date",
        "sampled_sequence": [{"value": 0.0}, {"value": 0.1}, {"value": -0.1}],
        "sample_rate": 8000,
    })

    assert result["success"] is True
    assert result["input_protocol"]["observed_input_type"] == "sampled_sequence"
    assert result["input_protocol"]["data_quality"]["status"] == "degraded"
    assert result["input_protocol"]["time_window"]["timestamp_valid"] is False


def test_empty_audio_buffer_returns_explainable_failure():
    plugin = create_plugin()
    result = plugin.process({"device_id": "acoustic_channel_01", "audio_buffer": []})

    assert result["success"] is False
    assert result["status"] == "error"
    assert "音频缓冲为空" in result["error_message"]
    assert_standard_output(result)


def test_threshold_change_affects_rule_output_when_model_unavailable():
    plugin = create_plugin({"detection_params": {"anomaly_threshold": 0.0}, "sample_rate": 8000, "audio_duration": 0.1})
    plugin._detector = None

    result = plugin.process({"device_id": "acoustic_channel_01"})

    assert result["success"] is True
    assert result["status"] == "warning"
    assert result["label"] == "warning"
    assert result["metadata"]["model_status"] == "unavailable"
    assert result["metadata"]["fallback_level"] == "rules"


def test_healthcheck_reflects_initialized_state():
    from plugins.acoustic_monitoring.plugin import Plugin

    plugin = Plugin()
    assert plugin.healthcheck().healthy is False
    plugin.init({})
    assert plugin.healthcheck().healthy is True
