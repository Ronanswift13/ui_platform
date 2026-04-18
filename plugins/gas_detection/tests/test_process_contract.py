import time


REQUIRED_METADATA = {
    "modality",
    "sensor_type",
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
    from plugins.gas_detection.plugin import Plugin

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
    assert result["metadata"]["modality"] == "gas"
    assert result["results"][0]["bbox"] == {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    assert result["results"][0]["metadata"]["virtual_roi"] is True


def test_single_sample_returns_standard_shell():
    plugin = create_plugin()
    result = plugin.process({
        "device_id": "gas_sensor_01",
        "timestamp": time.time(),
        "readings": {"SF6": 120.0, "H2": 20.0},
        "context": {"task_id": "gas-contract", "site_id": "site-a"},
    })

    assert result["success"] is True
    assert result["label"] == "normal"
    assert result["metadata"]["trend_prediction_available"] is False
    assert result["predictions"]["available"] is False
    assert_standard_output(result)


def test_empty_sample_returns_explainable_failure():
    plugin = create_plugin()
    result = plugin.process({"device_id": "gas_sensor_01", "readings": {}})

    assert result["success"] is False
    assert result["status"] == "error"
    assert "缺少气体读数" in result["error_message"]
    assert_standard_output(result)


def test_threshold_breach_returns_abnormal_label():
    plugin = create_plugin()
    result = plugin.process({"device_id": "gas_sensor_01", "readings": {"SF6": 1600.0}})

    assert result["success"] is True
    assert result["status"] == "alarm"
    assert result["label"] == "abnormal"
    assert result["alarms"]
    assert result["anomaly_events"]
    assert result["abnormal_intervals"]
    assert "GAS_THRESHOLD_ALARM_SF6" in result["reason_codes"]
    assert_standard_output(result)


def test_structured_timeseries_input_and_bad_timestamp_are_tolerated():
    plugin = create_plugin()
    result = plugin.process({
        "device_id": "gas_sensor_01",
        "timestamp": "bad-time",
        "structured_timeseries": {
            "variables": {
                "SF6": [100.0, 120.0, 180.0],
                "H2": [20.0, 22.0, 25.0],
            }
        },
    })

    assert result["success"] is True
    assert result["input_protocol"]["observed_input_type"] == "structured_timeseries"
    assert result["input_protocol"]["data_quality"]["status"] == "degraded"
    assert result["input_protocol"]["time_window"]["timestamp_valid"] is False


def test_config_threshold_change_affects_output():
    plugin = create_plugin({
        "thresholds": {
            "SF6": {"attention": 10, "warning": 20, "alarm": 30, "critical": 40}
        }
    })

    result = plugin.process({"device_id": "gas_sensor_01", "readings": {"SF6": 35.0}})

    assert result["success"] is True
    assert result["status"] == "alarm"
    assert result["metadata"]["threshold_snapshot"]["SF6"]["alarm"] == 30


def test_healthcheck_reflects_initialized_state():
    from plugins.gas_detection.plugin import Plugin

    plugin = Plugin()
    assert plugin.healthcheck().healthy is False
    plugin.init({})
    assert plugin.healthcheck().healthy is True
