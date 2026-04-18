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
    from plugins.device_monitoring.plugin import Plugin

    return Plugin.create_standalone(config)


def reading(**metrics):
    base = {
        "cpu_temp": 42,
        "cpu_usage": 30,
        "memory_usage": 35,
        "network_quality": 95,
        "error_count": 0,
        "last_heartbeat": time.time(),
    }
    base.update(metrics)
    return {
        "device_id": "edge_01",
        "device_name": "Edge Node",
        "device_type": "edge_computing_node",
        "status": "online",
        "metrics": base,
    }


def assert_standard_output(result):
    assert "success" in result
    assert "status" in result
    assert "label" in result
    assert "value" in result
    assert 0 <= result["confidence"] <= 1
    assert REQUIRED_METADATA.issubset(result["metadata"])
    assert REQUIRED_TEMPORAL_FIELDS.issubset(result)
    assert REQUIRED_PLACEHOLDERS.issubset(result["placeholders"])
    assert result["metadata"]["modality"] == "device"
    assert result["results"][0]["bbox"] == {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    assert result["results"][0]["metadata"]["virtual_roi"] is True


def test_single_reading_returns_standard_shell():
    plugin = create_plugin()
    result = plugin.process({
        "device_id": "edge_01",
        "readings": [reading()],
        "context": {"task_id": "device-contract", "site_id": "site-a"},
    })

    assert result["success"] is True
    assert result["label"] == "normal"
    assert result["metadata"]["trend_prediction_available"] is False
    assert result["metadata"]["model_status"] == "placeholder"
    assert_standard_output(result)


def test_empty_reading_returns_explainable_failure():
    plugin = create_plugin()
    result = plugin.process({"device_id": "edge_01", "readings": []})

    assert result["success"] is False
    assert result["status"] == "error"
    assert "缺少设备读数" in result["error_message"]
    assert_standard_output(result)


def test_threshold_breach_returns_abnormal_label():
    plugin = create_plugin()
    result = plugin.process({
        "device_id": "edge_01",
        "readings": [reading(cpu_temp=95, cpu_usage=96, memory_usage=96, network_quality=20, error_count=20)],
    })

    assert result["success"] is True
    assert result["label"] == "abnormal"
    assert result["alarms"]
    assert result["anomaly_events"]
    assert result["abnormal_intervals"]
    assert_standard_output(result)


def test_structured_timeseries_input_and_bad_timestamp_are_tolerated():
    plugin = create_plugin()
    result = plugin.process({
        "device_id": "edge_01",
        "timestamp": "bad-ts",
        "structured_timeseries": {
            "device_type": "edge_computing_node",
            "variables": {
                "cpu_temp": [40, 42],
                "cpu_usage": [25, 30],
                "memory_usage": [35, 36],
                "network_quality": [95, 94],
                "error_count": [0, 0],
                "last_heartbeat": [time.time(), time.time()],
            },
        },
    })

    assert result["success"] is True
    assert result["input_protocol"]["observed_input_type"] == "structured_timeseries"
    assert result["input_protocol"]["data_quality"]["status"] == "degraded"
    assert result["input_protocol"]["time_window"]["timestamp_valid"] is False


def test_config_threshold_change_affects_output():
    plugin = create_plugin({"thresholds": {"health_warning": 99, "health_alarm": 50, "health_critical": 30}})
    result = plugin.process({"device_id": "edge_01", "readings": [reading(cpu_temp=71)]})

    assert result["success"] is True
    assert result["status"] == "warning"
    assert result["metadata"]["threshold_snapshot"]["health_warning"] == 99


def test_healthcheck_reflects_initialized_state():
    from plugins.device_monitoring.plugin import Plugin

    plugin = Plugin()
    assert plugin.healthcheck().healthy is False
    plugin.init({})
    assert plugin.healthcheck().healthy is True
