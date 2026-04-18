from datetime import datetime


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
    from plugins.action_event_monitoring.plugin import ActionEventMonitoringPlugin

    plugin = ActionEventMonitoringPlugin.create_standalone(config)
    return plugin


def signal_change(**updates):
    event = {
        "signal_id": "SOE-001",
        "signal_name": "Remote signal change",
        "signal_group": "other",
        "action_type": "signal_change",
        "action_desc": "remote signal changed",
        "value_after": "1",
        "severity_hint": "info",
    }
    event.update(updates)
    return event


def assert_standard_output(result):
    assert "success" in result
    assert "status" in result
    assert "label" in result
    assert "value" in result
    assert 0 <= result["confidence"] <= 1
    assert REQUIRED_METADATA.issubset(result["metadata"])
    assert REQUIRED_TEMPORAL_FIELDS.issubset(result)
    assert REQUIRED_PLACEHOLDERS.issubset(result["placeholders"])
    assert result["metadata"]["modality"] == "action_event"
    assert result["results"][0]["bbox"] == {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    assert result["results"][0]["metadata"]["virtual_roi"] is True


def test_single_signal_change_returns_standard_shell():
    plugin = create_plugin()
    result = plugin.process({
        "device_id": "relay_01",
        "timestamp": datetime.now().isoformat(),
        "signal_changes": [signal_change()],
        "context": {"task_id": "action-contract", "site_id": "site-a"},
    })

    assert result["success"] is True
    assert result["label"] == "normal"
    assert result["stored_event_ids"]
    assert result["metadata"]["model_status"] == "unavailable"
    assert result["metadata"]["fallback_level"] == "rules"
    assert_standard_output(result)


def test_empty_events_return_explainable_failure():
    plugin = create_plugin()
    result = plugin.process({"device_id": "relay_01", "events": []})

    assert result["success"] is False
    assert result["status"] == "error"
    assert "缺少动作事件" in result["error_message"]
    assert_standard_output(result)


def test_alarm_signal_returns_abnormal_label():
    plugin = create_plugin()
    result = plugin.process({
        "device_id": "relay_01",
        "signal_changes": [signal_change(
            signal_group="protection",
            action_type="protection_trip",
            action_desc="protection trip",
            severity_hint="alarm",
        )],
    })

    assert result["success"] is True
    assert result["status"] == "alarm"
    assert result["label"] == "abnormal"
    assert result["anomaly_events"]
    assert result["abnormal_intervals"]
    assert_standard_output(result)


def test_state_change_events_input_and_bad_timestamp_are_tolerated():
    plugin = create_plugin()
    result = plugin.process({
        "device_id": "relay_01",
        "timestamp": "bad-ts",
        "state_change_events": [signal_change()],
    })

    assert result["success"] is True
    assert result["input_protocol"]["observed_input_type"] == "state_change_events"
    assert result["input_protocol"]["data_quality"]["status"] == "degraded"
    assert result["input_protocol"]["time_window"]["timestamp_valid"] is False


def test_config_threshold_change_affects_output():
    plugin = create_plugin({"thresholds": {"event_count_warning": 1}, "analysis": {}})
    result = plugin.process({"device_id": "relay_01", "signal_changes": [signal_change()]})

    assert result["success"] is True
    assert result["status"] == "warning"
    assert result["metadata"]["threshold_snapshot"]["event_count_warning"] == 1


def test_healthcheck_reflects_initialized_state():
    from plugins.action_event_monitoring.plugin import ActionEventMonitoringPlugin

    plugin = ActionEventMonitoringPlugin()
    assert plugin.healthcheck().healthy is False
    plugin.init({})
    assert plugin.healthcheck().healthy is True
