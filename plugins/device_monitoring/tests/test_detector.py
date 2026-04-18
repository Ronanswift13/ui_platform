import time

from plugins.device_monitoring.detector import (
    DeviceHealth,
    DeviceHealthCalculator,
    DeviceMonitorDetector,
)


def _isolated_weights():
    return {
        "cpu_temp": 0.0,
        "cpu_usage": 0.15,
        "memory_usage": 0.0,
        "network_quality": 0.0,
        "error_rate": 0.0,
        "uptime": 0.0,
    }


def _safe_metrics(**overrides):
    metrics = {
        "cpu_temp": 40,
        "cpu_usage": 30,
        "memory_usage": 35,
        "network_quality": 95,
        "error_count": 0,
        "last_heartbeat": time.time(),
        "uptime_hours": 100,
    }
    metrics.update(overrides)
    return metrics


def _detector_config(**threshold_overrides):
    thresholds = {
        "health_warning": 70,
        "health_alarm": 50,
        "health_critical": 30,
        "heartbeat_timeout_sec": 60,
        "cpu_temp_warning": 70,
        "cpu_temp_alarm": 85,
        "cpu_usage_warning_percent": 70,
        "cpu_usage_alarm_percent": 90,
        "memory_warning_percent": 85,
        "error_rate_warning": 5,
    }
    thresholds.update(threshold_overrides)
    return {
        "health_weights": _isolated_weights(),
        "thresholds": thresholds,
        "prediction": {"enabled": False},
        "maintenance": {
            "auto_ticket": True,
            "ticket_priority_map": {"critical": "P1", "alarm": "P2", "warning": "P3"},
        },
    }


def test_calculator_uses_configured_cpu_usage_thresholds():
    calculator = DeviceHealthCalculator(
        _isolated_weights(),
        {
            "cpu_usage_warning_percent": 60,
            "cpu_usage_alarm_percent": 80,
            "heartbeat_timeout_sec": 60,
        },
    )

    warning_health, warning_issues, _ = calculator.calculate(_safe_metrics(cpu_usage=65))
    alarm_health, alarm_issues, alarm_recs = calculator.calculate(_safe_metrics(cpu_usage=85))

    assert warning_health == 94.0
    assert warning_issues == ["CPU使用率偏高: 65%"]
    assert alarm_health == 85.0
    assert alarm_issues == ["CPU使用率过高: 85%"]
    assert alarm_recs == ["检查是否有异常进程，考虑升级硬件"]


def test_calculator_cpu_usage_boundary_is_strictly_greater_than_threshold():
    calculator = DeviceHealthCalculator(_isolated_weights(), _detector_config()["thresholds"])

    at_warning, warning_issues, _ = calculator.calculate(_safe_metrics(cpu_usage=70))
    above_warning, above_warning_issues, _ = calculator.calculate(_safe_metrics(cpu_usage=70.1))
    at_alarm, alarm_issues, _ = calculator.calculate(_safe_metrics(cpu_usage=90))
    above_alarm, above_alarm_issues, _ = calculator.calculate(_safe_metrics(cpu_usage=90.1))

    assert at_warning == 100.0
    assert warning_issues == []
    assert above_warning == 94.0
    assert above_warning_issues == ["CPU使用率偏高: 70.1%"]
    assert at_alarm == 94.0
    assert alarm_issues == ["CPU使用率偏高: 90%"]
    assert above_alarm == 85.0
    assert above_alarm_issues == ["CPU使用率过高: 90.1%"]


def test_calculator_missing_metrics_use_safe_defaults_and_score_is_clamped():
    calculator = DeviceHealthCalculator(_isolated_weights(), _detector_config()["thresholds"])
    missing_health, missing_issues, missing_recs = calculator.calculate({})

    severe_health, severe_issues, _ = DeviceHealthCalculator(
        {
            "cpu_temp": 1.0,
            "cpu_usage": 1.0,
            "memory_usage": 1.0,
            "network_quality": 1.0,
            "error_rate": 1.0,
            "uptime": 0.0,
        },
        _detector_config()["thresholds"],
    ).calculate(
        _safe_metrics(
            cpu_temp=120,
            cpu_usage=99,
            memory_usage=99,
            network_quality=1,
            error_count=99,
            last_heartbeat=time.time() - 10_000,
        )
    )

    assert missing_health == 100.0
    assert missing_issues == []
    assert missing_recs == []
    assert severe_health == 0
    assert severe_issues


def test_detector_status_boundaries_follow_health_thresholds():
    detector = DeviceMonitorDetector(_detector_config())

    assert detector._determine_status(70, "online") == "online"
    assert detector._determine_status(69.9, "online") == "warning"
    assert detector._determine_status(50, "online") == "warning"
    assert detector._determine_status(49.9, "online") == "error"
    assert detector._determine_status(30, "online") == "error"
    assert detector._determine_status(29.9, "online") == "error"
    assert detector._determine_status(100, "offline") == "offline"


def test_detector_ticket_trigger_is_below_alarm_boundary_and_deduplicated():
    config = _detector_config(health_warning=90, health_alarm=85, health_critical=30)
    detector = DeviceMonitorDetector(config)
    reading = {
        "device_id": "edge_ticket",
        "device_name": "Edge Ticket",
        "device_type": "edge_computing_node",
        "status": "online",
        "metrics": _safe_metrics(cpu_usage=90.1),
    }

    boundary = detector.detect([reading])
    assert boundary["devices"][0].health_index == 85.0
    assert boundary["new_tickets"] == []

    detector.th_alarm = 85.1
    first = detector.detect([reading])
    second = detector.detect([reading])

    assert first["new_tickets"][0].priority == "P2"
    assert second["new_tickets"] == []


def test_detector_critical_ticket_uses_configured_priority_map():
    detector = DeviceMonitorDetector(_detector_config())
    ticket = detector._create_ticket(
        DeviceHealth(
            device_id="edge_critical",
            device_name="Edge Critical",
            device_type="edge_computing_node",
            status="error",
            health_index=29.9,
            anomaly_score=0.0,
            metrics=_safe_metrics(),
            issues=["设备长时间无心跳"],
            recommendations=["检查设备供电和网络连接"],
        )
    )

    assert ticket is not None
    assert ticket.priority == "P1"


def test_detector_anomaly_score_stays_inside_contract_bounds():
    detector = DeviceMonitorDetector(_detector_config())
    now = time.time()
    detector._history["edge_anomaly"].extend(
        {"timestamp": now - i, "health": 95.0, "metrics": _safe_metrics()} for i in range(12)
    )
    detector._history["edge_anomaly"].append(
        {"timestamp": now, "health": 10.0, "metrics": _safe_metrics(cpu_usage=99)}
    )

    score = detector._calc_anomaly("edge_anomaly", _safe_metrics(cpu_usage=99))

    assert 0.0 <= score <= 1.0
    assert score == 1.0
