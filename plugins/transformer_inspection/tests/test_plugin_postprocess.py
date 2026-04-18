"""Postprocess and alarm mapping tests for transformer_inspection."""

from __future__ import annotations

from darkbreaker_sdk.schemas import AlarmLevel, BoundingBox, RecognitionResult


def test_postprocess_maps_alarm_levels(plugin):
    """Known labels should map to the plugin's current alarm severities."""
    results = [
        RecognitionResult(
            task_id="task-1",
            site_id="site-1",
            device_id="device-1",
            component_id="transformer-01",
            roi_id="roi-1",
            bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
            label="oil_leak",
            confidence=0.92,
        ),
        RecognitionResult(
            task_id="task-1",
            site_id="site-1",
            device_id="device-1",
            component_id="transformer-01",
            roi_id="roi-2",
            bbox=BoundingBox(x=0.2, y=0.2, width=0.2, height=0.2),
            label="silica_gel_abnormal",
            confidence=0.88,
        ),
        RecognitionResult(
            task_id="task-1",
            site_id="site-1",
            device_id="device-1",
            component_id="transformer-01",
            roi_id="thermal",
            bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            label="overtemp",
            value=120.0,
            confidence=0.9,
        ),
    ]

    alarms = plugin.postprocess(results, [])

    assert len(alarms) == 3
    assert alarms[0].level == AlarmLevel.ERROR
    assert alarms[1].level == AlarmLevel.WARNING
    assert alarms[2].level == AlarmLevel.ERROR
