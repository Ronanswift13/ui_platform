"""Postprocess and alarm mapping tests for capacitor_inspection."""

from __future__ import annotations

from darkbreaker_sdk.schemas import AlarmLevel, BoundingBox, RecognitionResult


def test_postprocess_maps_alarm_levels(plugin):
    """Known labels should map to the plugin's current alarm severities."""
    results = [
        RecognitionResult(
            task_id="task-1",
            site_id="site-1",
            device_id="device-1",
            component_id="capacitor-bank-01",
            roi_id="roi-1",
            bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
            label="tilt_error",
            confidence=0.92,
        ),
        RecognitionResult(
            task_id="task-1",
            site_id="site-1",
            device_id="device-1",
            component_id="capacitor-bank-01",
            roi_id="roi-2",
            bbox=BoundingBox(x=0.2, y=0.2, width=0.2, height=0.2),
            label="intrusion_animal",
            confidence=0.88,
        ),
    ]

    alarms = plugin.postprocess(results, [])

    assert len(alarms) == 2
    assert alarms[0].level == AlarmLevel.ERROR
    assert alarms[1].level == AlarmLevel.WARNING
