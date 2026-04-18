#!/usr/bin/env python3
"""Plugin postprocess tests for switch_inspection."""

from darkbreaker_sdk.schemas.alarm import AlarmLevel
from darkbreaker_sdk.schemas.detection import BoundingBox, RecognitionResult

from plugins.switch_inspection.plugin import Plugin


def _make_result(label: str, metadata=None):
    return RecognitionResult(
        task_id="task-001",
        site_id="site-001",
        device_id="device-001",
        component_id="component-001",
        roi_id="roi-001",
        bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
        label=label,
        confidence=0.9,
        metadata=metadata or {},
    )


def test_postprocess_maps_clarity_and_logic_alarms(default_config):
    plugin = Plugin.create_standalone(default_config)
    results = [
        _make_result("clarity_low"),
        _make_result("logic_error", {"rule": "防止带电合接地刀", "message": "检测到逻辑异常"}),
        _make_result("breaker_intermediate"),
    ]
    alarms = plugin.postprocess(results, [])
    assert [alarm.level for alarm in alarms] == [
        AlarmLevel.WARNING,
        AlarmLevel.ERROR,
        AlarmLevel.INFO,
    ]
