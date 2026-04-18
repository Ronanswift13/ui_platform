"""Plugin contract tests for animal_detection.

Validates that the plugin's infer/postprocess interface conforms to
the expected contract: infer returns an AnimalEvent, postprocess
returns a list of alarm dicts.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_infer_returns_animal_event(blank_frame):
    """infer() on a blank frame should return an AnimalEvent."""
    from plugins.animal_detection.plugin import AnimalDetectionPlugin
    from plugins.animal_detection.core.event_schema import AnimalEvent

    plugin = AnimalDetectionPlugin.create_standalone()
    result = plugin.infer(frame=blank_frame, rois=[], context=None)
    assert isinstance(result, AnimalEvent), (
        f"infer should return AnimalEvent, got {type(result).__name__}"
    )


def test_infer_event_has_required_fields(blank_frame):
    """AnimalEvent produced by infer should contain required attributes."""
    from plugins.animal_detection.plugin import AnimalDetectionPlugin

    plugin = AnimalDetectionPlugin.create_standalone()
    event = plugin.infer(frame=blank_frame)
    for attr in ("type", "detections", "timestamp"):
        assert hasattr(event, attr), f"AnimalEvent missing attribute: {attr}"
    assert isinstance(event.detections, list)


def test_postprocess_returns_list(blank_frame):
    """postprocess() should always return a list."""
    from plugins.animal_detection.plugin import AnimalDetectionPlugin

    plugin = AnimalDetectionPlugin.create_standalone()
    event = plugin.infer(frame=blank_frame)
    alarms = plugin.postprocess(event)
    assert isinstance(alarms, list)


def test_postprocess_no_alarms_on_blank_frame(blank_frame):
    """A blank frame should produce no detections and therefore no alarms."""
    from plugins.animal_detection.plugin import AnimalDetectionPlugin

    plugin = AnimalDetectionPlugin.create_standalone()
    event = plugin.infer(frame=blank_frame)
    alarms = plugin.postprocess(event)
    assert len(alarms) == 0, "blank frame should not produce alarms"
