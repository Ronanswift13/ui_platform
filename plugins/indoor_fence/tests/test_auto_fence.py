"""Tests for automatic fence generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
from plugins.indoor_fence.detection.auto_fence_generator import (
    AutoFenceGenerator, FenceZone, FenceLevel,
)
from plugins.indoor_fence.detection.object_detector import DetectionBox


def test_fence_generation_from_equipment():
    gen = AutoFenceGenerator(
        warning_buffer_m=0.5,
        danger_buffer_m=0.3,
    )
    equipment = DetectionBox(
        x1=100, y1=100, x2=300, y2=400,
        confidence=0.9,
        class_id=1,
        class_name="cabinet",
    )
    fences = gen.generate_from_detection(equipment, image_size=(640, 480))
    assert len(fences) >= 2  # warning + danger zones
    warning = [f for f in fences if f.level == FenceLevel.WARNING][0]
    danger = [f for f in fences if f.level == FenceLevel.DANGER][0]
    # Warning zone should be larger than danger zone
    assert warning.area > danger.area


def test_fence_zone_contains():
    zone = FenceZone(
        vertices=[(0, 0), (4, 0), (4, 4), (0, 4)],
        level=FenceLevel.WARNING,
        equipment_id="cab_1",
    )
    assert zone.contains(2.0, 2.0) is True
    assert zone.contains(5.0, 5.0) is False


def test_fence_merge():
    gen = AutoFenceGenerator(warning_buffer_m=0.5, danger_buffer_m=0.3)
    eq1 = DetectionBox(x1=100, y1=100, x2=200, y2=300, confidence=0.9, class_id=1, class_name="cabinet")
    eq2 = DetectionBox(x1=180, y1=100, x2=280, y2=300, confidence=0.9, class_id=1, class_name="cabinet")
    fences1 = gen.generate_from_detection(eq1, image_size=(640, 480))
    fences2 = gen.generate_from_detection(eq2, image_size=(640, 480))
    # Overlapping fences should be merge-able
    merged = gen.merge_overlapping(fences1 + fences2)
    assert len(merged) <= len(fences1) + len(fences2)
