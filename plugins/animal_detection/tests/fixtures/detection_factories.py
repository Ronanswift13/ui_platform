"""
检测结果工厂
============

生成各种测试用检测结果、事件和跟踪数据。
所有数据结构符合 core/event_schema.py 契约。
"""

import time
from typing import Optional

from core.event_schema import (
    AnimalClass,
    AnimalDetectionResult,
    BoundingBox,
    EvidenceItem,
    WorldPosition,
)
from core.tracker import AnimalTrack


# --- AnimalDetectionResult 工厂 ---

def make_detection(
    animal_class: str = AnimalClass.MOUSE,
    confidence: float = 0.85,
    bbox: Optional[BoundingBox] = None,
    track_id: Optional[str] = None,
    is_thermal_validated: bool = False,
    thermal_diff_c: Optional[float] = None,
    stay_duration_s: float = 0.0,
) -> AnimalDetectionResult:
    """通用检测结果工厂"""
    if bbox is None:
        bbox = BoundingBox(x=0.3, y=0.4, width=0.1, height=0.08)
    return AnimalDetectionResult(
        animal_class=animal_class,
        confidence=confidence,
        bbox=bbox,
        track_id=track_id,
        is_thermal_validated=is_thermal_validated,
        thermal_diff_c=thermal_diff_c,
        stay_duration_s=stay_duration_s,
    )


def make_mouse(confidence: float = 0.85) -> AnimalDetectionResult:
    """鼠检测结果"""
    return make_detection(
        animal_class=AnimalClass.MOUSE,
        confidence=confidence,
        bbox=BoundingBox(x=0.3, y=0.4, width=0.1, height=0.08),
    )


def make_snake(confidence: float = 0.75) -> AnimalDetectionResult:
    """蛇检测结果（CRITICAL 风险）"""
    return make_detection(
        animal_class=AnimalClass.SNAKE,
        confidence=confidence,
        bbox=BoundingBox(x=0.2, y=0.5, width=0.3, height=0.05),
    )


def make_cat(confidence: float = 0.80) -> AnimalDetectionResult:
    """猫检测结果"""
    return make_detection(
        animal_class=AnimalClass.CAT,
        confidence=confidence,
        bbox=BoundingBox(x=0.5, y=0.5, width=0.15, height=0.12),
    )


def make_bird(confidence: float = 0.70) -> AnimalDetectionResult:
    """鸟检测结果（LOW 风险）"""
    return make_detection(
        animal_class=AnimalClass.BIRD,
        confidence=confidence,
        bbox=BoundingBox(x=0.6, y=0.3, width=0.08, height=0.06),
    )


def make_dog(confidence: float = 0.80) -> AnimalDetectionResult:
    """狗检测结果"""
    return make_detection(
        animal_class=AnimalClass.DOG,
        confidence=confidence,
        bbox=BoundingBox(x=0.4, y=0.4, width=0.2, height=0.15),
    )


# --- AnimalTrack 工厂 ---

def make_track(
    track_id: str = "AT-000001",
    animal_class: str = AnimalClass.MOUSE,
    confidence: float = 0.8,
    bbox: Optional[BoundingBox] = None,
    stay_duration_s: float = 0.0,
    state: str = "confirmed",
) -> AnimalTrack:
    """跟踪轨迹工厂"""
    if bbox is None:
        bbox = BoundingBox(x=0.5, y=0.5, width=0.1, height=0.1)
    track = AnimalTrack(
        track_id=track_id,
        animal_class=animal_class,
        bbox=bbox,
        confidence=confidence,
    )
    track.state = state
    if stay_duration_s > 0:
        track.first_seen = time.time() - stay_duration_s
        track.update_stay()
    return track


def make_overstayed_track(
    track_id: str = "AT-000001",
    animal_class: str = AnimalClass.MOUSE,
    stay_s: float = 60.0,
) -> AnimalTrack:
    """已超时的跟踪轨迹"""
    return make_track(
        track_id=track_id,
        animal_class=animal_class,
        stay_duration_s=stay_s,
    )


# --- EvidenceItem 工厂 ---

def make_evidence(
    evidence_type: str = "raw_image",
    path: str = "/evidence/test_img.jpg",
    description: str = "测试证据图像",
) -> EvidenceItem:
    """证据项工厂"""
    return EvidenceItem(
        type=evidence_type,
        path=path,
        description=description,
    )
