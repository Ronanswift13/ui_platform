#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内动物入侵检测 - 核心算法模块
"""

from .event_schema import (
    AnimalClass, EventType, RiskLevel, ANIMAL_RISK_MAP, SUGGESTION_MAP,
    BoundingBox, WorldPosition, EvidenceItem,
    AnimalDetectionResult, AnimalEvent, build_intrusion_event,
)
from .detector import YOLOv8Detector, DEFAULT_CLASS_MAP, CLASS_NAMES_CN
from .thermal_validator import ThermalValidator
from .tracker import AnimalTracker, AnimalTrack
from .deterrent import DeterrentController, DeterrentAction, DeterrentResult, DETERRENT_STRATEGY
from .statistics import StatisticsCollector, InvasionRecord

__all__ = [
    # Schema
    "AnimalClass", "EventType", "RiskLevel", "ANIMAL_RISK_MAP", "SUGGESTION_MAP",
    "BoundingBox", "WorldPosition", "EvidenceItem",
    "AnimalDetectionResult", "AnimalEvent", "build_intrusion_event",
    # Detector
    "YOLOv8Detector", "DEFAULT_CLASS_MAP", "CLASS_NAMES_CN",
    # Thermal
    "ThermalValidator",
    # Tracker
    "AnimalTracker", "AnimalTrack",
    # Deterrent
    "DeterrentController", "DeterrentAction", "DeterrentResult", "DETERRENT_STRATEGY",
    # Statistics
    "StatisticsCollector", "InvasionRecord",
]
