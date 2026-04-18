"""
Indoor Fence V3.0 - Object Detection Module
=============================================

Abstract base for object detectors + YOLO implementation.
Extracted from camera_adapter for modularity.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DetectionBox:
    """A single object detection bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str = "person"

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def foot_point(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)


class ObjectDetector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionBox]:
        """Detect objects in a frame."""
        ...
