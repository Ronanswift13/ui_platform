"""
Indoor Fence V3.0 - Automatic Fence Generator
===============================================

Generates warning/danger buffer zones from equipment detections
using morphological dilation. Supports zone merging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .object_detector import DetectionBox


class FenceLevel(str, Enum):
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


@dataclass
class FenceZone:
    """A generated fence zone polygon."""
    vertices: List[Tuple[float, float]]
    level: FenceLevel
    equipment_id: str = ""

    @property
    def area(self) -> float:
        """Compute area using shoelace formula."""
        n = len(self.vertices)
        if n < 3:
            return 0.0
        a = 0.0
        for i in range(n):
            j = (i + 1) % n
            a += self.vertices[i][0] * self.vertices[j][1]
            a -= self.vertices[j][0] * self.vertices[i][1]
        return abs(a) / 2.0

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon test using ray casting."""
        n = len(self.vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


class AutoFenceGenerator:
    """Generates fence zones from equipment detections."""

    def __init__(
        self,
        warning_buffer_m: float = 0.5,
        danger_buffer_m: float = 0.3,
        pixels_per_meter: float = 100.0,
    ):
        self._warning_buffer = warning_buffer_m
        self._danger_buffer = danger_buffer_m
        self._ppm = pixels_per_meter

    def generate_from_detection(
        self,
        detection: DetectionBox,
        image_size: Tuple[int, int],
        equipment_id: Optional[str] = None,
    ) -> List[FenceZone]:
        """Generate warning and danger zones from an equipment detection."""
        eq_id = equipment_id or f"eq_{detection.class_name}_{detection.x1}"

        # Convert pixel bbox to normalized coordinates
        w, h = image_size
        cx = (detection.x1 + detection.x2) / 2.0
        cy = (detection.y1 + detection.y2) / 2.0
        bw = detection.width
        bh = detection.height

        zones = []

        # Danger zone (smaller buffer around equipment)
        db = self._danger_buffer * self._ppm
        danger_verts = [
            (cx - bw / 2 - db, cy - bh / 2 - db),
            (cx + bw / 2 + db, cy - bh / 2 - db),
            (cx + bw / 2 + db, cy + bh / 2 + db),
            (cx - bw / 2 - db, cy + bh / 2 + db),
        ]
        zones.append(FenceZone(
            vertices=danger_verts,
            level=FenceLevel.DANGER,
            equipment_id=eq_id,
        ))

        # Warning zone (larger buffer)
        wb = self._warning_buffer * self._ppm
        warning_verts = [
            (cx - bw / 2 - wb, cy - bh / 2 - wb),
            (cx + bw / 2 + wb, cy - bh / 2 - wb),
            (cx + bw / 2 + wb, cy + bh / 2 + wb),
            (cx - bw / 2 - wb, cy + bh / 2 + wb),
        ]
        zones.append(FenceZone(
            vertices=warning_verts,
            level=FenceLevel.WARNING,
            equipment_id=eq_id,
        ))

        return zones

    def merge_overlapping(self, zones: List[FenceZone]) -> List[FenceZone]:
        """Merge overlapping zones of the same level (simplified bounding box merge)."""
        by_level: dict = {}
        for z in zones:
            by_level.setdefault(z.level, []).append(z)

        merged = []
        for level, level_zones in by_level.items():
            # Simple merge: combine overlapping bboxes
            used = [False] * len(level_zones)
            for i in range(len(level_zones)):
                if used[i]:
                    continue
                bb = self._bbox_of(level_zones[i].vertices)
                eq_ids = [level_zones[i].equipment_id]
                for j in range(i + 1, len(level_zones)):
                    if used[j]:
                        continue
                    bb2 = self._bbox_of(level_zones[j].vertices)
                    if self._overlaps(bb, bb2):
                        bb = self._union(bb, bb2)
                        eq_ids.append(level_zones[j].equipment_id)
                        used[j] = True
                used[i] = True
                verts = [
                    (bb[0], bb[1]), (bb[2], bb[1]),
                    (bb[2], bb[3]), (bb[0], bb[3]),
                ]
                merged.append(FenceZone(
                    vertices=verts,
                    level=level,
                    equipment_id="+".join(eq_ids),
                ))

        return merged

    @staticmethod
    def _bbox_of(verts: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _overlaps(a: Tuple, b: Tuple) -> bool:
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    @staticmethod
    def _union(a: Tuple, b: Tuple) -> Tuple[float, float, float, float]:
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
