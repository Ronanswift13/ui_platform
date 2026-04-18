"""
Indoor Fence V3.0 - Multi-Target Tracker V3
=============================================

3D multi-target tracker using Hungarian assignment.
Manages track lifecycle: INIT -> ACTIVE -> LOST -> DELETED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...protocols import FusionOutput, SensorType
from .hungarian import linear_assignment


class TrackState(str, Enum):
    INIT = "init"
    ACTIVE = "active"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class Track:
    """A tracked target."""
    track_id: int
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    state: TrackState = TrackState.INIT
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    confidence: float = 0.5
    sources: List[SensorType] = field(default_factory=list)


class MultiTargetTrackerV3:
    """3D multi-target tracker with Hungarian assignment."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        distance_threshold: float = 3.0,
    ):
        self._max_age = max_age
        self._min_hits = min_hits
        self._distance_threshold = distance_threshold
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 0

    def update(self, fusion_outputs: List[FusionOutput]) -> List[Track]:
        """Update tracker with new fusion outputs.

        Returns list of all managed tracks (including lost ones).
        """
        # Increment age for all tracks
        for t in self._tracks.values():
            t.age += 1
            t.time_since_update += 1

        if len(fusion_outputs) == 0 and len(self._tracks) == 0:
            return []

        # Build cost matrix
        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(fusion_outputs)

        if n_tracks > 0 and n_dets > 0:
            cost_matrix = np.zeros((n_tracks, n_dets))
            for i, tid in enumerate(track_ids):
                t = self._tracks[tid]
                for j, fo in enumerate(fusion_outputs):
                    dx = t.position[0] - fo.position_3d[0]
                    dy = t.position[1] - fo.position_3d[1]
                    dz = t.position[2] - fo.position_3d[2]
                    cost_matrix[i, j] = (dx**2 + dy**2 + dz**2) ** 0.5

            # Run assignment
            assignments = linear_assignment(cost_matrix)

            matched_tracks: set = set()
            matched_dets: set = set()

            for row, col in assignments:
                if cost_matrix[row, col] > self._distance_threshold:
                    continue
                tid = track_ids[row]
                fo = fusion_outputs[col]
                t = self._tracks[tid]
                t.position = fo.position_3d
                t.velocity = fo.velocity_3d
                t.hits += 1
                t.time_since_update = 0
                t.confidence = fo.confidence
                t.sources = fo.sources
                if t.hits >= self._min_hits:
                    t.state = TrackState.ACTIVE
                matched_tracks.add(tid)
                matched_dets.add(col)

            # Handle unmatched detections -> new tracks
            for j in range(n_dets):
                if j not in matched_dets:
                    self._create_track(fusion_outputs[j])

        elif n_dets > 0:
            # No existing tracks, create all
            for fo in fusion_outputs:
                self._create_track(fo)

        # Update track states for unmatched tracks
        for tid, t in list(self._tracks.items()):
            if t.time_since_update > 0:
                if t.state == TrackState.ACTIVE:
                    t.state = TrackState.LOST

        # Delete old tracks
        to_delete = []
        for tid, t in self._tracks.items():
            if t.time_since_update > self._max_age:
                t.state = TrackState.DELETED
                to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

        return list(self._tracks.values())

    def _create_track(self, fo: FusionOutput) -> Track:
        """Create a new track from a fusion output."""
        tid = self._next_id
        self._next_id += 1
        track = Track(
            track_id=tid,
            position=fo.position_3d,
            velocity=fo.velocity_3d,
            state=TrackState.INIT,
            hits=1,
            age=1,
            time_since_update=0,
            confidence=fo.confidence,
            sources=fo.sources,
        )
        self._tracks[tid] = track
        return track

    def get_active_tracks(self) -> List[Track]:
        """Get only active (confirmed) tracks."""
        return [t for t in self._tracks.values() if t.state == TrackState.ACTIVE]

    def get_all_tracks(self) -> List[Track]:
        """Get all managed tracks."""
        return list(self._tracks.values())
