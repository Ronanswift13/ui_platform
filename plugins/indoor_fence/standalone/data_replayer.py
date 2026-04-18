"""
Indoor Fence V3.0 - Data Replay Engine
========================================

Replays recorded sensor data from JSONL files,
synchronized by timestamp. Supports reset and
frame-by-frame playback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..protocols import SensorData, SensorType


class DataReplayer:
    """Replays recorded multi-sensor data."""

    def __init__(self, data_dir: str):
        self._data_dir = Path(data_dir)
        self._frames: List[Dict[SensorType, SensorData]] = []
        self._index = 0
        self._load()

    def _load(self) -> None:
        """Load all JSONL files and merge by timestamp."""
        # Collect all sensor data entries
        all_entries: List[SensorData] = []

        for jsonl_file in self._data_dir.glob("*.jsonl"):
            with open(jsonl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = SensorData.model_validate_json(line)
                        all_entries.append(data)
                    except Exception:
                        continue

        if not all_entries:
            return

        # Sort by timestamp
        all_entries.sort(key=lambda d: d.timestamp)

        # Group into frames by timestamp proximity (within 10ms)
        frames: List[Dict[SensorType, SensorData]] = []
        current_frame: Dict[SensorType, SensorData] = {}
        current_ts = all_entries[0].timestamp

        for entry in all_entries:
            if abs(entry.timestamp - current_ts) > 0.01:
                # New frame
                if current_frame:
                    frames.append(current_frame)
                current_frame = {}
                current_ts = entry.timestamp
            current_frame[entry.sensor_type] = entry

        if current_frame:
            frames.append(current_frame)

        self._frames = frames

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def current_index(self) -> int:
        return self._index

    def next(self) -> Optional[Dict[SensorType, SensorData]]:
        """Get next frame of sensor data.

        Returns None when all frames have been consumed.
        """
        if self._index >= len(self._frames):
            return None

        frame = self._frames[self._index]
        self._index += 1
        return frame

    def reset(self) -> None:
        """Reset playback to the beginning."""
        self._index = 0

    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame index."""
        self._index = max(0, min(frame_index, len(self._frames)))
