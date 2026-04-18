"""
Indoor Fence V3.0 - Multi-Sensor Data Recorder
=================================================

Records sensor data streams to JSONL files for replay
and training pipeline. Supports per-sensor output files
and session metadata.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..protocols import SensorData, SensorType


class DataRecorder:
    """Records multi-sensor data to JSONL files."""

    def __init__(self, output_dir: Optional[str] = None):
        self._output_dir = output_dir or os.path.join(os.getcwd(), "recordings")
        self._is_recording = False
        self._session_id: Optional[str] = None
        self._files: Dict[str, Any] = {}
        self._record_count = 0

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def record_count(self) -> int:
        return self._record_count

    def start(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new recording session.

        Args:
            metadata: Optional session metadata (scene, operator, etc.)

        Returns:
            Session ID string.
        """
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self._output_dir, self._session_id)
        os.makedirs(session_dir, exist_ok=True)

        self._is_recording = True
        self._record_count = 0
        self._files = {}

        # Write metadata
        if metadata:
            meta_path = os.path.join(session_dir, f"{self._session_id}_metadata.json")
            with open(meta_path, "w") as f:
                meta = {
                    "session_id": self._session_id,
                    "start_time": time.time(),
                    "start_time_iso": datetime.now().isoformat(),
                    **metadata,
                }
                json.dump(meta, f, indent=2)

        return self._session_id

    def record(self, data: SensorData) -> None:
        """Record a single sensor data frame."""
        if not self._is_recording:
            return

        sensor_name = data.sensor_type.value
        session_dir = os.path.join(self._output_dir, self._session_id or "unknown")

        if sensor_name not in self._files:
            filepath = os.path.join(session_dir, f"{sensor_name}.jsonl")
            self._files[sensor_name] = open(filepath, "a")

        f = self._files[sensor_name]
        f.write(data.model_dump_json() + "\n")
        f.flush()
        self._record_count += 1

    def stop(self) -> None:
        """Stop recording and close all files."""
        self._is_recording = False
        for f in self._files.values():
            f.close()
        self._files = {}
