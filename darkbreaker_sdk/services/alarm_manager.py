"""
Local Alarm Manager - Alarm lifecycle management for standalone plugins.

Manages alarm creation, acknowledgement, and statistics locally.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AlarmRecord:
    """A single alarm record."""
    alarm_id: str
    level: str          # "info", "warning", "critical", "emergency"
    source_plugin: str
    device_id: str
    alarm_type: str
    message: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    acknowledged_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Alarm level ordering (higher = more severe)
LEVEL_ORDER = {"info": 0, "warning": 1, "critical": 2, "emergency": 3}


class LocalAlarmManager:
    """
    Lightweight alarm lifecycle manager for standalone plugin operation.

    Maintains an in-memory alarm store with persistence to a JSON file
    in the data directory. Supports alarm creation, acknowledgement,
    filtering by level, and statistics.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        max_alarms: int = 1000,
    ):
        self._alarms: Dict[str, AlarmRecord] = {}
        self._max_alarms = max_alarms
        self._data_dir = data_dir

        # Statistics counters
        self._total_created = 0
        self._total_acknowledged = 0
        self._by_level: Dict[str, int] = {
            "info": 0,
            "warning": 0,
            "critical": 0,
            "emergency": 0,
        }

    # ==================== Alarm Lifecycle ====================

    def create_alarm(
        self,
        alarm_id: str,
        level: str,
        source_plugin: str,
        device_id: str,
        alarm_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AlarmRecord:
        """
        Create and register a new alarm.

        Args:
            alarm_id: Unique alarm identifier
            level: Severity level (info, warning, critical, emergency)
            source_plugin: Plugin that raised the alarm
            device_id: Device associated with the alarm
            alarm_type: Type of alarm (e.g., "detection", "threshold", "anomaly")
            message: Human-readable alarm description
            metadata: Optional additional data

        Returns:
            The created AlarmRecord
        """
        level = level.lower() if level else "warning"
        if level not in LEVEL_ORDER:
            level = "warning"

        record = AlarmRecord(
            alarm_id=alarm_id,
            level=level,
            source_plugin=source_plugin,
            device_id=device_id,
            alarm_type=alarm_type,
            message=message,
            metadata=metadata or {},
        )

        self._alarms[alarm_id] = record
        self._total_created += 1
        self._by_level[level] = self._by_level.get(level, 0) + 1

        # Evict old acknowledged alarms if over limit
        if len(self._alarms) > self._max_alarms:
            self._evict_old()

        logger.info(f"Alarm created: [{level}] {alarm_id} - {message}")
        return record

    def acknowledge(self, alarm_id: str) -> bool:
        """
        Acknowledge an alarm.

        Returns True if the alarm was found and acknowledged.
        """
        record = self._alarms.get(alarm_id)
        if record is None:
            return False

        if not record.acknowledged:
            record.acknowledged = True
            record.acknowledged_at = time.time()
            self._total_acknowledged += 1
            logger.info(f"Alarm acknowledged: {alarm_id}")

        return True

    def dismiss(self, alarm_id: str) -> bool:
        """Remove an alarm entirely. Returns True if found and removed."""
        return self._alarms.pop(alarm_id, None) is not None

    # ==================== Query ====================

    def get_alarm(self, alarm_id: str) -> Optional[Dict[str, Any]]:
        """Get a single alarm by ID."""
        record = self._alarms.get(alarm_id)
        return self._to_dict(record) if record else None

    def get_active_alarms(
        self,
        level: Optional[str] = None,
        source_plugin: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return active (unacknowledged) alarms, optionally filtered.

        Args:
            level: Filter by minimum level (e.g., "warning" includes warning+critical+emergency)
            source_plugin: Filter by source plugin ID
            limit: Maximum number of alarms to return
        """
        min_level = LEVEL_ORDER.get(level.lower(), 0) if level else 0

        results = []
        for record in self._alarms.values():
            if record.acknowledged:
                continue
            if LEVEL_ORDER.get(record.level, 0) < min_level:
                continue
            if source_plugin and record.source_plugin != source_plugin:
                continue
            results.append(self._to_dict(record))

        # Sort by severity (descending), then by timestamp (newest first)
        results.sort(
            key=lambda a: (-LEVEL_ORDER.get(a["level"], 0), -a["timestamp"])
        )
        return results[:limit]

    def get_all_alarms(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Return all alarms (both active and acknowledged)."""
        results = [self._to_dict(r) for r in self._alarms.values()]
        results.sort(key=lambda a: -a["timestamp"])
        return results[:limit]

    def get_alarm_stats(self) -> Dict[str, Any]:
        """Return alarm statistics summary."""
        active = sum(1 for a in self._alarms.values() if not a.acknowledged)
        acknowledged = sum(1 for a in self._alarms.values() if a.acknowledged)

        active_by_level: Dict[str, int] = {}
        for record in self._alarms.values():
            if not record.acknowledged:
                active_by_level[record.level] = active_by_level.get(record.level, 0) + 1

        return {
            "total_created": self._total_created,
            "total_acknowledged": self._total_acknowledged,
            "current_active": active,
            "current_acknowledged": acknowledged,
            "current_total": len(self._alarms),
            "active_by_level": active_by_level,
            "total_by_level": dict(self._by_level),
        }

    # ==================== Internal ====================

    @staticmethod
    def _to_dict(record: AlarmRecord) -> Dict[str, Any]:
        """Convert an AlarmRecord to a JSON-serializable dict."""
        return {
            "alarm_id": record.alarm_id,
            "level": record.level,
            "source_plugin": record.source_plugin,
            "device_id": record.device_id,
            "alarm_type": record.alarm_type,
            "message": record.message,
            "timestamp": record.timestamp,
            "acknowledged": record.acknowledged,
            "acknowledged_at": record.acknowledged_at,
            "metadata": record.metadata,
        }

    def _evict_old(self) -> None:
        """Evict oldest acknowledged alarms to stay under max_alarms."""
        acknowledged = [
            (aid, r) for aid, r in self._alarms.items() if r.acknowledged
        ]
        acknowledged.sort(key=lambda x: x[1].timestamp)

        to_remove = len(self._alarms) - self._max_alarms
        for i in range(min(to_remove, len(acknowledged))):
            del self._alarms[acknowledged[i][0]]
