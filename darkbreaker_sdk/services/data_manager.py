"""
Local Data Manager - Lightweight data persistence for standalone plugins.

Uses SQLite for results storage and provides voltage-level-aware data isolation.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import sqlite3
except ImportError:
    sqlite3 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Standard voltage levels used in power equipment monitoring
DEFAULT_VOLTAGE_LEVELS = [
    {"id": "10kV", "name": "10kV", "description": "配电网"},
    {"id": "35kV", "name": "35kV", "description": "中压输电"},
    {"id": "110kV", "name": "110kV", "description": "高压输电"},
    {"id": "220kV", "name": "220kV", "description": "超高压输电"},
    {"id": "500kV", "name": "500kV", "description": "特高压直流"},
    {"id": "1000kV", "name": "1000kV", "description": "特高压交流"},
]


class LocalDataManager:
    """
    Lightweight data persistence for standalone plugin operation.

    Stores detection results in a local SQLite database with
    voltage-level-aware data isolation. Provides result queries
    with pagination and time-range filtering.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        voltage_levels: Optional[List[Dict[str, str]]] = None,
    ):
        self._data_dir = Path(data_dir or "data")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "results.db"
        self._voltage_levels = voltage_levels or DEFAULT_VOLTAGE_LEVELS
        self._current_voltage_level: Optional[str] = None
        self._use_sqlite = sqlite3 is not None
        self._memory_results: List[Dict[str, Any]] = []  # fallback when sqlite unavailable
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with results table."""
        if not self._use_sqlite:
            logger.warning("sqlite3 not available, using in-memory fallback for data storage")
            return
        try:
            with self._connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS detection_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plugin_id TEXT NOT NULL,
                        voltage_level TEXT,
                        timestamp REAL NOT NULL,
                        data TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_results_plugin
                    ON detection_results(plugin_id, timestamp DESC)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_results_voltage
                    ON detection_results(voltage_level, timestamp DESC)
                """)
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self._use_sqlite = False

    def _connect(self):
        """Create a database connection."""
        conn = sqlite3.connect(str(self._db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ==================== Voltage Level Management ====================

    def list_voltage_levels(self) -> List[Dict[str, str]]:
        """Return available voltage levels."""
        return list(self._voltage_levels)

    def get_voltage_level(self) -> Optional[str]:
        """Return current voltage level."""
        return self._current_voltage_level

    def set_voltage_level(self, level_id: str) -> None:
        """Set the active voltage level for data filtering."""
        self._current_voltage_level = level_id
        logger.info(f"Voltage level set to: {level_id}")

    # ==================== Result Storage ====================

    def save_result(
        self,
        plugin_id: str,
        data: Dict[str, Any],
        voltage_level: Optional[str] = None,
    ) -> bool:
        """
        Save a detection result.

        Args:
            plugin_id: ID of the plugin that generated the result
            data: Result data to store (will be JSON-serialized)
            voltage_level: Override voltage level (defaults to current)

        Returns:
            True if saved successfully
        """
        level = voltage_level or self._current_voltage_level
        if not self._use_sqlite:
            self._memory_results.append({
                "id": len(self._memory_results) + 1,
                "plugin_id": plugin_id,
                "voltage_level": level,
                "timestamp": time.time(),
                "data": data,
            })
            return True
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO detection_results
                       (plugin_id, voltage_level, timestamp, data)
                       VALUES (?, ?, ?, ?)""",
                    (plugin_id, level, time.time(), json.dumps(data, ensure_ascii=False)),
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            return False

    def get_results(
        self,
        plugin_id: str,
        limit: int = 100,
        offset: int = 0,
        voltage_level: Optional[str] = None,
        since: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve detection results with optional filtering.

        Args:
            plugin_id: Filter by plugin ID
            limit: Maximum number of results to return
            offset: Pagination offset
            voltage_level: Filter by voltage level
            since: Only return results after this Unix timestamp

        Returns:
            List of result dicts with id, plugin_id, voltage_level, timestamp, data
        """
        level = voltage_level or self._current_voltage_level

        if not self._use_sqlite:
            # In-memory fallback query
            filtered = [r for r in self._memory_results if r["plugin_id"] == plugin_id]
            if level:
                filtered = [r for r in filtered if r.get("voltage_level") == level]
            if since is not None:
                filtered = [r for r in filtered if r["timestamp"] > since]
            filtered.sort(key=lambda r: r["timestamp"], reverse=True)
            return filtered[offset:offset + limit]

        conditions = ["plugin_id = ?"]
        params: list[Any] = [plugin_id]

        if level:
            conditions.append("voltage_level = ?")
            params.append(level)

        if since is not None:
            conditions.append("timestamp > ?")
            params.append(since)

        where = " AND ".join(conditions)
        params.extend([limit, offset])

        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""SELECT id, plugin_id, voltage_level, timestamp, data
                        FROM detection_results
                        WHERE {where}
                        ORDER BY timestamp DESC
                        LIMIT ? OFFSET ?""",
                    params,
                ).fetchall()

            results = []
            for row in rows:
                try:
                    data = json.loads(row["data"])
                except (json.JSONDecodeError, TypeError):
                    data = row["data"]
                results.append({
                    "id": row["id"],
                    "plugin_id": row["plugin_id"],
                    "voltage_level": row["voltage_level"],
                    "timestamp": row["timestamp"],
                    "data": data,
                })
            return results

        except Exception as e:
            logger.error(f"Failed to query results: {e}")
            return []

    def count_results(self, plugin_id: str) -> int:
        """Return total number of stored results for a plugin."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM detection_results WHERE plugin_id = ?",
                    (plugin_id,),
                ).fetchone()
                return row["cnt"] if row else 0
        except Exception:
            return 0

    def clear_results(self, plugin_id: str) -> int:
        """Delete all results for a plugin. Returns number deleted."""
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM detection_results WHERE plugin_id = ?",
                    (plugin_id,),
                )
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Failed to clear results: {e}")
            return 0
