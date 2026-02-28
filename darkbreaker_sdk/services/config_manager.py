"""
Local Config Manager - Configuration management for standalone plugins.

Manages plugin configuration, voltage level selection, and device type settings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard power equipment device types
DEFAULT_DEVICE_TYPES = [
    {"id": "transformer", "name": "变压器", "category": "primary"},
    {"id": "circuit_breaker", "name": "断路器", "category": "primary"},
    {"id": "disconnector", "name": "隔离开关", "category": "primary"},
    {"id": "busbar", "name": "母线", "category": "primary"},
    {"id": "capacitor", "name": "电容器", "category": "primary"},
    {"id": "reactor", "name": "电抗器", "category": "primary"},
    {"id": "arrester", "name": "避雷器", "category": "protection"},
    {"id": "ct", "name": "电流互感器", "category": "measurement"},
    {"id": "pt", "name": "电压互感器", "category": "measurement"},
    {"id": "cable", "name": "电缆", "category": "transmission"},
    {"id": "tower", "name": "杆塔", "category": "transmission"},
    {"id": "insulator", "name": "绝缘子", "category": "transmission"},
    {"id": "gis", "name": "GIS设备", "category": "enclosed"},
    {"id": "switchgear", "name": "开关柜", "category": "enclosed"},
]


class LocalConfigManager:
    """
    Lightweight configuration manager for standalone plugin operation.

    Manages plugin-specific configuration, voltage level selection,
    and device type settings. Configuration is persisted to a JSON
    file in the plugin directory.
    """

    def __init__(
        self,
        plugin_dir: Optional[str] = None,
        device_types: Optional[List[Dict[str, str]]] = None,
    ):
        self._plugin_dir = Path(plugin_dir) if plugin_dir else None
        self._config: Dict[str, Any] = {}
        self._device_types = device_types or DEFAULT_DEVICE_TYPES
        self._current_voltage_level: Optional[str] = None
        self._current_device_type: Optional[str] = None
        self._config_path: Optional[Path] = None

        if self._plugin_dir:
            self._config_path = self._plugin_dir / "config" / "standalone_config.json"

    # ==================== Config File Operations ====================

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Args:
            config_path: Override path to config file

        Returns:
            The loaded configuration dict
        """
        path = Path(config_path) if config_path else self._config_path
        if path and path.exists():
            try:
                self._config = json.loads(path.read_text(encoding="utf-8"))
                self._current_voltage_level = self._config.get("voltage_level")
                self._current_device_type = self._config.get("device_type")
                logger.info(f"Config loaded from {path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")
                self._config = {}
        else:
            self._config = {}

        return dict(self._config)

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to JSON file.

        Returns True if saved successfully.
        """
        path = Path(config_path) if config_path else self._config_path
        if not path:
            logger.warning("No config path set, cannot save")
            return False

        try:
            # Merge current settings into config
            self._config["voltage_level"] = self._current_voltage_level
            self._config["device_type"] = self._current_device_type

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Config saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    # ==================== Configuration Access ====================

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value

    def get_config(self) -> Dict[str, Any]:
        """Return full configuration dict."""
        return dict(self._config)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Merge updates into current configuration."""
        self._config.update(updates)

    # ==================== Voltage Level ====================

    def get_voltage_level(self) -> Optional[str]:
        """Return current voltage level."""
        return self._current_voltage_level

    def set_voltage_level(self, level_id: str) -> None:
        """Set the active voltage level."""
        self._current_voltage_level = level_id
        self._config["voltage_level"] = level_id
        logger.info(f"Voltage level set to: {level_id}")

    # ==================== Device Type ====================

    def list_device_types(self) -> List[Dict[str, str]]:
        """Return available device types."""
        return list(self._device_types)

    def get_device_type(self) -> Optional[str]:
        """Return current device type."""
        return self._current_device_type

    def set_device_type(self, device_type: str) -> None:
        """Set the active device type."""
        self._current_device_type = device_type
        self._config["device_type"] = device_type
        logger.info(f"Device type set to: {device_type}")

    def get_device_type_info(self, device_type_id: str) -> Optional[Dict[str, str]]:
        """Get device type details by ID."""
        for dt in self._device_types:
            if dt["id"] == device_type_id:
                return dict(dt)
        return None

    def list_device_types_by_category(self, category: str) -> List[Dict[str, str]]:
        """Return device types filtered by category."""
        return [dt for dt in self._device_types if dt.get("category") == category]
