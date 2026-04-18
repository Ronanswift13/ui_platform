"""
Plugin Installer - Enable/Disable Controller

MATLAB-style plugin management: users can enable/disable individual plugins.
Manages plugin state persistence and dependency checking.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Plugin information for display."""
    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    category: str = "other"  # indoor / outdoor / monitoring / other
    enabled: bool = True
    capabilities: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    standalone_port: int = 0
    has_standalone: bool = False
    has_demo: bool = False
    has_tests: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Plugin category mapping
PLUGIN_CATEGORIES = {
    # Indoor plugins
    "indoor_fence": "indoor",
    "animal_detection": "indoor",
    "fire_detection": "indoor",
    "slam_mapping": "indoor",
    "temperature_monitoring": "indoor",
    "device_monitoring": "indoor",
    # Outdoor plugins
    "transformer_inspection": "outdoor",
    "switch_inspection": "outdoor",
    "busbar_inspection": "outdoor",
    "capacitor_inspection": "outdoor",
    "meter_reading": "outdoor",
    "bird_monitoring": "outdoor",
    # Monitoring plugins (shared indoor/outdoor)
    "acoustic_monitoring": "monitoring",
    "gas_detection": "monitoring",
    "action_event_monitoring": "monitoring",
    "hyperspectral_detection": "monitoring",
    "multimodal_fusion": "monitoring",
}

# Standalone port mapping
PLUGIN_PORTS = {
    "indoor_fence": 8081,
    "animal_detection": 8082,
    "fire_detection": 8083,
    "slam_mapping": 8084,
    "temperature_monitoring": 8085,
    "device_monitoring": 8086,
    "transformer_inspection": 8087,
    "switch_inspection": 8088,
    "busbar_inspection": 8089,
    "capacitor_inspection": 8090,
    "meter_reading": 8091,
    "bird_monitoring": 8092,
    "acoustic_monitoring": 8093,
    "gas_detection": 8094,
    "action_event_monitoring": 8097,
    "hyperspectral_detection": 8095,
    "multimodal_fusion": 8096,
}


class PluginInstaller:
    """
    Plugin enable/disable controller (MATLAB Add-On Explorer style).

    Manages which plugins are active in the platform.
    State is persisted in a JSON config file.
    """

    def __init__(
        self,
        plugins_dir: str | Path = "plugins",
        config_path: str | Path | None = None,
    ):
        self.plugins_dir = Path(plugins_dir)
        self.config_path = Path(config_path) if config_path else self.plugins_dir / ".enabled_plugins.json"
        self._enabled: set[str] = set()
        self._load_state()

    def _load_state(self) -> None:
        """Load enabled/disabled state from config file."""
        if self.config_path.exists():
            try:
                data = json.loads(self.config_path.read_text(encoding="utf-8"))
                self._enabled = set(data.get("enabled", []))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load plugin state: {e}")
                self._enabled = set()
        else:
            # Default: all discovered plugins are enabled
            self._enabled = set(self._discover_plugin_ids())
            self._save_state()

    def _save_state(self) -> None:
        """Persist enabled/disabled state."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "enabled": sorted(self._enabled),
                "version": "1.0.0",
            }
            self.config_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to save plugin state: {e}")

    def _discover_plugin_ids(self) -> list[str]:
        """Discover all plugin directories with manifest.json."""
        plugin_ids = []
        if not self.plugins_dir.exists():
            return plugin_ids
        for entry in sorted(self.plugins_dir.iterdir()):
            if entry.is_dir() and (entry / "manifest.json").exists():
                plugin_ids.append(entry.name)
        return plugin_ids

    def _load_manifest(self, plugin_id: str) -> dict[str, Any]:
        """Load a plugin's manifest.json."""
        manifest_path = self.plugins_dir / plugin_id / "manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def list_available(self) -> list[PluginInfo]:
        """List all available plugins with their enable status."""
        plugins = []
        for plugin_id in self._discover_plugin_ids():
            manifest = self._load_manifest(plugin_id)
            plugin_dir = self.plugins_dir / plugin_id

            info = PluginInfo(
                id=plugin_id,
                name=manifest.get("name", plugin_id),
                version=manifest.get("version", "0.0.0"),
                description=manifest.get("description", ""),
                author=manifest.get("author", ""),
                category=PLUGIN_CATEGORIES.get(plugin_id, "other"),
                enabled=plugin_id in self._enabled,
                capabilities=[str(c) for c in manifest.get("capabilities", [])],
                dependencies=manifest.get("dependencies", []),
                standalone_port=PLUGIN_PORTS.get(plugin_id, 0),
                has_standalone=(plugin_dir / "standalone" / "app.py").exists(),
                has_demo=(plugin_dir / "demo" / "run_demo.py").exists(),
                has_tests=(plugin_dir / "tests").is_dir(),
            )
            plugins.append(info)

        return plugins

    def list_by_category(self) -> dict[str, list[PluginInfo]]:
        """List plugins grouped by category."""
        plugins = self.list_available()
        categories: dict[str, list[PluginInfo]] = {
            "indoor": [],
            "outdoor": [],
            "monitoring": [],
            "other": [],
        }
        for p in plugins:
            categories.setdefault(p.category, []).append(p)
        return categories

    def enable(self, plugin_id: str) -> bool:
        """Enable a plugin."""
        available = self._discover_plugin_ids()
        if plugin_id not in available:
            logger.warning(f"Plugin '{plugin_id}' not found")
            return False

        self._enabled.add(plugin_id)
        self._save_state()
        logger.info(f"Plugin '{plugin_id}' enabled")
        return True

    def disable(self, plugin_id: str) -> bool:
        """Disable a plugin."""
        if plugin_id not in self._enabled:
            logger.warning(f"Plugin '{plugin_id}' is not enabled")
            return False

        self._enabled.discard(plugin_id)
        self._save_state()
        logger.info(f"Plugin '{plugin_id}' disabled")
        return True

    def is_enabled(self, plugin_id: str) -> bool:
        """Check if a plugin is enabled."""
        return plugin_id in self._enabled

    def get_enabled(self) -> list[str]:
        """Get list of enabled plugin IDs."""
        return sorted(self._enabled)

    def get_disabled(self) -> list[str]:
        """Get list of disabled plugin IDs."""
        all_ids = set(self._discover_plugin_ids())
        return sorted(all_ids - self._enabled)

    def enable_all(self) -> None:
        """Enable all plugins."""
        self._enabled = set(self._discover_plugin_ids())
        self._save_state()

    def disable_all(self) -> None:
        """Disable all plugins."""
        self._enabled.clear()
        self._save_state()

    def check_dependencies(self, plugin_id: str) -> list[str]:
        """
        Check if a plugin's Python dependencies are installed.

        Returns list of missing packages.
        """
        manifest = self._load_manifest(plugin_id)
        deps = manifest.get("dependencies", [])
        missing = []

        for dep in deps:
            # Normalize package name (e.g., opencv-python -> cv2)
            pkg_name = dep.split(">=")[0].split("==")[0].strip()
            import_map = {
                "opencv-python": "cv2",
                "pillow": "PIL",
                "scikit-learn": "sklearn",
                "pyyaml": "yaml",
                "darkbreaker-sdk": "darkbreaker_sdk",
            }
            import_name = import_map.get(pkg_name, pkg_name.replace("-", "_"))

            try:
                __import__(import_name)
            except ImportError:
                missing.append(dep)

        return missing

    def get_plugin_info(self, plugin_id: str) -> PluginInfo | None:
        """Get detailed info about a specific plugin."""
        for info in self.list_available():
            if info.id == plugin_id:
                return info
        return None
