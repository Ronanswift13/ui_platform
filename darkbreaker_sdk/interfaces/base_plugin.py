"""
Base plugin definition.

Contains BasePlugin ABC, PluginManifest, and PluginContext - extracted from
platform_core/plugin_manager/base.py with imports redirected to the SDK.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from darkbreaker_sdk.interfaces.lifecycle import (
    HealthStatus,
    PluginCapability,
    PluginStatus,
)
from darkbreaker_sdk.schemas.alarm import Alarm, AlarmRule
from darkbreaker_sdk.schemas.common import ROI
from darkbreaker_sdk.schemas.detection import RecognitionResult
from darkbreaker_sdk.schemas.plugin_io import PluginOutput


@dataclass
class PluginManifest:
    """Plugin manifest - loaded from manifest.json."""
    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    category: str = ""  # "indoor" or "outdoor"
    entrypoint: str = "plugin.py"
    plugin_class: str = "Plugin"
    capabilities: list[PluginCapability] = field(default_factory=list)
    device_types: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    min_platform_version: str = "1.0.0"
    config_schema: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginManifest":
        """Create manifest from dictionary."""
        capabilities = [
            PluginCapability(c) if isinstance(c, str) else c
            for c in data.get("capabilities", [])
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            author=data.get("author", ""),
            category=data.get("category", ""),
            entrypoint=data.get("entrypoint", "plugin.py"),
            plugin_class=data.get("plugin_class", "Plugin"),
            capabilities=capabilities,
            device_types=data.get("device_types", []),
            dependencies=data.get("dependencies", []),
            min_platform_version=data.get("min_platform_version", "1.0.0"),
            config_schema=data.get("config_schema", {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "PluginManifest":
        """Load manifest from a JSON file.

        Args:
            path: Path to manifest JSON file.

        Returns:
            PluginManifest instance.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class PluginContext:
    """Plugin runtime context."""
    task_id: str
    site_id: str
    device_id: str
    component_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "site_id": self.site_id,
            "device_id": self.device_id,
            "component_id": self.component_id,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config,
            "metadata": self.metadata,
        }


class BasePlugin(ABC):
    """
    Plugin base class.

    All plugins must inherit from this class and implement:
    - init(config): Initialize plugin
    - infer(frame, rois, context): Run inference
    - postprocess(results, rules): Post-process and generate alarms
    - healthcheck(): Health check
    """

    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        self.manifest = manifest
        self.plugin_dir = plugin_dir
        self.status = PluginStatus.UNLOADED
        self._config: dict[str, Any] = {}
        self._code_hash: str = ""
        self._last_error: str = ""

    @classmethod
    def create_standalone(cls, config: dict[str, Any] | None = None) -> "BasePlugin":
        """Create a plugin instance for standalone operation.

        Creates the plugin with a minimal manifest and initializes it
        with the provided config.

        Args:
            config: Optional configuration dictionary.

        Returns:
            An initialized plugin instance.
        """
        manifest = PluginManifest(
            id=getattr(cls, "PLUGIN_ID", cls.__name__.lower()),
            name=getattr(cls, "PLUGIN_NAME", cls.__name__),
            version=getattr(cls, "PLUGIN_VERSION", "1.0.0"),
            description=getattr(cls, "PLUGIN_DESCRIPTION", ""),
        )
        plugin_dir = Path(getattr(cls, "PLUGIN_DIR", "."))
        instance = cls(manifest, plugin_dir)
        instance.init(config or {})
        return instance

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def code_hash(self) -> str:
        """Calculate plugin code hash (for traceability)."""
        if not self._code_hash:
            self._code_hash = self._calculate_code_hash()
        return self._code_hash

    def _calculate_code_hash(self) -> str:
        """Calculate hash of all Python files in plugin directory."""
        hasher = hashlib.sha256()
        plugin_dir = Path(self.plugin_dir)
        if plugin_dir.exists():
            for py_file in sorted(plugin_dir.rglob("*.py")):
                hasher.update(py_file.read_bytes())
        return hasher.hexdigest()[:12]

    # ============== Required interface methods ==============

    @abstractmethod
    def init(self, config: dict[str, Any]) -> bool:
        """
        Initialize plugin.

        Args:
            config: Plugin configuration dictionary.

        Returns:
            Whether initialization succeeded.
        """
        pass

    @abstractmethod
    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """
        Run inference.

        Args:
            frame: Input image frame (BGR format, numpy array).
            rois: List of regions of interest.
            context: Runtime context.

        Returns:
            List of recognition results.
        """
        pass

    @abstractmethod
    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """
        Post-process and generate alarms.

        Args:
            results: List of inference results.
            rules: List of alarm rules.

        Returns:
            List of alarms.
        """
        pass

    @abstractmethod
    def healthcheck(self) -> HealthStatus:
        """
        Health check.

        Returns:
            Health status.
        """
        pass

    # ============== Optional interface methods ==============

    def cleanup(self) -> None:
        """Clean up resources (optional)."""
        pass

    def on_config_update(self, new_config: dict[str, Any]) -> None:
        """Config update callback (optional)."""
        self._config = new_config

    def analyze_thermal(
        self, frame: np.ndarray, config: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Thermal image analysis (optional)."""
        return None

    def get_model_version(self) -> str:
        """Get model version (optional)."""
        return self.version

    def get_ui_config(self) -> dict[str, Any] | None:
        """Get plugin UI configuration (optional)."""
        return None

    def get_standalone_routes(self) -> list:
        """Return additional standalone routes for this plugin.

        Override in subclasses to add plugin-specific API routes.
        Each item should be a dict with keys: path, endpoint, methods, summary.

        Returns:
            List of route definitions (empty by default).
        """
        return []

    # ============== Helper methods ==============

    def create_output(
        self,
        task_id: str,
        results: list[RecognitionResult],
        alarms: list[Alarm],
        processing_time_ms: float = 0,
        success: bool = True,
        error_message: str = "",
    ) -> PluginOutput:
        """Create standard plugin output."""
        return PluginOutput(
            task_id=task_id,
            plugin_id=self.id,
            plugin_version=self.version,
            code_hash=self.code_hash,
            success=success,
            results=results,
            alarms=alarms,
            error_message=error_message,
            processing_time_ms=processing_time_ms,
        )

    def set_status(self, status: PluginStatus, error: str = "") -> None:
        """Set plugin status."""
        self.status = status
        if error:
            self._last_error = error

    def __repr__(self) -> str:
        return f"<Plugin {self.id} v{self.version} [{self.status.value}]>"
