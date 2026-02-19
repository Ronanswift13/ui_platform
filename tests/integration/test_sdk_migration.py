"""Verify SDK migration is complete and platform_core is a thin compatibility layer."""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest


class TestSDKMigration:
    """Verify the SDK migration is complete."""

    def test_darkbreaker_sdk_imports(self):
        """SDK core modules must be importable."""
        from darkbreaker_sdk.interfaces import BasePlugin, PluginManifest, PluginContext
        from darkbreaker_sdk.interfaces import PluginCapability, PluginStatus, HealthStatus
        from darkbreaker_sdk.schemas import RecognitionResult, Alarm, AlarmLevel, BoundingBox
        from darkbreaker_sdk.schemas import ROI, ROIType, BaseEntity, Evidence
        from darkbreaker_sdk.standalone import StandalonePluginRunner
        from darkbreaker_sdk.utils import load_plugin_config, setup_plugin_logger

    def test_platform_core_reexports(self):
        """platform_core.plugin_manager.base must re-export from SDK."""
        from platform_core.plugin_manager.base import BasePlugin, PluginManifest, PluginContext
        from platform_core.plugin_manager.base import PluginStatus, HealthStatus
        from platform_core.plugin_manager.base import Alarm, RecognitionResult

    def test_sdk_and_platform_core_identical(self):
        """SDK and platform_core re-exports must be the same classes."""
        from darkbreaker_sdk.interfaces import BasePlugin as SDKBasePlugin
        from platform_core.plugin_manager.base import BasePlugin as PCBasePlugin
        assert SDKBasePlugin is PCBasePlugin

        from darkbreaker_sdk.interfaces import PluginStatus as SDKStatus
        from platform_core.plugin_manager.base import PluginStatus as PCStatus
        assert SDKStatus is PCStatus

    def test_no_direct_platform_core_in_plugins(self):
        """No plugin.py should directly import from platform_core (only SDK)."""
        plugins_dir = _project_root / "plugins"
        violations = []
        for plugin_dir in sorted(plugins_dir.iterdir()):
            if not plugin_dir.is_dir() or plugin_dir.name.startswith(('.', '_')):
                continue
            plugin_file = plugin_dir / "plugin.py"
            if not plugin_file.exists():
                continue
            content = plugin_file.read_text()
            # Check for direct platform_core imports (not in comments)
            for i, line in enumerate(content.split('\n'), 1):
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue
                if 'from platform_core' in stripped and 'import' in stripped:
                    violations.append(f"{plugin_dir.name}/plugin.py:{i}: {stripped}")

        assert len(violations) == 0, (
            f"Found {len(violations)} plugin(s) still importing from platform_core:\n"
            + "\n".join(violations)
        )

    def test_installer_available(self):
        """PluginInstaller must be importable and functional."""
        from platform_core.plugin_manager.installer import PluginInstaller
        installer = PluginInstaller(plugins_dir="plugins/")
        available = installer.list_available()
        assert len(available) >= 16, f"Expected >= 16 plugins, found {len(available)}"

    def test_all_plugins_have_manifest(self):
        """All plugins with plugin.py should also have manifest.json."""
        plugins_dir = _project_root / "plugins"
        missing = []
        for plugin_dir in sorted(plugins_dir.iterdir()):
            if not plugin_dir.is_dir() or plugin_dir.name.startswith(('.', '_')):
                continue
            if (plugin_dir / "plugin.py").exists() and not (plugin_dir / "manifest.json").exists():
                missing.append(plugin_dir.name)

        assert len(missing) == 0, f"Plugins missing manifest.json: {missing}"
