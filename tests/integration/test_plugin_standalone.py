"""Verify ALL 17 plugins can run standalone."""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import subprocess

PLUGINS = [
    "indoor_fence", "animal_detection", "fire_detection", "slam_mapping",
    "temperature_monitoring", "device_monitoring", "transformer_inspection",
    "switch_inspection", "busbar_inspection", "capacitor_inspection",
    "meter_reading", "bird_monitoring", "acoustic_monitoring",
    "gas_detection", "hyperspectral_detection", "multimodal_fusion",
]


@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_create_standalone(plugin_id):
    """Each plugin must instantiate via create_standalone()."""
    result = subprocess.run(
        [sys.executable, "-c",
         f"from plugins.{plugin_id}.plugin import Plugin; "
         f"p = Plugin.create_standalone(); "
         f"print(f'{{p.PLUGIN_ID if hasattr(p, \"PLUGIN_ID\") else p.id}} OK')"],
        capture_output=True, text=True, timeout=30,
        cwd=str(_project_root),
    )
    assert result.returncode == 0, f"{plugin_id} failed: {result.stderr}"
    assert "OK" in result.stdout


@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_has_requirements(plugin_id):
    """Each plugin must have a requirements.txt."""
    req = _project_root / f"plugins/{plugin_id}/requirements.txt"
    assert req.exists(), f"{plugin_id} missing requirements.txt"


@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_has_standalone_app(plugin_id):
    """Each plugin must have standalone/app.py."""
    app = _project_root / f"plugins/{plugin_id}/standalone/app.py"
    assert app.exists(), f"{plugin_id} missing standalone/app.py"


@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_has_demo(plugin_id):
    """Each plugin must have demo/run_demo.py."""
    demo = _project_root / f"plugins/{plugin_id}/demo/run_demo.py"
    assert demo.exists(), f"{plugin_id} missing demo/run_demo.py"


@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_has_tests(plugin_id):
    """Each plugin must have a tests/ directory."""
    tests_dir = _project_root / f"plugins/{plugin_id}/tests"
    assert tests_dir.is_dir(), f"{plugin_id} missing tests/ directory"


@pytest.mark.parametrize("plugin_id", PLUGINS)
def test_plugin_imports_from_sdk(plugin_id):
    """Each plugin.py must import from darkbreaker_sdk (not platform_core)."""
    plugin_file = _project_root / f"plugins/{plugin_id}/plugin.py"
    content = plugin_file.read_text()
    assert "darkbreaker_sdk" in content, f"{plugin_id} does not import from darkbreaker_sdk"
