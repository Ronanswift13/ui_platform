"""
Verify ALL 16 plugins boot standalone with working endpoints.

Tests that each plugin can:
1. Import and create_standalone()
2. Create a StandalonePluginRunner
3. Serve a dashboard at GET /
4. Respond to GET /api/health
5. Respond to GET /api/status
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from starlette.testclient import TestClient

PLUGIN_CONFIGS = [
    ("indoor_fence", 8081),
    ("animal_detection", 8082),
    ("fire_detection", 8083),
    ("slam_mapping", 8084),
    ("temperature_monitoring", 8085),
    ("device_monitoring", 8086),
    ("transformer_inspection", 8087),
    ("switch_inspection", 8088),
    ("busbar_inspection", 8089),
    ("capacitor_inspection", 8090),
    ("meter_reading", 8091),
    ("bird_monitoring", 8092),
    ("acoustic_monitoring", 8093),
    ("gas_detection", 8094),
    ("hyperspectral_detection", 8095),
    ("multimodal_fusion", 8096),
]


def _boot_plugin(plugin_id: str):
    """Boot a single plugin and return (runner, client)."""
    import importlib
    plugin_module = importlib.import_module(f"plugins.{plugin_id}.plugin")
    PluginCls = getattr(plugin_module, "Plugin")
    plugin = PluginCls.create_standalone()

    from darkbreaker_sdk.standalone import StandalonePluginRunner
    plugin_dir = _project_root / "plugins" / plugin_id
    templates_dir = plugin_dir / "standalone" / "templates"
    static_dir = plugin_dir / "standalone" / "static"

    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=templates_dir if templates_dir.exists() else None,
        plugin_static_dir=static_dir if static_dir.exists() else None,
    )
    client = TestClient(runner.app)
    return runner, client


@pytest.mark.parametrize("plugin_id,port", PLUGIN_CONFIGS)
def test_plugin_dashboard(plugin_id, port):
    """Dashboard GET / should return 200."""
    runner, client = _boot_plugin(plugin_id)
    resp = client.get("/")
    assert resp.status_code == 200, f"{plugin_id} dashboard failed: {resp.text[:200]}"
    assert len(resp.text) > 100, f"{plugin_id} dashboard too small"


@pytest.mark.parametrize("plugin_id,port", PLUGIN_CONFIGS)
def test_plugin_health(plugin_id, port):
    """GET /api/health should return healthy status."""
    runner, client = _boot_plugin(plugin_id)
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"{plugin_id} health failed"
    data = resp.json()
    assert "healthy" in data, f"{plugin_id} health missing 'healthy' key"


@pytest.mark.parametrize("plugin_id,port", PLUGIN_CONFIGS)
def test_plugin_status(plugin_id, port):
    """GET /api/status should return plugin status."""
    runner, client = _boot_plugin(plugin_id)
    resp = client.get("/api/status")
    assert resp.status_code == 200, f"{plugin_id} status failed"
    data = resp.json()
    assert data.get("success") is True, f"{plugin_id} status not success: {data}"
