from pathlib import Path

from fastapi.testclient import TestClient


def test_standalone_smoke_route_accepts_simulated_sample():
    from darkbreaker_sdk.standalone import StandalonePluginRunner
    from plugins.action_event_monitoring.plugin import ActionEventMonitoringPlugin

    plugin = ActionEventMonitoringPlugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parents[1] / "standalone" / "templates",
        plugin_static_dir=Path(__file__).parents[1] / "standalone" / "static",
    )
    client = TestClient(runner.app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["healthy"] is True

    response = client.post("/api/action-event/smoke", json={"device_id": "relay_smoke"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["metadata"]["modality"] == "action_event"
