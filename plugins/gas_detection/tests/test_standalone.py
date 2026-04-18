from pathlib import Path

from fastapi.testclient import TestClient


def test_standalone_smoke_route_accepts_simulated_sample():
    from darkbreaker_sdk.standalone import StandalonePluginRunner
    from plugins.gas_detection.plugin import Plugin

    plugin = Plugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=Path(__file__).parents[1] / "standalone" / "templates",
        plugin_static_dir=Path(__file__).parents[1] / "standalone" / "static",
    )
    client = TestClient(runner.app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["healthy"] is True

    response = client.post("/api/gas/smoke", json={
        "device_id": "gas_smoke",
        "readings": {"SF6": 100.0, "H2": 20.0},
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["metadata"]["modality"] == "gas"
