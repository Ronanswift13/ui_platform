from fastapi.testclient import TestClient

from plugins.multimodal_fusion.plugin import Plugin
from plugins.multimodal_fusion.standalone.app import create_app, create_runner


def isolated_plugin(tmp_path):
    plugin = Plugin.create_standalone()
    plugin.plugin_dir = tmp_path
    return plugin


def test_create_standalone_runner_exposes_health_status_and_demo_routes(tmp_path):
    runner = create_runner(plugin=isolated_plugin(tmp_path))
    client = TestClient(runner.app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["healthy"] is True
    assert client.get("/health").json()["healthy"] is True

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["plugin_id"] == "multimodal_fusion"
    assert client.get("/status").json()["plugin_id"] == "multimodal_fusion"

    smoke = client.post("/smoke")
    assert smoke.status_code == 200
    assert smoke.json()["success"] is True
    assert "fused_status" in smoke.json()

    demo = client.post("/fuse-demo")
    assert demo.status_code == 200
    payload = demo.json()
    assert payload["success"] is True
    assert payload["metadata"]["algorithm_stage"] == "stage_1_rule_fusion_contract"


def test_create_app_returns_fastapi_app_with_fuse_demo(tmp_path):
    client = TestClient(create_app(plugin=isolated_plugin(tmp_path)))

    response = client.post("/api/multimodal/fuse-demo")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["evidence_chain"][0]["simulated"] is True
