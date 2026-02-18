"""
Tests for darkbreaker_sdk.standalone.runner

Validates StandalonePluginRunner FastAPI application.
"""

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from darkbreaker_sdk.interfaces.base_plugin import (
    BasePlugin,
    PluginContext,
    PluginManifest,
)
from darkbreaker_sdk.interfaces.lifecycle import HealthStatus, PluginStatus
from darkbreaker_sdk.schemas.alarm import Alarm, AlarmLevel, AlarmRule
from darkbreaker_sdk.schemas.common import ROI
from darkbreaker_sdk.schemas.detection import BoundingBox, RecognitionResult
from darkbreaker_sdk.standalone.runner import StandalonePluginRunner


# ===================== Test Plugin =====================


class TestPlugin(BasePlugin):
    """Concrete plugin for testing the standalone runner."""

    def init(self, config: dict[str, Any]) -> bool:
        self._config = config
        self.status = PluginStatus.READY
        return True

    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        return [
            RecognitionResult(
                task_id=context.task_id,
                site_id=context.site_id,
                device_id=context.device_id,
                component_id="comp-1",
                roi_id="roi-1",
                bbox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
                label="test_object",
                confidence=0.85,
            )
        ]

    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        if not results:
            return []
        return [
            Alarm(
                task_id=results[0].task_id,
                title="Test Alarm",
                message="Object detected",
                site_id="site-1",
                device_id="dev-1",
                level=AlarmLevel.WARNING,
            )
        ]

    def healthcheck(self) -> HealthStatus:
        return HealthStatus(healthy=True, message="All systems go")


# ===================== Fixtures =====================


@pytest.fixture
def plugin():
    manifest = PluginManifest(
        id="test-plugin",
        name="Test Plugin",
        version="1.0.0",
        config_schema={"threshold": {"type": "number", "default": 0.5}},
    )
    p = TestPlugin(manifest, Path("."))
    p.init({"threshold": 0.5, "debug": False})
    return p


@pytest.fixture
def runner(plugin):
    return StandalonePluginRunner(plugin)


@pytest.fixture
def client(runner):
    return TestClient(runner.app)


# ===================== Tests =====================


class TestStandaloneRunnerAPI:
    """Tests for the REST API endpoints."""

    def test_dashboard(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Test Plugin" in response.text

    def test_get_status(self, client):
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["plugin_id"] == "test-plugin"
        assert data["plugin_name"] == "Test Plugin"
        assert data["status"] == "ready"
        assert data["health"]["healthy"] is True
        assert "stats" in data

    def test_get_config(self, client):
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert data["config"]["threshold"] == 0.5

    def test_update_config(self, client):
        response = client.put(
            "/api/config",
            json={"config": {"threshold": 0.8, "debug": True}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"
        assert data["config"]["threshold"] == 0.8

        # Verify the update persisted
        response2 = client.get("/api/config")
        assert response2.json()["config"]["threshold"] == 0.8

    def test_detect_with_image(self, client):
        # Create a simple test image (100x100 black PNG-compatible JPEG)
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not installed")

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        image_bytes = buf.tobytes()

        response = client.post(
            "/api/detect",
            files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["label"] == "test_object"
        assert len(data["alarms"]) == 1
        assert data["inference_time_ms"] >= 0

    def test_detect_updates_stats(self, client):
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 not installed")

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        image_bytes = buf.tobytes()

        # Run detection twice
        client.post(
            "/api/detect",
            files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )
        client.post(
            "/api/detect",
            files={"file": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )

        # Check stats
        response = client.get("/api/status")
        stats = response.json()["stats"]
        assert stats["total_frames"] == 2
        assert stats["total_detections"] == 2
        assert stats["total_alarms"] == 2

    def test_detect_invalid_image(self, client):
        try:
            import cv2  # noqa: F401
        except ImportError:
            pytest.skip("cv2 not installed")

        response = client.post(
            "/api/detect",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data


class TestStandaloneRunnerInit:
    """Tests for runner initialization."""

    def test_default_title(self, plugin):
        runner = StandalonePluginRunner(plugin)
        assert runner.title == "Test Plugin - Standalone"

    def test_custom_title(self, plugin):
        runner = StandalonePluginRunner(plugin, title="My Custom Title")
        assert runner.title == "My Custom Title"

    def test_initial_stats(self, runner):
        assert runner._stats["total_frames"] == 0
        assert runner._stats["total_detections"] == 0
        assert runner._stats["total_alarms"] == 0
        assert runner._stats["avg_inference_time_ms"] == 0.0

    def test_plugin_reference(self, runner, plugin):
        assert runner.plugin is plugin

    def test_update_stats(self, runner):
        runner._update_stats(3, 1, 25.0)
        assert runner._stats["total_frames"] == 1
        assert runner._stats["total_detections"] == 3
        assert runner._stats["total_alarms"] == 1
        assert runner._stats["avg_inference_time_ms"] == 25.0

        runner._update_stats(2, 0, 15.0)
        assert runner._stats["total_frames"] == 2
        assert runner._stats["total_detections"] == 5
        assert runner._stats["avg_inference_time_ms"] == 20.0
