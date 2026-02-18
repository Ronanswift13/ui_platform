"""
Tests for darkbreaker_sdk.interfaces.base_plugin

Validates BasePlugin, PluginManifest, PluginContext, and BaseAdapter.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from darkbreaker_sdk.interfaces.base_adapter import AdapterStatus, BaseAdapter
from darkbreaker_sdk.interfaces.base_plugin import (
    BasePlugin,
    PluginContext,
    PluginManifest,
)
from darkbreaker_sdk.interfaces.lifecycle import (
    HealthStatus,
    PluginCapability,
    PluginStatus,
)
from darkbreaker_sdk.schemas.alarm import Alarm, AlarmLevel, AlarmRule
from darkbreaker_sdk.schemas.common import ROI, ROIType
from darkbreaker_sdk.schemas.detection import BoundingBox, RecognitionResult


# ===================== Test Plugin Implementation =====================


class DummyPlugin(BasePlugin):
    """Minimal concrete plugin for testing."""

    PLUGIN_ID = "dummy-plugin"
    PLUGIN_NAME = "Dummy Plugin"
    PLUGIN_VERSION = "0.1.0"
    PLUGIN_DESCRIPTION = "A test plugin"

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
                label="test_detection",
                confidence=0.99,
            )
        ]

    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        return [
            Alarm(
                task_id=results[0].task_id if results else "unknown",
                title="Test Alarm",
                message="Test alarm message",
                site_id="site-1",
                device_id="dev-1",
            )
        ] if results else []

    def healthcheck(self) -> HealthStatus:
        return HealthStatus(healthy=True, message="OK")


# ===================== Tests =====================


class TestPluginManifest:
    """Tests for PluginManifest."""

    def test_basic_creation(self):
        m = PluginManifest(id="test", name="Test", version="1.0.0")
        assert m.id == "test"
        assert m.name == "Test"
        assert m.version == "1.0.0"
        assert m.entrypoint == "plugin.py"
        assert m.capabilities == []

    def test_from_dict(self):
        data = {
            "id": "my-plugin",
            "name": "My Plugin",
            "version": "2.0.0",
            "capabilities": ["defect_detection", "thermal_analysis"],
            "device_types": ["transformer"],
            "author": "Test Author",
        }
        m = PluginManifest.from_dict(data)
        assert m.id == "my-plugin"
        assert len(m.capabilities) == 2
        assert PluginCapability.DEFECT_DETECTION in m.capabilities
        assert m.author == "Test Author"

    def test_from_file(self):
        data = {
            "id": "file-plugin",
            "name": "File Plugin",
            "version": "1.0.0",
            "capabilities": ["meter_reading"],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            m = PluginManifest.from_file(path)
            assert m.id == "file-plugin"
            assert PluginCapability.METER_READING in m.capabilities
        finally:
            os.unlink(path)

    def test_from_dict_defaults(self):
        data = {"id": "min", "name": "Min", "version": "0.1.0"}
        m = PluginManifest.from_dict(data)
        assert m.description == ""
        assert m.entrypoint == "plugin.py"
        assert m.plugin_class == "Plugin"
        assert m.min_platform_version == "1.0.0"


class TestPluginContext:
    """Tests for PluginContext."""

    def test_basic_context(self):
        ctx = PluginContext(
            task_id="task-1",
            site_id="site-1",
            device_id="dev-1",
        )
        assert ctx.task_id == "task-1"
        assert ctx.component_id == ""
        assert isinstance(ctx.timestamp, datetime)

    def test_to_dict(self):
        ctx = PluginContext(
            task_id="task-1",
            site_id="site-1",
            device_id="dev-1",
            config={"threshold": 0.5},
        )
        d = ctx.to_dict()
        assert d["task_id"] == "task-1"
        assert d["config"]["threshold"] == 0.5
        assert "timestamp" in d


class TestBasePlugin:
    """Tests for BasePlugin ABC."""

    def _make_plugin(self, **config) -> DummyPlugin:
        manifest = PluginManifest(id="dummy", name="Dummy", version="1.0.0")
        plugin = DummyPlugin(manifest, Path("."))
        plugin.init(config)
        return plugin

    def test_instantiation(self):
        plugin = self._make_plugin()
        assert plugin.id == "dummy"
        assert plugin.name == "Dummy"
        assert plugin.version == "1.0.0"
        assert plugin.status == PluginStatus.READY

    def test_infer(self):
        plugin = self._make_plugin()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = PluginContext(task_id="t1", site_id="s1", device_id="d1")
        results = plugin.infer(frame, [], ctx)
        assert len(results) == 1
        assert results[0].label == "test_detection"
        assert results[0].confidence == 0.99

    def test_postprocess(self):
        plugin = self._make_plugin()
        result = RecognitionResult(
            task_id="t1",
            site_id="s1",
            device_id="d1",
            component_id="c1",
            roi_id="r1",
            bbox=BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4),
            label="test",
            confidence=0.9,
        )
        alarms = plugin.postprocess([result], [])
        assert len(alarms) == 1
        assert alarms[0].title == "Test Alarm"

    def test_healthcheck(self):
        plugin = self._make_plugin()
        health = plugin.healthcheck()
        assert health.healthy is True

    def test_create_output(self):
        plugin = self._make_plugin()
        output = plugin.create_output(
            task_id="t1",
            results=[],
            alarms=[],
            processing_time_ms=10.5,
        )
        assert output.task_id == "t1"
        assert output.plugin_id == "dummy"
        assert output.processing_time_ms == 10.5

    def test_set_status(self):
        plugin = self._make_plugin()
        plugin.set_status(PluginStatus.ERROR, "something broke")
        assert plugin.status == PluginStatus.ERROR
        assert plugin._last_error == "something broke"

    def test_create_standalone(self):
        plugin = DummyPlugin.create_standalone(config={"threshold": 0.7})
        assert plugin.id == "dummy-plugin"
        assert plugin.name == "Dummy Plugin"
        assert plugin._config == {"threshold": 0.7}
        assert plugin.status == PluginStatus.READY

    def test_get_standalone_routes_default_empty(self):
        plugin = self._make_plugin()
        assert plugin.get_standalone_routes() == []

    def test_repr(self):
        plugin = self._make_plugin()
        r = repr(plugin)
        assert "dummy" in r
        assert "1.0.0" in r

    def test_cleanup_no_error(self):
        plugin = self._make_plugin()
        plugin.cleanup()  # Should not raise

    def test_on_config_update(self):
        plugin = self._make_plugin()
        plugin.on_config_update({"new_key": "new_val"})
        assert plugin._config == {"new_key": "new_val"}

    def test_get_ui_config_default_none(self):
        plugin = self._make_plugin()
        assert plugin.get_ui_config() is None

    def test_analyze_thermal_default_none(self):
        plugin = self._make_plugin()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert plugin.analyze_thermal(frame) is None


class TestBaseAdapter:
    """Tests for BaseAdapter and AdapterStatus."""

    def test_adapter_status_values(self):
        expected = {"disconnected", "connecting", "connected", "error", "simulated"}
        actual = {s.value for s in AdapterStatus}
        assert actual == expected

    def test_adapter_subclass(self):
        class DummyAdapter(BaseAdapter):
            async def connect(self, config=None):
                self._status = AdapterStatus.CONNECTED
                return True

            async def disconnect(self):
                self._status = AdapterStatus.DISCONNECTED

        adapter = DummyAdapter()
        assert adapter.status == AdapterStatus.DISCONNECTED
        assert adapter.is_simulated is False
        assert repr(adapter).startswith("<DummyAdapter")

    def test_adapter_simulated(self):
        class SimAdapter(BaseAdapter):
            async def connect(self, config=None):
                self._status = AdapterStatus.SIMULATED
                return True

            async def disconnect(self):
                self._status = AdapterStatus.DISCONNECTED

        adapter = SimAdapter()
        import asyncio
        asyncio.get_event_loop().run_until_complete(adapter.connect())
        assert adapter.is_simulated is True
