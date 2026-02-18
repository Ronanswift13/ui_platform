"""
Tests for darkbreaker_sdk.interfaces.lifecycle

Validates PluginCapability, PluginStatus, and HealthStatus.
"""

import pytest
from datetime import datetime

from darkbreaker_sdk.interfaces.lifecycle import (
    HealthStatus,
    PluginCapability,
    PluginStatus,
)


class TestPluginCapability:
    """Tests for PluginCapability enum."""

    def test_is_string_enum(self):
        assert isinstance(PluginCapability.DEFECT_DETECTION, str)
        assert PluginCapability.DEFECT_DETECTION == "defect_detection"

    def test_all_values_are_strings(self):
        for cap in PluginCapability:
            assert isinstance(cap.value, str)

    def test_known_capabilities_exist(self):
        expected = [
            "defect_detection",
            "state_recognition",
            "meter_reading",
            "thermal_analysis",
            "intrusion_detection",
            "bird_detection",
            "person_detection",
            "fire_detection",
            "smoke_detection",
            "animal_detection",
        ]
        values = [c.value for c in PluginCapability]
        for exp in expected:
            assert exp in values, f"Missing capability: {exp}"

    def test_construct_from_string(self):
        cap = PluginCapability("defect_detection")
        assert cap == PluginCapability.DEFECT_DETECTION

    def test_invalid_capability_raises(self):
        with pytest.raises(ValueError):
            PluginCapability("nonexistent_capability")


class TestPluginStatus:
    """Tests for PluginStatus enum."""

    def test_is_string_enum(self):
        assert isinstance(PluginStatus.READY, str)
        assert PluginStatus.READY == "ready"

    def test_all_statuses(self):
        expected = {"unloaded", "loading", "ready", "running", "error", "disabled"}
        actual = {s.value for s in PluginStatus}
        assert actual == expected

    def test_construct_from_string(self):
        s = PluginStatus("running")
        assert s == PluginStatus.RUNNING


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_defaults(self):
        hs = HealthStatus(healthy=True)
        assert hs.healthy is True
        assert hs.message == ""
        assert isinstance(hs.last_check, datetime)
        assert hs.details == {}

    def test_with_all_fields(self):
        now = datetime.now()
        hs = HealthStatus(
            healthy=False,
            message="Model not loaded",
            last_check=now,
            details={"gpu_mem": "2GB"},
        )
        assert hs.healthy is False
        assert hs.message == "Model not loaded"
        assert hs.last_check == now
        assert hs.details["gpu_mem"] == "2GB"
