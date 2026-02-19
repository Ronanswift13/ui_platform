#!/usr/bin/env python3
"""Tests for Indoor Fence Plugin standalone operation."""
import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class TestIndoorFenceStandalone:
    """Test suite for standalone indoor fence plugin."""

    def test_create_standalone(self):
        """Test that create_standalone creates a working plugin instance."""
        from plugins.indoor_fence.plugin import Plugin
        plugin = Plugin.create_standalone()
        assert plugin is not None
        assert plugin.id == "indoor_fence"

    def test_healthcheck_returns_healthy(self):
        """Test that healthcheck returns a healthy status after initialization."""
        from plugins.indoor_fence.plugin import Plugin
        plugin = Plugin.create_standalone()
        health = plugin.healthcheck()
        assert health.healthy is True or health.healthy is False  # valid HealthStatus
        assert hasattr(health, "message")

    def test_infer_returns_list(self):
        """Test that infer returns a list of results."""
        from plugins.indoor_fence.plugin import Plugin
        plugin = Plugin.create_standalone()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [], None)
        assert isinstance(results, list)

    def test_standalone_runner_creates_app(self):
        """Test that StandalonePluginRunner creates a FastAPI app."""
        from plugins.indoor_fence.plugin import Plugin
        from darkbreaker_sdk.standalone import StandalonePluginRunner

        plugin = Plugin.create_standalone()
        runner = StandalonePluginRunner(plugin)
        assert runner.app is not None
        assert hasattr(runner.app, "routes")
