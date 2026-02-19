"""Tests for transformer_inspection standalone operation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_create_standalone(plugin):
    """Test that create_standalone creates a working plugin instance."""
    assert plugin is not None
    assert plugin.id == "transformer_inspection"


def test_healthcheck(plugin):
    """Test that healthcheck returns a healthy status."""
    health = plugin.healthcheck()
    assert health.healthy is True
    assert health.message is not None


def test_infer(plugin, dummy_frame, sample_context, sample_rois):
    """Test that infer returns a list of results."""
    results = plugin.infer(frame=dummy_frame, rois=sample_rois, context=sample_context)
    assert isinstance(results, list)


def test_runner_creation(plugin):
    """Test that a StandalonePluginRunner can be created."""
    from darkbreaker_sdk.standalone import StandalonePluginRunner

    runner = StandalonePluginRunner(
        plugin=plugin,
        title="Test Runner",
        port=18087,
    )
    assert runner is not None
    assert runner.app is not None
