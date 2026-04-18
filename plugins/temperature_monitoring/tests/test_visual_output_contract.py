"""统一视觉输出协议合约测试 — temperature_monitoring"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_core.visual_output_protocol import (
    REQUIRED_META_KEYS,
    REQUIRED_PLACEHOLDER_KEYS,
    REQUIRED_TRAINING_KEYS,
    validate_visual_meta,
)


@pytest.fixture
def plugin():
    from plugins.temperature_monitoring.plugin import Plugin

    p = Plugin()
    p.init({})
    return p


@pytest.fixture
def dummy_context():
    class Ctx:
        task_id = "test_task"
        site_id = "test_site"
        device_id = "test_device"
        component_id = ""

    return Ctx()


class TestVisualOutputContract:
    """temperature_monitoring infer() returns results only when hotspots are found.

    We use a thermal frame with deliberate hotspots to trigger detections.
    """

    def test_infer_returns_list(self, plugin, dummy_context):
        frame = np.full((64, 64), 25.0, dtype=np.float32)
        results = plugin.infer(frame, [], dummy_context)
        assert isinstance(results, list)

    def test_metadata_when_hotspots_present(self, plugin, dummy_context):
        """Create a frame with a synthetic hotspot to get results."""
        frame = np.full((64, 64), 25.0, dtype=np.float32)
        frame[20:30, 20:30] = 95.0  # hotspot
        results = plugin.infer(frame, [], dummy_context)
        for r in results:
            meta = r.metadata or {}
            missing = REQUIRED_META_KEYS - meta.keys()
            assert not missing, f"Missing keys: {missing}"
            errors = validate_visual_meta(meta)
            assert errors == [], f"Validation errors: {errors}"

    def test_placeholders_when_hotspots_present(self, plugin, dummy_context):
        frame = np.full((64, 64), 25.0, dtype=np.float32)
        frame[20:30, 20:30] = 95.0
        results = plugin.infer(frame, [], dummy_context)
        for r in results:
            ph = (r.metadata or {}).get("placeholders", {})
            missing = REQUIRED_PLACEHOLDER_KEYS - ph.keys()
            assert not missing, f"Missing placeholder keys: {missing}"

    def test_plugin_name_when_hotspots_present(self, plugin, dummy_context):
        frame = np.full((64, 64), 25.0, dtype=np.float32)
        frame[20:30, 20:30] = 95.0
        results = plugin.infer(frame, [], dummy_context)
        for r in results:
            assert (r.metadata or {}).get("plugin_name") == "temperature_monitoring"
