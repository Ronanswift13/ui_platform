"""统一视觉输出协议合约测试 — bird_monitoring"""

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
    from plugins.bird_monitoring.plugin import Plugin

    return Plugin.create_standalone()


@pytest.fixture
def dummy_context():
    from darkbreaker_sdk.interfaces import PluginContext

    return PluginContext(
        task_id="test_task",
        site_id="test_site",
        device_id="test_device",
        component_id="line_01",
    )


@pytest.fixture
def dummy_roi():
    from darkbreaker_sdk.schemas import BoundingBox, ROI
    from darkbreaker_sdk.schemas.common import ROIType

    return ROI(
        id="roi_1",
        name="test_roi",
        component_id="line_01",
        roi_type=ROIType.INTRUSION,
        bbox=BoundingBox(x=0.1, y=0.1, width=0.5, height=0.5),
    )


class TestVisualOutputContract:
    def test_infer_returns_list(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_metadata_has_required_keys(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            meta = r.metadata or {}
            missing = REQUIRED_META_KEYS - meta.keys()
            assert not missing, f"Missing keys: {missing}"

    def test_metadata_passes_validation(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            errors = validate_visual_meta(r.metadata or {})
            assert errors == [], f"Validation errors: {errors}"

    def test_placeholders_complete(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            ph = (r.metadata or {}).get("placeholders", {})
            missing = REQUIRED_PLACEHOLDER_KEYS - ph.keys()
            assert not missing, f"Missing placeholder keys: {missing}"

    def test_training_placeholders_complete(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            tp = (r.metadata or {}).get("training_placeholders", {})
            missing = REQUIRED_TRAINING_KEYS - tp.keys()
            assert not missing, f"Missing training keys: {missing}"

    def test_plugin_name_matches(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            assert (r.metadata or {}).get("plugin_name") == "bird_monitoring"
