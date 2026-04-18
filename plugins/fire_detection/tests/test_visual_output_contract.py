"""统一视觉输出协议合约测试 — fire_detection"""

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
    from plugins.fire_detection.plugin import Plugin

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


@pytest.fixture
def dummy_roi():
    class R:
        id = "roi_1"

    return R()


class TestVisualOutputContract:
    """fire_detection uses detect() internally — infer() wraps it.

    Note: if detect() returns no detections, infer() returns [].
    We test metadata when detections are present.
    """

    def test_infer_returns_list(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        assert isinstance(results, list)

    def test_metadata_when_detections_present(self, plugin, dummy_context, dummy_roi):
        """If detections exist, metadata must conform to protocol."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            meta = r.metadata or {}
            missing = REQUIRED_META_KEYS - meta.keys()
            assert not missing, f"Missing keys: {missing}"
            errors = validate_visual_meta(meta)
            assert errors == [], f"Validation errors: {errors}"

    def test_placeholders_when_detections_present(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            ph = (r.metadata or {}).get("placeholders", {})
            missing = REQUIRED_PLACEHOLDER_KEYS - ph.keys()
            assert not missing, f"Missing placeholder keys: {missing}"

    def test_plugin_name_when_detections_present(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            assert (r.metadata or {}).get("plugin_name") == "fire_detection"
