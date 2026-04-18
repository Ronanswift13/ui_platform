"""统一视觉输出协议合约测试 — busbar_inspection"""

import sys
from pathlib import Path

import numpy as np
import pytest

# 确保项目根目录在 sys.path
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
    from plugins.busbar_inspection.plugin import Plugin

    return Plugin.create_standalone()


@pytest.fixture
def dummy_context():
    class Ctx:
        task_id = "test_task"
        site_id = "test_site"
        device_id = "test_device"
        component_id = "busbar_01"

    return Ctx()


@pytest.fixture
def dummy_roi():
    from darkbreaker_sdk.schemas import BoundingBox

    class R:
        id = "roi_1"
        bbox = BoundingBox(x=0.1, y=0.1, width=0.5, height=0.5)
        type = "visual"
        metadata = {}

    return R()


class TestVisualOutputContract:
    """验证 busbar_inspection 的 infer() 输出符合 A 类统一协议。"""

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
            meta = r.metadata or {}
            ph = meta.get("placeholders", {})
            missing = REQUIRED_PLACEHOLDER_KEYS - ph.keys()
            assert not missing, f"Missing placeholder keys: {missing}"

    def test_training_placeholders_complete(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            meta = r.metadata or {}
            tp = meta.get("training_placeholders", {})
            missing = REQUIRED_TRAINING_KEYS - tp.keys()
            assert not missing, f"Missing training keys: {missing}"

    def test_plugin_name_matches(self, plugin, dummy_context, dummy_roi):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = plugin.infer(frame, [dummy_roi], dummy_context)
        for r in results:
            meta = r.metadata or {}
            assert meta.get("plugin_name") == "busbar_inspection"
