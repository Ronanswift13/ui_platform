"""统一视觉输出协议合约测试 — hyperspectral_detection"""

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
    from plugins.hyperspectral_detection.plugin import Plugin

    p = Plugin()
    p.init({})
    return p


@pytest.fixture
def dummy_context():
    class Ctx:
        task_id = "test_task"
        site_id = "test_site"
        device_id = "test_device"

    return Ctx()


class TestVisualOutputContract:
    def test_infer_returns_list(self, plugin, dummy_context):
        frame = np.random.rand(224, 64, 64).astype(np.float32)
        results = plugin.infer(frame, [], dummy_context)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_metadata_has_required_keys(self, plugin, dummy_context):
        frame = np.random.rand(224, 64, 64).astype(np.float32)
        results = plugin.infer(frame, [], dummy_context)
        for r in results:
            meta = r.metadata or {}
            missing = REQUIRED_META_KEYS - meta.keys()
            assert not missing, f"Missing keys: {missing}"

    def test_metadata_passes_validation(self, plugin, dummy_context):
        frame = np.random.rand(224, 64, 64).astype(np.float32)
        results = plugin.infer(frame, [], dummy_context)
        for r in results:
            errors = validate_visual_meta(r.metadata or {})
            assert errors == [], f"Validation errors: {errors}"

    def test_placeholders_complete(self, plugin, dummy_context):
        frame = np.random.rand(224, 64, 64).astype(np.float32)
        results = plugin.infer(frame, [], dummy_context)
        for r in results:
            ph = (r.metadata or {}).get("placeholders", {})
            missing = REQUIRED_PLACEHOLDER_KEYS - ph.keys()
            assert not missing, f"Missing placeholder keys: {missing}"

    def test_training_placeholders_complete(self, plugin, dummy_context):
        frame = np.random.rand(224, 64, 64).astype(np.float32)
        results = plugin.infer(frame, [], dummy_context)
        for r in results:
            tp = (r.metadata or {}).get("training_placeholders", {})
            missing = REQUIRED_TRAINING_KEYS - tp.keys()
            assert not missing, f"Missing training keys: {missing}"

    def test_plugin_name_matches(self, plugin, dummy_context):
        frame = np.random.rand(224, 64, 64).astype(np.float32)
        results = plugin.infer(frame, [], dummy_context)
        for r in results:
            assert (r.metadata or {}).get("plugin_name") == "hyperspectral_detection"

    def test_infer_returns_empty_for_none_frame(self, plugin, dummy_context):
        results = plugin.infer(None, [], dummy_context)
        assert results == []
