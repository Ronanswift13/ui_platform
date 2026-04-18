"""统一视觉输出协议合约测试 — animal_detection

animal_detection.infer() 返回 AnimalEvent (非 RecognitionResult)，
统一视觉协议元数据注入到 event.value["visual_meta"]。
"""

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
    from plugins.animal_detection.plugin import Plugin

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
    """验证 animal_detection 的 infer() 事件输出包含 visual_meta。"""

    def test_event_contains_visual_meta(self, plugin, dummy_context):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        event = plugin.infer(frame, context=dummy_context)
        assert hasattr(event, "value")
        if isinstance(event.value, dict):
            meta = event.value.get("visual_meta")
            assert meta is not None, "event.value should contain 'visual_meta'"

    def test_visual_meta_has_required_keys(self, plugin, dummy_context):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        event = plugin.infer(frame, context=dummy_context)
        if isinstance(event.value, dict) and "visual_meta" in event.value:
            meta = event.value["visual_meta"]
            missing = REQUIRED_META_KEYS - meta.keys()
            assert not missing, f"Missing keys: {missing}"

    def test_visual_meta_passes_validation(self, plugin, dummy_context):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        event = plugin.infer(frame, context=dummy_context)
        if isinstance(event.value, dict) and "visual_meta" in event.value:
            errors = validate_visual_meta(event.value["visual_meta"])
            assert errors == [], f"Validation errors: {errors}"

    def test_visual_meta_plugin_name(self, plugin, dummy_context):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        event = plugin.infer(frame, context=dummy_context)
        if isinstance(event.value, dict) and "visual_meta" in event.value:
            assert event.value["visual_meta"].get("plugin_name") == "animal_detection"

    def test_visual_meta_placeholders(self, plugin, dummy_context):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        event = plugin.infer(frame, context=dummy_context)
        if isinstance(event.value, dict) and "visual_meta" in event.value:
            ph = event.value["visual_meta"].get("placeholders", {})
            missing = REQUIRED_PLACEHOLDER_KEYS - ph.keys()
            assert not missing, f"Missing placeholder keys: {missing}"
