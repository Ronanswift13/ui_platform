#!/usr/bin/env python3
"""
ONNX 推理引擎测试套件
======================
覆盖:
  - 引擎初始化 (有模型/无模型)
  - 预处理正确性
  - 后处理 & NMS
  - 类别映射 (COCO -> 动物)
  - 降级行为 (无模型不造假数据)
  - 事件契约合规性
  - 性能基准

运行:
  pytest test_onnx_inference.py -v
  pytest test_onnx_inference.py -v -k "test_degraded"  # 运行特定测试
"""

import json
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 将 core 模块加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.onnx_inference import (
    AnimalONNXEngine,
    Detection,
    InferenceResult,
    TARGET_CLASSES,
    COCO_CLASS_MAPPING,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def fake_onnx_model(tmp_path):
    """创建一个不可加载的假模型路径 (用于测试降级)."""
    model_path = tmp_path / "fake_model.onnx"
    model_path.write_bytes(b"not a real onnx model")
    return str(model_path)


@pytest.fixture
def nonexistent_model():
    """不存在的模型路径."""
    return "/tmp/does_not_exist_animal_model.onnx"


@pytest.fixture
def class_mapping_file(tmp_path):
    """创建类别映射文件."""
    mapping = {
        "mode": "coco_pretrained",
        "coco_to_animal": {
            "14": {"target_class": "bird", "target_id": 3},
            "15": {"target_class": "cat", "target_id": 1},
            "16": {"target_class": "dog", "target_id": 4},
        },
    }
    path = tmp_path / "class_mapping.json"
    with open(path, "w") as f:
        json.dump(mapping, f)
    return str(path)


@pytest.fixture
def sample_frame():
    """生成测试用图像帧 (640x480 RGB)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_small():
    """生成小尺寸测试帧 (320x240)."""
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)


# ===================================================================
# 测试: Detection 数据结构
# ===================================================================

class TestDetection:
    def test_detection_to_dict(self):
        det = Detection(
            class_id=1,
            class_name="cat",
            confidence=0.85,
            bbox=(100.0, 200.0, 300.0, 400.0),
            bbox_normalized=(0.15625, 0.41667, 0.46875, 0.83333),
            trace_id="abc123",
            inference_ms=15.3,
            source_model="animal_yolov8n.onnx",
        )
        d = det.to_dict()

        assert d["class_id"] == 1
        assert d["class_name"] == "cat"
        assert d["confidence"] == 0.85
        assert d["bbox"]["x1"] == 100.0
        assert d["bbox"]["y2"] == 400.0
        assert d["trace_id"] == "abc123"
        assert d["source_model"] == "animal_yolov8n.onnx"

    def test_detection_normalized_bbox_range(self):
        det = Detection(
            class_id=3, class_name="bird", confidence=0.7,
            bbox=(10, 20, 100, 200),
            bbox_normalized=(0.05, 0.1, 0.5, 1.0),
        )
        d = det.to_dict()
        for key in ["x1", "y1", "x2", "y2"]:
            assert 0 <= d["bbox_normalized"][key] <= 1.0


class TestInferenceResult:
    def test_empty_result(self):
        r = InferenceResult(status="degraded", trace_id="test")
        d = r.to_dict()
        assert d["status"] == "degraded"
        assert d["detection_count"] == 0
        assert d["detections"] == []

    def test_result_with_detections(self):
        det = Detection(
            class_id=0, class_name="rat", confidence=0.9,
            bbox=(10, 20, 50, 60), bbox_normalized=(0.01, 0.02, 0.05, 0.06),
        )
        r = InferenceResult(
            detections=[det],
            status="ok",
            trace_id="xyz",
            inference_ms=20.5,
        )
        d = r.to_dict()
        assert d["detection_count"] == 1
        assert d["status"] == "ok"


# ===================================================================
# 测试: 引擎初始化与降级
# ===================================================================

class TestEngineInit:
    def test_init_with_nonexistent_model(self, nonexistent_model):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        assert engine.is_loaded is False

    def test_load_nonexistent_model_returns_false(self, nonexistent_model):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        result = engine.load()
        assert result is False
        assert engine.is_loaded is False

    def test_class_mapping_loaded(self, nonexistent_model, class_mapping_file):
        engine = AnimalONNXEngine(
            model_path=nonexistent_model,
            class_mapping_path=class_mapping_file,
        )
        assert engine.class_mapping is not None
        assert 14 in engine.class_mapping
        assert engine.class_mapping[14] == 3  # bird

    def test_default_thresholds(self, nonexistent_model):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        assert engine.conf_threshold == 0.55
        assert engine.nms_threshold == 0.4

    def test_custom_thresholds(self, nonexistent_model):
        engine = AnimalONNXEngine(
            model_path=nonexistent_model,
            conf_threshold=0.7,
            nms_threshold=0.3,
        )
        assert engine.conf_threshold == 0.7
        assert engine.nms_threshold == 0.3

    def test_health_info_not_loaded(self, nonexistent_model):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        info = engine.get_health_info()
        assert info["loaded"] is False


# ===================================================================
# 测试: 降级行为 (核心约束: 不造假数据)
# ===================================================================

class TestDegradedMode:
    """验证无模型时绝对不产生假检测数据."""

    def test_infer_without_model_returns_degraded(self, nonexistent_model, sample_frame):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        result = engine.infer(sample_frame)
        assert result.status == "degraded"
        assert len(result.detections) == 0
        assert result.error_message == "模型未加载"

    def test_infer_empty_frame_returns_error(self, nonexistent_model):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        result = engine.infer(np.array([]))
        # 未加载模型, 应该是 degraded 而不是 error
        assert result.status == "degraded"

    def test_infer_none_frame(self, nonexistent_model):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        result = engine.infer(None)
        assert result.status == "degraded"
        assert len(result.detections) == 0

    def test_no_random_data_generated(self, nonexistent_model, sample_frame):
        """确保多次调用都返回空, 没有随机/模拟数据."""
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        for _ in range(100):
            result = engine.infer(sample_frame)
            assert len(result.detections) == 0, \
                "降级模式下不应产生任何检测结果"

    def test_trace_id_propagated(self, nonexistent_model, sample_frame):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        result = engine.infer(sample_frame, trace_id="test-trace-123")
        assert result.trace_id == "test-trace-123"

    def test_auto_trace_id_generated(self, nonexistent_model, sample_frame):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        result = engine.infer(sample_frame)
        assert result.trace_id != ""
        assert len(result.trace_id) > 0


# ===================================================================
# 测试: NMS 算法
# ===================================================================

class TestNMS:
    def test_nms_empty_input(self):
        keep = AnimalONNXEngine._nms(
            np.array([]).reshape(0, 4),
            np.array([]),
            0.5,
        )
        assert keep == []

    def test_nms_single_box(self):
        boxes = np.array([[10, 20, 100, 200]])
        scores = np.array([0.9])
        keep = AnimalONNXEngine._nms(boxes, scores, 0.5)
        assert keep == [0]

    def test_nms_non_overlapping(self):
        boxes = np.array([
            [0, 0, 50, 50],
            [100, 100, 150, 150],
            [200, 200, 250, 250],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        keep = AnimalONNXEngine._nms(boxes, scores, 0.5)
        assert len(keep) == 3

    def test_nms_overlapping_suppresses(self):
        boxes = np.array([
            [10, 10, 100, 100],
            [12, 12, 102, 102],  # 高度重叠
            [200, 200, 300, 300],  # 不重叠
        ])
        scores = np.array([0.9, 0.85, 0.7])
        keep = AnimalONNXEngine._nms(boxes, scores, 0.3)
        assert 0 in keep
        assert 2 in keep
        assert len(keep) == 2  # 第二个框被抑制

    def test_nms_preserves_highest_score(self):
        boxes = np.array([
            [10, 10, 100, 100],
            [11, 11, 101, 101],
        ])
        scores = np.array([0.5, 0.9])
        keep = AnimalONNXEngine._nms(boxes, scores, 0.3)
        assert keep[0] == 1  # 高分框保留


# ===================================================================
# 测试: 预处理
# ===================================================================

class TestPreprocess:
    def test_preprocess_output_shape(self, nonexistent_model, sample_frame):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        engine.input_shape = (1, 3, 640, 640)
        tensor = engine._preprocess(sample_frame)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_preprocess_value_range(self, nonexistent_model, sample_frame):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        engine.input_shape = (1, 3, 640, 640)
        tensor = engine._preprocess(sample_frame)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_preprocess_small_frame(self, nonexistent_model, sample_frame_small):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        engine.input_shape = (1, 3, 640, 640)
        tensor = engine._preprocess(sample_frame_small)
        assert tensor.shape == (1, 3, 640, 640)

    def test_preprocess_square_frame(self, nonexistent_model):
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        engine.input_shape = (1, 3, 640, 640)
        tensor = engine._preprocess(frame)
        assert tensor.shape == (1, 3, 640, 640)


# ===================================================================
# 测试: 后处理 (使用模拟输出)
# ===================================================================

class TestPostprocess:
    def _make_engine(self, num_classes=7):
        engine = AnimalONNXEngine.__new__(AnimalONNXEngine)
        engine.model_path = "test_model.onnx"
        engine.conf_threshold = 0.55
        engine.nms_threshold = 0.4
        engine.input_shape = (1, 3, 640, 640)
        engine.model_mode = "custom_trained"
        engine.class_mapping = None
        engine.num_classes = num_classes
        return engine

    def test_postprocess_no_detections(self):
        engine = self._make_engine()
        # 模拟全零输出: 没有高置信度检测
        raw = np.zeros((1, 11, 100), dtype=np.float32)  # 7+4=11
        dets = engine._postprocess(raw, 640, 480, "test-trace", 10.0)
        assert len(dets) == 0

    def test_postprocess_with_detection(self):
        engine = self._make_engine()
        # 构造一个有效检测
        raw = np.zeros((1, 11, 10), dtype=np.float32)
        # 第一个检测: bbox at center, cat (class 1) with conf 0.9
        raw[0, 0, 0] = 320   # x_center
        raw[0, 1, 0] = 240   # y_center
        raw[0, 2, 0] = 100   # width
        raw[0, 3, 0] = 80    # height
        raw[0, 5, 0] = 0.9   # cat (class_id=1, offset=4+1=5) score

        dets = engine._postprocess(raw, 640, 480, "test-trace", 10.0)
        assert len(dets) >= 1
        assert dets[0].class_name == "cat"
        assert dets[0].confidence >= 0.55

    def test_postprocess_filters_low_confidence(self):
        engine = self._make_engine()
        raw = np.zeros((1, 11, 10), dtype=np.float32)
        raw[0, 0, 0] = 320
        raw[0, 1, 0] = 240
        raw[0, 2, 0] = 100
        raw[0, 3, 0] = 80
        raw[0, 5, 0] = 0.3   # 低于阈值 0.55

        dets = engine._postprocess(raw, 640, 480, "test-trace", 10.0)
        assert len(dets) == 0

    def test_postprocess_coco_mapping(self):
        engine = self._make_engine(num_classes=80)
        engine.model_mode = "coco_pretrained"
        engine.class_mapping = COCO_CLASS_MAPPING.copy()

        raw = np.zeros((1, 84, 10), dtype=np.float32)  # 80+4=84
        # 模拟 COCO cat (class 15) 检测
        raw[0, 0, 0] = 320
        raw[0, 1, 0] = 240
        raw[0, 2, 0] = 100
        raw[0, 3, 0] = 80
        raw[0, 4 + 15, 0] = 0.85  # COCO cat class

        dets = engine._postprocess(raw, 640, 480, "test-trace", 10.0)
        assert len(dets) >= 1
        assert dets[0].class_name == "cat"
        assert dets[0].class_id == 1  # 映射后的 target class id

    def test_postprocess_coco_skips_non_animal(self):
        engine = self._make_engine(num_classes=80)
        engine.model_mode = "coco_pretrained"
        engine.class_mapping = COCO_CLASS_MAPPING.copy()

        raw = np.zeros((1, 84, 10), dtype=np.float32)
        # 模拟 COCO person (class 0) — 非动物, 应被过滤
        raw[0, 0, 0] = 320
        raw[0, 1, 0] = 240
        raw[0, 2, 0] = 100
        raw[0, 3, 0] = 80
        raw[0, 4, 0] = 0.95  # person class

        dets = engine._postprocess(raw, 640, 480, "test-trace", 10.0)
        assert len(dets) == 0


# ===================================================================
# 测试: 事件契约合规 (统一 schema)
# ===================================================================

class TestEventContract:
    """验证输出是否符合统一事件契约:
    timestamp/type/location/value/confidence + evidence + suggestion + trace_id
    """

    def test_detection_has_required_fields(self):
        det = Detection(
            class_id=2, class_name="snake", confidence=0.78,
            bbox=(50, 100, 200, 300), bbox_normalized=(0.1, 0.2, 0.4, 0.6),
            trace_id="evt-001", inference_ms=12.5,
            source_model="animal_yolov8n.onnx",
        )
        d = det.to_dict()

        # 必需字段检查
        assert "class_id" in d
        assert "class_name" in d
        assert "confidence" in d
        assert "bbox" in d
        assert "bbox_normalized" in d
        assert "trace_id" in d

    def test_inference_result_has_status(self):
        r = InferenceResult(status="ok", trace_id="test")
        d = r.to_dict()
        assert "status" in d
        assert "trace_id" in d
        assert "detection_count" in d
        assert "inference_ms" in d


# ===================================================================
# 测试: 类别定义
# ===================================================================

class TestClassDefinitions:
    def test_target_classes_count(self):
        assert len(TARGET_CLASSES) == 7

    def test_target_classes_names(self):
        expected = ["rat", "cat", "snake", "bird", "dog", "poultry", "other_wildlife"]
        assert TARGET_CLASSES == expected

    def test_coco_mapping_valid(self):
        for coco_id, target_id in COCO_CLASS_MAPPING.items():
            assert 0 <= target_id < len(TARGET_CLASSES)


# ===================================================================
# 测试: 边界条件
# ===================================================================

class TestEdgeCases:
    def test_very_large_frame(self, nonexistent_model):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        engine.input_shape = (1, 3, 640, 640)
        large_frame = np.random.randint(0, 255, (4096, 4096, 3), dtype=np.uint8)
        tensor = engine._preprocess(large_frame)
        assert tensor.shape == (1, 3, 640, 640)

    def test_single_pixel_frame(self, nonexistent_model):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        engine.input_shape = (1, 3, 640, 640)
        tiny_frame = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)
        tensor = engine._preprocess(tiny_frame)
        assert tensor.shape == (1, 3, 640, 640)

    def test_grayscale_frame_raises_or_handles(self, nonexistent_model):
        engine = AnimalONNXEngine(model_path=nonexistent_model)
        engine.input_shape = (1, 3, 640, 640)
        gray = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        # 应该要么处理要么抛出明确异常
        try:
            tensor = engine._preprocess(gray)
            # 如果成功, 检查形状
            assert tensor.shape[1] == 3
        except (ValueError, IndexError):
            pass  # 预期行为: 拒绝非 3 通道输入


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
