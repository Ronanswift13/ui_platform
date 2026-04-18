"""
QR-11 置信度消毒测试 (busbar_inspection)
==========================================
验证 _sanitize_confidence 和 _map_class_to_defect_type 的防御行为。
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector_enhanced import (
    _sanitize_confidence,
    BusbarDetectorEnhanced,
    BusbarDefectType,
)


class TestSanitizeConfidence:
    """_sanitize_confidence 纯函数测试"""

    def test_nan_returns_zero(self):
        assert _sanitize_confidence(float("nan")) == 0.0

    def test_inf_returns_zero(self):
        assert _sanitize_confidence(float("inf")) == 0.0

    def test_neg_inf_returns_zero(self):
        assert _sanitize_confidence(float("-inf")) == 0.0

    def test_negative_returns_zero(self):
        assert _sanitize_confidence(-0.5) == 0.0

    def test_greater_than_one_returns_one(self):
        assert _sanitize_confidence(1.5) == 1.0

    def test_valid_unchanged(self):
        assert _sanitize_confidence(0.0) == 0.0
        assert _sanitize_confidence(0.75) == 0.75
        assert _sanitize_confidence(1.0) == 1.0

    def test_zero_boundary(self):
        assert _sanitize_confidence(0.0) == 0.0

    def test_one_boundary(self):
        assert _sanitize_confidence(1.0) == 1.0


class TestMapClassToDefectType:
    """_map_class_to_defect_type 返回值测试 (QR-10)"""

    @pytest.fixture
    def detector(self):
        config = {
            "confidence_threshold": 0.4,
            "nms_threshold": 0.4,
        }
        return BusbarDetectorEnhanced(config)

    def test_known_class_returns_correct_type(self, detector):
        defect_type, is_unknown = detector._map_class_to_defect_type("pin_missing")
        assert defect_type == BusbarDefectType.PIN_MISSING
        assert is_unknown is False

    def test_known_class_crack(self, detector):
        defect_type, is_unknown = detector._map_class_to_defect_type("crack")
        assert defect_type == BusbarDefectType.CRACK
        assert is_unknown is False

    def test_known_class_case_insensitive(self, detector):
        defect_type, is_unknown = detector._map_class_to_defect_type("PIN_MISSING")
        assert defect_type == BusbarDefectType.PIN_MISSING
        assert is_unknown is False

    def test_deformation_maps_to_crack(self, detector):
        defect_type, is_unknown = detector._map_class_to_defect_type("deformation")
        assert defect_type == BusbarDefectType.CRACK
        assert is_unknown is False

    def test_unknown_class_returns_none_with_flag(self, detector):
        """未知类名不得再默认回退到 CRACK"""
        defect_type, is_unknown = detector._map_class_to_defect_type("totally_unknown_defect")
        assert defect_type is None
        assert is_unknown is True

    def test_empty_string_is_unknown(self, detector):
        defect_type, is_unknown = detector._map_class_to_defect_type("")
        assert defect_type is None
        assert is_unknown is True

    def test_unknown_class_id_is_suppressed_in_legacy_model_registry_path(self, detector):
        """未知 class_id 不得通过 legacy model_registry 路径回退成 crack"""
        class DummyRegistry:
            def infer(self, model_id, image):
                class Result:
                    detections = [
                        {
                            "confidence": 0.92,
                            "class_id": 999,
                            "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                        }
                    ]

                return Result()

        detector._model_registry = DummyRegistry()
        detections = detector._detect_by_deep_learning(image=None)
        assert detections == []

    def test_all_known_classes_not_unknown(self, detector):
        """所有已知类名都不应标记为未知"""
        known_classes = [
            "pin_missing", "crack", "foreign_object", "corrosion",
            "flashover", "broken_strand", "insulator_damage", "fitting_loose",
            "deformation",
        ]
        for cls_name in known_classes:
            _, is_unknown = detector._map_class_to_defect_type(cls_name)
            assert is_unknown is False, f"{cls_name} should not be unknown"
