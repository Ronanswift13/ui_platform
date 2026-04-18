"""质量门禁契约测试

测试质量门禁是否正确触发并返回正确的原因码。
遵循 02_algorithm_contract.md 第 2.4 节和 03_test_strategy.md 第 5 节。
"""

import pytest
import numpy as np
import sys
import os

try:
    import cv2
except ImportError:
    cv2 = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector_enhanced import (
    BusbarDetectorEnhanced,
    QualityGateDecision,
    QualityGateStatus,
)
from reason_code_mapper import ReasonCodeMapper


class TestQualityGateContract:
    """质量门禁契约测试"""

    @pytest.fixture
    def detector(self):
        """创建检测器实例"""
        config = {
            'thresholds': {'conf_thr': 0.25, 'nms_iou': 0.50},
            'quality': {'blur_thr': 0.35},
            'zoom': {'min_obj_px': 18, 'target_px': 90, 'zmin': 1.0, 'zmax': 12.0},
            'clarity_threshold': 0.35,
            'brightness_range': (0.2, 0.8),
            'contrast_threshold': 0.3,
        }
        return BusbarDetectorEnhanced(config)

    def _create_textured_image(self, base_value, noise_range=20):
        """创建具有纹理的图像，避免纯色导致的低清晰度"""
        image = np.ones((480, 640, 3), dtype=np.uint8) * base_value
        noise = np.random.default_rng(0).integers(
            -noise_range, noise_range, (480, 640, 3), dtype=np.int16
        )
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return image

    def _create_structured_blur_image(self):
        """创建保留动态范围但清晰度明显下降的模糊样本。"""
        if cv2 is None:
            pytest.skip("cv2 is required to synthesize realistic blur samples")

        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        for x in range(40, 600, 80):
            cv2.rectangle(image, (x, 60), (x + 30, 420), (40, 40, 40), -1)
        for y in range(80, 400, 70):
            cv2.line(image, (20, y), (620, y), (220, 220, 220), 6)
        cv2.circle(image, (320, 240), 70, (240, 240, 240), -1)
        cv2.circle(image, (320, 240), 35, (30, 30, 30), -1)
        return cv2.GaussianBlur(image, (21, 21), 7)

    def test_blur_image_triggers_quality_fail(self, detector):
        """测试模糊图像触发质量门禁失败（T-003）"""
        blur_image = self._create_structured_blur_image()
        result = detector.check_quality_gate(blur_image)

        assert result.status == QualityGateStatus.FAIL_BLUR
        assert result.quality_gate_status == QualityGateDecision.HARD_FAIL
        assert result.reason_code == "1001"

        external_code = ReasonCodeMapper.map_to_external(int(result.reason_code))
        assert external_code == 103

    def test_overexposed_image_triggers_quality_fail(self, detector):
        """测试过曝图像触发质量门禁失败（T-004）"""
        overexp_image = self._create_textured_image(240, noise_range=30)
        result = detector.check_quality_gate(overexp_image)

        assert result.status == QualityGateStatus.FAIL_OVEREXPOSED
        assert result.quality_gate_status == QualityGateDecision.HARD_FAIL
        assert result.reason_code == "1002"

        external_code = ReasonCodeMapper.map_to_external(int(result.reason_code))
        assert external_code == 101

    def test_low_contrast_image_triggers_quality_fail(self, detector):
        """测试低对比图像触发质量门禁失败（T-005）"""
        low_contrast_image = self._create_textured_image(128, noise_range=30)
        result = detector.check_quality_gate(low_contrast_image)

        assert result.status == QualityGateStatus.FAIL_LOW_CONTRAST
        assert result.quality_gate_status == QualityGateDecision.SOFT_FAIL
        assert result.reason_code == "1005"

        external_code = ReasonCodeMapper.map_to_external(int(result.reason_code))
        assert external_code == 101

    def test_underexposed_image_triggers_quality_fail(self, detector):
        """测试欠曝图像触发质量门禁失败"""
        underexp_image = self._create_textured_image(30, noise_range=20)
        result = detector.check_quality_gate(underexp_image)

        assert result.status == QualityGateStatus.FAIL_UNDEREXPOSED
        assert result.quality_gate_status == QualityGateDecision.SOFT_FAIL
        assert result.reason_code == "1003"

        external_code = ReasonCodeMapper.map_to_external(int(result.reason_code))
        assert external_code == 101

    def test_good_quality_image_passes(self, detector):
        """测试高质量图像通过质量门禁"""
        good_image = np.random.default_rng(1).integers(50, 200, (480, 640, 3), dtype=np.uint8)
        result = detector.check_quality_gate(good_image)

        assert result.status == QualityGateStatus.PASS
        assert result.quality_gate_status == QualityGateDecision.PASS
        assert result.reason_code == ""

    def test_quality_result_has_required_fields(self, detector):
        """测试质量门禁结果包含必需字段"""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        result = detector.check_quality_gate(image)

        assert hasattr(result, 'status')
        assert hasattr(result, 'quality_gate_status')
        assert hasattr(result, 'clarity_score')
        assert hasattr(result, 'brightness_score')
        assert hasattr(result, 'contrast_score')
        assert hasattr(result, 'occlusion_ratio')
        assert hasattr(result, 'reason_code')
        assert hasattr(result, 'failure_reason')
        assert hasattr(result, 'metadata')
        assert "suggested_action" in result.metadata

    def test_clarity_score_in_valid_range(self, detector):
        """测试清晰度评分在有效范围内"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.check_quality_gate(image)

        assert 0.0 <= result.clarity_score <= 1.0

    def test_brightness_score_in_valid_range(self, detector):
        """测试亮度评分在有效范围内"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.check_quality_gate(image)

        assert 0.0 <= result.brightness_score <= 1.0

    def test_contrast_score_in_valid_range(self, detector):
        """测试对比度评分在有效范围内"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.check_quality_gate(image)

        assert 0.0 <= result.contrast_score <= 1.0

    def test_occlusion_ratio_in_valid_range(self, detector):
        """测试遮挡比例在有效范围内"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.check_quality_gate(image)

        assert 0.0 <= result.occlusion_ratio <= 1.0

    def test_reason_code_mapping_consistency(self, detector):
        """测试原因码映射一致性"""
        test_cases = [
            ("blur", "1001", 103),
            (self._create_textured_image(240, noise_range=30), "1002", 101),
            (self._create_textured_image(30, noise_range=20), "1003", 101),
        ]

        for image, expected_internal, expected_external in test_cases:
            if isinstance(image, str) and image == "blur":
                image = self._create_structured_blur_image()
            result = detector.check_quality_gate(image)
            if result.reason_code:
                internal_code = int(result.reason_code)
                external_code = ReasonCodeMapper.map_to_external(internal_code)
                assert result.reason_code == expected_internal
                assert external_code == expected_external

    def test_nested_quality_config_allows_realistic_medium_contrast_upload(self):
        """嵌套 YAML 阈值必须真正影响现场上传图的质量门禁判断。"""
        detector = BusbarDetectorEnhanced({
            'thresholds': {'conf_thr': 0.25, 'nms_iou': 0.50},
            'quality': {
                'blur_thr': 0.35,
                'y_high': 245,
                'overexp_ratio': 0.25,
                'dr_min': 35,
                'edge_thr': 10,
            },
            'zoom': {'min_obj_px': 18, 'target_px': 90, 'zmin': 1.0, 'zmax': 12.0},
        })
        image = np.random.default_rng(7).integers(
            85, 175, size=(480, 640, 3), dtype=np.uint8
        )

        result = detector.check_quality_gate(image)

        assert result.status == QualityGateStatus.PASS
        assert result.reason_code == ""
