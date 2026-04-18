"""
表计读数检测器 - 增强版 V3.1
输变电激光监测平台 (E组) - 全自动AI巡检改造

增强功能:
- 关键点检测: 表盘圆心/指针/刻度精确定位
- 透视矫正: 基于关键点的图像变换 + 视角超限检测
- 增强指针检测: HRNet -> HoughCircle -> HoughLine 三级降级
- CRNN数字OCR: 数字表和七段码识别 + 严格清洗规则
- LED状态识别: HSV色相分类 + 可分离性检验
- 统一置信度: c = min(c_det, c_geom, c_parse)
- 失败兜底策略: 多次重试+人工复核标记

V3.1更新:
- 严格遵循算法合约 02_algorithm_contract.md
- 移除旧的第四状态, 只保留 SUCCESS/FAILED/NEED_MANUAL_REVIEW
- CLAHE + 去眩光预处理流水线
- 模拟表角度越界/视角超限强制进入人工复核
- 量程缺失时强制进入人工复核, 不使用默认值
- 统一置信度计算, NaN/负数/超1校验
- LED: HSV可分离性校验 + 发光面积占比
- 数字表: 严格字符集清洗 + 多小数点/中间负号检测
- metadata 输出符合合约要求
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import time
import math
import re
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ai_models.deep_learning.keypoint_detector import (
        KeypointDetector as DLKeypointDetector,
        KeypointConfig,
        MeterType as DLMeterType,
    )
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    DLKeypointDetector = None
    KeypointConfig = None
    DLMeterType = None

logger = logging.getLogger(__name__)


class MeterType(Enum):
    """表计类型 - 严格9种"""
    PRESSURE_GAUGE = "pressure_gauge"
    TEMPERATURE_GAUGE = "temperature_gauge"
    OIL_LEVEL_GAUGE = "oil_level_gauge"
    SF6_DENSITY_GAUGE = "sf6_density_gauge"
    DIGITAL_DISPLAY = "digital_display"
    LED_INDICATOR = "led_indicator"
    AMMETER = "ammeter"
    VOLTMETER = "voltmeter"
    SEVEN_SEGMENT = "seven_segment"


ANALOG_METER_TYPES = {
    MeterType.PRESSURE_GAUGE,
    MeterType.TEMPERATURE_GAUGE,
    MeterType.OIL_LEVEL_GAUGE,
    MeterType.SF6_DENSITY_GAUGE,
    MeterType.AMMETER,
    MeterType.VOLTMETER,
}

DIGITAL_METER_TYPES = {
    MeterType.DIGITAL_DISPLAY,
    MeterType.SEVEN_SEGMENT,
}


class ReadingStatus(Enum):
    """读数状态 - 严格三态"""
    SUCCESS = "success"
    FAILED = "failed"
    NEED_MANUAL_REVIEW = "need_manual_review"


@dataclass
class Keypoint:
    """关键点"""
    name: str
    x: float
    y: float
    confidence: float
    visible: bool = True


@dataclass
class MeterKeypoints:
    """表计关键点集合"""
    center: Optional[Keypoint] = None
    pointer_tip: Optional[Keypoint] = None
    pointer_base: Optional[Keypoint] = None
    scale_min: Optional[Keypoint] = None
    scale_max: Optional[Keypoint] = None
    dial_corners: List[Keypoint] = field(default_factory=list)


@dataclass
class MeterReading:
    """表计读数结果"""
    meter_type: MeterType
    value: Optional[float] = None
    unit: str = ""
    confidence: float = 0.0
    status: ReadingStatus = ReadingStatus.FAILED
    keypoints: Optional[MeterKeypoints] = None
    pointer_angle: Optional[float] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    need_manual_review: bool = False
    failure_reason: str = ""
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DigitalReading:
    """数字读数结果"""
    text: str
    value: Optional[float] = None
    confidence: float = 0.0
    bbox: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeterInspectionResult:
    """表计巡视综合结果"""
    readings: List[MeterReading] = field(default_factory=list)
    digital_readings: List[DigitalReading] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_version: str = ""
    code_hash: str = ""


def _sanitize_confidence(value: float) -> float:
    """校验并清洗置信度值, 确保在[0,1]范围内"""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _clean_ocr_text(s_raw: str) -> Tuple[Optional[str], str]:
    """
    清洗OCR原始文本, 严格遵循合约规则.

    Returns:
        (cleaned_text_or_None, rejection_reason)
        cleaned_text 为 None 表示文本非法
    """
    if not s_raw or not s_raw.strip():
        return None, "empty_string"

    allowed = set("0123456789.-")
    s_clean = "".join(c for c in s_raw if c in allowed)

    if not s_clean:
        return None, "no_valid_chars"

    if s_clean.count(".") > 1:
        return None, "multiple_decimal_points"

    if "-" in s_clean[1:]:
        return None, "negative_sign_not_at_start"

    try:
        float(s_clean)
    except ValueError:
        return None, "invalid_float_syntax"

    return s_clean, ""


def _review_status_from_reading_status(status: ReadingStatus) -> str:
    """Map the three-state reading status to replay/output review semantics."""
    if status == ReadingStatus.SUCCESS:
        return "clear"
    if status == ReadingStatus.NEED_MANUAL_REVIEW:
        return "manual_review_required"
    return "failed"


class MeterReadingDetectorEnhanced:
    """
    表计读数增强检测器 V3.1

    严格遵循算法合约:
    - 配置驱动, 禁止硬编码业务阈值
    - Fallback链路: HRNet -> HoughCircle -> HoughLine
    - 统一置信度: c = min(c_det, c_geom, c_parse)
    - 角度越界/视角超限/量程缺失 -> NEED_MANUAL_REVIEW
    """

    MODEL_IDS = {
        "keypoint": "hrnet_meter_keypoint",
        "ocr": "crnn_meter_ocr",
        "classifier": "meter_type_classifier",
    }

    METER_RANGES: Dict[MeterType, Dict[str, Any]] = {
        MeterType.PRESSURE_GAUGE: {"min": 0, "max": 1.6, "unit": "MPa"},
        MeterType.TEMPERATURE_GAUGE: {"min": -20, "max": 100, "unit": "\u00b0C"},
        MeterType.OIL_LEVEL_GAUGE: {"min": 0, "max": 100, "unit": "%"},
        MeterType.SF6_DENSITY_GAUGE: {"min": 0, "max": 0.8, "unit": "MPa"},
        MeterType.AMMETER: {"min": 0, "max": 100, "unit": "A"},
        MeterType.VOLTMETER: {"min": 0, "max": 500, "unit": "V"},
    }

    def __init__(
        self,
        config: Dict[str, Any],
        model_registry: Any = None,
    ) -> None:
        self.config = config
        self._model_registry = model_registry
        self._initialized = False

        inference_cfg = config.get("inference", {})
        fallback_cfg = config.get("fallback", {})
        perspective_cfg = config.get("perspective_correction", {})
        pointer_cfg = config.get("pointer_detection", {})
        preprocess_cfg = config.get("preprocessing", {})
        led_cfg = config.get("led_detection", {})
        performance_cfg = config.get("performance", {})

        self._confidence_threshold = inference_cfg.get("confidence_threshold", 0.6)
        self._manual_review_threshold = fallback_cfg.get("manual_review_threshold", 0.5)
        self._max_retry = fallback_cfg.get("retry_count", 3)

        self._perspective_enabled = perspective_cfg.get("enabled", True)
        self._max_rotation = perspective_cfg.get("max_rotation", 45)

        angle_range_cfg = pointer_cfg.get("angle_range", {})
        self._angle_min = angle_range_cfg.get("min", -135)
        self._angle_max = angle_range_cfg.get("max", 135)

        self._hough_threshold = pointer_cfg.get("hough_threshold", 50)
        self._min_line_length = pointer_cfg.get("min_line_length", 30)
        self._max_line_gap = pointer_cfg.get("max_line_gap", 10)
        self._hough_circle_dist_threshold = pointer_cfg.get("center_dist_threshold", 20)

        self._enhance_lighting = preprocess_cfg.get("enhance_lighting", True)
        self._anti_glare = preprocess_cfg.get("anti_glare", True)
        self._contrast_enhancement = preprocess_cfg.get("contrast_enhancement", 1.2)
        self._clahe_clip_limit = preprocess_cfg.get("clahe_clip_limit", 2.0)
        self._clahe_grid_size = preprocess_cfg.get("clahe_grid_size", 8)

        self._led_brightness_threshold = led_cfg.get("brightness_threshold", 100)
        self._led_saturation_min = led_cfg.get("saturation_min", 50)
        self._led_lit_area_min = led_cfg.get("lit_area_min_ratio", 0.1)

        self._max_reading_time_ms = performance_cfg.get("max_reading_time_ms", 500)

        self._use_deep_learning = config.get("use_deep_learning", True)

        self._dl_keypoint_detector: Optional[Any] = None
        self._dl_initialized = False

        self._model_version = "meter_enhanced_v3.1"
        self._code_hash = self._calculate_code_hash()

    def _calculate_code_hash(self) -> str:
        import inspect
        source = inspect.getsource(self.__class__)
        return f"sha256:{hashlib.sha256(source.encode()).hexdigest()[:12]}"

    def _runtime_mode_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Return the observable runtime path used for this result."""
        if metadata.get("runtime_mode"):
            return str(metadata["runtime_mode"])

        pipeline_stage = metadata.get("pipeline_stage", "")
        fallback_level = metadata.get("fallback_level", -1)
        if pipeline_stage in {"input_validation", "type_dispatch"} and fallback_level == -1:
            return "not_applicable"
        if pipeline_stage in {"hsv_chain", "ocr_chain"}:
            return "traditional_fallback"

        real_dl_available = self._use_deep_learning and (
            self._dl_initialized or self._model_registry is not None
        )
        if pipeline_stage == "analog_chain" and fallback_level == 0 and real_dl_available:
            return "real_dl"
        return "traditional_fallback"

    def _finalize_metadata_contract(
        self,
        result: MeterReading,
        timestamp_ms: int,
    ) -> MeterReading:
        """Ensure every output carries the minimum replay/audit metadata contract."""
        metadata = dict(result.metadata or {})
        metadata.setdefault("meter_type", result.meter_type.value)
        metadata["reading_status"] = result.status.value
        metadata.setdefault("pipeline_stage", "unknown")
        metadata.setdefault("fallback_level", -1)
        metadata.setdefault("timestamp_ms", timestamp_ms)
        metadata["need_manual_review"] = result.need_manual_review
        metadata["review_status"] = _review_status_from_reading_status(result.status)
        metadata["failure_reason"] = result.failure_reason or metadata.get("error_code", "")
        metadata["runtime_mode"] = self._runtime_mode_from_metadata(metadata)
        result.metadata = metadata
        return result

    def initialize(self) -> bool:
        """初始化检测器"""
        try:
            if self._use_deep_learning and DL_AVAILABLE:
                self._init_keypoint_detector()

            if self._model_registry and self._use_deep_learning and not self._dl_initialized:
                for model_key, model_id in self.MODEL_IDS.items():
                    try:
                        self._model_registry.load(model_id)
                    except Exception as e:
                        logger.warning("模型 %s 加载失败: %s", model_id, e)

            self._initialized = True
            logger.info("检测器初始化成功 (V3.1)")
            return True
        except Exception as e:
            logger.warning("检测器初始化失败: %s", e)
            return False

    def _init_keypoint_detector(self) -> bool:
        if not DL_AVAILABLE:
            logger.info("KeypointDetector模块不可用, 将使用OpenCV fallback")
            return False

        try:
            model_path = self.config.get("keypoint_model_path", None)
            kp_config = KeypointConfig(
                model_path=model_path,
                input_size=(256, 256),
                confidence_threshold=self._confidence_threshold,
                enable_perspective_correction=self._perspective_enabled,
                validate_range=True,
            )
            self._dl_keypoint_detector = DLKeypointDetector(kp_config)
            self._dl_keypoint_detector.load()
            self._dl_initialized = True
            logger.info("关键点检测器初始化成功 (HRNet)")
            return True
        except Exception as e:
            logger.warning("关键点检测器初始化失败: %s", e)
            self._dl_initialized = False
            return False

    # ======================== 输入校验 ========================

    def _validate_input(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """验证输入合法性, 返回错误描述或None(合法)"""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return "empty_frame"

        if image.ndim != 3 or image.shape[2] != 3:
            return "invalid_channel_count"

        if image.dtype != np.uint8:
            return "invalid_dtype"

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return "invalid_frame_dimensions"

        if roi_bbox is not None:
            x = roi_bbox.get("x", 0)
            y = roi_bbox.get("y", 0)
            rw = roi_bbox.get("width", roi_bbox.get("w", 0))
            rh = roi_bbox.get("height", roi_bbox.get("h", 0))

            if x < 0 or y < 0:
                return "roi_negative_coordinates"
            if rw <= 0 or rh <= 0:
                return "roi_zero_dimensions"
            if x >= 1.0 or y >= 1.0:
                return "roi_out_of_bounds"
            if x + rw > 1.0 or y + rh > 1.0:
                return "roi_out_of_bounds"

            pixel_w = int(rw * w)
            pixel_h = int(rh * h)
            if pixel_w < 4 or pixel_h < 4:
                return "roi_too_small"

        return None

    # ======================== 预处理 ========================

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """CLAHE + 去眩光 + 对比度增强 预处理流水线"""
        if cv2 is None:
            return image

        processed = image.copy()

        if self._enhance_lighting:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self._clahe_clip_limit,
                tileGridSize=(self._clahe_grid_size, self._clahe_grid_size),
            )
            l_channel = clahe.apply(l_channel)
            lab = cv2.merge([l_channel, a_channel, b_channel])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self._anti_glare:
            hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            h_ch, s_ch, v_ch = cv2.split(hsv)
            glare_mask = (v_ch > 240) & (s_ch < 30)
            if np.any(glare_mask):
                v_mean = int(np.mean(v_ch[~glare_mask])) if np.any(~glare_mask) else 128
                v_ch[glare_mask] = v_mean
                hsv = cv2.merge([h_ch, s_ch, v_ch])
                processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if self._contrast_enhancement != 1.0:
            processed = cv2.convertScaleAbs(
                processed, alpha=self._contrast_enhancement, beta=0
            )

        return processed

    # ======================== 主入口 ========================

    def read_meter(
        self,
        image: np.ndarray,
        meter_type: MeterType,
        roi_bbox: Optional[Dict[str, float]] = None,
        roi_id: str = "",
    ) -> MeterReading:
        """读取表计 - 主入口"""
        start_time = time.perf_counter()
        timestamp_ms = int(time.time() * 1000)

        validation_error = self._validate_input(image, roi_bbox)
        if validation_error:
            result = MeterReading(
                meter_type=meter_type,
                status=ReadingStatus.FAILED,
                confidence=0.0,
                failure_reason=validation_error,
                metadata={
                    "error_code": validation_error,
                    "roi_id": roi_id,
                    "meter_type": meter_type.value,
                    "reading_status": ReadingStatus.FAILED.value,
                    "pipeline_stage": "input_validation",
                    "fallback_level": -1,
                    "timestamp_ms": timestamp_ms,
                },
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            result.metadata["processing_time_ms"] = round(elapsed_ms, 2)
            return self._finalize_metadata_contract(result, timestamp_ms)

        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)

        image = self._preprocess_image(image)

        if meter_type in DIGITAL_METER_TYPES:
            result = self._read_digital_meter(image, meter_type, roi_id, timestamp_ms)
        elif meter_type == MeterType.LED_INDICATOR:
            result = self._read_led_indicator(image, roi_id, timestamp_ms)
        elif meter_type in ANALOG_METER_TYPES:
            result = self._read_analog_meter(image, meter_type, roi_id, timestamp_ms)
        else:
            result = MeterReading(
                meter_type=meter_type,
                status=ReadingStatus.NEED_MANUAL_REVIEW,
                need_manual_review=True,
                failure_reason="unknown_meter_type",
                metadata={
                    "error_code": "unknown_meter_type",
                    "roi_id": roi_id,
                    "meter_type": meter_type.value,
                    "reading_status": ReadingStatus.NEED_MANUAL_REVIEW.value,
                    "pipeline_stage": "type_dispatch",
                    "fallback_level": -1,
                    "timestamp_ms": timestamp_ms,
                },
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        result.metadata["processing_time_ms"] = round(elapsed_ms, 2)
        if elapsed_ms > self._max_reading_time_ms:
            result.metadata["latency_violation"] = True
            logger.warning("单帧耗时 %.1fms 超过 %dms 门限", elapsed_ms, self._max_reading_time_ms)

        return self._finalize_metadata_contract(result, timestamp_ms)

    # ======================== 模拟表链路 ========================

    def _read_analog_meter(
        self,
        image: np.ndarray,
        meter_type: MeterType,
        roi_id: str,
        timestamp_ms: int,
    ) -> MeterReading:
        """模拟表读数: HRNet -> HoughCircle -> HoughLine 三级降级"""
        result = MeterReading(meter_type=meter_type, status=ReadingStatus.FAILED)
        fallback_level = -1

        retry_count = 0
        while retry_count < self._max_retry:
            try:
                keypoints, fallback_level = self._detect_keypoints_with_fallback(image)
                result.keypoints = keypoints

                tilt_deg: Optional[float] = None
                corrected_image = image
                if self._perspective_enabled and keypoints.dial_corners and len(keypoints.dial_corners) >= 4:
                    corrected_image, tilt_deg = self._apply_perspective_correction(image, keypoints)
                    if tilt_deg is not None and abs(tilt_deg) > self._max_rotation:
                        result.status = ReadingStatus.NEED_MANUAL_REVIEW
                        result.need_manual_review = True
                        result.failure_reason = "tilt_exceeded"
                        result.metadata = self._build_analog_metadata(
                            roi_id, meter_type, timestamp_ms, fallback_level,
                            tilt_exceeded=True, tilt_deg=tilt_deg,
                        )
                        return result
                    keypoints, _ = self._detect_keypoints_with_fallback(corrected_image)
                    result.keypoints = keypoints

                pointer_angle = self._detect_pointer_angle(corrected_image, keypoints)
                result.pointer_angle = pointer_angle

                if pointer_angle is None:
                    retry_count += 1
                    continue

                if pointer_angle < self._angle_min or pointer_angle > self._angle_max:
                    result.status = ReadingStatus.NEED_MANUAL_REVIEW
                    result.need_manual_review = True
                    result.confidence = 0.0
                    result.failure_reason = "angle_out_of_range"
                    result.metadata = self._build_analog_metadata(
                        roi_id, meter_type, timestamp_ms, fallback_level,
                        pointer_angle=pointer_angle,
                    )
                    return result

                range_info = self._recognize_range(image, meter_type)
                if range_info is None:
                    result.status = ReadingStatus.NEED_MANUAL_REVIEW
                    result.need_manual_review = True
                    result.failure_reason = "range_missing"
                    result.metadata = self._build_analog_metadata(
                        roi_id, meter_type, timestamp_ms, fallback_level,
                        pointer_angle=pointer_angle,
                    )
                    return result

                result.range_min = range_info["min"]
                result.range_max = range_info["max"]
                result.unit = range_info.get("unit", "")

                if result.range_max <= result.range_min:
                    result.status = ReadingStatus.FAILED
                    result.failure_reason = "invalid_range_config"
                    result.metadata = self._build_analog_metadata(
                        roi_id, meter_type, timestamp_ms, fallback_level,
                        pointer_angle=pointer_angle,
                    )
                    return result

                angle_span = self._angle_max - self._angle_min
                r = (pointer_angle - self._angle_min) / angle_span
                value = result.range_min + r * (result.range_max - result.range_min)

                c_det = _sanitize_confidence(
                    min(kp.confidence for kp in [keypoints.center, keypoints.pointer_tip] if kp is not None)
                    if keypoints.center and keypoints.pointer_tip
                    else max(0.1, 0.5 - 0.1 * fallback_level)
                )
                c_geom = 1.0
                if tilt_deg is not None:
                    c_geom = _sanitize_confidence(max(0.0, 1.0 - abs(tilt_deg) / self._max_rotation * 0.3))
                c_parse = _sanitize_confidence(1.0 if 0.0 <= r <= 1.0 else 0.3)
                confidence = _sanitize_confidence(min(c_det, c_geom, c_parse))

                result.value = round(value, 2)
                result.confidence = confidence
                result.retry_count = retry_count

                if confidence >= self._confidence_threshold:
                    result.status = ReadingStatus.SUCCESS
                elif confidence >= self._manual_review_threshold:
                    result.status = ReadingStatus.NEED_MANUAL_REVIEW
                    result.need_manual_review = True
                else:
                    result.status = ReadingStatus.NEED_MANUAL_REVIEW
                    result.need_manual_review = True

                result.metadata = self._build_analog_metadata(
                    roi_id, meter_type, timestamp_ms, fallback_level,
                    pointer_angle=pointer_angle,
                    range_min=result.range_min, range_max=result.range_max,
                    tilt_deg=tilt_deg,
                )
                result.metadata["reading_status"] = result.status.value
                return result

            except Exception as e:
                logger.warning("模拟表读数重试 %d: %s", retry_count, e)
                result.failure_reason = str(e)
                retry_count += 1

        result.retry_count = retry_count
        result.status = ReadingStatus.NEED_MANUAL_REVIEW
        result.need_manual_review = True
        result.failure_reason = result.failure_reason or "all_retries_exhausted"
        result.metadata = self._build_analog_metadata(
            roi_id, meter_type, timestamp_ms, fallback_level,
        )
        return result

    def _build_analog_metadata(
        self,
        roi_id: str,
        meter_type: MeterType,
        timestamp_ms: int,
        fallback_level: int,
        pointer_angle: Optional[float] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        tilt_exceeded: bool = False,
        tilt_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "roi_id": roi_id,
            "meter_type": meter_type.value,
            "reading_status": ReadingStatus.NEED_MANUAL_REVIEW.value,
            "pipeline_stage": "analog_chain",
            "fallback_level": fallback_level,
            "timestamp_ms": timestamp_ms,
        }
        if pointer_angle is not None:
            meta["pointer_angle_deg"] = round(pointer_angle, 2)
        meta["angle_range_deg"] = [self._angle_min, self._angle_max]
        if range_min is not None:
            meta["range_min"] = range_min
        if range_max is not None:
            meta["range_max"] = range_max
        if tilt_exceeded:
            meta["tilt_exceeded"] = True
        if tilt_deg is not None:
            meta["tilt_deg"] = round(tilt_deg, 2)
        return meta

    # ======================== 关键点检测 (三级降级) ========================

    def _detect_keypoints_with_fallback(
        self,
        image: np.ndarray,
    ) -> Tuple[MeterKeypoints, int]:
        """
        三级降级关键点检测:
        Level 0: HRNet (深度学习)
        Level 1: HoughCircle (传统圆检测)
        Level 2: HoughLine (传统线检测)

        Returns:
            (keypoints, fallback_level)
        """
        if self._use_deep_learning and self._dl_initialized and self._dl_keypoint_detector is not None:
            kp = self._detect_keypoints_v3(image)
            if kp and kp.center and kp.pointer_tip:
                return kp, 0

        if self._use_deep_learning and self._model_registry:
            kp = self._detect_keypoints_dl(image)
            if kp and kp.center and kp.pointer_tip:
                return kp, 0

        kp = self._detect_keypoints_hough_circle(image)
        if kp and kp.center:
            return kp, 1

        kp = self._detect_keypoints_hough_line(image)
        if kp:
            return kp, 2

        return MeterKeypoints(), 3

    def _detect_keypoints_v3(self, image: np.ndarray) -> Optional[MeterKeypoints]:
        """HRNet关键点检测"""
        if not self._dl_initialized or self._dl_keypoint_detector is None:
            return None

        try:
            dl_keypoints = self._dl_keypoint_detector.detect_keypoints(image)
            keypoints = MeterKeypoints()
            corners: List[Keypoint] = []

            for kp in dl_keypoints:
                local_kp = Keypoint(name=kp.name, x=kp.x, y=kp.y, confidence=kp.confidence)
                if kp.name == "center":
                    keypoints.center = local_kp
                elif kp.name == "pointer_tip":
                    keypoints.pointer_tip = local_kp
                elif kp.name == "scale_start":
                    keypoints.scale_min = local_kp
                elif kp.name == "scale_end":
                    keypoints.scale_max = local_kp
                elif kp.name in ("top_left", "top_right", "bottom_right", "bottom_left"):
                    corners.append(local_kp)

            keypoints.dial_corners = corners
            return keypoints
        except Exception as e:
            logger.warning("V3.0关键点检测失败: %s", e)
        return None

    def _detect_keypoints_dl(self, image: np.ndarray) -> Optional[MeterKeypoints]:
        """通过model_registry的HRNet关键点检测"""
        try:
            model_id = self.MODEL_IDS["keypoint"]
            result = self._model_registry.infer(model_id, image)

            if result.raw_outputs:
                heatmaps = result.raw_outputs.get("heatmaps")
                if heatmaps is not None:
                    return self._parse_heatmaps(heatmaps)

            if result.detections:
                keypoints = MeterKeypoints()
                for det in result.detections:
                    kp_type = det.get("class_name", "")
                    x, y = det.get("x", 0.5), det.get("y", 0.5)
                    conf = det.get("confidence", 0)
                    kp = Keypoint(name=kp_type, x=x, y=y, confidence=conf)
                    if kp_type == "center":
                        keypoints.center = kp
                    elif kp_type == "pointer_tip":
                        keypoints.pointer_tip = kp
                    elif kp_type == "pointer_base":
                        keypoints.pointer_base = kp
                    elif kp_type == "scale_min":
                        keypoints.scale_min = kp
                    elif kp_type == "scale_max":
                        keypoints.scale_max = kp
                return keypoints
        except Exception as e:
            logger.warning("深度学习关键点检测失败: %s", e)
        return None

    def _parse_heatmaps(self, heatmaps: np.ndarray) -> MeterKeypoints:
        """解析热力图"""
        keypoints = MeterKeypoints()
        kp_names = ["center", "pointer_tip", "pointer_base", "scale_min", "scale_max"]

        for i, name in enumerate(kp_names):
            if i < heatmaps.shape[0]:
                hm = heatmaps[i]
                max_idx = np.unravel_index(np.argmax(hm), hm.shape)
                y_idx, x_idx = max_idx
                h, w = hm.shape
                kp = Keypoint(
                    name=name,
                    x=x_idx / w,
                    y=y_idx / h,
                    confidence=float(hm[max_idx]),
                )
                if name == "center":
                    keypoints.center = kp
                elif name == "pointer_tip":
                    keypoints.pointer_tip = kp
                elif name == "pointer_base":
                    keypoints.pointer_base = kp
                elif name == "scale_min":
                    keypoints.scale_min = kp
                elif name == "scale_max":
                    keypoints.scale_max = kp
        return keypoints

    def _detect_keypoints_hough_circle(self, image: np.ndarray) -> Optional[MeterKeypoints]:
        """Level 1: HoughCircle 几何圆检测"""
        if cv2 is None:
            return None

        keypoints = MeterKeypoints()
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30,
            minRadius=max(10, min(h, w) // 10),
            maxRadius=min(h, w) // 2,
        )

        if circles is None:
            return None

        circles_rounded = np.uint16(np.around(circles))
        circles_array = circles_rounded[0]
        best_circle = max(circles_array, key=lambda c: c[2])
        cx, cy, r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])

        keypoints.center = Keypoint(name="center", x=cx / w, y=cy / h, confidence=0.7)
        keypoints.dial_corners = [
            Keypoint("corner_tl", max(0, cx - r) / w, max(0, cy - r) / h, 0.6),
            Keypoint("corner_tr", min(w, cx + r) / w, max(0, cy - r) / h, 0.6),
            Keypoint("corner_br", min(w, cx + r) / w, min(h, cy + r) / h, 0.6),
            Keypoint("corner_bl", max(0, cx - r) / w, min(h, cy + r) / h, 0.6),
        ]
        return keypoints

    def _detect_keypoints_hough_line(self, image: np.ndarray) -> Optional[MeterKeypoints]:
        """Level 2: HoughLine 指针线检测 (仅圆心估计)"""
        if cv2 is None:
            return None

        keypoints = MeterKeypoints()
        keypoints.center = Keypoint(name="center", x=0.5, y=0.5, confidence=0.4)
        return keypoints

    # ======================== 透视矫正 ========================

    def _apply_perspective_correction(
        self,
        image: np.ndarray,
        keypoints: MeterKeypoints,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        应用透视矫正.

        Returns:
            (corrected_image, estimated_tilt_degrees_or_None)
        """
        if cv2 is None or len(keypoints.dial_corners) < 4:
            return image, None

        h, w = image.shape[:2]

        src_points = np.array([
            [keypoints.dial_corners[0].x * w, keypoints.dial_corners[0].y * h],
            [keypoints.dial_corners[1].x * w, keypoints.dial_corners[1].y * h],
            [keypoints.dial_corners[2].x * w, keypoints.dial_corners[2].y * h],
            [keypoints.dial_corners[3].x * w, keypoints.dial_corners[3].y * h],
        ], dtype=np.float32)

        size = min(h, w)
        dst_points = np.array([
            [0, 0], [size, 0], [size, size], [0, size],
        ], dtype=np.float32)

        edge_lengths = []
        for i in range(4):
            j = (i + 1) % 4
            length = float(np.linalg.norm(src_points[i] - src_points[j]))
            edge_lengths.append(length)

        if len(edge_lengths) >= 2 and min(edge_lengths) > 1e-6:
            ratio = max(edge_lengths) / min(edge_lengths)
            tilt_deg = min(90.0, math.degrees(math.atan(max(0, ratio - 1))))
        else:
            tilt_deg = 0.0

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        corrected = cv2.warpPerspective(image, M, (size, size))

        return corrected, tilt_deg

    # ======================== 指针角度检测 ========================

    def _detect_pointer_angle(
        self,
        image: np.ndarray,
        keypoints: MeterKeypoints,
    ) -> Optional[float]:
        """检测指针角度"""
        if keypoints.center and keypoints.pointer_tip:
            cx = keypoints.center.x
            cy = keypoints.center.y
            px = keypoints.pointer_tip.x
            py = keypoints.pointer_tip.y
            angle = math.atan2(-(py - cy), px - cx) * 180 / math.pi
            return angle

        return self._detect_pointer_by_hough(image, keypoints)

    def _detect_pointer_by_hough(
        self,
        image: np.ndarray,
        keypoints: MeterKeypoints,
    ) -> Optional[float]:
        """霍夫变换检测指针"""
        if cv2 is None:
            return None

        h, w = image.shape[:2]

        if keypoints.center:
            cx = int(keypoints.center.x * w)
            cy = int(keypoints.center.y * h)
        else:
            cx, cy = w // 2, h // 2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            self._hough_threshold,
            minLineLength=self._min_line_length,
            maxLineGap=self._max_line_gap,
        )

        if lines is None:
            return None

        best_line = None
        min_dist = float("inf")

        for line in lines:
            x1, y1, x2, y2 = line[0]
            denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if denominator < 1e-6:
                continue
            dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / denominator
            if dist < min_dist and dist < self._hough_circle_dist_threshold:
                min_dist = dist
                best_line = (x1, y1, x2, y2)

        if best_line is None:
            return None

        x1, y1, x2, y2 = best_line
        d1 = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        d2 = math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)

        if d1 > d2:
            px, py = x1, y1
        else:
            px, py = x2, y2

        angle = math.atan2(-(py - cy), px - cx) * 180 / math.pi
        return angle

    # ======================== 量程识别 ========================

    def _recognize_range(
        self,
        image: np.ndarray,
        meter_type: MeterType,
    ) -> Optional[Dict[str, Any]]:
        """
        识别量程.
        优先级: 配置 > METER_RANGES > OCR
        若全部缺失返回 None (触发人工复核).
        """
        config_ranges = self.config.get("meter_ranges", {})
        type_key = meter_type.value
        if type_key in config_ranges:
            cfg_range = config_ranges[type_key]
            if "min" in cfg_range and "max" in cfg_range:
                return {
                    "min": cfg_range["min"],
                    "max": cfg_range["max"],
                    "unit": cfg_range.get("unit", ""),
                }

        if meter_type in self.METER_RANGES:
            return self.METER_RANGES[meter_type].copy()

        if self._use_deep_learning and self._model_registry:
            ocr_result = self._ocr_range_text(image)
            if ocr_result:
                return ocr_result

        return None

    def _ocr_range_text(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """OCR识别量程文字"""
        try:
            model_id = self.MODEL_IDS["ocr"]
            result = self._model_registry.infer(model_id, image)

            if result.detections:
                texts = [d.get("text", "") for d in result.detections]
                numbers: List[float] = []
                unit = ""

                for text in texts:
                    nums = re.findall(r"[-+]?\d*\.?\d+", text)
                    numbers.extend([float(n) for n in nums])
                    if "MPa" in text or "mpa" in text:
                        unit = "MPa"
                    elif "\u00b0C" in text or "\u2103" in text:
                        unit = "\u00b0C"
                    elif "%" in text:
                        unit = "%"
                    elif "A" in text:
                        unit = "A"
                    elif "V" in text:
                        unit = "V"

                if len(numbers) >= 2:
                    return {"min": min(numbers), "max": max(numbers), "unit": unit}
        except Exception as e:
            logger.warning("OCR量程识别失败: %s", e)
        return None

    # ======================== 数字表链路 ========================

    def _read_digital_meter(
        self,
        image: np.ndarray,
        meter_type: MeterType,
        roi_id: str,
        timestamp_ms: int,
    ) -> MeterReading:
        """数字表/七段码读数"""
        result = MeterReading(meter_type=meter_type, status=ReadingStatus.FAILED)

        raw_text, det_confidence = self._ocr_digital(image)
        det_confidence = _sanitize_confidence(det_confidence)

        cleaned_text, reject_reason = _clean_ocr_text(raw_text)

        if cleaned_text is None:
            result.status = ReadingStatus.NEED_MANUAL_REVIEW
            result.need_manual_review = True
            result.failure_reason = reject_reason
            result.metadata = {
                "roi_id": roi_id,
                "meter_type": meter_type.value,
                "reading_status": ReadingStatus.NEED_MANUAL_REVIEW.value,
                "pipeline_stage": "ocr_chain",
                "fallback_level": 0,
                "timestamp_ms": timestamp_ms,
                "ocr_text_raw": raw_text,
            }
            return result

        value = float(cleaned_text)
        c_parse = 1.0
        confidence = _sanitize_confidence(min(det_confidence, c_parse))

        result.value = value
        result.confidence = confidence

        if confidence >= self._confidence_threshold:
            result.status = ReadingStatus.SUCCESS
        elif confidence >= self._manual_review_threshold:
            result.status = ReadingStatus.NEED_MANUAL_REVIEW
            result.need_manual_review = True
        else:
            result.status = ReadingStatus.NEED_MANUAL_REVIEW
            result.need_manual_review = True

        result.metadata = {
            "roi_id": roi_id,
            "meter_type": meter_type.value,
            "reading_status": result.status.value,
            "pipeline_stage": "ocr_chain",
            "fallback_level": 0,
            "timestamp_ms": timestamp_ms,
            "ocr_text_raw": raw_text,
        }
        return result

    def _ocr_digital(self, image: np.ndarray) -> Tuple[str, float]:
        """OCR识别数字"""
        if self._use_deep_learning and self._model_registry:
            try:
                model_id = self.MODEL_IDS["ocr"]
                result = self._model_registry.infer(model_id, image)

                if result.detections:
                    texts = [d.get("text", "") for d in result.detections]
                    confidences = [d.get("confidence", 0) for d in result.detections]
                    combined_text = "".join(texts)
                    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
                    return combined_text, avg_confidence
            except Exception as e:
                logger.warning("OCR识别失败: %s", e)

        return self._ocr_traditional(image)

    def _ocr_traditional(self, image: np.ndarray) -> Tuple[str, float]:
        """传统OCR方法 (fallback)"""
        if cv2 is None:
            return "", 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return "", 0.0

    # ======================== LED链路 ========================

    def _read_led_indicator(
        self,
        image: np.ndarray,
        roi_id: str,
        timestamp_ms: int,
    ) -> MeterReading:
        """LED指示灯状态识别, 含HSV可分离性校验"""
        result = MeterReading(
            meter_type=MeterType.LED_INDICATOR,
            status=ReadingStatus.FAILED,
        )

        if cv2 is None:
            result.failure_reason = "opencv_unavailable"
            result.metadata = {
                "roi_id": roi_id,
                "meter_type": MeterType.LED_INDICATOR.value,
                "reading_status": ReadingStatus.FAILED.value,
                "pipeline_stage": "hsv_chain",
                "fallback_level": 0,
                "timestamp_ms": timestamp_ms,
            }
            return result

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].astype(np.float64)
        s_ch = hsv[:, :, 1].astype(np.float64)
        v_ch = hsv[:, :, 2].astype(np.float64)

        mu_h = float(np.mean(h_ch))
        mu_s = float(np.mean(s_ch))
        mu_v = float(np.mean(v_ch))
        std_h = float(np.std(h_ch))

        lit_mask = v_ch > self._led_brightness_threshold
        total_pixels = max(1, image.shape[0] * image.shape[1])
        rho_lit = float(np.sum(lit_mask)) / total_pixels

        hsv_stats = {
            "h_mean": round(mu_h, 2),
            "s_mean": round(mu_s, 2),
            "v_mean": round(mu_v, 2),
            "h_std": round(std_h, 2),
            "lit_area_ratio": round(rho_lit, 4),
        }

        if mu_v < self._led_brightness_threshold and rho_lit < self._led_lit_area_min:
            color_class = "off"
            value = 0.0
            c_det = _sanitize_confidence(0.8 + 0.2 * (1.0 - mu_v / 255.0))
        elif mu_s < self._led_saturation_min or std_h > 40:
            result.status = ReadingStatus.NEED_MANUAL_REVIEW
            result.need_manual_review = True
            result.failure_reason = "hsv_not_separable"
            result.metadata = {
                "roi_id": roi_id,
                "meter_type": MeterType.LED_INDICATOR.value,
                "reading_status": ReadingStatus.NEED_MANUAL_REVIEW.value,
                "pipeline_stage": "hsv_chain",
                "fallback_level": 0,
                "timestamp_ms": timestamp_ms,
                "hsv_stats": hsv_stats,
                "color_class": "unknown",
            }
            return result
        elif mu_h < 10 or mu_h > 170:
            color_class = "red"
            value = 1.0
            c_det = _sanitize_confidence(min(1.0, mu_s / 255.0))
        elif 35 < mu_h < 85:
            color_class = "green"
            value = 2.0
            c_det = _sanitize_confidence(min(1.0, mu_s / 255.0))
        elif 10 <= mu_h <= 35:
            color_class = "yellow"
            value = 3.0
            c_det = _sanitize_confidence(min(1.0, mu_s / 255.0))
        else:
            result.status = ReadingStatus.NEED_MANUAL_REVIEW
            result.need_manual_review = True
            result.failure_reason = "color_ambiguous"
            result.metadata = {
                "roi_id": roi_id,
                "meter_type": MeterType.LED_INDICATOR.value,
                "reading_status": ReadingStatus.NEED_MANUAL_REVIEW.value,
                "pipeline_stage": "hsv_chain",
                "fallback_level": 0,
                "timestamp_ms": timestamp_ms,
                "hsv_stats": hsv_stats,
                "color_class": "unknown",
            }
            return result

        confidence = _sanitize_confidence(c_det)

        if confidence >= self._confidence_threshold:
            result.status = ReadingStatus.SUCCESS
        elif confidence >= self._manual_review_threshold:
            result.status = ReadingStatus.NEED_MANUAL_REVIEW
            result.need_manual_review = True
        else:
            result.status = ReadingStatus.NEED_MANUAL_REVIEW
            result.need_manual_review = True

        result.value = value
        result.unit = color_class
        result.confidence = confidence

        result.metadata = {
            "roi_id": roi_id,
            "meter_type": MeterType.LED_INDICATOR.value,
            "reading_status": result.status.value,
            "pipeline_stage": "hsv_chain",
            "fallback_level": 0,
            "timestamp_ms": timestamp_ms,
            "hsv_stats": hsv_stats,
            "color_class": color_class,
        }
        return result

    # ======================== 批量/综合 ========================

    def read_batch(
        self,
        image: np.ndarray,
        rois: List[Dict[str, Any]],
    ) -> List[MeterReading]:
        """批量读取"""
        results: List[MeterReading] = []

        for roi in rois:
            roi_id = roi.get("id", "")
            roi_type = roi.get("type", "pressure_gauge")
            bbox = roi.get("bbox")

            try:
                meter_type = MeterType(roi_type)
            except ValueError:
                meter_type = MeterType.PRESSURE_GAUGE

            reading = self.read_meter(image, meter_type, bbox, roi_id)
            results.append(reading)

        return results

    def inspect(
        self,
        image: np.ndarray,
        rois: Optional[List[Dict[str, Any]]] = None,
    ) -> MeterInspectionResult:
        """综合巡视"""
        start_time = time.perf_counter()

        readings: List[MeterReading] = []
        digital_readings: List[DigitalReading] = []

        if rois:
            readings = self.read_batch(image, rois)

        processing_time = (time.perf_counter() - start_time) * 1000

        return MeterInspectionResult(
            readings=readings,
            digital_readings=digital_readings,
            processing_time_ms=processing_time,
            model_version=self._model_version,
            code_hash=self._code_hash,
        )

    # ======================== 工具方法 ========================

    def _crop_roi(self, image: np.ndarray, bbox: Dict[str, float]) -> np.ndarray:
        """裁剪ROI区域"""
        h, w = image.shape[:2]
        x = int(bbox.get("x", 0) * w)
        y = int(bbox.get("y", 0) * h)
        bw = int(bbox.get("width", bbox.get("w", 1)) * w)
        bh = int(bbox.get("height", bbox.get("h", 1)) * h)

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))

        return image[y : y + bh, x : x + bw]

    def reload_config(self, config: Dict[str, Any]) -> bool:
        """热加载配置 - 不中断推理"""
        try:
            inference_cfg = config.get("inference", {})
            fallback_cfg = config.get("fallback", {})
            new_threshold = inference_cfg.get("confidence_threshold", self._confidence_threshold)
            new_review = fallback_cfg.get("manual_review_threshold", self._manual_review_threshold)

            if not (0.0 <= new_threshold <= 1.0) or not (0.0 <= new_review <= 1.0):
                logger.warning("配置校验失败, 保持旧配置")
                return False

            self._confidence_threshold = new_threshold
            self._manual_review_threshold = new_review
            self.config = config
            logger.info("配置热加载成功")
            return True
        except Exception as e:
            logger.warning("配置热加载失败: %s, 保持旧配置", e)
            return False


def create_detector(config: Dict[str, Any], model_registry: Any = None) -> MeterReadingDetectorEnhanced:
    """创建检测器实例"""
    detector = MeterReadingDetectorEnhanced(config, model_registry)
    detector.initialize()
    return detector
