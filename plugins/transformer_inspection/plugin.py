"""
主变自主巡视插件 - 完整实现
输变电激光星芒破夜绘明监测平台 (A组)

当前主链路:
- 外观缺陷识别: 破损、锈蚀、渗漏油、异物悬挂
- 状态识别: 呼吸器硅胶、油位计读数
- 热成像集成: 可选热像阈值告警

说明:
- `plugin.py` 负责 SDK 适配、ROI 编排、结果/告警封装
- 识别算法与回退链路保持在 `detector_enhanced.py`
"""

from __future__ import annotations

from datetime import datetime
import importlib.util
import logging
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

from darkbreaker_sdk.interfaces import (
    BasePlugin,
    HealthStatus,
    PluginContext,
    PluginManifest,
    PluginStatus,
)
from darkbreaker_sdk.schemas import (
    Alarm,
    AlarmLevel,
    AlarmRule,
    BoundingBox,
    ROI,
    ROIType,
    RecognitionResult,
)

def _get_model_resolver():
    """延迟加载 model resolution，避免模块级 import 触发 training → torch 初始化。"""
    try:
        from plugins._model_resolution import resolve_plugin_model_config
        return resolve_plugin_model_config
    except ImportError:  # pragma: no cover - standalone fallback
        def _fallback(**kwargs):
            config = kwargs.get("config") or {}
            return config, {
                "enabled": False,
                "attempted": False,
                "resolved": False,
                "error_code": "RESOLVER_IMPORT_FAILED",
                "error_message": "plugins._model_resolution import failed",
            }
        return _fallback

from platform_core.visual_output_protocol import build_visual_meta

logger = logging.getLogger(__name__)


def _load_detector_class():
    """动态加载检测器类，解决相对导入问题。"""
    detector_path = Path(__file__).parent / "detector_enhanced.py"
    if not detector_path.exists():
        detector_path = Path(__file__).parent / "detector.py"

    spec = importlib.util.spec_from_file_location("transformer_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["transformer_detector"] = module
    spec.loader.exec_module(module)

    if hasattr(module, "TransformerDetectorEnhanced"):
        return module.TransformerDetectorEnhanced
    return module.TransformerDetector


_TransformerDetector = None


def get_detector_class():
    global _TransformerDetector
    if _TransformerDetector is None:
        _TransformerDetector = _load_detector_class()
    return _TransformerDetector


class TransformerInspectionPlugin(BasePlugin):
    """
    主变自主巡视插件。

    实现主变压器的缺陷检测和状态识别功能。
    """

    LABEL_NAMES = {
        "oil_leak": "渗漏油",
        "damage": "破损",
        "rust": "锈蚀",
        "foreign_object": "异物悬挂",
        "silica_gel_normal": "硅胶正常",
        "silica_gel_abnormal": "硅胶变色",
        "silica_gel_unknown": "硅胶状态未知",
        "valve_open": "阀门开启",
        "valve_closed": "阀门关闭",
        "valve_intermediate": "阀门中间态",
        "valve_unknown": "阀门状态未知",
        "oil_level_reading": "油位读数",
        "overtemp": "温度异常",
        "normal": "正常",
        "unknown": "未知",
    }

    ALARM_LEVELS = {
        "oil_leak": AlarmLevel.ERROR,
        "damage": AlarmLevel.ERROR,
        "rust": AlarmLevel.WARNING,
        "foreign_object": AlarmLevel.WARNING,
        "silica_gel_abnormal": AlarmLevel.WARNING,
        "valve_intermediate": AlarmLevel.INFO,
        "overtemp": AlarmLevel.ERROR,
    }

    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        super().__init__(manifest, plugin_dir)
        self._detector: Any = None
        self._thermal_enabled = False
        self._thermal_threshold = 80.0
        self._thermal_range = (-20.0, 150.0)
        self._initialized = False
        self._last_inference_time: Optional[datetime] = None
        self._inference_count = 0
        self._error_count = 0

        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.max_detections = 100
        self._model_resolution: dict[str, Any] = {}

    @classmethod
    def create_standalone(cls, config=None):
        """Create plugin instance for standalone operation."""
        plugin_dir = Path(__file__).resolve().parent
        manifest = PluginManifest.from_file(plugin_dir / "manifest.json")
        instance = cls(manifest, plugin_dir)
        if config is None:
            from darkbreaker_sdk.utils import load_plugin_config

            config = load_plugin_config(plugin_dir / "configs" / "default.yaml")
        instance.init(config)
        return instance

    def init(self, config: dict[str, Any]) -> bool:
        """初始化插件。"""
        try:
            resolve_plugin_model_config = _get_model_resolver()
            resolved_config, self._model_resolution = resolve_plugin_model_config(
                plugin_id=self.id,
                plugin_dir=self.plugin_dir,
                config=config,
                default_task_type="detection",
                expected_modality="rgb",
                top_level_path_keys=("yolov8_model_path",),
            )
            self._config = resolved_config
            config = resolved_config

            inference_config = config.get("inference", {})
            self.confidence_threshold = inference_config.get("confidence_threshold", 0.5)
            self.nms_threshold = inference_config.get("nms_threshold", 0.4)
            self.max_detections = inference_config.get("max_detections", 100)

            thermal_config = config.get("thermal", {})
            self._thermal_enabled = thermal_config.get("enabled", False)
            self._thermal_threshold = float(thermal_config.get("temperature_threshold", 80.0))
            self._thermal_range = (
                float(thermal_config.get("min_temp", -20.0)),
                float(thermal_config.get("max_temp", 150.0)),
            )

            detector_cls = get_detector_class()
            self._detector = detector_cls(config)

            detector_initialize = getattr(self._detector, "initialize", None)
            if callable(detector_initialize):
                initialized = bool(detector_initialize())
                if not initialized:
                    logger.warning(
                        "[%s] detector initialize() returned False; fallback paths remain available",
                        self.id,
                    )

            self.status = PluginStatus.READY
            self._initialized = True

            logger.info(
                "[%s] initialized: confidence_threshold=%s thermal_enabled=%s",
                self.id,
                self.confidence_threshold,
                self._thermal_enabled,
            )
            if self._model_resolution.get("resolved"):
                logger.info(
                    "[%s] registry resolved %s/%s@%s -> %s (source=%s)",
                    self.id,
                    self._model_resolution.get("plugin_id"),
                    self._model_resolution.get("task_type"),
                    self._model_resolution.get("version"),
                    self._model_resolution.get("model_path"),
                    self._model_resolution.get("source"),
                )
            elif self._model_resolution.get("attempted") and self._model_resolution.get("error_code"):
                logger.warning(
                    "[%s] registry resolution failed [%s]: %s; keeping configured path",
                    self.id,
                    self._model_resolution.get("error_code"),
                    self._model_resolution.get("error_message"),
                )
            return True

        except Exception as exc:
            self.status = PluginStatus.ERROR
            self._last_error = str(exc)
            logger.exception("[%s] 初始化失败", self.id)
            return False

    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: Optional[PluginContext],
    ) -> list[RecognitionResult]:
        """执行推理。"""
        if not self._initialized or self._detector is None:
            logger.error("[%s] 插件未初始化", self.id)
            return []

        runtime_context = self._ensure_context(context)

        self.status = PluginStatus.RUNNING
        self._last_inference_time = datetime.now()
        self._inference_count += 1

        results: list[RecognitionResult] = []
        thermal_frame = self._extract_thermal_frame(runtime_context)

        for roi in rois:
            try:
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None or roi_image.size == 0:
                    continue

                roi_key = self._get_roi_key(roi)
                defects = self._detector.detect_defects(roi_image)

                for defect in defects[: self.max_detections]:
                    label = self._normalize_defect_label(defect)
                    if label is None or defect.confidence < self.confidence_threshold:
                        continue

                    abs_bbox = self._convert_bbox_to_absolute(defect.bbox, roi.bbox)
                    metadata = dict(getattr(defect, "metadata", {}) or {})
                    metadata.update(
                        {
                            "roi_key": roi_key,
                            "class_name": getattr(defect, "class_name", ""),
                            "defect_type": getattr(getattr(defect, "defect_type", None), "value", None),
                        }
                    )
                    results.append(
                        self._build_result(
                            context=runtime_context,
                            roi_id=roi.id,
                            bbox=abs_bbox,
                            label=label,
                            confidence=float(defect.confidence),
                            metadata=metadata,
                        )
                    )

                results.extend(
                    self._build_state_results(
                        roi_image=roi_image,
                        roi=roi,
                        roi_key=roi_key,
                        context=runtime_context,
                    )
                )

            except Exception as exc:
                self._error_count += 1
                logger.warning("[%s] 处理 ROI %s 时出错: %s", self.id, roi.id, exc)
                continue

        if self._thermal_enabled and thermal_frame is not None:
            results.extend(self._build_thermal_results(frame, thermal_frame, runtime_context))

        self.status = PluginStatus.READY
        logger.info("[%s] 检测完成，共 %s 个结果", self.id, len(results))
        return results

    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """后处理和告警生成。"""
        alarms: list[Alarm] = []

        for result in results:
            alarm_level = self.ALARM_LEVELS.get(result.label)

            if alarm_level is not None:
                label_name = self.LABEL_NAMES.get(result.label, result.label)
                alarms.append(
                    Alarm(
                        task_id=result.task_id,
                        result_id=None,
                        level=alarm_level,
                        title=f"检测到{label_name}",
                        message=f"在 {result.roi_id} 区域检测到{label_name}，置信度: {result.confidence:.2f}",
                        site_id=result.site_id,
                        device_id=result.device_id,
                        component_id=result.component_id,
                    )
                )

        return alarms

    def healthcheck(self) -> HealthStatus:
        """健康检查。"""
        if not self._initialized:
            return HealthStatus(
                healthy=False,
                message="插件未初始化",
                details={"status": self.status.value},
            )

        if self._detector is None:
            return HealthStatus(
                healthy=False,
                message="检测器未就绪",
                details={"status": self.status.value},
            )

        return HealthStatus(
            healthy=True,
            message="插件运行正常",
            details={
                "status": self.status.value,
                "inference_count": self._inference_count,
                "error_count": self._error_count,
                "last_inference": self._last_inference_time.isoformat()
                if self._last_inference_time
                else None,
                "thermal_enabled": self._thermal_enabled,
                "config": {
                    "confidence_threshold": self.confidence_threshold,
                    "nms_threshold": self.nms_threshold,
                    "max_detections": self.max_detections,
                    "thermal_threshold": self._thermal_threshold,
                },
                "model_resolution": dict(self._model_resolution),
            },
        )

    def _ensure_context(self, context: Optional[PluginContext]) -> PluginContext:
        """为 demo/standalone smoke 提供最小上下文。"""
        if context is not None:
            return context

        return PluginContext(
            task_id="standalone-task",
            site_id="standalone-site",
            device_id="standalone-device",
            component_id="standalone-component",
        )

    def _extract_thermal_frame(self, context: PluginContext) -> Optional[np.ndarray]:
        """从 runtime context 中提取热像。"""
        metadata = getattr(context, "metadata", {})
        if isinstance(metadata, dict):
            thermal_frame = metadata.get("thermal_frame")
            if isinstance(thermal_frame, np.ndarray):
                return thermal_frame

        thermal_frame = getattr(context, "thermal_frame", None)
        if isinstance(thermal_frame, np.ndarray):
            return thermal_frame
        return None

    def _build_result(
        self,
        context: PluginContext,
        roi_id: str,
        bbox: BoundingBox,
        label: str,
        confidence: float,
        value: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> RecognitionResult:
        unified = build_visual_meta(
            plugin_name="transformer_inspection",
            task_type="inspection",
            modality="visual",
            runtime_mode="traditional_fallback",
            algorithm_stage="baseline",
            quality_gate_status="pass",
        )
        merged = {**unified, **(metadata or {})}
        return RecognitionResult(
            task_id=context.task_id,
            site_id=context.site_id,
            device_id=context.device_id,
            component_id=context.component_id,
            roi_id=roi_id,
            bbox=bbox,
            label=label,
            value=value,
            confidence=confidence,
            model_version=self.version,
            code_version=self.code_hash,
            metadata=merged,
        )

    def _build_state_results(
        self,
        roi_image: np.ndarray,
        roi: ROI,
        roi_key: str,
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """按当前 detector 真实能力组装状态类结果。"""
        results: list[RecognitionResult] = []

        if any(token in roi_key for token in ("breather", "silica")):
            silica_result = self._detector.recognize_silica_gel(roi_image)
            if silica_result.confidence >= self.confidence_threshold:
                label_map = {
                    "normal": "silica_gel_normal",
                    "warning": "silica_gel_abnormal",
                    "alarm": "silica_gel_abnormal",
                    "unknown": "silica_gel_unknown",
                }
                label = label_map.get(silica_result.state.value, "silica_gel_unknown")
                metadata = dict(silica_result.metadata or {})
                if silica_result.color_rgb is not None:
                    metadata["color_rgb"] = list(silica_result.color_rgb)
                results.append(
                    self._build_result(
                        context=context,
                        roi_id=roi.id,
                        bbox=roi.bbox,
                        label=label,
                        value=silica_result.state.value,
                        confidence=float(silica_result.confidence),
                        metadata=metadata,
                    )
                )

        elif any(token in roi_key for token in ("oil_level", "meter", "gauge")):
            oil_result = self._detector.detect_oil_level(roi_image)
            if oil_result.confidence >= self.confidence_threshold:
                metadata = dict(oil_result.metadata or {})
                metadata["level_status"] = oil_result.level_status
                results.append(
                    self._build_result(
                        context=context,
                        roi_id=roi.id,
                        bbox=roi.bbox,
                        label="oil_level_reading",
                        value=oil_result.level_ratio,
                        confidence=float(oil_result.confidence),
                        metadata=metadata,
                    )
                )

        return results

    def _build_thermal_results(
        self,
        frame: np.ndarray,
        thermal_frame: np.ndarray,
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """热像阈值结果适配。"""
        thermal_result = self._detector.analyze_thermal(
            thermal_frame,
            visible_image=frame,
            temperature_range=self._thermal_range,
        )
        if thermal_result.max_temperature < self._thermal_threshold:
            return []

        results: list[RecognitionResult] = []
        hotspots = thermal_result.hotspots or []
        if not hotspots:
            hotspots = [
                {
                    "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0},
                    "max_temp": thermal_result.max_temperature,
                    "avg_temp": thermal_result.avg_temperature,
                }
            ]

        for hotspot in hotspots:
            bbox = hotspot.get("bbox", {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0})
            results.append(
                self._build_result(
                    context=context,
                    roi_id="thermal",
                    bbox=BoundingBox(
                        x=float(bbox["x"]),
                        y=float(bbox["y"]),
                        width=float(bbox["width"]),
                        height=float(bbox["height"]),
                    ),
                    label="overtemp",
                    value=hotspot.get("max_temp", thermal_result.max_temperature),
                    confidence=0.9,
                    metadata={
                        "thermal_level": thermal_result.level.value,
                        "avg_temperature": thermal_result.avg_temperature,
                        "threshold": self._thermal_threshold,
                        "hotspot_count": thermal_result.hotspot_count,
                    },
                )
            )

        return results

    def _extract_roi(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """从帧中提取 ROI 区域。"""
        h, w = frame.shape[:2]

        x1 = int(bbox.x * w)
        y1 = int(bbox.y * h)
        x2 = int((bbox.x + bbox.width) * w)
        y2 = int((bbox.y + bbox.height) * h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2].copy()

    def _convert_bbox_to_absolute(
        self,
        rel_bbox: dict[str, float],
        roi_bbox: BoundingBox,
    ) -> BoundingBox:
        """将 ROI 内相对坐标转换为全图相对坐标。"""
        abs_x = roi_bbox.x + rel_bbox["x"] * roi_bbox.width
        abs_y = roi_bbox.y + rel_bbox["y"] * roi_bbox.height
        abs_w = rel_bbox["width"] * roi_bbox.width
        abs_h = rel_bbox["height"] * roi_bbox.height

        return BoundingBox(
            x=max(0.0, min(1.0, abs_x)),
            y=max(0.0, min(1.0, abs_y)),
            width=max(0.0, min(1.0 - max(0.0, abs_x), abs_w)),
            height=max(0.0, min(1.0 - max(0.0, abs_y), abs_h)),
        )

    def _get_roi_key(self, roi: ROI) -> str:
        """收敛 ROI 语义键，优先用 name / id / metadata。"""
        tokens: list[str] = []

        roi_type = getattr(roi, "roi_type", None)
        if isinstance(roi_type, ROIType):
            tokens.append(roi_type.value)
        elif roi_type is not None:
            tokens.append(str(roi_type))

        for attr_name in ("name", "id", "description"):
            value = getattr(roi, attr_name, "")
            if value:
                tokens.append(str(value))

        metadata = getattr(roi, "metadata", {})
        if isinstance(metadata, dict):
            for key in ("roi_type", "semantic_type", "kind"):
                value = metadata.get(key)
                if value:
                    tokens.append(str(value))

        return " ".join(tokens).lower()

    def _normalize_defect_label(self, defect: Any) -> Optional[str]:
        """统一 detector 输出标签到插件输出标签。"""
        label_map = {
            "oil_leak": "oil_leak",
            "rust": "rust",
            "damage": "damage",
            "foreign": "foreign_object",
            "foreign_object": "foreign_object",
            "crack": "damage",
            "deformation": "damage",
            "discoloration": "damage",
            "油泄漏": "oil_leak",
            "锈蚀": "rust",
            "破损": "damage",
            "异物": "foreign_object",
        }

        candidates = [
            getattr(getattr(defect, "defect_type", None), "value", None),
            getattr(defect, "class_name", None),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            normalized = label_map.get(str(candidate).lower()) or label_map.get(str(candidate))
            if normalized:
                return normalized
        return None

    def _get_label_name(self, label: str) -> str:
        """获取标签的中文名称。"""
        return self.LABEL_NAMES.get(label, label)


Plugin = TransformerInspectionPlugin
