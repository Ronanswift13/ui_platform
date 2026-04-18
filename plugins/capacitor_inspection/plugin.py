"""
电容器自主巡视插件。

当前主链路:
- 结构完整性检测: 倾斜 / 倒塌 / 单元缺失
- 区域入侵检测: 人员 / 车辆 / 动物

说明:
- `plugin.py` 负责 SDK 适配、ROI 路由、结果/告警封装
- 检测算法与降级链保持在 `detector_enhanced.py`
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
    """动态加载检测器类。"""
    detector_path = Path(__file__).parent / "detector_enhanced.py"
    if not detector_path.exists():
        detector_path = Path(__file__).parent / "detector.py"

    spec = importlib.util.spec_from_file_location("capacitor_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["capacitor_detector"] = module
    spec.loader.exec_module(module)

    if hasattr(module, "CapacitorDetectorEnhanced"):
        return module.CapacitorDetectorEnhanced
    return module.CapacitorDetector


_CapacitorDetector = None


def get_detector_class():
    global _CapacitorDetector
    if _CapacitorDetector is None:
        _CapacitorDetector = _load_detector_class()
    return _CapacitorDetector


class CapacitorInspectionPlugin(BasePlugin):
    """电容器自主巡视插件。"""

    LABEL_NAMES = {
        "tilt_warning": "电容器倾斜(警告)",
        "tilt_error": "电容器倾斜(严重)",
        "collapse": "电容器倒塌",
        "missing_unit": "电容器单元缺失",
        "intrusion_person": "人员入侵",
        "intrusion_vehicle": "车辆入侵",
        "intrusion_animal": "动物入侵",
        "intrusion_unknown": "未知入侵",
    }

    ALARM_LEVELS = {
        "tilt_warning": AlarmLevel.WARNING,
        "tilt_error": AlarmLevel.ERROR,
        "collapse": AlarmLevel.ERROR,
        "missing_unit": AlarmLevel.ERROR,
        "intrusion_person": AlarmLevel.ERROR,
        "intrusion_vehicle": AlarmLevel.ERROR,
        "intrusion_animal": AlarmLevel.WARNING,
        "intrusion_unknown": AlarmLevel.INFO,
    }

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

    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        super().__init__(manifest, plugin_dir)
        self._detector: Any | None = None
        self._initialized = False
        self._last_inference_time: Optional[datetime] = None
        self._inference_count = 0
        self._error_count = 0
        self.confidence_threshold = 0.55
        self._model_resolution: dict[str, Any] = {}

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
            self.confidence_threshold = inference_config.get("confidence_threshold", 0.55)

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
            logger.info("[%s] 插件初始化成功", self.id)
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
            logger.exception("[%s] 插件初始化失败", self.id)
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
        timestamp = self._last_inference_time.timestamp()

        results: list[RecognitionResult] = []
        for roi in rois:
            try:
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None or roi_image.size == 0:
                    continue

                roi_key = self._get_roi_key(roi)

                if self._is_structural_roi(roi_key, getattr(roi, "roi_type", None)):
                    defects = self._detector.detect_structural_defects(roi_image)
                    for defect in defects:
                        if defect.confidence < self.confidence_threshold:
                            continue
                        metadata = dict(getattr(defect, "metadata", {}) or {})
                        if getattr(defect, "tilt_angle", None) is not None:
                            metadata["tilt_angle"] = defect.tilt_angle
                        metadata["roi_key"] = roi_key
                        results.append(
                            self._build_result(
                                context=runtime_context,
                                roi=roi,
                                bbox=self._convert_bbox_to_absolute(defect.bbox, roi.bbox),
                                label=defect.defect_type.value,
                                confidence=float(defect.confidence),
                                metadata=metadata,
                            )
                        )

                if self._is_intrusion_roi(roi_key, getattr(roi, "roi_type", None)):
                    intrusions = self._detector.detect_intrusion(roi_image, timestamp=timestamp)
                    for intrusion in intrusions:
                        if intrusion.confidence < self.confidence_threshold:
                            continue
                        metadata = dict(getattr(intrusion, "metadata", {}) or {})
                        metadata.update(
                            {
                                "zone": intrusion.zone.value,
                                "track_id": intrusion.track_id,
                                "duration_sec": intrusion.duration_sec,
                                "confirmed": intrusion.confirmed,
                                "roi_key": roi_key,
                            }
                        )
                        results.append(
                            self._build_result(
                                context=runtime_context,
                                roi=roi,
                                bbox=self._convert_bbox_to_absolute(intrusion.bbox, roi.bbox),
                                label=f"intrusion_{intrusion.intrusion_type.value}",
                                confidence=float(intrusion.confidence),
                                metadata=metadata,
                            )
                        )
            except Exception as exc:
                self._error_count += 1
                logger.warning("[%s] 处理 ROI %s 时出错: %s", self.id, roi.id, exc)

        self.status = PluginStatus.READY
        return results

    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """根据识别结果生成告警。"""
        alarms: list[Alarm] = []
        for result in results:
            level = self.ALARM_LEVELS.get(result.label)
            if level is None:
                continue
            alarms.append(
                Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=level,
                    title=f"检测到{self.LABEL_NAMES.get(result.label, result.label)}",
                    message=f"在 {result.roi_id} 区域检测到异常",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
            )
        return alarms

    def healthcheck(self) -> HealthStatus:
        """返回插件健康状态。"""
        if not self._initialized:
            return HealthStatus(
                healthy=False,
                message="插件未初始化",
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
                "confidence_threshold": self.confidence_threshold,
                "detector_ready": self._detector is not None,
                "model_resolution": dict(self._model_resolution),
            },
        )

    def get_ui_config(self) -> dict[str, Any]:
        """获取 UI 配置。"""
        return {
            "detection_types": [
                {
                    "id": "structural",
                    "name": "结构完整性检测",
                    "icon": "exclamation-octagon",
                    "description": "电容器倾斜、倒塌、部件缺失检测",
                    "enabled": True,
                    "capabilities": [
                        {"label": "倾斜检测", "tags": ["tilt_warning", "tilt_error"], "level": "warning"},
                        {"label": "倒塌检测", "tags": ["collapse"], "level": "error"},
                        {"label": "部件缺失", "tags": ["missing_unit"], "level": "error"},
                    ],
                },
                {
                    "id": "intrusion",
                    "name": "区域入侵检测",
                    "icon": "shield-exclamation",
                    "description": "人员、车辆、动物入侵告警",
                    "enabled": True,
                    "capabilities": [
                        {"label": "人员入侵", "tags": ["intrusion_person"], "level": "error"},
                        {"label": "车辆入侵", "tags": ["intrusion_vehicle"], "level": "error"},
                        {"label": "动物入侵", "tags": ["intrusion_animal"], "level": "warning"},
                    ],
                },
            ],
            "parameters": [
                {
                    "name": "confidence_threshold",
                    "label": "置信度阈值",
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "default": self.confidence_threshold,
                    "description": "检测结果的最小置信度",
                },
                {
                    "name": "tilt_threshold",
                    "label": "倾斜角度阈值(°)",
                    "type": "number",
                    "min": 1.0,
                    "max": 30.0,
                    "step": 1.0,
                    "default": 5.0,
                    "description": "倾斜检测的最小角度",
                },
            ],
        }

    def _ensure_context(self, context: Optional[PluginContext]) -> PluginContext:
        """为 demo / smoke 提供最小上下文。"""
        if context is not None:
            return context

        return PluginContext(
            task_id="standalone-task",
            site_id="standalone-site",
            device_id="standalone-device",
            component_id="standalone-component",
        )

    def _extract_roi(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """从帧中提取 ROI 区域。"""
        h, w = frame.shape[:2]
        x1 = int(bbox.x * w)
        y1 = int(bbox.y * h)
        x2 = int((bbox.x + bbox.width) * w)
        y2 = int((bbox.y + bbox.height) * h)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()

    def _get_roi_key(self, roi: ROI) -> str:
        """提取可用于路由的 ROI 语义键。"""
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
            for key in ("roi_type", "semantic_type", "zone_type"):
                value = metadata.get(key)
                if value:
                    tokens.append(str(value))

        return " ".join(tokens).lower()

    def _is_structural_roi(self, roi_key: str, roi_type: Any) -> bool:
        """判断是否走结构缺陷链路。"""
        if roi_type == ROIType.DEFECT:
            return True
        return any(token in roi_key for token in ("capacitor_bank", "capacitor_unit", "fuse", "connecting_bar", "insulator"))

    def _is_intrusion_roi(self, roi_key: str, roi_type: Any) -> bool:
        """判断是否走入侵检测链路。"""
        if roi_type == ROIType.INTRUSION:
            return True
        return any(token in roi_key for token in ("fence", "warning_zone", "restricted_zone", "intrusion"))

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

    def _build_result(
        self,
        context: PluginContext,
        roi: ROI,
        bbox: BoundingBox,
        label: str,
        confidence: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> RecognitionResult:
        unified = build_visual_meta(
            plugin_name="capacitor_inspection",
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
            roi_id=roi.id,
            bbox=bbox,
            label=label,
            confidence=confidence,
            model_version=self.version,
            code_version=self.code_hash,
            metadata=merged,
        )


Plugin = CapacitorInspectionPlugin
