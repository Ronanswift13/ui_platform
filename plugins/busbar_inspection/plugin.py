"""
母线自主巡视插件 - 完整实现
输变电激光星芒破夜绘明监测平台 (C组)

功能范围:
- 远距小目标检测: 销钉缺失、裂纹、异物
- 多目标并发处理: 批量ROI高效推理
- 环境过滤: 逆光/雨雾/遮挡场景识别
- 变焦建议: 目标过小时输出zoom建议

性能指标:
- pin_missing: Recall >= 0.85, Precision >= 0.85
- crack: Recall >= 0.70, Precision >= 0.80
- GPU P95 <= 800ms, CPU P95 <= 5s
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
from datetime import datetime
import importlib.util
import logging
import sys
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
    RecognitionResult,
    ROI,
    BoundingBox,
)
from darkbreaker_sdk.schemas.common import ROIType

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

try:
    from .reason_code_mapper import ReasonCodeMapper
except ImportError:
    from reason_code_mapper import ReasonCodeMapper

try:
    from .label_contract import (
        LABEL_DISPLAY_NAMES,
        RUNTIME_SUPPORTED_DEFECT_LABELS,
        RUNTIME_SUPPORTED_LABELS,
        canonicalize_label,
        get_label_contract_snapshot,
        is_runtime_supported_defect_label,
    )
except ImportError:
    from label_contract import (  # type: ignore[no-redef]
        LABEL_DISPLAY_NAMES,
        RUNTIME_SUPPORTED_DEFECT_LABELS,
        RUNTIME_SUPPORTED_LABELS,
        canonicalize_label,
        get_label_contract_snapshot,
        is_runtime_supported_defect_label,
    )


logger = logging.getLogger(__name__)


def _load_detector_class():
    """动态加载检测器类"""
    # 优先加载增强版检测器
    detector_path = Path(__file__).parent / "detector_enhanced.py"
    if not detector_path.exists():
        detector_path = Path(__file__).parent / "detector.py"

    module_dir = str(detector_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("busbar_detector", detector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载检测器模块: {detector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["busbar_detector"] = module
    spec.loader.exec_module(module)

    # 优先返回增强版检测器类
    if hasattr(module, 'BusbarDetectorEnhanced'):
        return module.BusbarDetectorEnhanced
    return module.BusbarDetector


_BusbarDetector = None

def get_detector_class():
    global _BusbarDetector
    if _BusbarDetector is None:
        _BusbarDetector = _load_detector_class()
    return _BusbarDetector


class BusbarInspectionPlugin(BasePlugin):
    """
    母线自主巡视插件
    
    实现远距小目标检测、环境过滤和变焦建议
    
    性能指标(按验收标准):
    - pin_missing: Recall >= 0.85, Precision >= 0.85
    - crack: Recall >= 0.70, Precision >= 0.80
    - GPU P95 <= 800ms
    """
    
    SUPPORTED_RUNTIME_LABELS = RUNTIME_SUPPORTED_LABELS
    SUPPORTED_RUNTIME_DEFECT_LABELS = RUNTIME_SUPPORTED_DEFECT_LABELS

    # 标签名称映射
    LABEL_NAMES = {
        label: LABEL_DISPLAY_NAMES[label]
        for label in SUPPORTED_RUNTIME_LABELS
    }
    
    # 告警级别映射
    ALARM_LEVELS = {
        "pin_missing": AlarmLevel.ERROR,
        "crack": AlarmLevel.WARNING,
        "foreign_object": AlarmLevel.WARNING,
    }
    
    # 原因码说明
    REASON_DESCRIPTIONS = {
        101: "逆光/过曝/低对比",
        102: "遮挡/不可见",
        103: "模糊/失焦",
        104: "雨雾/霾导致低能见度",
        105: "运动干扰",
        201: "目标过小,需要变焦",
        202: "检测不稳定",
        301: "结果不可信",
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
        self._detector: Any = None
        self._initialized = False
        self._last_inference_time: Optional[datetime] = None
        self._inference_count = 0
        self._error_count = 0
        self._quality_fail_count = 0
        self._model_resolution: dict[str, Any] = {}

        # 配置参数
        self.confidence_threshold = 0.5

    def _get_detector_runtime_status(self, quality_blocked: bool = False) -> dict[str, Any]:
        """获取 detector 暴露的运行真实性快照。"""
        if self._detector is None:
            return {
                "runtime_mode": "quality_blocked" if quality_blocked else "traditional_fallback",
                "model_path_configured": None,
                "model_path_resolved": None,
                "model_file_exists": False,
                "real_model_loaded": False,
                "onnx_session_ready": False,
                "fallback_enabled": False,
                "dl_preflight_checked": False,
                "dl_preflight_passed": False,
                "dl_failure_reason": None,
                "dl_failure_details": [],
                "manifest_path": None,
                "manifest_exists": False,
                "class_map_path": None,
                "class_map_exists": False,
                "class_map_compatible": False,
                "model_version_validated": None,
                "input_size_validated": None,
                "output_format_validated": None,
                "validated_class_names": [],
                "requested_providers": [],
                "session_providers": [],
                "input_tensor_shape": None,
                "output_tensor_shapes": [],
                "output_probe_shape": None,
                "output_structure_compatible": False,
                "session_error": None,
                "runtime_supported_labels": list(self.SUPPORTED_RUNTIME_LABELS),
                "runtime_supported_defect_labels": list(self.SUPPORTED_RUNTIME_DEFECT_LABELS),
                "label_contract": get_label_contract_snapshot(),
                "model_resolution": dict(self._model_resolution),
            }

        getter = getattr(self._detector, "get_runtime_status", None)
        if callable(getter):
            status = dict(getter(quality_blocked=quality_blocked))
        else:
            status = {
                "runtime_mode": "quality_blocked" if quality_blocked else "traditional_fallback",
                "model_path_configured": None,
                "model_path_resolved": None,
                "model_file_exists": False,
                "real_model_loaded": False,
                "onnx_session_ready": False,
                "fallback_enabled": False,
                "runtime_supported_labels": list(self.SUPPORTED_RUNTIME_LABELS),
                "runtime_supported_defect_labels": list(self.SUPPORTED_RUNTIME_DEFECT_LABELS),
            }
        status["label_contract"] = get_label_contract_snapshot()
        status["model_resolution"] = dict(self._model_resolution)
        return status

    def _log_runtime_status(self, quality_blocked: bool = False) -> None:
        """初始化/健康检查时输出关键运行真实性字段。"""
        status = self._get_detector_runtime_status(quality_blocked=quality_blocked)
        logger.info(
            (
                "[%s] runtime_mode=%s model_path_configured=%s model_path_resolved=%s "
                "model_file_exists=%s real_model_loaded=%s onnx_session_ready=%s "
                "fallback_enabled=%s dl_preflight_passed=%s dl_failure_reason=%s"
            ),
            self.id,
            status["runtime_mode"],
            status["model_path_configured"],
            status["model_path_resolved"],
            status["model_file_exists"],
            status["real_model_loaded"],
            status["onnx_session_ready"],
            status["fallback_enabled"],
            status.get("dl_preflight_passed"),
            status.get("dl_failure_reason"),
        )

    def _get_quality_gate_status(self, quality: Any) -> str:
        """读取 detector 暴露的三态质量门禁结果。"""
        gate_status = getattr(quality, "quality_gate_status", None)
        return str(getattr(gate_status, "value", gate_status or "pass"))
    
    def init(self, config: dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            resolve_plugin_model_config = _get_model_resolver()
            resolved_config, self._model_resolution = resolve_plugin_model_config(
                plugin_id=self.id,
                plugin_dir=self.plugin_dir,
                config=config,
                default_task_type="detection",
                expected_modality="rgb",
                model_path_keys=("model_path",),
                top_level_path_keys=("yolov8_model_path",),
            )
            self._config = resolved_config
            config = resolved_config
            
            inference_config = config.get("inference", {})
            thresholds_config = config.get("thresholds", {})
            self.confidence_threshold = thresholds_config.get(
                "conf_thr",
                inference_config.get("confidence_threshold", 0.5),
            )
            
            # 创建检测器
            BusbarDetector = get_detector_class()
            self._detector = BusbarDetector(config)
            detector_initialize = getattr(self._detector, "initialize", None)
            if callable(detector_initialize):
                detector_initialize()
            
            self.status = PluginStatus.READY
            self._initialized = True
            
            logger.info("[%s] 插件初始化成功", self.id)
            logger.info("[%s] 置信度阈值: %.2f", self.id, self.confidence_threshold)
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
            self._log_runtime_status()
            
            return True
            
        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            logger.exception("[%s] 初始化失败: %s", self.id, e)
            return False
    
    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """
        执行推理
        
        支持4K大视场切片检测,输出质量门禁结果
        """
        if not self._initialized or self._detector is None:
            logger.error("[%s] 插件未初始化", self.id)
            return []
        
        self.status = PluginStatus.RUNNING
        self._last_inference_time = datetime.now()
        self._inference_count += 1

        results: list[RecognitionResult] = []

        # 空 ROI 列表 → 合成一个全帧 ROI
        # 背景: standalone runner /api/detect 调用 infer(frame, [], ctx)，调用方
        # 未指定 ROI 时语义是"对整帧做检测"；不合成则管道不会执行。
        if not rois:
            rois = [ROI(
                id="full_frame",
                name="full_frame",
                component_id=context.component_id or "full_frame",
                roi_type=ROIType.DEFECT,
                bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
            )]

        for roi in rois:
            try:
                # 提取ROI区域
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None or roi_image.size == 0:
                    continue
                
                roi_type = self._get_roi_type(roi)
                
                # 执行检测(包含质量门禁)
                tiling_cfg = self._config.get("tiling") or self._config.get("slicing") or {}
                use_tiling = bool(tiling_cfg.get("enabled", True)) if isinstance(tiling_cfg, dict) else True
                det_result = self._detector.detect_roi(
                    roi_image,
                    use_tiling=use_tiling,
                    roi_type=roi_type,
                )

                quality = det_result.quality
                zoom = det_result.zoom_suggestion
                debug_info = det_result.debug_info or {}
                quality_gate_status = str(
                    debug_info.get(
                        "quality_gate_status",
                        self._get_quality_gate_status(quality),
                    )
                )
                runtime_mode = str(
                    debug_info.get(
                        "runtime_mode",
                        self._get_detector_runtime_status()["runtime_mode"],
                    )
                )
                component_id = context.component_id or roi.component_id or "busbar_component"

                quality_dict = self._adapt_quality(quality)
                zoom_dict = self._adapt_zoom(zoom, quality, det_result.reason_code)

                # 处理 HARD_FAIL 质量门禁失败
                if quality_gate_status == "hard_fail":
                    self._quality_fail_count += 1
                    external_reason_code = self._to_external_reason_code(det_result.reason_code)
                    reason_desc = self._describe_reason_code(external_reason_code)
                    detection_id = self._build_detection_id(roi.id, 0, prefix="qg")

                    _hard_fail_meta = {
                        "detection_id": detection_id,
                        "pred_label": "quality_failed",
                        "candidate_labels": ["quality_failed"],
                        "source": "quality_gate",
                        "quality_gate_status": "hard_fail",
                        "runtime_mode": runtime_mode,
                        "review_status": "blocked",
                        "reason_code": external_reason_code,
                        "reason": reason_desc,
                        "suggested_action": zoom_dict["suggested_action"],
                        "quality_suggested_action": zoom_dict["quality_suggested_action"],
                        "zoom_suggested_action": zoom_dict["zoom_suggested_action"],
                        "suggested_zoom": zoom_dict["suggested_zoom"],
                        "roi_type": roi_type,
                        "roi_name": getattr(roi, "name", roi.id),
                        "quality": quality_dict,
                        "details": {
                            "quality": quality_dict,
                            "debug": debug_info,
                        },
                    }
                    result = RecognitionResult(
                        task_id=context.task_id,
                        site_id=context.site_id,
                        device_id=context.device_id,
                        component_id=component_id,
                        roi_id=roi.id,
                        bbox=roi.bbox,
                        label="quality_failed",
                        confidence=1.0,
                        model_version=self.version,
                        code_version=self.code_hash,
                        failure_reason=(
                            str(external_reason_code)
                            if external_reason_code is not None
                            else None
                        ),
                        metadata={
                            **build_visual_meta(
                                plugin_name="busbar_inspection",
                                task_type="inspection",
                                modality="visual",
                                runtime_mode=runtime_mode,
                                algorithm_stage="baseline",
                                quality_gate_status="hard_fail",
                            ),
                            **_hard_fail_meta,
                        },
                    )
                    results.append(result)
                    continue

                # 处理检测结果
                roi_results_added = 0
                for det_index, det in enumerate(det_result.detections):
                    det_label = self._adapt_defect_label(det)
                    if not is_runtime_supported_defect_label(det_label):
                        logger.warning(
                            "[%s] 跳过未纳入当前 runtime baseline 的标签: %s",
                            self.id,
                            det_label,
                        )
                        continue
                    det_conf = float(getattr(det, "confidence", 0.0))
                    if det_conf < self.confidence_threshold:
                        continue
                    # 转换坐标
                    abs_bbox = self._convert_bbox_to_absolute(det.bbox, roi.bbox)
                    internal_reason_code = self._extract_internal_reason_code(
                        det,
                        det_result.reason_code,
                    )
                    external_reason_code = self._to_external_reason_code(internal_reason_code)
                    metadata = self._adapt_detection_metadata(
                        det=det,
                        det_label=det_label,
                        det_conf=det_conf,
                        roi=roi,
                        roi_type=roi_type,
                        quality_dict=quality_dict,
                        zoom_dict=zoom_dict,
                        debug_info=debug_info,
                        runtime_mode=runtime_mode,
                        detection_index=det_index,
                        external_reason_code=external_reason_code,
                    )

                    result = RecognitionResult(
                        task_id=context.task_id,
                        site_id=context.site_id,
                        device_id=context.device_id,
                        component_id=component_id,
                        roi_id=roi.id,
                        bbox=abs_bbox,
                        label=det_label,
                        confidence=det_conf,
                        model_version=self.version,
                        code_version=self.code_hash,
                        failure_reason=(
                            str(external_reason_code)
                            if external_reason_code is not None
                            else None
                        ),
                        metadata=metadata,
                    )
                    results.append(result)
                    roi_results_added += 1

                if quality_gate_status == "soft_fail" and roi_results_added == 0:
                    self._quality_fail_count += 1
                    external_reason_code = self._to_external_reason_code(det_result.reason_code)
                    reason_desc = self._describe_reason_code(external_reason_code)
                    detection_id = self._build_detection_id(roi.id, 0, prefix="qgsoft")

                    _soft_fail_extra = {
                        "detection_id": detection_id,
                        "pred_label": "quality_failed",
                        "candidate_labels": ["quality_failed"],
                        "source": "quality_gate",
                        "review_status": "review_required",
                        "review_reason": "质量门禁软失败且未形成稳定检测，需人工复核",
                        "reason_code": external_reason_code,
                        "reason": reason_desc,
                        "suggested_action": zoom_dict["suggested_action"],
                        "quality_suggested_action": zoom_dict["quality_suggested_action"],
                        "zoom_suggested_action": zoom_dict["zoom_suggested_action"],
                        "suggested_zoom": zoom_dict["suggested_zoom"],
                        "roi_type": roi_type,
                        "roi_name": getattr(roi, "name", roi.id),
                        "quality": quality_dict,
                        "details": {
                            "quality": quality_dict,
                            "debug": debug_info,
                        },
                    }
                    results.append(RecognitionResult(
                        task_id=context.task_id,
                        site_id=context.site_id,
                        device_id=context.device_id,
                        component_id=component_id,
                        roi_id=roi.id,
                        bbox=roi.bbox,
                        label="quality_failed",
                        confidence=1.0,
                        model_version=self.version,
                        code_version=self.code_hash,
                        failure_reason=(
                            str(external_reason_code)
                            if external_reason_code is not None
                            else None
                        ),
                        metadata={
                            **build_visual_meta(
                                plugin_name="busbar_inspection",
                                task_type="inspection",
                                modality="visual",
                                runtime_mode=runtime_mode,
                                algorithm_stage="baseline",
                                quality_gate_status="soft_fail",
                                review_required=True,
                                reason_codes=(
                                    [str(external_reason_code)]
                                    if external_reason_code is not None else []
                                ),
                                recommended_actions=[zoom_dict["suggested_action"]],
                            ),
                            **_soft_fail_extra,
                        },
                    ))
                    
            except Exception as e:
                self._error_count += 1
                logger.exception("[%s] 处理ROI %s 时出错: %s", self.id, roi.id, e)
                continue
        
        self.status = PluginStatus.READY
        logger.info("[%s] 检测完成，共 %d 个结果", self.id, len(results))
        
        return results
    
    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """后处理和告警生成"""
        alarms: list[Alarm] = []
        
        for result in results:
            # 质量门禁告警
            if result.label == "quality_failed":
                reason_code_str = result.failure_reason
                reason_code_int = int(reason_code_str) if reason_code_str is not None else None
                reason_desc = self._describe_reason_code(reason_code_int)
                quality_gate_status = result.metadata.get("quality_gate_status", "hard_fail")
                title_prefix = "质量门禁需复核" if quality_gate_status == "soft_fail" else "质量门禁未通过"
                
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title=f"{title_prefix}: {reason_desc}",
                    message=(
                        f"ROI {result.roi_id} 图像质量状态={quality_gate_status}，"
                        f"建议: {result.metadata.get('suggested_action', '重新抓拍')}"
                    ),
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
                continue
            
            # 缺陷告警
            alarm_level = self.ALARM_LEVELS.get(result.label)
            if alarm_level is not None:
                label_name = self.LABEL_NAMES.get(result.label, result.label)
                
                message = f"在 {result.roi_id} 区域检测到{label_name}，置信度: {result.confidence:.2f}"
                if result.metadata.get("review_status") == "review_required":
                    message += "。结果存在类别冲突，建议人工复核"
                
                # 添加变焦建议
                if result.metadata and result.metadata.get("suggested_zoom"):
                    message += f"。建议变焦 {result.metadata['suggested_zoom']}x 进行复核"
                
                alarm = Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=alarm_level,
                    title=f"检测到{label_name}",
                    message=message,
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                )
                alarms.append(alarm)
        
        return alarms
    
    def healthcheck(self) -> HealthStatus:
        """健康检查"""
        runtime_status = self._get_detector_runtime_status()
        if not self._initialized:
            return HealthStatus(
                healthy=False,
                message="插件未初始化",
                details={
                    "status": self.status.value,
                    **runtime_status,
                }
            )

        if self._detector is None:
            return HealthStatus(
                healthy=False,
                message="检测器未就绪",
                details={
                    "status": self.status.value,
                    **runtime_status,
                }
            )

        healthy = bool(runtime_status["real_model_loaded"] or runtime_status["fallback_enabled"])
        return HealthStatus(
            healthy=healthy,
            message=f"插件运行正常 ({runtime_status['runtime_mode']})" if healthy else "插件不可用",
            details={
                "status": self.status.value,
                "inference_count": self._inference_count,
                "error_count": self._error_count,
                "quality_fail_count": self._quality_fail_count,
                "last_inference": self._last_inference_time.isoformat() if self._last_inference_time else None,
                **runtime_status,
            }
        )

    def get_ui_config(self) -> dict[str, Any]:
        """获取UI配置"""
        return {
            "detection_types": [
                {
                    "id": "defect",
                    "name": "缺陷检测",
                    "icon": "exclamation-triangle",
                    "description": "销钉缺失、裂纹、异物等远距小目标检测",
                    "enabled": True,
                    "capabilities": [
                        {
                            "label": self.LABEL_NAMES["pin_missing"],
                            "tags": ["pin_missing"],
                            "level": "error",
                        },
                        {
                            "label": self.LABEL_NAMES["crack"],
                            "tags": ["crack"],
                            "level": "warning",
                        },
                        {
                            "label": self.LABEL_NAMES["foreign_object"],
                            "tags": ["foreign_object"],
                            "level": "warning",
                        },
                    ]
                },
                {
                    "id": "quality",
                    "name": "图像质量评估",
                    "icon": "image",
                    "description": "逆光/雨雾/遮挡等环境过滤",
                    "enabled": True,
                    "capabilities": [
                        {"label": self.LABEL_NAMES["quality_failed"], "tags": ["quality_failed"]},
                        {"label": "清晰度评价", "tags": ["clarity_score"]},
                        {"label": "环境过滤", "tags": ["quality_gate"]},
                        {"label": "变焦建议", "tags": ["zoom_suggestion"]},
                    ]
                }
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
                    "description": "检测结果的最小置信度"
                },
                {
                    "name": "use_tiling",
                    "label": "启用切片检测",
                    "type": "boolean",
                    "default": True,
                    "description": "对大视场图像进行切片检测"
                }
            ]
        }
    
    # ==================== 辅助方法 ====================

    # 原因码 → 建议动作 映射 (与 02_algorithm_contract.md 一致)
    _QUALITY_ACTION_MAP = {
        1001: "REFOCUS",       # 模糊/失焦 → 重新对焦
        1002: "RECAPTURE",     # 过曝 → 重新抓拍
        1003: "RECAPTURE",     # 欠曝 → 重新抓拍
        1004: "CHANGE_VIEW",   # 遮挡 → 换角度
        1005: "RECAPTURE",     # 低对比 → 重新抓拍
    }

    def _adapt_quality(self, quality: Any) -> dict[str, Any]:
        """将 detector QualityGateResult 适配到插件输出契约.

        detector 的 QualityGateResult 字段: status/clarity_score/
        brightness_score/contrast_score/occlusion_ratio。
        契约要求: clarity_score/is_overexposed/is_low_contrast/
        is_occluded/edge_energy。从 status 枚举派生布尔位。
        """
        if quality is None:
            return {
                "quality_gate_status": "pass",
                "status": "pass",
                "clarity_score": 0.0,
                "is_overexposed": False,
                "is_low_contrast": False,
                "is_occluded": False,
                "edge_energy": 0.0,
            }

        status = getattr(quality, "status", None)
        status_val = getattr(status, "value", status)
        quality_gate_status = self._get_quality_gate_status(quality)
        metadata = getattr(quality, "metadata", {}) or {}
        return {
            "quality_gate_status": quality_gate_status,
            "status": status_val,
            "failure_reason": str(getattr(quality, "failure_reason", "") or metadata.get("failure_reason", "")),
            "suggested_action": str(metadata.get("suggested_action", "NONE")),
            "clarity_score": float(getattr(quality, "clarity_score", 0.0)),
            "is_blurred": status_val == "fail_blur",
            "is_overexposed": status_val == "fail_overexposed",
            "is_underexposed": status_val == "fail_underexposed",
            "is_low_contrast": status_val == "fail_low_contrast",
            "is_occluded": status_val == "fail_occluded",
            "brightness_score": float(getattr(quality, "brightness_score", 0.0)),
            "contrast_score": float(getattr(quality, "contrast_score", 0.0)),
            "occlusion_ratio": float(getattr(quality, "occlusion_ratio", 0.0)),
            "edge_energy": float(
                getattr(getattr(quality, "metadata", {}), "get", lambda *_: 0.0)("edge_energy", 0.0)
            ),
        }

    def _adapt_zoom(
        self,
        zoom: Any,
        quality: Any,
        reason_code: Optional[int],
    ) -> dict[str, Any]:
        """将 detector ZoomSuggestion 适配到插件输出契约.

        detector 的 ZoomSuggestion 字段: current_zoom/recommended_zoom/
        reason/target_area/priority。
        契约要求: suggested_zoom/suggested_action/min_object_size_px/
        target_size_px。
        """
        zoom_cfg = self._config.get("zoom", {}) if isinstance(self._config, dict) else {}
        min_px = int(zoom_cfg.get("min_obj_px", 18))
        target_px = int(zoom_cfg.get("target_px", 90))
        quality_gate_status = self._get_quality_gate_status(quality)
        quality_action = (
            self._QUALITY_ACTION_MAP.get(reason_code, "RECAPTURE")
            if quality_gate_status in {"soft_fail", "hard_fail"}
            else "NONE"
        )

        # 质量失败路径: 从 reason_code 派生动作
        if zoom is None:
            return {
                "suggested_zoom": 1.0,
                "suggested_action": quality_action,
                "quality_suggested_action": quality_action,
                "zoom_suggested_action": "NONE",
                "min_object_size_px": min_px,
                "target_size_px": target_px,
            }

        rec_zoom = float(getattr(zoom, "recommended_zoom", 1.0))
        cur_zoom = float(getattr(zoom, "current_zoom", 1.0))
        # 变焦动作: 需要放大则 ZOOM_IN，否则 NONE
        if rec_zoom > cur_zoom * 1.2:
            zoom_action = "ZOOM_IN"
        else:
            zoom_action = "NONE"
        action = quality_action if quality_gate_status in {"soft_fail", "hard_fail"} else zoom_action
        return {
            "suggested_zoom": rec_zoom,
            "suggested_action": action,
            "quality_suggested_action": quality_action,
            "zoom_suggested_action": zoom_action,
            "min_object_size_px": min_px,
            "target_size_px": target_px,
        }

    def _extract_internal_reason_code(
        self,
        det: Any,
        fallback_reason_code: Optional[int],
    ) -> Optional[int]:
        """提取 detector 内部原因码。"""
        raw_reason_code = getattr(det, "reason_code", None)
        if raw_reason_code in (None, ""):
            raw_reason_code = fallback_reason_code
        if raw_reason_code in (None, ""):
            return None
        try:
            return int(raw_reason_code)
        except (TypeError, ValueError):
            return None

    def _to_external_reason_code(self, internal_reason_code: Optional[int]) -> Optional[int]:
        """统一将内部原因码映射为插件输出原因码。"""
        if internal_reason_code is None:
            return None
        mapped = ReasonCodeMapper.map_to_external(internal_reason_code)
        if mapped is not None:
            return mapped
        if internal_reason_code in self.REASON_DESCRIPTIONS:
            return internal_reason_code
        return None

    def _describe_reason_code(self, reason_code: Optional[int]) -> str:
        """获取原因码说明。"""
        if reason_code is None:
            return "未知原因"
        return self.REASON_DESCRIPTIONS.get(
            reason_code,
            ReasonCodeMapper.get_description(reason_code),
        )

    def _normalize_source(self, raw_source: Any) -> str:
        """规范化检测来源。"""
        source = str(raw_source or "").lower()
        if not source:
            return "unknown"
        if source == "simulator":
            return "simulator"
        if "traditional" in source:
            return "traditional"
        if "quality" in source:
            return "quality_gate"
        if "yolo" in source or "model_registry" in source or source == "dl":
            return "dl"
        return source

    def _build_detection_id(
        self,
        roi_id: str,
        detection_index: int,
        prefix: str = "det",
    ) -> str:
        """生成检测唯一标识。"""
        return (
            f"{self.id}-{prefix}-{self._inference_count:06d}-"
            f"{roi_id}-{detection_index:03d}"
        )

    def _adapt_detection_metadata(
        self,
        det: Any,
        det_label: str,
        det_conf: float,
        roi: ROI,
        roi_type: str,
        quality_dict: dict[str, Any],
        zoom_dict: dict[str, Any],
        debug_info: dict[str, Any],
        runtime_mode: str,
        detection_index: int,
        external_reason_code: Optional[int],
    ) -> dict[str, Any]:
        """将 detector 元数据适配为 UI/standalone 可直接消费的明细结构。"""
        raw_metadata = dict(getattr(det, "metadata", {}) or {})
        quality_gate_status = str(
            raw_metadata.get(
                "quality_gate_status",
                quality_dict.get("quality_gate_status", "pass"),
            )
        )
        review_status = str(raw_metadata.get("review_status", "confirmed"))
        candidate_labels = [
            canonicalize_label(label)
            for label in (raw_metadata.get("candidate_labels") or [det_label])
        ]
        candidate_labels = [
            label for label in candidate_labels
            if is_runtime_supported_defect_label(label)
        ] or [det_label]
        suggested_action = str(raw_metadata.get("suggested_action") or zoom_dict["suggested_action"])
        if quality_gate_status == "soft_fail":
            review_status = "review_required"
            if suggested_action == "NONE":
                suggested_action = str(
                    raw_metadata.get("quality_suggested_action")
                    or zoom_dict["quality_suggested_action"]
                    or "RECAPTURE"
                )
        if review_status == "review_required" and suggested_action == "NONE":
            suggested_action = "MANUAL_REVIEW"

        metadata = {
            "detection_id": self._build_detection_id(roi.id, detection_index),
            "pred_label": det_label,
            "candidate_labels": candidate_labels,
            "candidate_scores": raw_metadata.get("candidate_scores", {det_label: det_conf}),
            "primary_score": raw_metadata.get("primary_score", det_conf),
            "secondary_label": canonicalize_label(raw_metadata.get("secondary_label")) or None,
            "secondary_score": raw_metadata.get("secondary_score"),
            "score_gap": raw_metadata.get("score_gap"),
            "review_status": review_status,
            "review_reason": raw_metadata.get("review_reason", ""),
            "source": self._normalize_source(raw_metadata.get("source")),
            "quality_gate_status": quality_gate_status,
            "runtime_mode": runtime_mode,
            "reason_code": external_reason_code,
            "reason": self._describe_reason_code(external_reason_code)
            if external_reason_code is not None
            else "",
            "suggested_zoom": zoom_dict["suggested_zoom"],
            "suggested_action": suggested_action,
            "quality_suggested_action": raw_metadata.get(
                "quality_suggested_action",
                zoom_dict["quality_suggested_action"],
            ),
            "zoom_suggested_action": zoom_dict["zoom_suggested_action"],
            "min_object_size_px": zoom_dict["min_object_size_px"],
            "target_size_px": zoom_dict["target_size_px"],
            "quality": quality_dict,
            "debug": debug_info,
            "severity": self._derive_severity(det_label, det_conf),
            "roi_type": roi_type,
            "roi_name": getattr(roi, "name", roi.id),
        }
        if quality_gate_status == "soft_fail" and not metadata["review_reason"]:
            metadata["review_reason"] = "质量门禁软失败，需人工复核"
        metadata.update(raw_metadata)
        metadata["candidate_labels"] = list(dict.fromkeys(candidate_labels))
        metadata["review_status"] = review_status
        if quality_gate_status == "soft_fail" and not metadata.get("review_reason"):
            metadata["review_reason"] = "质量门禁软失败，需人工复核"
        metadata["source"] = self._normalize_source(metadata.get("source"))
        metadata["quality_gate_status"] = quality_gate_status
        metadata["runtime_mode"] = runtime_mode
        metadata["secondary_label"] = canonicalize_label(metadata.get("secondary_label")) or None
        metadata["reason_code"] = external_reason_code
        metadata["reason"] = (
            self._describe_reason_code(external_reason_code)
            if external_reason_code is not None
            else metadata.get("reason", "")
        )
        metadata["suggested_action"] = suggested_action
        metadata["quality_suggested_action"] = metadata.get(
            "quality_suggested_action",
            zoom_dict["quality_suggested_action"],
        )
        metadata["zoom_suggested_action"] = zoom_dict["zoom_suggested_action"]
        metadata["suggested_zoom"] = zoom_dict["suggested_zoom"]
        metadata["detection_id"] = self._build_detection_id(roi.id, detection_index)

        # 统一视觉输出协议字段
        unified = build_visual_meta(
            plugin_name="busbar_inspection",
            task_type="inspection",
            modality="visual",
            runtime_mode=runtime_mode,
            algorithm_stage="baseline",
            quality_gate_status=quality_gate_status,
            review_required=(review_status == "review_required"),
            reason_codes=(
                [str(external_reason_code)]
                if external_reason_code is not None else []
            ),
            recommended_actions=[suggested_action] if suggested_action != "NONE" else [],
            evidence_hint="visual_frame",
        )
        # 统一协议字段作为底层，插件特有字段覆盖
        unified.update(metadata)
        return unified

    def _adapt_defect_label(self, det: Any) -> str:
        """从 detector BusbarDetection 提取标签字符串.

        detector 使用 defect_type (Enum)，契约要求小写字符串。
        """
        defect_type = getattr(det, "defect_type", None)
        if defect_type is not None:
            val = getattr(defect_type, "value", None)
            if val:
                return canonicalize_label(val)
            return canonicalize_label(defect_type)
        # 降级: 类名 / 显式 label
        return canonicalize_label(
            getattr(det, "class_name", getattr(det, "label", "unknown"))
        )

    def _derive_severity(self, label: str, confidence: float) -> str:
        """根据标签和置信度派生告警严重度 (UI 展示用)."""
        if label == "pin_missing":
            return "high" if confidence > 0.80 else "medium"
        if label == "crack":
            return "high" if confidence > 0.75 else "medium"
        if label == "foreign_object":
            return "medium" if confidence > 0.70 else "low"
        return "low"

    def _extract_roi(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """从帧中提取ROI区域"""
        h, w = frame.shape[:2]
        
        x1 = int(bbox.x * w)
        y1 = int(bbox.y * h)
        x2 = int((bbox.x + bbox.width) * w)
        y2 = int((bbox.y + bbox.height) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2].copy()
    
    def _get_roi_type(self, roi: ROI) -> str:
        """获取ROI类型"""
        roi_name = getattr(roi, 'name', 'unknown').lower()
        roi_type = getattr(roi, 'roi_type', None)
        if roi_type is not None:
            roi_type_value = roi_type.value if hasattr(roi_type, 'value') else str(roi_type)
            if roi_name not in {"", "unknown"} and roi_type_value in {
                "defect", "state", "meter", "thermal", "intrusion",
            }:
                return roi_name
            return roi_type_value
        return roi_name
    
    def _convert_bbox_to_absolute(
        self,
        rel_bbox: Any,
        roi_bbox: BoundingBox
    ) -> BoundingBox:
        """转换坐标"""
        if hasattr(rel_bbox, "x") and hasattr(rel_bbox, "width"):
            rel_x = rel_bbox.x
            rel_y = rel_bbox.y
            rel_w = rel_bbox.width
            rel_h = rel_bbox.height
        else:
            rel_x = rel_bbox["x"]
            rel_y = rel_bbox["y"]
            rel_w = rel_bbox["width"]
            rel_h = rel_bbox["height"]

        abs_x = roi_bbox.x + rel_x * roi_bbox.width
        abs_y = roi_bbox.y + rel_y * roi_bbox.height
        abs_w = rel_w * roi_bbox.width
        abs_h = rel_h * roi_bbox.height

        return BoundingBox(x=abs_x, y=abs_y, width=abs_w, height=abs_h)


Plugin = BusbarInspectionPlugin
