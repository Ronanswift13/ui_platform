"""
母线自主巡视检测器 - 增强版 V3.5
输变电激光监测平台 (C组) - 全自动AI巡检改造

增强功能:
- YOLOv8-ViT小目标检测 (UPDATE.md短期计划)
- 4K图像切片处理: 重叠瓦片分解
- 多尺度特征融合: FPN改进
- 质量门禁增强: 模糊/过曝/遮挡检测
- 智能变焦建议: 自动计算推荐倍数

V3.0更新:
- 集成YOLOv8-ViT深度学习模型
- 增强远距小目标检测能力
- 支持多尺度推理

V3.5更新 (室外监测迭代):
- 时序ReID模块: 跨帧缺陷重识别和轨迹追踪
- 裂纹增长分析: 扩展趋势预测和风险评估
- 缺陷轨迹管理: 长期监测和历史对比
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import time
import numpy as np

logger = logging.getLogger(__name__)

try:
    from .config_adapter import ConfigAdapter
except ImportError:
    from config_adapter import ConfigAdapter

try:
    from .label_contract import (
        RUNTIME_SUPPORTED_DEFECT_LABELS,
        RUNTIME_SUPPORTED_LABELS,
        canonicalize_label,
        is_runtime_supported_defect_label,
    )
except ImportError:
    from label_contract import (  # type: ignore[no-redef]
        RUNTIME_SUPPORTED_DEFECT_LABELS,
        RUNTIME_SUPPORTED_LABELS,
        canonicalize_label,
        is_runtime_supported_defect_label,
    )

try:
    from .onnx_preflight import OnnxPreflightResult, run_onnx_preflight
except ImportError:
    from onnx_preflight import (  # type: ignore[no-redef]
        OnnxPreflightResult,
        run_onnx_preflight,
    )


def _sanitize_confidence(value: float) -> float:
    """置信度消毒: NaN/Inf/负数/超1 → 安全区间 [0, 1] (QR-11)"""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)

try:
    import cv2
except ImportError:
    cv2 = None

# V3.0: 导入YOLOv8-ViT深度学习模型
try:
    from ai_models.deep_learning.yolov8_vit import (
        YOLOv8ViTDetector, YOLOv8ViTConfig, DetectionTask
    )
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    YOLOv8ViTDetector = None
    YOLOv8ViTConfig = None
    DetectionTask = None

# V3.5: 导入时序ReID模块和裂纹增长分析
try:
    from ai_models.deep_learning.temporal_reid import (
        TemporalReIDModule,
        TemporalReIDConfig,
        CrackGrowthAnalyzer,
        CrackGrowthAnalysis,
        DefectType as ReIDDefectType
    )
    TEMPORAL_REID_AVAILABLE = True
except ImportError:
    TEMPORAL_REID_AVAILABLE = False
    TemporalReIDModule = None
    TemporalReIDConfig = None
    CrackGrowthAnalyzer = None
    CrackGrowthAnalysis = None
    ReIDDefectType = None


class BusbarDefectType(Enum):
    """母线缺陷类型"""
    PIN_MISSING = "pin_missing"         # 销钉缺失
    CRACK = "crack"                     # 裂纹
    FOREIGN_OBJECT = "foreign_object"   # 异物悬挂
    CORROSION = "corrosion"             # 腐蚀
    FLASHOVER = "flashover"             # 闪络痕迹
    BROKEN_STRAND = "broken_strand"     # 断股
    INSULATOR_DAMAGE = "insulator_damage"  # 绝缘子损坏
    FITTING_LOOSE = "fitting_loose"     # 金具松动


class QualityGateStatus(Enum):
    """质量门禁状态"""
    PASS = "pass"
    FAIL_BLUR = "fail_blur"
    FAIL_OVEREXPOSED = "fail_overexposed"
    FAIL_UNDEREXPOSED = "fail_underexposed"
    FAIL_OCCLUDED = "fail_occluded"
    FAIL_LOW_CONTRAST = "fail_low_contrast"


class QualityGateDecision(Enum):
    """质量门禁决策级别。"""
    PASS = "pass"
    SOFT_FAIL = "soft_fail"
    HARD_FAIL = "hard_fail"


@dataclass
class BusbarDetection:
    """母线检测结果"""
    defect_type: BusbarDefectType
    bbox: Dict[str, float]          # 归一化坐标
    confidence: float
    class_name: str
    reason_code: str = ""           # 失败原因码
    tile_info: Optional[Dict] = None  # 切片信息
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityGateResult:
    """质量门禁结果"""
    status: QualityGateStatus
    clarity_score: float            # 清晰度评分 0-1
    brightness_score: float         # 亮度评分 0-1
    contrast_score: float           # 对比度评分 0-1
    occlusion_ratio: float          # 遮挡比例 0-1
    quality_gate_status: QualityGateDecision = QualityGateDecision.PASS
    reason_code: str = ""
    failure_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZoomSuggestion:
    """变焦建议"""
    current_zoom: float
    recommended_zoom: float
    reason: str
    target_area: Optional[Dict[str, float]] = None
    priority: int = 0               # 优先级 0-10


@dataclass
class BusbarInspectionResult:
    """母线巡视综合结果"""
    detections: List[BusbarDetection] = field(default_factory=list)
    quality_gate: Optional[QualityGateResult] = None
    zoom_suggestions: List[ZoomSuggestion] = field(default_factory=list)
    total_tiles: int = 0
    processed_tiles: int = 0
    processing_time_ms: float = 0.0
    model_version: str = ""
    code_hash: str = ""
    # V3.5: 时序ReID和裂纹增长分析
    temporal_reid_stats: Optional[Dict[str, Any]] = None
    crack_growth_analyses: Optional[Dict[str, Any]] = None


@dataclass
class DetectROIResult:
    """detect_roi方法的返回结果 (兼容plugin.py)"""
    detections: List[BusbarDetection] = field(default_factory=list)
    quality: Optional[QualityGateResult] = None
    zoom_suggestion: Optional[ZoomSuggestion] = None
    reason_code: Optional[int] = None
    debug_info: Optional[Dict[str, Any]] = None


class BusbarDetectorEnhanced:
    """
    母线巡视增强检测器
    
    支持4K大视场图像的切片处理和小目标检测
    """
    
    # 模型ID映射
    MODEL_IDS = {
        "detector": "busbar_yolov8m_small",     # YOLOv8m小目标检测
        "classifier": "busbar_defect_classifier",  # 缺陷分类器
    }

    RUNTIME_SUPPORTED_LABELS = RUNTIME_SUPPORTED_LABELS
    RUNTIME_SUPPORTED_DEFECT_LABELS = RUNTIME_SUPPORTED_DEFECT_LABELS

    # 缺陷类别映射（保留历史兼容，当前交付标签由 label_contract.py 冻结）
    DEFECT_CLASSES = {
        0: BusbarDefectType.PIN_MISSING,
        1: BusbarDefectType.CRACK,
        2: BusbarDefectType.FOREIGN_OBJECT,
        3: BusbarDefectType.CORROSION,
        4: BusbarDefectType.FLASHOVER,
        5: BusbarDefectType.BROKEN_STRAND,
        6: BusbarDefectType.INSULATOR_DAMAGE,
        7: BusbarDefectType.FITTING_LOOSE,
    }
    
    # 原因码定义
    REASON_CODES = {
        "1001": "图像模糊",
        "1002": "过度曝光",
        "1003": "曝光不足",
        "1004": "目标遮挡",
        "1005": "对比度过低",
        "2001": "目标过小需放大",
        "2002": "检测置信度过低",
        "2003": "多目标重叠",
        "3001": "环境干扰(鸟类)",
        "3002": "环境干扰(飞虫)",
        "3003": "环境干扰(水滴)",
    }

    QUALITY_ACTION_MAP = {
        "1001": "REFOCUS",
        "1002": "RECAPTURE",
        "1003": "RECAPTURE",
        "1004": "CHANGE_VIEW",
        "1005": "RECAPTURE",
    }
    
    # 默认切片参数
    DEFAULT_TILE_SIZE = 1280
    DEFAULT_OVERLAP = 128
    MIN_TARGET_SIZE = 32  # 最小目标像素
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_registry=None,
    ):
        """
        初始化增强检测器

        Args:
            config: 配置字典
            model_registry: 模型注册表实例 (已废弃，保留兼容性)
        """
        self.config = config
        self._model_registry = model_registry
        self._initialized = False
        self._config_adapter = ConfigAdapter(config)

        # 配置参数
        self._confidence_threshold = float(self._get_runtime_config("confidence_threshold", 0.25))
        self._nms_threshold = float(self._get_runtime_config("nms_threshold", 0.50))
        self._tile_size = int(self._get_runtime_config("tile_size", self.DEFAULT_TILE_SIZE))
        self._tile_overlap = int(self._get_runtime_config("tile_overlap", 320))
        self._use_slicing = config.get("use_slicing", True)
        self._use_deep_learning = config.get("use_deep_learning", True)

        # 质量门禁阈值
        self._clarity_threshold = float(self._get_runtime_config("clarity_threshold", 0.35))
        self._brightness_range = config.get("brightness_range", (0.2, 0.8))
        self._contrast_threshold_explicit = "contrast_threshold" in config
        self._dynamic_range_min = float(self._get_quality_config("dr_min", 35))
        self._contrast_threshold = float(
            config.get("contrast_threshold", self._dynamic_range_min / 255.0)
        )
        self._brightness_high = int(self._get_quality_config("y_high", 245))
        self._overexposed_ratio = float(self._get_quality_config("overexp_ratio", 0.25))
        self._edge_threshold = float(self._get_quality_config("edge_thr", 10))
        self._decision_config = self._get_nested_config("decision", {})
        conflict_cfg = self._get_nested_config("decision.conflict_arbitration", {})
        output_cfg = self._get_nested_config("decision.output_limits", {})
        crack_validation_cfg = self._get_nested_config(
            "decision.traditional_crack_validation",
            {},
        )
        self._uncertain_reason_code = str(
            self._decision_config.get("unstable_reason_code", 2002)
        )
        self._conflict_iou_threshold = float(conflict_cfg.get("iou_threshold", 0.15))
        self._review_score_gap = float(conflict_cfg.get("review_score_gap", 0.12))
        self._scene_max_detections = int(output_cfg.get("scene_max_detections", 12))
        self._per_class_topk = {
            str(label): int(limit)
            for label, limit in (output_cfg.get("per_class_topk", {}) or {}).items()
        }
        self._roi_priors = self._decision_config.get("roi_priors", {}) or {}
        self._traditional_crack_validation = {
            "min_length_px": float(crack_validation_cfg.get("min_length_px", 40.0)),
            "min_slenderness": float(crack_validation_cfg.get("min_slenderness", 4.0)),
            "min_local_contrast": float(crack_validation_cfg.get("min_local_contrast", 12.0)),
            "axis_aligned_tolerance_deg": float(
                crack_validation_cfg.get("axis_aligned_tolerance_deg", 8.0)
            ),
            "reject_axis_aligned": bool(
                crack_validation_cfg.get("reject_axis_aligned", False)
            ),
            "reject_vertical_axis_aligned": bool(
                crack_validation_cfg.get("reject_vertical_axis_aligned", False)
            ),
            "default_max_axis_aligned_span_ratio": float(
                crack_validation_cfg.get("default_max_axis_aligned_span_ratio", 0.18)
            ),
            "roi_overrides": crack_validation_cfg.get("roi_overrides", {}) or {},
        }

        # V3.0: YOLOv8-ViT深度学习检测器
        self._yolov8_vit_detector: Optional[YOLOv8ViTDetector] = None
        self._dl_initialized = False
        self._model_path_configured: Optional[str] = None
        self._model_path_resolved: Optional[str] = None
        self._model_file_exists = False
        self._real_model_loaded = False
        self._onnx_session_ready = False
        self._fallback_enabled = cv2 is not None
        self._runtime_mode = "traditional_fallback"
        self._dl_preflight = OnnxPreflightResult()

        # V3.5: 时序ReID模块和裂纹增长分析
        self._temporal_reid: Optional[Any] = None
        self._crack_analyzer: Optional[Any] = None
        self._reid_enabled = config.get("temporal_reid", {}).get("enabled", True)
        self._growth_analysis_enabled = config.get("crack_growth", {}).get("enabled", True)
        self._frame_counter = 0

        # 版本信息 (V3.5更新)
        self._model_version = "busbar_enhanced_v3.5"
        self._code_hash = self._calculate_code_hash()
        self._refresh_model_path_state()

    def _get_runtime_config(self, unified_key: str, default: Any) -> Any:
        """读取统一配置视图，优先兼容已扁平化的测试配置。"""
        if unified_key in self.config:
            return self.config[unified_key]
        return self._config_adapter.get(unified_key, default)

    def _get_quality_config(self, key: str, default: Any) -> Any:
        """读取质量门禁配置，兼容嵌套 YAML 与旧式扁平键。"""
        quality_cfg = self.config.get("quality", {})
        if isinstance(quality_cfg, dict) and key in quality_cfg:
            return quality_cfg[key]
        return self.config.get(key, default)

    def _get_nested_config(self, path: str, default: Any) -> Any:
        """读取嵌套配置，兼容旧式扁平键。"""
        current: Any = self.config
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    
    def _calculate_code_hash(self) -> str:
        """计算代码版本hash"""
        import inspect
        source = inspect.getsource(self.__class__)
        return f"sha256:{hashlib.sha256(source.encode()).hexdigest()[:12]}"

    def _get_configured_model_path(self) -> Optional[str]:
        """读取配置中的模型路径字符串。"""
        model_cfg = self.config.get("model", {})
        model_path = self.config.get("yolov8_model_path")
        if model_path is None and isinstance(model_cfg, dict):
            model_path = model_cfg.get("model_path")
        if model_path in (None, ""):
            return None
        return str(model_path)

    def _refresh_model_path_state(self) -> Optional[Path]:
        """解析模型路径并记录真实存在性。"""
        model_path = self._get_configured_model_path()
        self._model_path_configured = model_path
        self._model_path_resolved = None
        self._model_file_exists = False

        if not model_path:
            return None

        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parent / candidate

        resolved = candidate.resolve(strict=False)
        self._model_path_resolved = str(resolved)
        self._model_file_exists = resolved.exists()
        return resolved

    def _update_runtime_mode(self) -> None:
        """根据真实加载状态刷新当前运行模式。"""
        if self._real_model_loaded and self._onnx_session_ready and self._dl_initialized:
            self._runtime_mode = "real_dl"
        else:
            self._runtime_mode = "traditional_fallback"

    def _get_requested_runtime_providers(self) -> List[str]:
        """提取 runtime providers 配置。"""
        runtime_cfg = self.config.get("runtime", {})
        providers = runtime_cfg.get("providers", []) if isinstance(runtime_cfg, dict) else []
        return [str(provider) for provider in (providers or [])]

    def _mark_no_real_dl(self) -> None:
        """明确标记当前不具备真实 DL 运行条件。"""
        self._dl_initialized = False
        self._real_model_loaded = False
        self._onnx_session_ready = False
        self._update_runtime_mode()

    def _apply_preflight_snapshot(self, result: OnnxPreflightResult) -> None:
        """将 preflight 结果同步到运行状态字段。"""
        self._dl_preflight = result
        self._model_path_configured = result.model_path_configured
        self._model_path_resolved = result.model_path_resolved
        self._model_file_exists = result.model_file_exists
        self._onnx_session_ready = result.session_ready
        self._real_model_loaded = False
        self._dl_initialized = False
        self._update_runtime_mode()

    def get_runtime_status(self, quality_blocked: bool = False) -> Dict[str, Any]:
        """暴露 detector 的真实运行状态。"""
        runtime_mode = "quality_blocked" if quality_blocked else self._runtime_mode
        status = {
            "runtime_mode": runtime_mode,
            "model_path_configured": self._model_path_configured,
            "model_path_resolved": self._model_path_resolved,
            "model_file_exists": self._model_file_exists,
            "real_model_loaded": self._real_model_loaded,
            "onnx_session_ready": self._onnx_session_ready,
            "fallback_enabled": self._fallback_enabled,
            "runtime_supported_labels": list(self.RUNTIME_SUPPORTED_LABELS),
            "runtime_supported_defect_labels": list(self.RUNTIME_SUPPORTED_DEFECT_LABELS),
        }
        status.update(self._dl_preflight.to_runtime_status())
        return status

    def _log_runtime_status(self, level: int = logging.INFO) -> None:
        """将关键运行真实性字段打到初始化日志。"""
        logger.log(
            level,
            (
                "runtime_mode=%s model_path_configured=%s model_path_resolved=%s "
                "model_file_exists=%s real_model_loaded=%s onnx_session_ready=%s "
                "fallback_enabled=%s dl_preflight_passed=%s dl_failure_reason=%s "
                "class_map_compatible=%s output_structure_compatible=%s"
            ),
            self._runtime_mode,
            self._model_path_configured,
            self._model_path_resolved,
            self._model_file_exists,
            self._real_model_loaded,
            self._onnx_session_ready,
            self._fallback_enabled,
            self._dl_preflight.passed,
            self._dl_preflight.failure_reason or None,
            self._dl_preflight.class_map_compatible,
            self._dl_preflight.output_structure_compatible,
        )
    
    def initialize(self) -> bool:
        """初始化检测器"""
        try:
            # V3.0: 优先初始化YOLOv8-ViT深度学习检测器
            if self._use_deep_learning and DL_AVAILABLE:
                self._init_yolov8_vit()

            # V3.5: 初始化时序ReID模块
            if self._reid_enabled and TEMPORAL_REID_AVAILABLE:
                self._init_temporal_reid()

            # V3.5: 初始化裂纹增长分析器
            if self._growth_analysis_enabled and TEMPORAL_REID_AVAILABLE:
                self._init_crack_analyzer()

            # 兼容旧版：如果有模型注册表，预加载模型
            if self._model_registry and self._use_deep_learning and not self._dl_initialized:
                for model_key, model_id in self.MODEL_IDS.items():
                    try:
                        self._model_registry.load(model_id)
                    except Exception as e:
                        logger.warning("模型 %s 加载失败: %s", model_id, e)

            self._update_runtime_mode()
            self._initialized = True
            self._log_runtime_status()
            return True
        except Exception as e:
            logger.exception("检测器初始化失败: %s", e)
            return False

    def _init_temporal_reid(self) -> bool:
        """初始化时序ReID模块 (V3.5)"""
        if not TEMPORAL_REID_AVAILABLE or TemporalReIDModule is None:
            logger.warning("时序ReID模块不可用")
            return False

        try:
            reid_config = self.config.get("temporal_reid", {})
            config = TemporalReIDConfig(
                feature_dim=reid_config.get("feature_dim", 256),
                match_threshold=reid_config.get("match_threshold", 0.7),
                use_temporal_attention=reid_config.get("use_temporal_attention", True),
                max_miss_count=reid_config.get("max_miss_count", 10),
                temporal_window=reid_config.get("temporal_window", 30)
            )

            self._temporal_reid = TemporalReIDModule(config)
            logger.info("时序ReID模块初始化成功 (V3.5)")
            return True

        except Exception as e:
            logger.exception("时序ReID初始化失败: %s", e)
            return False

    def _init_crack_analyzer(self) -> bool:
        """初始化裂纹增长分析器 (V3.5)"""
        if not TEMPORAL_REID_AVAILABLE or CrackGrowthAnalyzer is None:
            logger.warning("裂纹增长分析模块不可用")
            return False

        try:
            growth_config = self.config.get("crack_growth", {})
            self._crack_analyzer = CrackGrowthAnalyzer(
                pixel_to_mm=growth_config.get("pixel_to_mm", 0.1),
                critical_length=growth_config.get("critical_length", 50.0),
                history_window_days=growth_config.get("history_window_days", 30)
            )
            logger.info("裂纹增长分析器初始化成功 (V3.5)")
            return True

        except Exception as e:
            logger.exception("裂纹增长分析器初始化失败: %s", e)
            return False

    def _init_yolov8_vit(self) -> bool:
        """初始化YOLOv8-ViT检测器 (V3.0)"""
        if not DL_AVAILABLE:
            logger.warning("YOLOv8-ViT模块不可用")
            self._mark_no_real_dl()
            self._log_runtime_status(logging.WARNING)
            return False

        try:
            runtime_cfg = self.config.get("runtime", {})
            model_cfg = self.config.get("model", {})
            preflight = run_onnx_preflight(
                plugin_dir=Path(__file__).resolve().parent,
                model_path_configured=self._get_configured_model_path(),
                runtime_providers=self._get_requested_runtime_providers(),
                runtime_supported_defect_labels=self.RUNTIME_SUPPORTED_DEFECT_LABELS,
                canonicalize_label=canonicalize_label,
                is_runtime_supported_defect_label=is_runtime_supported_defect_label,
                model_config=model_cfg if isinstance(model_cfg, dict) else {},
            )
            self._apply_preflight_snapshot(preflight)

            if not preflight.passed:
                logger.warning(
                    (
                        "real_dl preflight 未通过，保持 traditional_fallback: "
                        "failure_reason=%s details=%s"
                    ),
                    preflight.failure_reason or "unknown",
                    preflight.failure_details,
                )
                self._yolov8_vit_detector = None
                self._dl_initialized = False
                self._real_model_loaded = False
                self._update_runtime_mode()
                self._log_runtime_status(logging.WARNING)
                return False

            device = self.config.get("device")
            if device is None and isinstance(runtime_cfg, dict):
                providers = preflight.session_providers or self._get_requested_runtime_providers()
                device = "cuda" if any("CUDA" in str(p) for p in providers) else "cpu"
            if device is None:
                device = "cpu"

            config = YOLOv8ViTConfig(
                model_path=preflight.model_path_resolved,
                task=DetectionTask.BUSBAR_DEFECT,
                input_size=tuple(preflight.input_size or (640, 640)),
                num_classes=len(preflight.class_names),
                class_names=list(preflight.class_names),
                confidence_threshold=self._confidence_threshold,
                nms_threshold=self._nms_threshold,
                device=device,
                use_vit_backbone=True,
                use_se_attention=True,
                use_faster_block=True,
                small_object_aug=True,  # 启用小目标增强
                multi_scale_inference=True,  # 启用多尺度推理
            )

            self._yolov8_vit_detector = YOLOv8ViTDetector(config)
            load_ok = self._yolov8_vit_detector.load()
            self._onnx_session_ready = bool(
                getattr(self._yolov8_vit_detector, "_session", None) is not None
            )
            self._real_model_loaded = bool(load_ok and self._onnx_session_ready)
            self._dl_initialized = self._real_model_loaded
            self._update_runtime_mode()

            if not self._real_model_loaded:
                logger.warning(
                    (
                        "YOLOv8-ViT未形成真实 ONNX 会话，禁止将 simulation 视为已加载成功: "
                        "preflight_passed=%s"
                    ),
                    preflight.passed,
                )
                self._yolov8_vit_detector = None
                self._mark_no_real_dl()
                self._dl_preflight.failure_reason = "onnx_session_failed"
                if "load() failed to preserve a real ONNX session" not in self._dl_preflight.failure_details:
                    self._dl_preflight.failure_details.append(
                        "load() failed to preserve a real ONNX session"
                    )
                self._log_runtime_status(logging.WARNING)
                return False

            logger.info("YOLOv8-ViT检测器初始化成功 (V3.0)")
            self._log_runtime_status()
            return True

        except Exception as e:
            logger.exception("YOLOv8-ViT初始化失败: %s", e)
            self._yolov8_vit_detector = None
            self._mark_no_real_dl()
            self._log_runtime_status(logging.WARNING)
            return False
    
    def detect_defects(
        self,
        image: np.ndarray,
        use_slicing: Optional[bool] = None,
        roi_bbox: Optional[Dict[str, float]] = None,
        roi_type: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> List[BusbarDetection]:
        """
        缺陷检测

        Args:
            image: BGR图像(支持4K)
            use_slicing: 是否使用切片(默认根据图像大小自动决定)
            roi_bbox: 可选的ROI区域
            timestamp: 时间戳 (用于时序ReID)

        Returns:
            检测结果列表
        """
        start_time = time.perf_counter()
        timestamp = timestamp or time.time()
        self._frame_counter += 1

        # 裁剪ROI
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)

        h, w = image.shape[:2]

        # 自动决定是否切片
        if use_slicing is None:
            use_slicing = self._use_slicing and (w > 2000 or h > 2000)
        elif use_slicing and not self._generate_tiles(w, h):
            use_slicing = False

        detections = []

        normalized_roi_type = self._normalize_roi_type(roi_type)

        if use_slicing:
            # 切片检测
            detections = self._detect_with_slicing(image, normalized_roi_type)
        else:
            # 整图检测
            detections = self._detect_single(image, normalized_roi_type)

        # 过滤环境干扰
        detections = self._filter_environmental_noise(detections)

        # NMS合并
        detections = self._apply_global_nms(detections)

        # 决策层仲裁：ROI先验、冲突消解、输出限流
        detections = self._apply_decision_postprocess(detections, normalized_roi_type)

        # V3.5: 时序ReID处理 - 跨帧缺陷跟踪
        if self._temporal_reid is not None and detections:
            detections = self._apply_temporal_reid(image, detections, timestamp)

        # V3.5: 裂纹增长分析
        if self._crack_analyzer is not None:
            detections = self._apply_crack_growth_analysis(detections, timestamp)

        processing_time = (time.perf_counter() - start_time) * 1000
        for det in detections:
            det.metadata["processing_time_ms"] = processing_time
            det.metadata["frame_id"] = self._frame_counter

        return detections

    def _apply_temporal_reid(
        self,
        image: np.ndarray,
        detections: List[BusbarDetection],
        timestamp: float
    ) -> List[BusbarDetection]:
        """应用时序ReID处理 (V3.5)"""
        if self._temporal_reid is None:
            return detections

        try:
            # 转换检测格式
            det_dicts = [{
                "bbox": det.bbox,
                "label": det.defect_type.value,
                "confidence": det.confidence
            } for det in detections]

            # 执行ReID匹配
            reid_results = self._temporal_reid.process_frame(
                frame=image,
                detections=det_dicts,
                frame_id=self._frame_counter,
                timestamp=timestamp
            )

            # 更新检测结果
            for i, det in enumerate(detections):
                if i < len(reid_results):
                    reid_info = reid_results[i]
                    det.metadata["track_id"] = reid_info.get("track_id")
                    det.metadata["match_type"] = reid_info.get("match_type")
                    det.metadata["track_age"] = reid_info.get("track_age", 1)

            return detections

        except Exception as e:
            logger.exception("时序ReID处理失败: %s", e)
            return detections

    def _apply_crack_growth_analysis(
        self,
        detections: List[BusbarDetection],
        timestamp: float
    ) -> List[BusbarDetection]:
        """应用裂纹增长分析 (V3.5)"""
        if self._crack_analyzer is None:
            return detections

        try:
            for det in detections:
                # 只分析裂纹类型的缺陷
                if det.defect_type != BusbarDefectType.CRACK:
                    continue

                # 获取track_id (需要先经过ReID处理)
                track_id = det.metadata.get("track_id")
                if not track_id:
                    continue

                # 添加观测数据
                self._crack_analyzer.add_observation(
                    track_id=track_id,
                    bbox=det.bbox,
                    timestamp=timestamp
                )

                # 分析增长趋势
                analysis = self._crack_analyzer.analyze_growth(track_id)
                if analysis:
                    det.metadata["growth_analysis"] = {
                        "growth_rate": analysis.growth_rate,
                        "growth_trend": analysis.growth_trend,
                        "predicted_length": analysis.predicted_length,
                        "days_to_critical": analysis.days_to_critical,
                        "risk_level": analysis.risk_level,
                        "confidence": analysis.confidence,
                        "current_length": analysis.metadata.get("current_length", 0),
                        "observation_count": analysis.metadata.get("observation_count", 0)
                    }

            return detections

        except Exception as e:
            logger.exception("裂纹增长分析失败: %s", e)
            return detections

    def get_temporal_reid_stats(self) -> Optional[Dict[str, Any]]:
        """获取时序ReID统计信息 (V3.5)"""
        if self._temporal_reid is None:
            return None
        return self._temporal_reid.get_statistics()

    def get_crack_growth_history(self, track_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取裂纹增长历史 (V3.5)"""
        if self._crack_analyzer is None:
            return None
        history = self._crack_analyzer.get_history(track_id)
        return [{
            "timestamp": p.timestamp,
            "length": p.length,
            "width": p.width,
            "area": p.area,
            "severity": p.severity
        } for p in history]

    def get_all_crack_analyses(self) -> Dict[str, Dict[str, Any]]:
        """获取所有裂纹的增长分析 (V3.5)"""
        if self._crack_analyzer is None:
            return {}

        results = {}
        if self._temporal_reid is not None:
            for track in self._temporal_reid.get_active_tracks():
                if track.defect_type.value == "crack":
                    analysis = self._crack_analyzer.analyze_growth(track.track_id)
                    if analysis:
                        results[track.track_id] = {
                            "growth_rate": analysis.growth_rate,
                            "growth_trend": analysis.growth_trend,
                            "risk_level": analysis.risk_level,
                            "days_to_critical": analysis.days_to_critical,
                            "first_seen": track.first_seen,
                            "last_seen": track.last_seen,
                            "observation_count": len(track.features)
                        }
        return results
    
    def _detect_with_slicing(
        self,
        image: np.ndarray,
        roi_type: Optional[str] = None,
    ) -> List[BusbarDetection]:
        """切片检测"""
        h, w = image.shape[:2]
        detections = []
        
        # 生成切片
        tiles = self._generate_tiles(w, h)
        
        for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
            # 裁剪切片
            tile_image = image[y1:y2, x1:x2]
            
            # 检测当前切片
            tile_detections = self._detect_single(tile_image, roi_type)
            
            # 映射回原图坐标
            for det in tile_detections:
                det.bbox = self._remap_bbox(det.bbox, x1, y1, x2-x1, y2-y1, w, h)
                det.tile_info = {
                    "tile_idx": tile_idx,
                    "tile_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                }
            
            detections.extend(tile_detections)
        
        return detections
    
    def _generate_tiles(self, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """生成切片坐标"""
        tiles = []
        stride = self._tile_size - self._tile_overlap
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                x1, y1 = x, y
                x2 = min(x + self._tile_size, width)
                y2 = min(y + self._tile_size, height)
                
                # 确保切片不太小
                if (x2 - x1) >= self._tile_size // 2 and (y2 - y1) >= self._tile_size // 2:
                    tiles.append((x1, y1, x2, y2))
        
        return tiles
    
    def _detect_single(
        self,
        image: np.ndarray,
        roi_type: Optional[str] = None,
    ) -> List[BusbarDetection]:
        """单图检测 (V3.0: 优先使用YOLOv8-ViT)"""
        # V3.0: 优先使用YOLOv8-ViT深度学习
        if self._use_deep_learning and self._dl_initialized and self._yolov8_vit_detector is not None:
            detections = self._detect_by_yolov8_vit(image)
            if detections:
                return detections

        # 兼容旧版：使用model_registry
        if self._use_deep_learning and self._model_registry:
            detections = self._detect_by_deep_learning(image)
            if detections:
                return detections

        # 回退到传统方法
        return self._filter_runtime_supported_detections(
            self._detect_by_traditional(image, roi_type)
        )

    def _detect_by_yolov8_vit(self, image: np.ndarray) -> List[BusbarDetection]:
        """YOLOv8-ViT深度学习检测 (V3.0新增)"""
        detections = []

        if not self._dl_initialized or self._yolov8_vit_detector is None:
            return detections

        try:
            # 使用多尺度检测(对小目标更有效)
            result = self._yolov8_vit_detector.detect_multi_scale(image)

            for det in result.detections:
                sanitized_conf = _sanitize_confidence(det.confidence)
                if sanitized_conf >= self._confidence_threshold:
                    # 将类名映射到缺陷类型 (QR-10)
                    defect_type, is_unknown = self._map_class_to_defect_type(det.class_name)
                    if defect_type is None:
                        logger.warning("抑制未知类名检测结果: %s", det.class_name)
                        continue

                    detections.append(BusbarDetection(
                        defect_type=defect_type,
                        bbox=det.bbox,
                        confidence=sanitized_conf,
                        class_name=det.class_name,
                        metadata={
                            "source": "yolov8_vit_v3",
                            "class_id": det.class_id,
                            "inference_time_ms": result.inference_time_ms,
                            "model_version": result.model_version,
                            "multi_scale": True,
                            "unknown_class_mapped": is_unknown,
                        }
                    ))

        except Exception as e:
            logger.exception("YOLOv8-ViT检测失败: %s", e)

        return self._filter_runtime_supported_detections(detections)

    def _map_class_to_defect_type(
        self,
        class_name: str,
    ) -> Tuple[Optional[BusbarDefectType], bool]:
        """将类名映射到缺陷类型 (V3.0, QR-10: 未知类名记录警告)

        Returns:
            (defect_type, is_unknown_mapped) 元组
        """
        normalized_class_name = canonicalize_label(class_name)
        class_mapping = {
            "pin_missing": BusbarDefectType.PIN_MISSING,
            "crack": BusbarDefectType.CRACK,
            "foreign_object": BusbarDefectType.FOREIGN_OBJECT,
            "corrosion": BusbarDefectType.CORROSION,
            "flashover": BusbarDefectType.FLASHOVER,
            "broken_strand": BusbarDefectType.BROKEN_STRAND,
            "insulator_damage": BusbarDefectType.INSULATOR_DAMAGE,
            "fitting_loose": BusbarDefectType.FITTING_LOOSE,
            "loose_fitting": BusbarDefectType.FITTING_LOOSE,
            "deformation": BusbarDefectType.CRACK,  # 映射到裂纹
        }
        mapped = class_mapping.get(normalized_class_name)
        if mapped is None:
            logger.warning(
                "未知缺陷类名 '%s' 将被抑制，不再默认回退到 CRACK", class_name,
            )
            return None, True
        if not is_runtime_supported_defect_label(mapped.value):
            logger.warning(
                "抑制未纳入当前 runtime supported baseline 的类名: raw=%s canonical=%s",
                class_name,
                normalized_class_name,
            )
            return None, False
        return mapped, False

    def _filter_runtime_supported_detections(
        self,
        detections: List[BusbarDetection],
    ) -> List[BusbarDetection]:
        """统一抑制当前基线外的标签，避免误宣称已支持。"""
        filtered: List[BusbarDetection] = []
        for det in detections:
            if not is_runtime_supported_defect_label(det.defect_type.value):
                logger.warning(
                    "抑制非 runtime supported detection label: %s",
                    det.defect_type.value,
                )
                continue
            filtered.append(det)
        return filtered

    def _detect_by_deep_learning(self, image: np.ndarray) -> List[BusbarDetection]:
        """深度学习检测 (兼容旧版model_registry)"""
        detections = []

        try:
            model_id = self.MODEL_IDS["detector"]
            result = self._model_registry.infer(model_id, image)  # type: ignore[union-attr]

            for det in result.detections:
                sanitized_conf = _sanitize_confidence(det["confidence"])
                if sanitized_conf >= self._confidence_threshold:
                    class_id = det.get("class_id")
                    class_name = str(det.get("class_name", "") or "")
                    defect_type = self.DEFECT_CLASSES.get(class_id) if class_id is not None else None
                    is_unknown = False
                    if defect_type is None and class_name:
                        defect_type, is_unknown = self._map_class_to_defect_type(class_name)
                    if defect_type is None:
                        logger.warning(
                            "抑制未知 class_id/class_name 检测结果: class_id=%s class_name=%s",
                            class_id,
                            class_name,
                        )
                        continue

                    detections.append(BusbarDetection(
                        defect_type=defect_type,
                        bbox=det["bbox"],
                        confidence=sanitized_conf,
                        class_name=class_name or defect_type.value,
                        metadata={
                            "source": "model_registry_legacy",
                            "model_id": model_id,
                            "class_id": class_id,
                            "unknown_class_mapped": is_unknown,
                        }
                    ))
        except Exception as e:
            logger.exception("model_registry检测失败: %s", e)

        return self._filter_runtime_supported_detections(detections)
    
    def _detect_by_traditional(
        self,
        image: np.ndarray,
        roi_type: Optional[str] = None,
    ) -> List[BusbarDetection]:
        """传统方法检测(回退方案)"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 1. 销钉缺失检测 - 圆形缺失
        pin_detections = self._detect_missing_pins(image)
        detections.extend(pin_detections)
        
        # 2. 裂纹检测 - 细长线条
        crack_detections = self._detect_cracks(image, roi_type)
        detections.extend(crack_detections)
        
        # 3. 异物检测 - 悬挂物
        foreign_detections = self._detect_foreign_objects(image)
        detections.extend(foreign_detections)
        
        return detections
    
    def _detect_missing_pins(self, image: np.ndarray) -> List[BusbarDetection]:
        """销钉缺失检测"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 霍夫圆检测
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=30
        )
        
        if circles is not None:
            circles_rounded = np.uint16(np.around(circles))
            circles_array = circles_rounded[0, :]  # type: ignore[index]

            # 分析圆形分布，检测缺失
            for i in circles_array:
                x, y, r = int(i[0]), int(i[1]), int(i[2])
                
                # 检查是否为空洞(销钉缺失位置)
                roi = gray[max(0,y-r):min(h,y+r), max(0,x-r):min(w,x+r)]
                if roi.size > 0:
                    mean_val = np.mean(roi)
                    if mean_val < 50:  # 暗区域表示缺失
                        confidence = _sanitize_confidence(float(0.6 + (50 - mean_val) / 100))

                        detections.append(BusbarDetection(
                            defect_type=BusbarDefectType.PIN_MISSING,
                            bbox={
                                "x": (x - r) / w,
                                "y": (y - r) / h,
                                "width": 2 * r / w,
                                "height": 2 * r / h
                            },
                            confidence=min(0.85, confidence),
                            class_name="销钉缺失",
                            metadata={"source": "traditional", "radius": int(r)}
                        ))
        
        return detections
    
    def _detect_cracks(
        self,
        image: np.ndarray,
        roi_type: Optional[str] = None,
    ) -> List[BusbarDetection]:
        """裂纹检测"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘增强
        edges = cv2.Canny(gray, 30, 100)
        
        # 形态学处理 - 连接断开的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 霍夫线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        crack_limits = self._get_crack_validation_limits(roi_type)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # 过滤短线
                if length > 50:
                    # 计算线条方向
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    minor_axis = max(min(dx, dy), 1)
                    slenderness = length / minor_axis
                    span_ratio = float(length / max(w, h, 1))
                    local_contrast = self._estimate_line_contrast(gray, x1, y1, x2, y2)
                    axis_aligned = self._is_axis_aligned(
                        float(angle),
                        float(crack_limits["axis_aligned_tolerance_deg"]),
                    )
                    vertical_axis_aligned = abs(float(angle) - 90.0) <= float(
                        crack_limits["axis_aligned_tolerance_deg"]
                    )

                    if slenderness < float(crack_limits["min_slenderness"]):
                        continue
                    if length < float(crack_limits["min_length_px"]):
                        continue
                    if local_contrast < float(crack_limits["min_local_contrast"]):
                        continue
                    if bool(crack_limits["reject_axis_aligned"]) and axis_aligned:
                        continue
                    if bool(crack_limits["reject_vertical_axis_aligned"]) and vertical_axis_aligned:
                        continue
                    if axis_aligned and span_ratio > float(
                        crack_limits["max_axis_aligned_span_ratio"]
                    ):
                        continue

                    if slenderness > 3:  # 裂纹通常是细长线条
                        confidence = _sanitize_confidence(min(0.7, 0.4 + length / 200))

                        detections.append(BusbarDetection(
                            defect_type=BusbarDefectType.CRACK,
                            bbox={
                                "x": min(x1, x2) / w,
                                "y": min(y1, y2) / h,
                                "width": abs(x2 - x1) / w + 0.01,
                                "height": abs(y2 - y1) / h + 0.01
                            },
                            confidence=confidence,
                            class_name="裂纹",
                            metadata={
                                "source": "traditional",
                                "length": length,
                                "angle": angle,
                                "slenderness": slenderness,
                                "span_ratio": span_ratio,
                                "local_contrast": local_contrast,
                                "axis_aligned": axis_aligned,
                                "roi_type": self._normalize_roi_type(roi_type),
                            }
                        ))
        
        return detections
    
    def _detect_foreign_objects(self, image: np.ndarray) -> List[BusbarDetection]:
        """异物检测"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 背景减除
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray, blur)
        
        # 阈值化
        _, binary = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 10000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = cw / (ch + 1e-6)
                
                # 悬挂物通常是垂直的
                if aspect_ratio < 0.5 or aspect_ratio > 2:
                    confidence = _sanitize_confidence(min(0.65, 0.35 + area / 5000))
                    
                    detections.append(BusbarDetection(
                        defect_type=BusbarDefectType.FOREIGN_OBJECT,
                        bbox={
                            "x": x / w,
                            "y": y / h,
                            "width": cw / w,
                            "height": ch / h
                        },
                        confidence=confidence,
                        class_name="异物",
                        metadata={"source": "traditional", "area": area}
                    ))
        
        return detections

    def _normalize_roi_type(self, roi_type: Optional[str]) -> str:
        """归一化 ROI 类型字符串。"""
        if roi_type is None:
            return "unknown"
        normalized = str(roi_type).strip().lower()
        return normalized or "unknown"

    def _lookup_roi_mapping(
        self,
        mapping: Dict[str, Any],
        roi_type: Optional[str],
    ) -> Dict[str, Any]:
        """按 ROI 类型查找最匹配的配置映射。"""
        normalized_roi_type = self._normalize_roi_type(roi_type)
        if normalized_roi_type in mapping and isinstance(mapping[normalized_roi_type], dict):
            return mapping[normalized_roi_type]

        for key, value in mapping.items():
            if not isinstance(value, dict):
                continue
            if key in normalized_roi_type or normalized_roi_type in key:
                return value
        return {}

    def _get_crack_validation_limits(self, roi_type: Optional[str]) -> Dict[str, float]:
        """获取 ROI 相关的裂纹二次校验阈值。"""
        base_limits = {
            "min_length_px": float(self._traditional_crack_validation["min_length_px"]),
            "min_slenderness": float(self._traditional_crack_validation["min_slenderness"]),
            "min_local_contrast": float(
                self._traditional_crack_validation["min_local_contrast"]
            ),
            "axis_aligned_tolerance_deg": float(
                self._traditional_crack_validation["axis_aligned_tolerance_deg"]
            ),
            "reject_axis_aligned": bool(
                self._traditional_crack_validation["reject_axis_aligned"]
            ),
            "reject_vertical_axis_aligned": bool(
                self._traditional_crack_validation["reject_vertical_axis_aligned"]
            ),
            "max_axis_aligned_span_ratio": float(
                self._traditional_crack_validation["default_max_axis_aligned_span_ratio"]
            ),
        }
        overrides = self._lookup_roi_mapping(
            self._traditional_crack_validation.get("roi_overrides", {}),
            roi_type,
        )
        for key, value in overrides.items():
            if key in base_limits:
                if isinstance(base_limits[key], bool):
                    base_limits[key] = bool(value)
                else:
                    base_limits[key] = float(value)
        return base_limits

    def _estimate_line_contrast(
        self,
        gray: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> float:
        """估计线条周围局部对比度。"""
        min_x = max(0, min(x1, x2) - 3)
        max_x = min(gray.shape[1], max(x1, x2) + 4)
        min_y = max(0, min(y1, y2) - 3)
        max_y = min(gray.shape[0], max(y1, y2) + 4)
        patch = gray[min_y:max_y, min_x:max_x]
        if patch.size == 0:
            return 0.0
        p10, p90 = np.percentile(patch, [10, 90])
        return float(max(0.0, p90 - p10))

    def _is_axis_aligned(self, angle: float, tolerance_deg: float) -> bool:
        """判断线段是否接近水平或垂直结构线。"""
        normalized = angle % 180.0
        return (
            normalized <= tolerance_deg
            or abs(normalized - 90.0) <= tolerance_deg
            or abs(normalized - 180.0) <= tolerance_deg
        )
    
    def _filter_environmental_noise(self, detections: List[BusbarDetection]) -> List[BusbarDetection]:
        """过滤环境干扰"""
        filtered = []
        
        for det in detections:
            # 检查是否为环境干扰
            if self._is_environmental_noise(det):
                det.reason_code = self._get_noise_reason_code(det)
                det.metadata["filtered"] = True
                continue
            
            filtered.append(det)
        
        return filtered
    
    def _is_environmental_noise(self, detection: BusbarDetection) -> bool:
        """判断是否为环境干扰"""
        # 根据检测框特征判断
        bbox = detection.bbox
        area = bbox["width"] * bbox["height"]
        aspect = bbox["width"] / (bbox["height"] + 1e-6)

        # 裂纹候选天然细长，不能直接按飞虫/轨迹规则过滤。
        if detection.defect_type == BusbarDefectType.CRACK:
            return area < 0.0002 and detection.confidence < 0.45

        # 非常小的检测可能是飞虫
        if area < 0.001 and detection.confidence < 0.5:
            return True
        
        # 非常细长的可能是飞行轨迹
        if aspect > 10 or aspect < 0.1:
            return True
        
        return False
    
    def _get_noise_reason_code(self, detection: BusbarDetection) -> str:
        """获取干扰原因码"""
        bbox = detection.bbox
        area = bbox["width"] * bbox["height"]
        
        if area < 0.001:
            return "3002"  # 飞虫
        
        return "3001"  # 鸟类
    
    def _remap_bbox(
        self,
        bbox: Dict[str, float],
        tile_x: int, tile_y: int,
        tile_w: int, tile_h: int,
        img_w: int, img_h: int
    ) -> Dict[str, float]:
        """将切片坐标映射回原图"""
        return {
            "x": (tile_x + bbox["x"] * tile_w) / img_w,
            "y": (tile_y + bbox["y"] * tile_h) / img_h,
            "width": bbox["width"] * tile_w / img_w,
            "height": bbox["height"] * tile_h / img_h,
        }
    
    def _apply_global_nms(self, detections: List[BusbarDetection]) -> List[BusbarDetection]:
        """全局NMS合并"""
        if not detections:
            return []
        
        # 按类别分组
        by_class: Dict[BusbarDefectType, List[BusbarDetection]] = {}
        for det in detections:
            if det.defect_type not in by_class:
                by_class[det.defect_type] = []
            by_class[det.defect_type].append(det)
        
        # 对每个类别执行NMS
        result = []
        for defect_type, class_detections in by_class.items():
            nms_result = self._nms(class_detections)
            result.extend(nms_result)
        
        return result
    
    def _nms(self, detections: List[BusbarDetection]) -> List[BusbarDetection]:
        """非极大值抑制"""
        if not detections:
            return []
        
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                d for d in detections
                if self._iou(best.bbox, d.bbox) < self._nms_threshold
            ]
        
        return keep
    
    def _iou(self, box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """计算IoU"""
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0

    def _apply_decision_postprocess(
        self,
        detections: List[BusbarDetection],
        roi_type: Optional[str],
    ) -> List[BusbarDetection]:
        """决策层后处理：ROI先验、冲突仲裁、review 标注与输出限流。"""
        if not detections:
            return []

        for det in detections:
            label = det.defect_type.value
            det.metadata.setdefault("candidate_labels", [label])
            det.metadata.setdefault("review_status", "confirmed")
            det.metadata.setdefault("review_reason", "")
            det.metadata.setdefault("suggested_action", "NONE")
            adjusted_score = self._get_adjusted_score(det, roi_type)
            det.metadata["adjusted_score"] = adjusted_score
            det.metadata.setdefault("candidate_scores", {label: adjusted_score})
            if det.reason_code:
                det.metadata["reason_code"] = str(det.reason_code)

        detections = self._arbitrate_crack_foreign_conflicts(detections, roi_type)
        detections = self._enforce_output_limits(detections)

        for det in detections:
            det.metadata["pred_label"] = det.defect_type.value
            det.metadata["candidate_labels"] = list(dict.fromkeys(det.metadata["candidate_labels"]))
            if det.reason_code:
                det.metadata["reason_code"] = str(det.reason_code)

        return detections

    def _get_adjusted_score(
        self,
        detection: BusbarDetection,
        roi_type: Optional[str],
    ) -> float:
        """计算用于仲裁/限流的调整后分数。"""
        label = detection.defect_type.value
        roi_biases = self._lookup_roi_mapping(self._roi_priors, roi_type)
        score = float(detection.confidence) + float(roi_biases.get(label, 0.0))
        return round(float(np.clip(score, 0.0, 1.0)), 6)

    def _arbitrate_crack_foreign_conflicts(
        self,
        detections: List[BusbarDetection],
        roi_type: Optional[str],
    ) -> List[BusbarDetection]:
        """对 crack / foreign_object 做跨类冲突仲裁。"""
        kept = [True] * len(detections)

        for i, left in enumerate(detections):
            if left.defect_type not in (
                BusbarDefectType.CRACK,
                BusbarDefectType.FOREIGN_OBJECT,
            ):
                continue
            for j in range(i + 1, len(detections)):
                right = detections[j]
                pair_types = {left.defect_type, right.defect_type}
                if pair_types != {
                    BusbarDefectType.CRACK,
                    BusbarDefectType.FOREIGN_OBJECT,
                }:
                    continue
                if self._iou(left.bbox, right.bbox) < self._conflict_iou_threshold:
                    continue

                left_score = float(left.metadata.get("adjusted_score", left.confidence))
                right_score = float(right.metadata.get("adjusted_score", right.confidence))
                if left_score >= right_score:
                    winner_idx, loser_idx = i, j
                else:
                    winner_idx, loser_idx = j, i

                winner = detections[winner_idx]
                loser = detections[loser_idx]
                primary_score = float(winner.metadata.get("adjusted_score", winner.confidence))
                secondary_score = float(loser.metadata.get("adjusted_score", loser.confidence))
                score_gap = round(abs(primary_score - secondary_score), 6)
                candidate_labels = [
                    winner.defect_type.value,
                    loser.defect_type.value,
                ]
                winner.metadata["candidate_labels"] = candidate_labels
                winner.metadata["candidate_scores"] = {
                    winner.defect_type.value: primary_score,
                    loser.defect_type.value: secondary_score,
                }
                winner.metadata["secondary_label"] = loser.defect_type.value
                winner.metadata["secondary_score"] = secondary_score
                winner.metadata["primary_score"] = primary_score
                winner.metadata["score_gap"] = score_gap
                winner.metadata["conflict_type"] = "crack_vs_foreign_object"

                if score_gap < self._review_score_gap:
                    winner.metadata["review_status"] = "review_required"
                    winner.metadata["review_reason"] = (
                        "crack 与 foreign_object 分数接近，需人工复核"
                    )
                    winner.metadata["suggested_action"] = "MANUAL_REVIEW"
                    winner.reason_code = self._uncertain_reason_code
                    winner.metadata["reason_code"] = self._uncertain_reason_code
                kept[loser_idx] = False

        return [det for keep, det in zip(kept, detections) if keep]

    def _enforce_output_limits(
        self,
        detections: List[BusbarDetection],
    ) -> List[BusbarDetection]:
        """限制单类与场景级输出数量，避免 crack 洪泛。"""
        if not detections:
            return []

        ordered = sorted(
            detections,
            key=lambda det: float(det.metadata.get("adjusted_score", det.confidence)),
            reverse=True,
        )
        per_class_counts: Dict[str, int] = {}
        kept: List[BusbarDetection] = []

        for det in ordered:
            label = det.defect_type.value
            class_limit = self._per_class_topk.get(label, self._scene_max_detections)
            current_count = per_class_counts.get(label, 0)
            if current_count >= class_limit:
                continue
            if len(kept) >= self._scene_max_detections:
                break
            per_class_counts[label] = current_count + 1
            kept.append(det)

        return kept

    def _get_quality_failure_reason(self, reason_code: str) -> str:
        """根据内部质量原因码获取说明。"""
        return self.REASON_CODES.get(str(reason_code or ""), "")

    def _get_quality_suggested_action(self, reason_code: str) -> str:
        """根据内部质量原因码给出建议动作。"""
        return self.QUALITY_ACTION_MAP.get(str(reason_code or ""), "RECAPTURE")

    def _select_quality_issue(
        self,
        issues: List[Tuple[QualityGateDecision, QualityGateStatus, str, Dict[str, Any]]],
    ) -> Tuple[QualityGateDecision, QualityGateStatus, str, Dict[str, Any]]:
        """从多个质量问题中选择最需要暴露的一项。"""
        if not issues:
            return (
                QualityGateDecision.PASS,
                QualityGateStatus.PASS,
                "",
                {},
            )

        decision_rank = {
            QualityGateDecision.PASS: 0,
            QualityGateDecision.SOFT_FAIL: 1,
            QualityGateDecision.HARD_FAIL: 2,
        }
        status_priority = {
            # 大面积遮挡与低对比更适合优先暴露，便于后续复拍/改视角决策。
            QualityGateStatus.FAIL_OCCLUDED: 5,
            QualityGateStatus.FAIL_OVEREXPOSED: 4,
            QualityGateStatus.FAIL_UNDEREXPOSED: 3,
            QualityGateStatus.FAIL_LOW_CONTRAST: 2,
            QualityGateStatus.FAIL_BLUR: 1,
            QualityGateStatus.PASS: 0,
        }
        return max(
            issues,
            key=lambda item: (
                decision_rank[item[0]],
                status_priority[item[1]],
            ),
        )

    def _apply_soft_fail_review(
        self,
        detections: List[BusbarDetection],
        quality: QualityGateResult,
    ) -> List[BusbarDetection]:
        """SOFT_FAIL 继续检测，但统一要求复核。"""
        quality_gate_status = getattr(quality.quality_gate_status, "value", quality.quality_gate_status)
        reason_code = str(quality.reason_code or "")
        review_reason = self._get_quality_failure_reason(reason_code)
        suggested_action = self._get_quality_suggested_action(reason_code)

        for det in detections:
            det.metadata["quality_gate_status"] = quality_gate_status
            det.metadata["quality_failure_reason"] = quality.failure_reason or review_reason
            det.metadata["quality_reason_code"] = reason_code or None
            det.metadata["quality_suggested_action"] = suggested_action
            det.metadata["review_status"] = "review_required"
            existing_reason = str(det.metadata.get("review_reason") or "").strip()
            if existing_reason:
                if review_reason and review_reason not in existing_reason:
                    det.metadata["review_reason"] = f"{existing_reason}; 质量门禁软失败: {review_reason}"
            else:
                det.metadata["review_reason"] = (
                    f"质量门禁软失败: {review_reason}" if review_reason else "质量门禁软失败，需人工复核"
                )
            if reason_code and not det.reason_code:
                det.reason_code = reason_code
            if det.metadata.get("suggested_action") in (None, "", "NONE"):
                det.metadata["suggested_action"] = suggested_action
        return detections
    
    def check_quality_gate(self, image: np.ndarray) -> QualityGateResult:
        """
        质量门禁检查
        
        Args:
            image: BGR图像
            
        Returns:
            质量门禁结果
        """
        if cv2 is None:
            return QualityGateResult(
                status=QualityGateStatus.PASS,
                quality_gate_status=QualityGateDecision.PASS,
                clarity_score=1.0,
                brightness_score=0.5,
                contrast_score=0.5,
                occlusion_ratio=0.0,
            )
        
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 清晰度评分 - 拉普拉斯方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())
        global_clarity_score = min(1.0, laplacian_var / 1500.0)
        local_clarity_score = self._compute_local_clarity_score(gray)
        clarity_score = max(global_clarity_score, local_clarity_score)

        # 2. 亮度评分
        mean_brightness = float(np.mean(gray) / 255.0)
        brightness_score = float(1.0 - 2 * abs(mean_brightness - 0.5))

        # 3. 对比度/曝光统计
        p5, p95 = np.percentile(gray, [5, 95])
        dynamic_range = float(p95 - p5)
        contrast_score = float(dynamic_range / 255.0)
        high_brightness_ratio = float((gray > self._brightness_high).mean())

        # 4. 遮挡检测
        edge_energy = self._compute_edge_energy(gray)
        local_edge_energy = self._compute_local_edge_energy(gray)
        effective_edge_energy = max(edge_energy, local_edge_energy)
        occlusion_ratio = self._occlusion_ratio_from_edge_energy(effective_edge_energy)
        
        # 三态决策：status 继续表示“失败类型”，quality_gate_status 表示“决策级别”
        issues: List[Tuple[QualityGateDecision, QualityGateStatus, str, Dict[str, Any]]] = []

        blur_hard_var = max(40.0, self._clarity_threshold * 350.0)
        blur_soft_var = max(60.0, blur_hard_var * 1.4)
        if (
            clarity_score < self._clarity_threshold * 0.75
            or (
                laplacian_var < blur_hard_var
                and local_clarity_score < self._clarity_threshold + 0.01
                and global_clarity_score < self._clarity_threshold * 0.25
            )
        ):
            issues.append((
                QualityGateDecision.HARD_FAIL,
                QualityGateStatus.FAIL_BLUR,
                "1001",
                {
                    "metric": "clarity_score",
                    "value": clarity_score,
                    "threshold": self._clarity_threshold * 0.75,
                    "laplacian_var": laplacian_var,
                },
            ))
        elif (
            clarity_score < self._clarity_threshold
            or (
                laplacian_var < blur_soft_var
                and local_clarity_score < self._clarity_threshold + 0.04
                and global_clarity_score < self._clarity_threshold * 0.50
            )
        ):
            issues.append((
                QualityGateDecision.SOFT_FAIL,
                QualityGateStatus.FAIL_BLUR,
                "1001",
                {
                    "metric": "clarity_score",
                    "value": clarity_score,
                    "threshold": self._clarity_threshold,
                    "laplacian_var": laplacian_var,
                },
            ))

        hard_brightness_high = min(0.98, self._brightness_range[1] + 0.12)
        hard_overexp_ratio = min(0.95, max(self._overexposed_ratio * 1.6, self._overexposed_ratio + 0.18))
        if (
            high_brightness_ratio > hard_overexp_ratio
            or mean_brightness > hard_brightness_high
        ):
            issues.append((
                QualityGateDecision.HARD_FAIL,
                QualityGateStatus.FAIL_OVEREXPOSED,
                "1002",
                {
                    "metric": "high_brightness_ratio",
                    "value": high_brightness_ratio,
                    "threshold": hard_overexp_ratio,
                    "mean_brightness": mean_brightness,
                },
            ))
        elif (
            high_brightness_ratio > self._overexposed_ratio
            or mean_brightness > self._brightness_range[1]
        ):
            issues.append((
                QualityGateDecision.SOFT_FAIL,
                QualityGateStatus.FAIL_OVEREXPOSED,
                "1002",
                {
                    "metric": "high_brightness_ratio",
                    "value": high_brightness_ratio,
                    "threshold": self._overexposed_ratio,
                    "mean_brightness": mean_brightness,
                },
            ))

        hard_underexposed = max(0.01, self._brightness_range[0] * 0.55)
        if mean_brightness < hard_underexposed:
            issues.append((
                QualityGateDecision.HARD_FAIL,
                QualityGateStatus.FAIL_UNDEREXPOSED,
                "1003",
                {
                    "metric": "mean_brightness",
                    "value": mean_brightness,
                    "threshold": hard_underexposed,
                },
            ))
        elif mean_brightness < self._brightness_range[0]:
            issues.append((
                QualityGateDecision.SOFT_FAIL,
                QualityGateStatus.FAIL_UNDEREXPOSED,
                "1003",
                {
                    "metric": "mean_brightness",
                    "value": mean_brightness,
                    "threshold": self._brightness_range[0],
                },
            ))

        soft_dynamic_range = max(self._dynamic_range_min, 55.0)
        hard_dynamic_range = max(8.0, self._dynamic_range_min * 0.60)
        if dynamic_range < hard_dynamic_range:
            issues.append((
                QualityGateDecision.HARD_FAIL,
                QualityGateStatus.FAIL_LOW_CONTRAST,
                "1005",
                {
                    "metric": "dynamic_range",
                    "value": dynamic_range,
                    "threshold": hard_dynamic_range,
                },
            ))
        elif self._is_low_contrast(dynamic_range, contrast_score, soft_dynamic_range):
            issues.append((
                QualityGateDecision.SOFT_FAIL,
                QualityGateStatus.FAIL_LOW_CONTRAST,
                "1005",
                {
                    "metric": "dynamic_range",
                    "value": dynamic_range,
                    "threshold": soft_dynamic_range,
                },
            ))

        hard_edge_threshold = max(1.0, self._edge_threshold * 0.55)
        if (
            effective_edge_energy < hard_edge_threshold
            or occlusion_ratio > 0.85
            or (
                edge_energy < self._edge_threshold * 0.75
                and local_edge_energy < self._edge_threshold
                and dynamic_range >= self._dynamic_range_min * 2.0
            )
        ):
            issues.append((
                QualityGateDecision.HARD_FAIL,
                QualityGateStatus.FAIL_OCCLUDED,
                "1004",
                {
                    "metric": "effective_edge_energy",
                    "value": effective_edge_energy,
                    "threshold": hard_edge_threshold,
                    "occlusion_ratio": occlusion_ratio,
                    "global_edge_energy": edge_energy,
                    "structural_edge_energy": local_edge_energy,
                    "dynamic_range": dynamic_range,
                },
            ))
        elif (
            effective_edge_energy < self._edge_threshold
            and dynamic_range >= self._dynamic_range_min * 1.6
        ):
            issues.append((
                QualityGateDecision.SOFT_FAIL,
                QualityGateStatus.FAIL_OCCLUDED,
                "1004",
                {
                    "metric": "effective_edge_energy",
                    "value": effective_edge_energy,
                    "threshold": self._edge_threshold,
                    "occlusion_ratio": occlusion_ratio,
                },
            ))

        quality_gate_status, status, reason_code, issue_detail = self._select_quality_issue(issues)
        failure_reason = self._get_quality_failure_reason(reason_code)

        return QualityGateResult(
            status=status,
            quality_gate_status=quality_gate_status,
            clarity_score=clarity_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            occlusion_ratio=occlusion_ratio,
            reason_code=reason_code,
            failure_reason=failure_reason,
            metadata={
                "quality_gate_status": quality_gate_status.value,
                "failure_reason": failure_reason,
                "suggested_action": self._get_quality_suggested_action(reason_code)
                if reason_code
                else "NONE",
                "selected_issue": status.value,
                "selected_issue_detail": issue_detail,
                "mean_brightness": mean_brightness,
                "laplacian_var": laplacian_var,
                "global_clarity_score": global_clarity_score,
                "local_clarity_score": local_clarity_score,
                "high_brightness_ratio": high_brightness_ratio,
                "dynamic_range": dynamic_range,
                "edge_energy": edge_energy,
                "local_edge_energy": local_edge_energy,
                "effective_edge_energy": effective_edge_energy,
            }
        )

    def _is_low_contrast(
        self,
        dynamic_range: float,
        contrast_score: float,
        dynamic_range_threshold: Optional[float] = None,
    ) -> bool:
        """低对比判断兼容旧测试配置与 YAML 动态范围配置。"""
        if self._contrast_threshold_explicit:
            return contrast_score < self._contrast_threshold
        threshold = self._dynamic_range_min if dynamic_range_threshold is None else dynamic_range_threshold
        return dynamic_range < threshold

    def _compute_edge_energy(self, gray: np.ndarray) -> float:
        """使用 Sobel 纹理能量估计可见结构强度。"""
        if cv2 is None:
            return 0.0
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        return float(np.mean(magnitude))

    def _compute_local_clarity_score(self, gray: np.ndarray) -> float:
        """小目标场景使用局部最清晰区域，避免全帧背景拖低清晰度。"""
        if cv2 is None:
            return 0.0

        h, w = gray.shape[:2]
        patch_h = max(h // 4, 64)
        patch_w = max(w // 4, 64)
        patch_vars: List[float] = []

        for y in range(0, h, patch_h):
            for x in range(0, w, patch_w):
                patch = gray[y:min(h, y + patch_h), x:min(w, x + patch_w)]
                if patch.size == 0:
                    continue
                patch_vars.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))

        if not patch_vars:
            return 0.0

        return min(1.0, float(np.percentile(patch_vars, 95)) / 1500.0)

    def _compute_local_edge_energy(self, gray: np.ndarray) -> float:
        """局部结构能量，避免整帧背景稀释目标纹理。"""
        if cv2 is None:
            return 0.0

        h, w = gray.shape[:2]
        patch_h = max(h // 4, 64)
        patch_w = max(w // 4, 64)
        patch_energies: List[float] = []

        for y in range(0, h, patch_h):
            for x in range(0, w, patch_w):
                patch = gray[y:min(h, y + patch_h), x:min(w, x + patch_w)]
                if patch.size == 0:
                    continue
                patch_energies.append(self._compute_edge_energy(patch))

        if not patch_energies:
            return 0.0

        # 使用较低分位数，避免少量清晰补丁掩盖大面积遮挡。
        return float(np.percentile(patch_energies, 80))

    def _occlusion_ratio_from_edge_energy(self, edge_energy: float) -> float:
        """将边缘能量映射为遮挡比例，能量越低越接近遮挡。"""
        if self._edge_threshold <= 0:
            return 0.0
        scale = max(self._edge_threshold * 3.0, 1.0)
        return float(np.clip(1.0 - edge_energy / scale, 0.0, 1.0))

    def _detect_occlusion(self, gray: np.ndarray) -> float:
        """检测遮挡比例"""
        if cv2 is None or self._edge_threshold <= 0:
            return 0.0

        effective_edge_energy = max(
            self._compute_edge_energy(gray),
            self._compute_local_edge_energy(gray),
        )
        return self._occlusion_ratio_from_edge_energy(effective_edge_energy)
    
    def compute_zoom_suggestion(
        self,
        image: np.ndarray,
        detections: List[BusbarDetection],
        current_zoom: float = 1.0,
    ) -> List[ZoomSuggestion]:
        """
        计算变焦建议
        
        Args:
            image: 当前图像
            detections: 检测结果
            current_zoom: 当前变焦倍数
            
        Returns:
            变焦建议列表
        """
        suggestions = []
        h, w = image.shape[:2]
        
        for det in detections:
            bbox = det.bbox
            det_w = bbox["width"] * w
            det_h = bbox["height"] * h
            det_size = max(det_w, det_h)
            
            # 如果目标太小，建议放大
            if det_size < self.MIN_TARGET_SIZE * 2:
                target_size = self.MIN_TARGET_SIZE * 4
                recommended_zoom = current_zoom * (target_size / det_size)
                
                suggestions.append(ZoomSuggestion(
                    current_zoom=current_zoom,
                    recommended_zoom=min(30.0, recommended_zoom),  # 最大30倍
                    reason=f"目标过小({det_size:.0f}px)，建议放大",
                    target_area=bbox,
                    priority=10 - int(det.confidence * 10)
                ))
        
        # 按优先级排序
        suggestions.sort(key=lambda s: s.priority)
        
        return suggestions
    
    def inspect(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None,
        current_zoom: float = 1.0,
    ) -> BusbarInspectionResult:
        """
        综合巡视
        
        Args:
            image: BGR图像(支持4K)
            roi_bbox: ROI区域
            current_zoom: 当前变焦倍数
            
        Returns:
            综合巡视结果
        """
        start_time = time.perf_counter()
        
        # 质量门禁
        quality_gate = self.check_quality_gate(image)
        quality_gate_status = getattr(
            quality_gate.quality_gate_status,
            "value",
            quality_gate.quality_gate_status,
        )

        # 仅 HARD_FAIL 直接阻断
        if quality_gate_status == QualityGateDecision.HARD_FAIL.value:
            return BusbarInspectionResult(
                detections=[],
                quality_gate=quality_gate,
                zoom_suggestions=[],
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                model_version=self._model_version,
                code_hash=self._code_hash,
            )
        
        # 缺陷检测
        detections = self.detect_defects(image, roi_bbox=roi_bbox)
        if quality_gate_status == QualityGateDecision.SOFT_FAIL.value:
            detections = self._apply_soft_fail_review(detections, quality_gate)
        
        # 变焦建议
        zoom_suggestions = self.compute_zoom_suggestion(image, detections, current_zoom)
        
        # 计算切片信息
        h, w = image.shape[:2]
        use_slicing = self._use_slicing and (w > 2000 or h > 2000)
        total_tiles = len(self._generate_tiles(w, h)) if use_slicing else 1
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return BusbarInspectionResult(
            detections=detections,
            quality_gate=quality_gate,
            zoom_suggestions=zoom_suggestions,
            total_tiles=total_tiles,
            processed_tiles=total_tiles,
            processing_time_ms=processing_time,
            model_version=self._model_version,
            code_hash=self._code_hash,
        )
    
    def _crop_roi(self, image: np.ndarray, bbox: Dict[str, float]) -> np.ndarray:
        """裁剪ROI区域"""
        h, w = image.shape[:2]
        x = int(bbox.get("x", 0) * w)
        y = int(bbox.get("y", 0) * h)
        bw = int(bbox.get("width", 1) * w)
        bh = int(bbox.get("height", 1) * h)
        
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        
        return image[y:y+bh, x:x+bw]


    def detect_roi(
        self,
        image: np.ndarray,
        use_tiling: bool = True,
        roi_type: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> DetectROIResult:
        """
        ROI区域检测 (兼容plugin.py调用)

        Args:
            image: 输入图像 (BGR格式)
            use_tiling: 是否使用切片检测
            timestamp: 时间戳

        Returns:
            检测结果
        """
        timestamp = timestamp or time.time()

        # 质量门禁检查
        quality = self.check_quality_gate(image)
        quality_gate_status = getattr(
            quality.quality_gate_status,
            "value",
            quality.quality_gate_status,
        )
        suggested_action = str(
            getattr(getattr(quality, "metadata", {}), "get", lambda *_: "NONE")(
                "suggested_action",
                "NONE",
            )
        )
        failure_reason = str(
            getattr(getattr(quality, "metadata", {}), "get", lambda *_: "")(
                "failure_reason",
                quality.failure_reason,
            )
        )

        if quality_gate_status == QualityGateDecision.HARD_FAIL.value:
            reason_code = int(quality.reason_code) if quality.reason_code else None
            return DetectROIResult(
                detections=[],
                quality=quality,
                zoom_suggestion=None,
                reason_code=reason_code,
                debug_info={
                    "quality_gate": "failed",
                    "quality_gate_status": quality_gate_status,
                    "quality_failure_reason": failure_reason,
                    "quality_reason_code": quality.reason_code or None,
                    "quality_suggested_action": suggested_action,
                    **self.get_runtime_status(quality_blocked=True),
                }
            )

        # 执行检测
        detections = self.detect_defects(
            image,
            use_slicing=use_tiling and self._use_slicing,
            roi_type=roi_type,
            timestamp=timestamp
        )
        if quality_gate_status == QualityGateDecision.SOFT_FAIL.value:
            detections = self._apply_soft_fail_review(detections, quality)

        # 变焦建议
        zoom_suggestions = self.compute_zoom_suggestion(image, detections)
        if zoom_suggestions:
            zoom_suggestion = zoom_suggestions[0]
        elif quality_gate_status == QualityGateDecision.PASS.value:
            zoom_suggestion = ZoomSuggestion(
                current_zoom=1.0,
                recommended_zoom=1.0,
                reason="none",
            )
        else:
            zoom_suggestion = None

        return DetectROIResult(
            detections=detections,
            quality=quality,
            zoom_suggestion=zoom_suggestion,
            reason_code=int(quality.reason_code) if quality.reason_code else None,
            debug_info={
                "quality_gate": "passed" if quality_gate_status == QualityGateDecision.PASS.value else "soft_failed",
                "quality_gate_status": quality_gate_status,
                "quality_failure_reason": failure_reason,
                "quality_reason_code": quality.reason_code or None,
                "quality_suggested_action": suggested_action,
                "frame_id": self._frame_counter,
                "detection_count": len(detections),
                "roi_type": self._normalize_roi_type(roi_type),
                "review_required_count": sum(
                    1
                    for det in detections
                    if det.metadata.get("review_status") == "review_required"
                ),
                "temporal_reid_enabled": self._temporal_reid is not None,
                "crack_analysis_enabled": self._crack_analyzer is not None,
                **self.get_runtime_status(),
            }
        )


# 便捷函数
def create_detector(config: Dict[str, Any], model_registry=None) -> BusbarDetectorEnhanced:
    """创建检测器实例"""
    detector = BusbarDetectorEnhanced(config, model_registry)
    detector.initialize()
    return detector
