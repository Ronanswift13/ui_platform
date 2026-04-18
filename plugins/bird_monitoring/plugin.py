"""
鸟类监控插件 - 工程基线实现
输变电激光星芒破夜绘明监测平台 (F组)

功能范围:
- 鸟类实时检测: 基于YOLOv8的多目标鸟类检测（当前模拟模式）
- 种类猜测: 与鸟类特征数据库对比识别鸟类种类（experimental）
- 风险评估: 综合距离、翼展、行为模式评估触电风险
- 驱离建议: 根据风险等级输出驱离建议（硬件联动为 blocked）

当前阶段: engineering_baseline
runtime_mode: simulation（无真实 ONNX 模型）
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import importlib.util
import sys
import numpy as np
import yaml

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
from platform_core.visual_output_protocol import build_visual_meta

logger = logging.getLogger(__name__)


# =============================================================================
# 鸟类数据库定义
# =============================================================================
@dataclass
class BirdSpecies:
    """鸟类种类信息"""
    id: str
    name: str
    scientific_name: str
    risk_level: str  # low, medium, high
    wingspan_cm: int
    body_length_cm: int
    weight_g: int
    behavior_pattern: str  # group_perching, nesting, perching_hunting, flying
    description: str
    safety_notes: str


# 内置鸟类数据库
BIRD_DATABASE: Dict[str, BirdSpecies] = {
    "sparrow": BirdSpecies(
        id="B001", name="麻雀", scientific_name="Passer montanus",
        risk_level="low", wingspan_cm=22, body_length_cm=14, weight_g=25,
        behavior_pattern="group_perching",
        description="常见小型鸟类，偶尔停留在输电线上",
        safety_notes="体型小，单只触电风险低，但群体停留可能导致线路污染"
    ),
    "magpie": BirdSpecies(
        id="B002", name="喜鹊", scientific_name="Pica pica",
        risk_level="medium", wingspan_cm=56, body_length_cm=45, weight_g=250,
        behavior_pattern="nesting",
        description="中型鸟类，喜在电塔筑巢",
        safety_notes="喜在电塔筑巢，巢材可能导致短路"
    ),
    "crow": BirdSpecies(
        id="B003", name="乌鸦", scientific_name="Corvus",
        risk_level="medium", wingspan_cm=100, body_length_cm=50, weight_g=500,
        behavior_pattern="group_perching",
        description="大型鸟类，翼展较大有触电风险",
        safety_notes="翼展较大，跨相短路风险中等"
    ),
    "eagle": BirdSpecies(
        id="B004", name="老鹰", scientific_name="Aquila",
        risk_level="high", wingspan_cm=200, body_length_cm=80, weight_g=4500,
        behavior_pattern="perching_hunting",
        description="大型猛禽，翼展极大，高风险",
        safety_notes="翼展极大，跨相短路风险高"
    ),
    "egret": BirdSpecies(
        id="B005", name="白鹭", scientific_name="Egretta garzetta",
        risk_level="high", wingspan_cm=95, body_length_cm=60, weight_g=500,
        behavior_pattern="flying",
        description="涉禽类，翼展大且腿长",
        safety_notes="翼展大且腿长，触电风险高"
    ),
    "swallow": BirdSpecies(
        id="B006", name="燕子", scientific_name="Hirundo rustica",
        risk_level="low", wingspan_cm=32, body_length_cm=17, weight_g=20,
        behavior_pattern="flying",
        description="飞行敏捷，风险较低",
        safety_notes="飞行敏捷，很少停留在电线上"
    ),
    "dove": BirdSpecies(
        id="B007", name="斑鸠", scientific_name="Streptopelia",
        risk_level="low", wingspan_cm=50, body_length_cm=30, weight_g=200,
        behavior_pattern="group_perching",
        description="中小型鸟类，常见于城郊",
        safety_notes="体型适中，单只风险较低"
    ),
    "heron": BirdSpecies(
        id="B008", name="苍鹭", scientific_name="Ardea cinerea",
        risk_level="high", wingspan_cm=185, body_length_cm=100, weight_g=1500,
        behavior_pattern="flying",
        description="大型涉禽，翼展极大",
        safety_notes="翼展极大，飞行时易触碰多相导线"
    ),
}

# 种类标签 → 内部 species_id 映射；未匹配的一律走 unknown_bird
SPECIES_LABEL_MAP: Dict[str, str] = {
    "sparrow": "sparrow",
    "magpie": "magpie",
    "crow": "crow",
    "eagle": "eagle",
    "egret": "egret",
    "swallow": "swallow",
    "dove": "dove",
    "heron": "heron",
    "nest": "nest",
    "bird_generic": "unknown_bird",
    "bird": "unknown_bird",
}

# 告警级别名称
ALARM_LABEL_NAMES = {
    "bird_detected": "检测到鸟类",
    "bird_approaching": "鸟类正在接近",
    "bird_perched": "鸟类停留",
    "high_risk_species": "高风险鸟类",
    "nest_detected": "检测到鸟巢",
    "unknown_bird": "未识别鸟类",
    "review_required": "需人工复核",
}


@dataclass
class BirdDetection:
    """鸟类检测结果"""
    track_id: int
    species_id: str
    species_name: str
    confidence: float
    bbox: Dict[str, float]
    position_3d: Optional[Dict[str, float]] = None
    distance_to_line_m: float = 0.0
    speed_ms: float = 0.0
    heading_deg: float = 0.0
    status: str = "flying"  # flying, perched, approaching
    risk_score: float = 0.0
    risk_level: str = "safe"  # safe, warning, danger, critical
    species_confidence: float = 0.0
    review_required: bool = False
    review_reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


def _load_detector_class():
    """动态加载生产检测器类。

    bird_monitoring 的可信生产主链固定为 plugin.py -> detector.py:BirdDetector。
    高级/实验检测器不得被自动接入主链，避免随机检测或硬件语义误入生产。
    """
    detector_path = Path(__file__).parent / "detector.py"

    if not detector_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("bird_detector", detector_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bird_detector"] = module
    spec.loader.exec_module(module)

    if hasattr(module, 'BirdDetector'):
        return module.BirdDetector
    return None


_BirdDetector = None

def get_detector_class():
    global _BirdDetector
    if _BirdDetector is None:
        _BirdDetector = _load_detector_class()
    return _BirdDetector


class BirdMonitoringPlugin(BasePlugin):
    """
    鸟类监控插件

    实现室外输电线路鸟类检测、种类识别和风险评估。
    当前阶段为 engineering_baseline，无真实 DL 模型。
    """

    # 告警级别映射
    ALARM_LEVELS = {
        "safe": AlarmLevel.INFO,
        "warning": AlarmLevel.WARNING,
        "danger": AlarmLevel.ERROR,
        "critical": AlarmLevel.CRITICAL,
    }

    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        """初始化插件"""
        super().__init__(manifest, plugin_dir)
        self._config: Dict[str, Any] = {}
        self._initialized = False
        self._detector = None
        self._bird_database = BIRD_DATABASE.copy()
        self._active_tracks: Dict[int, BirdDetection] = {}
        self._inference_count = 0
        self._last_inference_time: Optional[datetime] = None
        self._runtime_mode = "unknown"

        # 风险评估权重 — 由 init 从 config 加载
        self._distance_weight = 0.4
        self._species_weight = 0.3
        self._wingspan_weight = 0.2
        self._behavior_weight = 0.1

        # 置信度阈值
        self._review_threshold = 0.5
        self._species_review_threshold = 0.6

        # 风险距离阈值
        self._risk_thresholds = {
            "safe_distance_m": 15.0,
            "warning_distance_m": 10.0,
            "danger_distance_m": 5.0,
            "critical_distance_m": 2.0,
        }

        # 计算代码版本 hash
        self._code_hash = self._calculate_code_hash()

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

    def _calculate_code_hash(self) -> str:
        """计算代码版本hash"""
        h = hashlib.sha256()
        files_to_hash = ["plugin.py", "detector.py", "detector_enhanced.py"]
        for fname in files_to_hash:
            fpath = self.plugin_dir / fname
            if fpath.exists():
                h.update(fpath.read_bytes())
        return f"sha256:{h.hexdigest()[:12]}"

    @property
    def code_hash(self) -> str:
        return self._code_hash

    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        config_path = self.plugin_dir / "configs" / "default.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _merge_config(self, base: Dict, override: Dict) -> Dict:
        """递归合并配置"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def init(self, config: dict[str, Any] | None = None) -> bool:
        """初始化插件"""
        config = config or {}

        try:
            # 加载默认配置并合并
            default_config = self._load_default_config()
            self._config = self._merge_config(default_config, config)

            # 加载风险评估参数
            risk_config = self._config.get("risk_assessment", {})
            self._risk_thresholds.update({
                "safe_distance_m": risk_config.get("safe_distance_m", 15.0),
                "warning_distance_m": risk_config.get("warning_distance_m", 10.0),
                "danger_distance_m": risk_config.get("danger_distance_m", 5.0),
                "critical_distance_m": risk_config.get("critical_distance_m", 2.0),
            })
            self._distance_weight = risk_config.get("distance_weight", 0.4)
            self._species_weight = risk_config.get("species_weight", 0.3)
            self._wingspan_weight = risk_config.get("wingspan_weight", 0.2)
            self._behavior_weight = risk_config.get("behavior_weight", 0.1)

            # 加载置信度阈值
            quality_config = self._config.get("quality", {})
            self._review_threshold = quality_config.get("review_confidence_threshold", 0.5)
            self._species_review_threshold = quality_config.get(
                "species_confidence_threshold", 0.6
            )

            # 加载检测器
            detector_class = get_detector_class()
            if detector_class:
                self._detector = detector_class(self._config)
                self._runtime_mode = getattr(self._detector, "runtime_mode", "traditional_fallback")
                logger.info("[%s] 检测器加载成功, runtime_mode=%s", self.id, self._runtime_mode)
            else:
                self._runtime_mode = "simulation"
                logger.info("[%s] 检测器未找到，使用模拟模式", self.id)

            # 加载自定义鸟类数据库
            self._load_custom_bird_database()

            self._initialized = True
            self.status = PluginStatus.READY

            logger.info(
                "[%s] 初始化成功, 鸟类数据库=%d种, runtime_mode=%s",
                self.id, len(self._bird_database), self._runtime_mode,
            )
            return True

        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            logger.error("[%s] 初始化失败: %s", self.id, e, exc_info=True)
            return False

    def _load_custom_bird_database(self):
        """加载自定义鸟类数据库"""
        db_path = self.plugin_dir / "data" / "bird_database.yaml"
        if db_path.exists():
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    custom_db = yaml.safe_load(f) or {}
                for bird_id, bird_data in custom_db.get("species", {}).items():
                    self._bird_database[bird_id] = BirdSpecies(**bird_data)
                logger.info(
                    "[%s] 自定义鸟类数据库: %d种",
                    self.id, len(custom_db.get("species", {})),
                )
            except Exception as e:
                logger.warning("[%s] 加载自定义鸟类数据库失败: %s", self.id, e)

    # =========================================================================
    # 输入质量评估
    # =========================================================================
    def _assess_input_quality(
        self, frame: np.ndarray, roi: ROI
    ) -> Dict[str, Any]:
        """评估输入图像质量（三态: pass / soft_fail / hard_fail）。

        - hard_fail: 致命缺陷（None / empty / 尺寸过小 / 过暗 / 过度模糊），
          `is_valid=False`，推理链会阻断并输出 `quality_failed`。
        - soft_fail: 边界缺陷（亮度或清晰度低于 soft 阈值但高于 hard 阈值），
          `is_valid=True`，推理会继续但检测结果会强制进入 `review_required`。
        - pass: 输入符合阈值要求，推理正常。

        兼容性: 返回字典仍暴露 `is_valid / issues / overall_score`；
        新增字段 `status / hard_issues / soft_issues`。
        """
        quality: Dict[str, Any] = {
            "status": "pass",
            "is_valid": True,
            "issues": [],
            "hard_issues": [],
            "soft_issues": [],
            "overall_score": 1.0,
        }

        # 帧有效性
        if frame is None or frame.size == 0:
            quality["status"] = "hard_fail"
            quality["is_valid"] = False
            quality["issues"].append("输入图像为空")
            quality["hard_issues"].append("输入图像为空")
            quality["overall_score"] = 0.0
            return quality

        h, w = frame.shape[:2]
        quality_cfg = self._config.get("quality", {})
        min_dim = quality_cfg.get("min_dimension", 64)
        if w < min_dim or h < min_dim:
            quality["status"] = "hard_fail"
            quality["is_valid"] = False
            msg = f"图像尺寸过小({w}x{h})"
            quality["issues"].append(msg)
            quality["hard_issues"].append(msg)
            quality["overall_score"] *= 0.5

        # 委托给 detector 做深度质量评估
        if self._detector and hasattr(self._detector, "assess_image_quality"):
            roi_image = self._extract_roi(frame, roi.bbox)
            if roi_image is not None and roi_image.size > 0:
                det_quality = self._detector.assess_image_quality(roi_image)
                det_issues = det_quality.get("issues", [])
                quality["issues"].extend(det_issues)
                quality["overall_score"] = min(
                    quality["overall_score"], det_quality.get("overall_score", 1.0)
                )
                quality["clarity_score"] = det_quality.get("clarity_score")
                quality["brightness_score"] = det_quality.get("brightness_score")
                blocking = [
                    issue for issue in det_issues if "cv2不可用" not in str(issue)
                ]
                if det_quality.get("is_valid") is False or blocking:
                    quality["status"] = "hard_fail"
                    quality["is_valid"] = False
                    quality["hard_issues"].extend(blocking)

        # soft_fail 判定：未到 hard_fail，但清晰度或整体分数低于 soft 阈值
        if quality["status"] == "pass":
            soft_overall = quality_cfg.get("soft_overall_threshold", 0.6)
            soft_clarity = quality_cfg.get("soft_clarity_threshold", 0.5)
            soft_brightness = quality_cfg.get("soft_brightness_threshold", 0.5)
            soft_hits: List[str] = []
            if quality.get("overall_score", 1.0) < soft_overall:
                soft_hits.append(
                    f"整体质量分({quality.get('overall_score', 1.0):.2f})"
                    f"低于软阈值({soft_overall})"
                )
            clarity_score = quality.get("clarity_score")
            if clarity_score is not None and clarity_score < soft_clarity:
                soft_hits.append(
                    f"清晰度分({clarity_score:.2f})低于软阈值({soft_clarity})"
                )
            brightness_score = quality.get("brightness_score")
            if brightness_score is not None and brightness_score < soft_brightness:
                soft_hits.append(
                    f"亮度分({brightness_score:.2f})低于软阈值({soft_brightness})"
                )
            if soft_hits:
                quality["status"] = "soft_fail"
                quality["soft_issues"].extend(soft_hits)
                quality["issues"].extend(soft_hits)

        return quality

    # =========================================================================
    # 推理主链路
    # =========================================================================
    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """执行推理"""
        if not self._initialized:
            return [self._make_error_result(
                roi, context, "9000", "插件未初始化"
            ) for roi in rois]

        self.status = PluginStatus.RUNNING
        self._last_inference_time = datetime.now()
        self._inference_count += 1

        results: list[RecognitionResult] = []

        for roi in rois:
            try:
                # 输入质量评估（三态）
                quality = self._assess_input_quality(frame, roi)
                if quality.get("status") == "hard_fail":
                    results.append(self._make_quality_fail_result(
                        roi, context, quality
                    ))
                    continue

                # 提取 ROI 区域
                roi_image = self._extract_roi(frame, roi.bbox)
                if roi_image is None or roi_image.size == 0:
                    results.append(self._make_error_result(
                        roi, context, "9002", "ROI提取失败: 区域无效"
                    ))
                    continue

                # 执行鸟类检测
                detections = self._detect_birds(roi_image, context)

                soft_fail = quality.get("status") == "soft_fail"
                soft_reason = (
                    "; ".join(quality.get("soft_issues", [])) if soft_fail else ""
                )

                # 处理每个检测结果
                for detection in detections:
                    # 评估风险
                    risk_score, risk_level = self._assess_risk(detection, context)
                    detection.risk_score = risk_score
                    detection.risk_level = risk_level

                    # soft_fail 质量门 → review_required
                    if soft_fail:
                        detection.review_required = True
                        detection.review_reason = (
                            f"质量软失败: {soft_reason}"
                        )

                    # 低置信度 → review_required
                    if detection.confidence < self._review_threshold:
                        detection.review_required = True
                        detection.review_reason = (
                            detection.review_reason or
                            f"检测置信度({detection.confidence:.2f})"
                            f"低于阈值({self._review_threshold})"
                        )

                    # 种类不确定 → review_required
                    if (detection.species_id == "unknown_bird"
                            and detection.species_confidence < self._species_review_threshold):
                        detection.review_required = True
                        detection.review_reason = (
                            detection.review_reason or
                            f"物种识别不确定(conf={detection.species_confidence:.2f})"
                        )

                    # 更新跟踪
                    self._active_tracks[detection.track_id] = detection

                    # 生成识别结果
                    result = self._create_result(roi, detection, context, quality)
                    results.append(result)

                # 如果没有检测到鸟类
                if not detections:
                    no_bird_meta = build_visual_meta(
                        plugin_name="bird_monitoring",
                        task_type="detection",
                        modality="bird",
                        runtime_mode=self._runtime_mode,
                        algorithm_stage="baseline",
                        quality_gate_status=quality.get("status", "pass"),
                        review_required=soft_fail,
                        reason_codes=(
                            [f"质量软失败: {soft_reason}"] if soft_fail else []
                        ),
                        evidence_hint="visual_frame",
                        extra={
                            "message": "未检测到鸟类",
                            "input_quality": quality,
                        },
                    )
                    results.append(RecognitionResult(
                        task_id=context.task_id,
                        site_id=context.site_id,
                        device_id=context.device_id,
                        component_id=context.component_id,
                        roi_id=roi.id,
                        bbox=roi.bbox,
                        label="no_bird",
                        confidence=1.0,
                        value="安全",
                        model_version=self.version,
                        code_version=self.code_hash,
                        metadata=no_bird_meta,
                    ))

            except Exception as e:
                logger.warning("[%s] ROI %s 推理失败: %s", self.id, roi.id, e)
                results.append(self._make_error_result(
                    roi, context, "9001", str(e)
                ))

        self.status = PluginStatus.READY
        return results

    # =========================================================================
    # 检测与物种识别
    # =========================================================================
    def _detect_birds(
        self, image: np.ndarray, context: PluginContext
    ) -> List[BirdDetection]:
        """检测鸟类"""
        detections = []

        if self._detector:
            raw_detections = self._detector.detect(image)
            for det in raw_detections:
                species_id, species_confidence = self._identify_species(det)
                species_info = self._bird_database.get(species_id)

                if species_info:
                    species_name = species_info.name
                else:
                    species_name = "未知鸟类"

                detection = BirdDetection(
                    track_id=det.get("track_id", 0),
                    species_id=species_id,
                    species_name=species_name,
                    confidence=det.get("confidence", 0.0),
                    species_confidence=species_confidence,
                    bbox=det.get("bbox", {}),
                    distance_to_line_m=det.get("distance", 20.0),
                    speed_ms=det.get("speed", 0.0),
                    heading_deg=det.get("heading", 0.0),
                    status=det.get("status", "flying"),
                )
                detections.append(detection)

        return detections

    def _identify_species(self, detection: Dict) -> tuple[str, float]:
        """
        识别鸟类种类。

        返回 (species_id, species_confidence)。
        未能匹配的类别返回 ("unknown_bird", 0.0)，不默认为麻雀。
        """
        class_name = detection.get("class_name", "").lower().strip()
        raw_confidence = detection.get("confidence", 0.0)

        species_id = SPECIES_LABEL_MAP.get(class_name, "unknown_bird")

        # 种类置信度: 若映射到已知种类则继承检测置信度；否则为 0
        if species_id == "unknown_bird":
            return "unknown_bird", 0.0

        return species_id, raw_confidence

    # =========================================================================
    # 风险评估
    # =========================================================================
    def _assess_risk(
        self, detection: BirdDetection, context: PluginContext
    ) -> tuple[float, str]:
        """
        评估鸟类风险

        综合考虑:
        1. 与输电线的距离
        2. 鸟类种类风险等级
        3. 翼展与相间距比例
        4. 行为模式
        """
        species_info = self._bird_database.get(detection.species_id)

        # unknown_bird 使用保守估计
        if not species_info:
            return self._assess_unknown_bird_risk(detection)

        # 1. 距离因子
        dist = detection.distance_to_line_m
        thresholds = self._risk_thresholds
        if dist > thresholds["safe_distance_m"]:
            distance_factor = 0.0
        elif dist > thresholds["warning_distance_m"]:
            distance_factor = 0.3
        elif dist > thresholds["danger_distance_m"]:
            distance_factor = 0.7
        else:
            distance_factor = 1.0

        # 2. 种类风险因子
        species_risk_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
        species_factor = species_risk_map.get(species_info.risk_level, 0.5)

        # 3. 翼展因子
        default_spacing = self._config.get("risk_assessment", {}).get(
            "default_phase_spacing_m", 3.0
        )
        phase_spacing = (
            context.metadata.get("phase_spacing_m", default_spacing)
            if context and context.metadata else default_spacing
        )
        wingspan_m = species_info.wingspan_cm / 100.0
        wingspan_factor = min(1.0, wingspan_m / phase_spacing)

        # 4. 行为因子
        behavior_factors = {
            "perched": 0.8,
            "approaching": 0.6,
            "flying": 0.3,
            "group_perching": 0.7,
            "nesting": 0.9,
        }
        behavior_factor = behavior_factors.get(detection.status, 0.3)

        # 综合风险分数
        risk_score = (
            distance_factor * self._distance_weight
            + species_factor * self._species_weight
            + wingspan_factor * self._wingspan_weight
            + behavior_factor * self._behavior_weight
        ) * 100

        return risk_score, self._score_to_level(risk_score)

    def _assess_unknown_bird_risk(
        self, detection: BirdDetection
    ) -> tuple[float, str]:
        """对未识别鸟类做保守风险估计"""
        dist = detection.distance_to_line_m
        if dist <= self._risk_thresholds["danger_distance_m"]:
            return 60.0, "danger"
        elif dist <= self._risk_thresholds["warning_distance_m"]:
            return 45.0, "warning"
        return 20.0, "safe"

    @staticmethod
    def _score_to_level(risk_score: float) -> str:
        if risk_score >= 80:
            return "critical"
        elif risk_score >= 60:
            return "danger"
        elif risk_score >= 40:
            return "warning"
        return "safe"

    # =========================================================================
    # 结果构建
    # =========================================================================
    def _create_result(
        self,
        roi: ROI,
        detection: BirdDetection,
        context: PluginContext,
        quality: Dict[str, Any],
    ) -> RecognitionResult:
        """创建识别结果 — 含详细 metadata 和训练占位符"""
        species_info = self._bird_database.get(detection.species_id)

        bbox = roi.bbox
        if detection.bbox:
            try:
                bbox = self._convert_bbox_to_absolute(detection.bbox, roi.bbox)
            except Exception:
                bbox = roi.bbox

        # 确定 label
        if detection.review_required:
            label = "review_required"
        elif detection.species_id == "unknown_bird":
            label = "unknown_bird"
        else:
            label = f"bird_{detection.risk_level}"

        # 确定 value（人类可读）
        if detection.review_required:
            value = f"需人工复核: {detection.review_reason}"
        else:
            value = f"风险评分: {detection.risk_score:.1f}"

        # 驱离建议
        deterrent_suggestion = self._make_deterrent_suggestion(detection)

        unified = build_visual_meta(
            plugin_name="bird_monitoring",
            task_type="detection",
            modality="bird",
            runtime_mode=self._runtime_mode,
            algorithm_stage="baseline",
            quality_gate_status=quality.get("status", "pass"),
            review_required=detection.review_required,
            reason_codes=[detection.review_reason] if detection.review_reason else [],
            recommended_actions=[deterrent_suggestion] if deterrent_suggestion else [],
            evidence_hint="visual_frame",
            extra={
                # 基础检测信息
                "track_id": detection.track_id,
                "species_id": detection.species_id,
                "species_name": detection.species_name,
                "species_confidence": detection.species_confidence,
                "scientific_name": species_info.scientific_name if species_info else "",
                # 风险评估
                "risk_level": detection.risk_level,
                "risk_score": detection.risk_score,
                "distance_m": detection.distance_to_line_m,
                "speed_ms": detection.speed_ms,
                "heading_deg": detection.heading_deg,
                "status": detection.status,
                "wingspan_cm": species_info.wingspan_cm if species_info else 0,
                "safety_notes": species_info.safety_notes if species_info else "",
                # 复核状态
                "review_reason": detection.review_reason,
                # 驱离建议
                "deterrent_suggestion": deterrent_suggestion,
                # 运行时信息
                "input_quality": quality,
            },
        )

        return RecognitionResult(
            task_id=context.task_id,
            site_id=context.site_id,
            device_id=context.device_id,
            component_id=context.component_id,
            roi_id=roi.id,
            label=label,
            confidence=detection.confidence,
            value=value,
            bbox=bbox,
            model_version=self.version,
            code_version=self.code_hash,
            metadata=unified,
        )

    def _make_quality_fail_result(
        self, roi: ROI, context: PluginContext, quality: Dict[str, Any]
    ) -> RecognitionResult:
        """输入质量 hard_fail 时的结构化失败结果（blocked 而非 review_required）。"""
        return RecognitionResult(
            task_id=context.task_id,
            site_id=context.site_id,
            device_id=context.device_id,
            component_id=context.component_id,
            roi_id=roi.id,
            bbox=roi.bbox,
            label="quality_failed",
            confidence=0.0,
            model_version=self.version,
            code_version=self.code_hash,
            failure_reason="9003",
            metadata=build_visual_meta(
                plugin_name="bird_monitoring",
                task_type="detection",
                modality="bird",
                runtime_mode=self._runtime_mode,
                algorithm_stage="baseline",
                quality_gate_status="hard_fail",
                review_required=False,
                reason_codes=quality.get("hard_issues", []),
                recommended_actions=["请确认图像清晰度和亮度后重新采集"],
                evidence_hint="visual_frame",
                extra={
                    "error": "输入质量不合格",
                    "quality_issues": quality.get("issues", []),
                    "input_quality": quality,
                },
            ),
        )

    def _make_error_result(
        self, roi: ROI, context: PluginContext, code: str, message: str
    ) -> RecognitionResult:
        return RecognitionResult(
            task_id=context.task_id,
            site_id=context.site_id,
            device_id=context.device_id,
            component_id=context.component_id,
            roi_id=roi.id,
            bbox=roi.bbox,
            label="error",
            confidence=0.0,
            model_version=self.version,
            code_version=self.code_hash,
            failure_reason=code,
            metadata=build_visual_meta(
                plugin_name="bird_monitoring",
                task_type="detection",
                modality="bird",
                runtime_mode=self._runtime_mode,
                algorithm_stage="baseline",
                quality_gate_status="hard_fail",
                reason_codes=[code],
                recommended_actions=["检查推理环境后重试"],
                evidence_hint="visual_frame",
                extra={
                    "error": message,
                    "input_quality": {
                        "is_valid": False,
                        "issues": [message],
                        "overall_score": 0.0,
                    },
                },
            ),
        )

    # =========================================================================
    # 驱离建议（不控制硬件，仅输出建议）
    # =========================================================================
    def _make_deterrent_suggestion(self, detection: BirdDetection) -> Dict[str, Any]:
        """根据风险等级生成驱离建议"""
        if detection.risk_level == "safe":
            return {"action": "none", "reason": "风险等级低，无需驱离"}

        deterrent_cfg = self._config.get("deterrent", {})
        auto_trigger = deterrent_cfg.get("auto_trigger_level", "danger")
        levels_order = ["safe", "warning", "danger", "critical"]
        should_trigger = (
            detection.risk_level in levels_order
            and levels_order.index(detection.risk_level) >= levels_order.index(auto_trigger)
        )

        if not should_trigger:
            return {
                "action": "monitor",
                "reason": f"风险等级({detection.risk_level})未达驱离阈值({auto_trigger})",
            }

        # 建议动作
        suggested_actions = []
        if deterrent_cfg.get("acoustic_enabled", True):
            suggested_actions.append("acoustic")
        if deterrent_cfg.get("laser_enabled", False):
            suggested_actions.append("laser")
        if not suggested_actions:
            suggested_actions.append("acoustic")

        return {
            "action": "suggest_deterrent",
            "methods": suggested_actions,
            "reason": f"{detection.species_name}风险等级={detection.risk_level},"
                      f"距离={detection.distance_to_line_m:.1f}m",
            "hardware_available": False,
            "hardware_control": "blocked",
            "note": "当前仅输出驱离建议，不控制任何硬件",
        }

    # =========================================================================
    # 训练数据占位符
    # =========================================================================
    @staticmethod
    def _make_training_placeholders(
        label: str, detection: Optional[BirdDetection]
    ) -> Dict[str, Any]:
        """为后续真实模型训练预留字段"""
        placeholders: Dict[str, Any] = {
            "hard_negative_candidate": False,
            "hard_positive_candidate": False,
            "suggested_label_for_dataset": label,
            "annotation_status": "unlabeled",
            "feature_placeholder": None,
            "model_placeholder": "bird_yolov8n_v1",
        }

        if detection and detection.review_required:
            placeholders["annotation_status"] = "needs_review"
            if detection.confidence > 0.3:
                placeholders["hard_positive_candidate"] = True
            else:
                placeholders["hard_negative_candidate"] = True

        return placeholders

    # =========================================================================
    # 后处理 — 告警生成
    # =========================================================================
    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """后处理和告警生成"""
        alarms: list[Alarm] = []

        for result in results:
            # review_required 生成复核提醒
            if result.label == "review_required":
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title="鸟类检测需人工复核",
                    message=result.value or "检测结果不确定，建议人工确认",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))
                continue

            # unknown_bird 生成提醒
            if result.label == "unknown_bird":
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title="检测到未识别鸟类",
                    message="检测到鸟类但无法识别种类，建议人工确认",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))
                continue

            # 质量失败
            if result.label == "quality_failed":
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title="图像质量不合格",
                    message="; ".join(result.metadata.get("quality_issues", ["未知问题"])),
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))
                continue

            # 标准鸟类风险告警
            if not result.label.startswith("bird_"):
                continue

            risk_level = result.metadata.get("risk_level")
            if not risk_level:
                risk_level = result.label.split("bird_", 1)[1] if "bird_" in result.label else None

            if risk_level not in ("danger", "critical"):
                continue

            alarm_level = self.ALARM_LEVELS.get(risk_level)
            if alarm_level is None:
                continue

            species_name = result.metadata.get("species_name", "鸟类")
            distance = result.metadata.get("distance_m")
            risk_score = result.metadata.get("risk_score")
            deterrent = result.metadata.get("deterrent_suggestion", {})

            message = f"{species_name}接近输电线"
            if distance is not None:
                message = f"{species_name}距离输电线约{distance:.1f}米"
            if risk_score is not None:
                message += f"，风险评分{risk_score:.1f}"
            if deterrent.get("action") == "suggest_deterrent":
                message += f"，建议驱离方式: {','.join(deterrent.get('methods', []))}"

            alarms.append(Alarm(
                task_id=result.task_id,
                result_id=None,
                level=alarm_level,
                title=f"{species_name}高风险接近",
                message=message,
                site_id=result.site_id,
                device_id=result.device_id,
                component_id=result.component_id,
            ))

        return alarms

    # =========================================================================
    # 工具方法
    # =========================================================================
    def _extract_roi(self, frame: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
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

    def _convert_bbox_to_absolute(
        self, rel_bbox: Dict[str, float] | BoundingBox, roi_bbox: BoundingBox
    ) -> BoundingBox:
        if isinstance(rel_bbox, BoundingBox):
            rel_x, rel_y, rel_w, rel_h = rel_bbox.x, rel_bbox.y, rel_bbox.width, rel_bbox.height
        else:
            rel_x = rel_bbox.get("x", 0.0)
            rel_y = rel_bbox.get("y", 0.0)
            rel_w = rel_bbox.get("width", 0.0)
            rel_h = rel_bbox.get("height", 0.0)

        return BoundingBox(
            x=roi_bbox.x + rel_x * roi_bbox.width,
            y=roi_bbox.y + rel_y * roi_bbox.height,
            width=rel_w * roi_bbox.width,
            height=rel_h * roi_bbox.height,
        )

    # =========================================================================
    # 健康检查
    # =========================================================================
    def _get_detector_runtime_status(self) -> Dict[str, Any]:
        """获取 detector 暴露的运行真实性快照。"""
        if self._detector and hasattr(self._detector, "get_runtime_status"):
            return dict(self._detector.get_runtime_status())
        return {
            "runtime_mode": self._runtime_mode,
            "model_path_configured": None,
            "model_path_resolved": None,
            "model_file_exists": False,
            "real_model_loaded": False,
            "onnx_session_ready": False,
            "fallback_enabled": False,
            "simulation_enabled": self._runtime_mode == "simulation",
            "model_load_error": None,
        }

    def healthcheck(self) -> HealthStatus:
        runtime_status = self._get_detector_runtime_status()
        if not self._initialized:
            return HealthStatus(
                healthy=False,
                message="插件未初始化",
                details={"status": "not_initialized", **runtime_status},
            )

        return HealthStatus(
            healthy=True,
            message=f"鸟类监控就绪，数据库含{len(self._bird_database)}种鸟类",
            details={
                "status": "ready",
                "runtime_mode": self._runtime_mode,
                "inference_count": self._inference_count,
                "active_tracks": len(self._active_tracks),
                "bird_database_size": len(self._bird_database),
                "last_inference": (
                    self._last_inference_time.isoformat()
                    if self._last_inference_time else None
                ),
                "review_threshold": self._review_threshold,
                "species_review_threshold": self._species_review_threshold,
                **runtime_status,
            },
        )

    def cleanup(self) -> None:
        self._active_tracks.clear()
        if self._detector and hasattr(self._detector, 'cleanup'):
            self._detector.cleanup()
        self._initialized = False
        self.status = PluginStatus.UNLOADED
        logger.info("[%s] 插件已清理", self.id)

    # =========================================================================
    # 额外 API
    # =========================================================================
    def get_bird_database(self) -> Dict[str, Dict]:
        return {
            bird_id: {
                "id": species.id,
                "name": species.name,
                "scientific_name": species.scientific_name,
                "risk_level": species.risk_level,
                "wingspan_cm": species.wingspan_cm,
                "body_length_cm": species.body_length_cm,
                "weight_g": species.weight_g,
                "behavior_pattern": species.behavior_pattern,
                "description": species.description,
                "safety_notes": species.safety_notes,
            }
            for bird_id, species in self._bird_database.items()
        }

    def get_active_tracks(self) -> List[Dict]:
        return [
            {
                "track_id": det.track_id,
                "species_name": det.species_name,
                "risk_level": det.risk_level,
                "risk_score": det.risk_score,
                "distance_m": det.distance_to_line_m,
                "status": det.status,
                "review_required": det.review_required,
                "timestamp": det.timestamp.isoformat(),
            }
            for det in self._active_tracks.values()
        ]

    def trigger_deterrent(self, level: str = "warning") -> bool:
        """兼容旧调用的驱离入口；本插件只允许建议模式，不执行硬件动作。"""
        logger.info("[%s] 驱鸟硬件控制已阻断，仅保留建议输出: level=%s", self.id, level)
        return False

    def get_ui_config(self) -> Dict[str, Any]:
        return {
            "detection_types": [
                {"id": "bird_detection", "name": "鸟类检测", "default": True},
                {"id": "species_id", "name": "种类识别", "default": True},
                {"id": "risk_assessment", "name": "风险评估", "default": True},
                {"id": "nest_detection", "name": "鸟巢检测", "default": False},
            ],
            "risk_levels": ["safe", "warning", "danger", "critical"],
            "deterrent_options": ["acoustic", "laser", "combined"],
            "bird_database_size": len(self._bird_database),
            "runtime_mode": self._runtime_mode,
        }


Plugin = BirdMonitoringPlugin
