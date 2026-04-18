"""
主变自主巡视检测器 - 增强版 V3.5
输变电激光监测平台 (A组) - 全自动AI巡检改造

增强功能:
- YOLOv8-ViT缺陷检测: 破损/锈蚀/油泄漏/异物 (UPDATE.md短期计划)
- U-Net油位分割: 精确油位标记检测
- CNN硅胶分类: 变色状态识别
- 热成像融合: 可见光与热成像双模态检测
- 多模型融合: 综合决策输出

V3.0更新:
- 集成YOLOv8-ViT深度学习模型
- 支持热成像融合检测
- 增强小目标检测能力

V3.5更新 (室外监测迭代):
- SegFormer语义分割: 变压器组件精确分割
- Gabor纹理分析: 表面锈蚀/油污/涂层剥落检测
- 多模态分析融合: 检测+分割+纹理综合判定
- 增强油位/硅胶状态识别精度
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# 导入深度学习模型
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

# V3.5: 导入SegFormer语义分割模块
try:
    from ai_models.deep_learning.segformer import (
        SegFormerSegmenter,
        SegFormerConfig,
        SegmentationTask,
        SegmentationMask
    )
    SEGFORMER_AVAILABLE = True
except ImportError:
    SEGFORMER_AVAILABLE = False
    SegFormerSegmenter = None
    SegFormerConfig = None
    SegmentationTask = None
    SegmentationMask = None

# V3.5: 导入Gabor纹理分析模块
try:
    from ai_models.deep_learning.gabor_texture import (
        GaborTextureAnalyzer,
        GaborFilterConfig,
        TextureAnomalyType,
        TextureAnomaly
    )
    GABOR_AVAILABLE = True
except ImportError:
    GABOR_AVAILABLE = False
    GaborTextureAnalyzer = None
    GaborFilterConfig = None
    TextureAnomalyType = None
    TextureAnomaly = None


logger = logging.getLogger(__name__)


class DefectType(Enum):
    """缺陷类型"""
    OIL_LEAK = "oil_leak"           # 油泄漏
    RUST = "rust"                   # 锈蚀
    DAMAGE = "damage"               # 破损
    FOREIGN_OBJECT = "foreign"      # 异物
    CRACK = "crack"                 # 裂纹
    DEFORMATION = "deformation"     # 变形
    DISCOLORATION = "discoloration" # 变色


class SilicaGelState(Enum):
    """硅胶状态"""
    NORMAL = "normal"               # 正常(蓝色)
    WARNING = "warning"             # 警告(淡蓝/粉红)
    ALARM = "alarm"                 # 告警(粉红/白色)
    UNKNOWN = "unknown"


class ThermalLevel(Enum):
    """热成像级别"""
    NORMAL = "normal"               # 正常
    ATTENTION = "attention"         # 注意
    WARNING = "warning"             # 警告
    ALARM = "alarm"                 # 告警
    CRITICAL = "critical"           # 危急


@dataclass
class Detection:
    """检测结果"""
    defect_type: DefectType
    bbox: Dict[str, float]          # {x, y, width, height} 归一化坐标
    confidence: float
    class_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OilLevelResult:
    """油位检测结果"""
    level_ratio: float              # 油位比例 0-1
    level_status: str               # 正常/偏低/偏高/严重
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SilicaGelResult:
    """硅胶检测结果"""
    state: SilicaGelState
    confidence: float
    color_rgb: Optional[Tuple[int, int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThermalResult:
    """热成像分析结果"""
    max_temperature: float
    avg_temperature: float
    hotspot_count: int
    level: ThermalLevel
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    aligned_image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformerInspectionResult:
    """主变巡视综合结果"""
    defects: List[Detection] = field(default_factory=list)
    oil_level: Optional[OilLevelResult] = None
    silica_gel: Optional[SilicaGelResult] = None
    thermal: Optional[ThermalResult] = None
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    model_version: str = ""
    code_hash: str = ""


class TransformerDetectorEnhanced:
    """
    主变巡视增强检测器
    
    集成深度学习模型进行缺陷检测、状态识别和热成像分析
    """
    
    # 模型ID映射
    MODEL_IDS = {
        "defect": "transformer_defect_yolov8",      # YOLOv8缺陷检测
        "oil_seg": "transformer_oil_unet",          # U-Net油位分割
        "silica": "transformer_silica_classifier",  # 硅胶分类器
        "thermal": "transformer_thermal_cnn",       # 热成像异常检测
    }
    
    # 缺陷类别映射
    DEFECT_CLASSES = {
        0: DefectType.OIL_LEAK,
        1: DefectType.RUST,
        2: DefectType.DAMAGE,
        3: DefectType.FOREIGN_OBJECT,
        4: DefectType.CRACK,
        5: DefectType.DEFORMATION,
    }
    
    # 硅胶颜色范围(HSV)
    SILICA_COLOR_RANGES: Dict[SilicaGelState, Dict[str, np.ndarray]] = {
        SilicaGelState.NORMAL: {
            "lower": np.array([100, 100, 50]),   # 蓝色
            "upper": np.array([130, 255, 255]),
        },
        SilicaGelState.WARNING: {
            "lower": np.array([140, 50, 100]),   # 粉红
            "upper": np.array([170, 150, 255]),
        },
        SilicaGelState.ALARM: {
            "lower": np.array([0, 0, 200]),      # 白色
            "upper": np.array([180, 30, 255]),
        },
    }
    
    # 温度阈值
    THERMAL_THRESHOLDS = {
        ThermalLevel.NORMAL: (0, 60),
        ThermalLevel.ATTENTION: (60, 80),
        ThermalLevel.WARNING: (80, 100),
        ThermalLevel.ALARM: (100, 130),
        ThermalLevel.CRITICAL: (130, float('inf')),
    }
    
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

        # 配置参数
        inference_config = config.get("inference", {})
        model_config = config.get("model", {})
        self._confidence_threshold = inference_config.get(
            "confidence_threshold",
            config.get("confidence_threshold", 0.5),
        )
        self._nms_threshold = inference_config.get(
            "nms_threshold",
            config.get("nms_threshold", 0.4),
        )
        self._use_deep_learning = config.get(
            "use_deep_learning",
            model_config.get("use_deep_learning", True),
        )

        # V3.0: YOLOv8-ViT深度学习检测器
        self._yolov8_vit_detector: Optional[YOLOv8ViTDetector] = None
        self._dl_initialized = False

        # V3.5: SegFormer语义分割器
        self._segformer: Optional[Any] = None
        self._segformer_enabled = config.get("segformer", {}).get("enabled", True)

        # V3.5: Gabor纹理分析器
        self._gabor_analyzer: Optional[Any] = None
        self._gabor_enabled = config.get("gabor_texture", {}).get("enabled", True)

        # 版本信息 (V3.5更新)
        self._model_version = "transformer_enhanced_v3.5"
        self._code_hash = self._calculate_code_hash()
    
    def _calculate_code_hash(self) -> str:
        """计算代码版本hash"""
        import inspect
        source = inspect.getsource(self.__class__)
        return f"sha256:{hashlib.sha256(source.encode()).hexdigest()[:12]}"
    
    def initialize(self) -> bool:
        """初始化检测器"""
        try:
            # V3.0: 优先初始化YOLOv8-ViT深度学习检测器
            if self._use_deep_learning and DL_AVAILABLE:
                self._init_yolov8_vit()

            # V3.5: 初始化SegFormer语义分割器
            if self._segformer_enabled and SEGFORMER_AVAILABLE:
                self._init_segformer()

            # V3.5: 初始化Gabor纹理分析器
            if self._gabor_enabled and GABOR_AVAILABLE:
                self._init_gabor_analyzer()

            # 兼容旧版：如果有模型注册表，预加载模型
            if self._model_registry and self._use_deep_learning and not self._dl_initialized:
                for model_key, model_id in self.MODEL_IDS.items():
                    try:
                        self._model_registry.load(model_id)
                    except Exception as e:
                        logger.warning("[TransformerDetector] 模型 %s 加载失败: %s", model_id, e)

            self._initialized = True
            return True
        except Exception as e:
            logger.exception("[TransformerDetector] 初始化失败")
            return False

    def _init_segformer(self) -> bool:
        """初始化SegFormer语义分割器 (V3.5)"""
        if not SEGFORMER_AVAILABLE or SegFormerSegmenter is None:
            logger.info("[TransformerDetector] SegFormer模块不可用")
            return False

        try:
            seg_config = self.config.get("segformer", {})

            config = SegFormerConfig(
                model_path=seg_config.get("model_path"),
                task=SegmentationTask.TRANSFORMER_COMPONENT,
                model_size=seg_config.get("model_size", "b0"),
                input_size=tuple(seg_config.get("input_size", [512, 512])),
                confidence_threshold=seg_config.get("confidence_threshold", 0.5),
                min_area=seg_config.get("min_area", 100)
            )

            self._segformer = SegFormerSegmenter(config)
            self._segformer.load()

            logger.info("[TransformerDetector] SegFormer语义分割器初始化成功 (V3.5)")
            return True

        except Exception as e:
            logger.warning("[TransformerDetector] SegFormer初始化失败: %s", e)
            return False

    def _init_gabor_analyzer(self) -> bool:
        """初始化Gabor纹理分析器 (V3.5)"""
        if not GABOR_AVAILABLE or GaborTextureAnalyzer is None:
            logger.info("[TransformerDetector] Gabor纹理分析模块不可用")
            return False

        try:
            gabor_config = self.config.get("gabor_texture", {})

            filter_config = GaborFilterConfig(
                wavelengths=gabor_config.get("wavelengths", [4.0, 8.0, 16.0, 32.0]),
                num_orientations=gabor_config.get("num_orientations", 8)
            )

            self._gabor_analyzer = GaborTextureAnalyzer(
                filter_config=filter_config,
                analysis_window_size=gabor_config.get("window_size", 64),
                window_stride=gabor_config.get("window_stride", 32)
            )

            logger.info("[TransformerDetector] Gabor纹理分析器初始化成功 (V3.5)")
            return True

        except Exception as e:
            logger.warning("[TransformerDetector] Gabor分析器初始化失败: %s", e)
            return False

    def _init_yolov8_vit(self) -> bool:
        """初始化YOLOv8-ViT检测器 (V3.0)"""
        if not DL_AVAILABLE:
            logger.info("[TransformerDetector] YOLOv8-ViT模块不可用")
            return False

        try:
            model_config = self.config.get("model", {})
            # 创建配置
            model_path = model_config.get("path", self.config.get("yolov8_model_path"))
            device = model_config.get("device", self.config.get("device", "cpu"))
            use_thermal = self.config.get("use_thermal_fusion", False)

            config = YOLOv8ViTConfig(
                model_path=model_path,
                task=DetectionTask.TRANSFORMER_DEFECT,
                confidence_threshold=self._confidence_threshold,
                nms_threshold=self._nms_threshold,
                device=device,
                use_vit_backbone=True,
                use_se_attention=True,
                use_faster_block=True,
                small_object_aug=True,
                thermal_fusion=use_thermal,
            )

            # 创建检测器
            self._yolov8_vit_detector = YOLOv8ViTDetector(config)
            self._yolov8_vit_detector.load()
            self._dl_initialized = True

            logger.info("[TransformerDetector] YOLOv8-ViT检测器初始化成功 (V3.0)")
            return True

        except Exception as e:
            logger.warning("[TransformerDetector] YOLOv8-ViT初始化失败: %s", e)
            self._dl_initialized = False
            return False
    
    def detect_defects(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None,
        thermal_image: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """
        缺陷检测 (V3.0: 支持热成像融合)

        Args:
            image: BGR图像
            roi_bbox: 可选的ROI区域
            thermal_image: 可选的热成像图像 (V3.0新增)

        Returns:
            检测结果列表
        """
        start_time = time.perf_counter()

        # 裁剪ROI
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)
            if thermal_image is not None:
                thermal_image = self._crop_roi(thermal_image, roi_bbox)

        detections = []

        # V3.0: 优先使用YOLOv8-ViT深度学习 (支持热成像融合)
        if self._use_deep_learning:
            if thermal_image is not None and self._dl_initialized and self._yolov8_vit_detector is not None:
                # 热成像融合检测
                dl_detections = self._detect_with_thermal_fusion(image, thermal_image)
            else:
                # 普通深度学习检测
                dl_detections = self._detect_by_deep_learning(image)

            if dl_detections:
                detections.extend(dl_detections)

        # 深度学习失败或未启用时，回退到传统方法
        if not detections:
            traditional_detections = self._detect_by_traditional(image)
            detections.extend(traditional_detections)

        # NMS去重
        detections = self._apply_nms(detections)

        processing_time = (time.perf_counter() - start_time) * 1000
        for det in detections:
            det.metadata["processing_time_ms"] = processing_time

        return detections

    def _detect_with_thermal_fusion(
        self,
        visible_image: np.ndarray,
        thermal_image: np.ndarray
    ) -> List[Detection]:
        """热成像融合检测 (V3.0新增)"""
        detections = []

        if not self._dl_initialized or self._yolov8_vit_detector is None:
            return detections

        try:
            result = self._yolov8_vit_detector.detect_with_thermal(
                visible_image, thermal_image
            )

            for det in result.detections:
                class_name = det.class_name
                defect_type = self._map_class_to_defect_type(class_name)

                detections.append(Detection(
                    defect_type=defect_type,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_name=class_name,
                    metadata={
                        "source": "yolov8_vit_thermal_fusion",
                        "class_id": det.class_id,
                        "inference_time_ms": result.inference_time_ms,
                        "model_version": result.model_version,
                        "fusion_metadata": det.metadata,
                    }
                ))

        except Exception as e:
            logger.warning("[TransformerDetector] 热成像融合检测失败: %s", e)

        return detections
    
    def _detect_by_deep_learning(self, image: np.ndarray) -> List[Detection]:
        """深度学习缺陷检测 (V3.0: 优先使用YOLOv8-ViT)"""
        detections = []

        # V3.0: 优先使用YOLOv8-ViT检测器
        if self._dl_initialized and self._yolov8_vit_detector is not None:
            try:
                result = self._yolov8_vit_detector.detect(
                    image,
                    confidence_threshold=self._confidence_threshold,
                    nms_threshold=self._nms_threshold
                )

                for det in result.detections:
                    # 将YOLOv8-ViT的类名映射到DefectType
                    class_name = det.class_name
                    defect_type = self._map_class_to_defect_type(class_name)

                    detections.append(Detection(
                        defect_type=defect_type,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_name=class_name,
                        metadata={
                            "source": "yolov8_vit_v3",
                            "class_id": det.class_id,
                            "inference_time_ms": result.inference_time_ms,
                            "model_version": result.model_version,
                        }
                    ))

                return detections

            except Exception as e:
                logger.warning("[TransformerDetector] YOLOv8-ViT检测失败: %s", e)

        # 兼容旧版：回退到model_registry
        if self._model_registry is not None:
            try:
                model_id = self.MODEL_IDS["defect"]
                result = self._model_registry.infer(model_id, image)

                for det in result.detections:
                    class_id = det.get("class_id", 0)
                    defect_type = self.DEFECT_CLASSES.get(class_id, DefectType.DAMAGE)

                    detections.append(Detection(
                        defect_type=defect_type,
                        bbox=det["bbox"],
                        confidence=det["confidence"],
                        class_name=det.get("class_name", defect_type.value),
                        metadata={
                            "source": "model_registry_legacy",
                            "model_id": model_id,
                        }
                    ))
            except Exception as e:
                logger.warning("[TransformerDetector] model_registry检测失败: %s", e)

        return detections

    def _map_class_to_defect_type(self, class_name: str) -> DefectType:
        """将类名映射到缺陷类型"""
        class_mapping = {
            "oil_leak": DefectType.OIL_LEAK,
            "rust": DefectType.RUST,
            "damage": DefectType.DAMAGE,
            "foreign_object": DefectType.FOREIGN_OBJECT,
            "crack": DefectType.CRACK,
            "deformation": DefectType.DEFORMATION,
            "discoloration": DefectType.DISCOLORATION,
        }
        return class_mapping.get(class_name.lower(), DefectType.DAMAGE)
    
    def _detect_by_traditional(self, image: np.ndarray) -> List[Detection]:
        """传统方法缺陷检测(回退方案)"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 1. 油泄漏检测 - 深色区域
        oil_detections = self._detect_oil_leak(image)
        detections.extend(oil_detections)
        
        # 2. 锈蚀检测 - 棕红色区域
        rust_detections = self._detect_rust(image)
        detections.extend(rust_detections)
        
        # 3. 破损检测 - 边缘异常
        damage_detections = self._detect_damage(image)
        detections.extend(damage_detections)
        
        # 4. 异物检测 - 轮廓分析
        foreign_detections = self._detect_foreign_object(image)
        detections.extend(foreign_detections)
        
        return detections
    
    def _detect_oil_leak(self, image: np.ndarray) -> List[Detection]:
        """油泄漏检测"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 深色区域(油渍)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 80])
        mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # 最小面积阈值
                x, y, cw, ch = cv2.boundingRect(cnt)
                confidence = min(0.9, 0.5 + area / (w * h) * 10)
                
                detections.append(Detection(
                    defect_type=DefectType.OIL_LEAK,
                    bbox={"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                    confidence=confidence,
                    class_name="油泄漏",
                    metadata={"source": "traditional", "area": area}
                ))
        
        return detections
    
    def _detect_rust(self, image: np.ndarray) -> List[Detection]:
        """锈蚀检测"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 棕红色区域(锈蚀)
        lower = np.array([0, 100, 50])
        upper = np.array([20, 255, 200])
        mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, cw, ch = cv2.boundingRect(cnt)
                confidence = min(0.85, 0.4 + area / (w * h) * 8)
                
                detections.append(Detection(
                    defect_type=DefectType.RUST,
                    bbox={"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                    confidence=confidence,
                    class_name="锈蚀",
                    metadata={"source": "traditional", "area": area}
                ))
        
        return detections
    
    def _detect_damage(self, image: np.ndarray) -> List[Detection]:
        """破损检测"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 膨胀边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                # 检查轮廓形状不规则性
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
                
                if circularity < 0.3:  # 不规则形状
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    confidence = min(0.7, 0.3 + (1 - circularity) * 0.5)
                    
                    detections.append(Detection(
                        defect_type=DefectType.DAMAGE,
                        bbox={"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                        confidence=confidence,
                        class_name="破损",
                        metadata={"source": "traditional", "circularity": circularity}
                    ))
        
        return detections
    
    def _detect_foreign_object(self, image: np.ndarray) -> List[Detection]:
        """异物检测"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 转换到灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 5000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = cw / (ch + 1e-6)
                
                # 排除太细长的对象
                if 0.3 < aspect_ratio < 3.0:
                    confidence = min(0.6, 0.3 + area / 2000)
                    
                    detections.append(Detection(
                        defect_type=DefectType.FOREIGN_OBJECT,
                        bbox={"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                        confidence=confidence,
                        class_name="异物",
                        metadata={"source": "traditional", "aspect_ratio": aspect_ratio}
                    ))
        
        return detections
    
    def detect_oil_level(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None,
    ) -> OilLevelResult:
        """
        油位检测
        
        Args:
            image: BGR图像
            roi_bbox: 油位计ROI区域
            
        Returns:
            油位检测结果
        """
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)
        
        # 优先使用深度学习分割
        if self._use_deep_learning and self._model_registry:
            result = self._detect_oil_level_dl(image)
            if result:
                return result
        
        # 回退到传统方法
        return self._detect_oil_level_traditional(image)
    
    def _detect_oil_level_dl(self, image: np.ndarray) -> Optional[OilLevelResult]:
        """深度学习油位分割"""
        try:
            model_id = self.MODEL_IDS["oil_seg"]
            result = self._model_registry.infer(model_id, image)  # type: ignore[union-attr]
            
            if result.raw_outputs:
                mask = result.raw_outputs.get("mask", None)
                if mask is not None:
                    # 计算油位比例
                    h = mask.shape[0]
                    oil_pixels = np.sum(mask > 0.5, axis=1)
                    total_pixels = mask.shape[1]
                    
                    # 找到油位线
                    oil_ratio = oil_pixels / total_pixels
                    level_line = np.argmax(oil_ratio > 0.5) / h if np.any(oil_ratio > 0.5) else 0.5
                    
                    return OilLevelResult(
                        level_ratio=1 - level_line,
                        level_status=self._get_level_status(1 - level_line),
                        mask=mask,
                        confidence=0.9,
                        metadata={"source": "deep_learning"}
                    )
        except Exception as e:
            logger.warning("[TransformerDetector] 深度学习油位检测失败: %s", e)
        
        return None
    
    def _detect_oil_level_traditional(self, image: np.ndarray) -> OilLevelResult:
        """传统方法油位检测"""
        if cv2 is None:
            return OilLevelResult(level_ratio=0.5, level_status="未知", confidence=0.0)
        
        h, w = image.shape[:2]
        
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测油的颜色(通常为黄色/琥珀色)
        lower = np.array([15, 50, 50])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # 计算每行的油像素比例
        row_ratio = np.sum(mask > 0, axis=1) / w
        
        # 找到油位线
        threshold = 0.3
        oil_rows = np.where(row_ratio > threshold)[0]
        
        if len(oil_rows) > 0:
            level_line = oil_rows[0] / h
            level_ratio = 1 - level_line
        else:
            level_ratio = 0.5
        
        return OilLevelResult(
            level_ratio=level_ratio,
            level_status=self._get_level_status(level_ratio),
            mask=mask,
            confidence=0.7,
            metadata={"source": "traditional"}
        )
    
    def _get_level_status(self, ratio: float) -> str:
        """获取油位状态"""
        if ratio < 0.2:
            return "严重偏低"
        elif ratio < 0.4:
            return "偏低"
        elif ratio <= 0.7:
            return "正常"
        elif ratio <= 0.85:
            return "偏高"
        else:
            return "严重偏高"
    
    def recognize_silica_gel(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None,
    ) -> SilicaGelResult:
        """
        硅胶状态识别
        
        Args:
            image: BGR图像
            roi_bbox: 硅胶罐ROI区域
            
        Returns:
            硅胶状态结果
        """
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)
        
        # 优先使用深度学习分类
        if self._use_deep_learning and self._model_registry:
            result = self._recognize_silica_dl(image)
            if result:
                return result
        
        # 回退到颜色分析
        return self._recognize_silica_by_color(image)
    
    def _recognize_silica_dl(self, image: np.ndarray) -> Optional[SilicaGelResult]:
        """深度学习硅胶分类"""
        try:
            model_id = self.MODEL_IDS["silica"]
            result = self._model_registry.infer(model_id, image)  # type: ignore[union-attr]
            
            if result.detections:
                det = result.detections[0]
                class_name = det.get("class_name", "unknown")
                
                state_map = {
                    "normal": SilicaGelState.NORMAL,
                    "warning": SilicaGelState.WARNING,
                    "alarm": SilicaGelState.ALARM,
                }
                state = state_map.get(class_name, SilicaGelState.UNKNOWN)
                
                return SilicaGelResult(
                    state=state,
                    confidence=det["confidence"],
                    metadata={"source": "deep_learning"}
                )
        except Exception as e:
            logger.warning("[TransformerDetector] 深度学习硅胶识别失败: %s", e)
        
        return None
    
    def _recognize_silica_by_color(self, image: np.ndarray) -> SilicaGelResult:
        """颜色分析硅胶状态"""
        if cv2 is None:
            return SilicaGelResult(state=SilicaGelState.UNKNOWN, confidence=0.0)
        
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        best_state = SilicaGelState.UNKNOWN
        best_ratio = 0.0
        
        for state, color_range in self.SILICA_COLOR_RANGES.items():
            mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])  # type: ignore[arg-type]
            ratio = np.sum(mask > 0) / mask.size
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_state = state
        
        # 计算平均颜色
        avg_color = cv2.mean(image)[:3]  # type: ignore[index]
        
        return SilicaGelResult(
            state=best_state,
            confidence=float(min(0.9, best_ratio * 2)),
            color_rgb=(int(avg_color[2]), int(avg_color[1]), int(avg_color[0])),
            metadata={"source": "color_analysis", "color_ratio": best_ratio}
        )
    
    def analyze_thermal(
        self,
        thermal_image: np.ndarray,
        visible_image: Optional[np.ndarray] = None,
        temperature_range: Tuple[float, float] = (-20, 150),
    ) -> ThermalResult:
        """
        热成像分析
        
        Args:
            thermal_image: 热成像图像(灰度或伪彩色)
            visible_image: 可见光图像(用于对齐)
            temperature_range: 温度范围
            
        Returns:
            热成像分析结果
        """
        if cv2 is None:
            return ThermalResult(
                max_temperature=0, avg_temperature=0,
                hotspot_count=0, level=ThermalLevel.NORMAL
            )
        
        # 转换为灰度(如果是彩色)
        if len(thermal_image.shape) == 3:
            gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = thermal_image.copy()
        
        # 灰度到温度映射
        min_temp, max_temp = temperature_range
        temp_map = gray.astype(np.float32) / 255.0 * (max_temp - min_temp) + min_temp
        
        # 统计温度
        max_temperature = float(np.max(temp_map))
        avg_temperature = float(np.mean(temp_map))
        
        # 检测热点
        hotspots = []
        hot_threshold = avg_temperature + 20  # 高于平均20度为热点
        hot_mask = (temp_map > hot_threshold).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50:
                x, y, cw, ch = cv2.boundingRect(cnt)
                region_temp = temp_map[y:y+ch, x:x+cw]
                
                hotspots.append({
                    "bbox": {"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                    "max_temp": float(np.max(region_temp)),
                    "avg_temp": float(np.mean(region_temp)),
                    "area": area,
                })
        
        # 确定告警级别
        level = ThermalLevel.NORMAL
        for lvl, (low, high) in self.THERMAL_THRESHOLDS.items():
            if low <= max_temperature < high:
                level = lvl
                break
        
        # 图像对齐(如果提供了可见光图像)
        aligned_image = None
        if visible_image is not None:
            aligned_image = self._align_thermal_visible(thermal_image, visible_image)
        
        return ThermalResult(
            max_temperature=max_temperature,
            avg_temperature=avg_temperature,
            hotspot_count=len(hotspots),
            level=level,
            hotspots=hotspots,
            aligned_image=aligned_image,
            metadata={"temperature_range": temperature_range}
        )
    
    def _align_thermal_visible(
        self,
        thermal: np.ndarray,
        visible: np.ndarray,
    ) -> Optional[np.ndarray]:
        """热成像与可见光对齐"""
        if cv2 is None:
            return None
        
        try:
            # 调整尺寸
            if thermal.shape[:2] != visible.shape[:2]:
                thermal = cv2.resize(thermal, (visible.shape[1], visible.shape[0]))
            
            # 简单融合(实际应用中可使用特征点匹配)
            if len(thermal.shape) == 2:
                thermal_color = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)
            else:
                thermal_color = thermal
            
            aligned = cv2.addWeighted(visible, 0.6, thermal_color, 0.4, 0)
            return aligned
        except Exception as e:
            logger.warning("[TransformerDetector] 图像对齐失败: %s", e)
            return None
    
    def inspect(
        self,
        image: np.ndarray,
        thermal_image: Optional[np.ndarray] = None,
        rois: Optional[List[Dict[str, Any]]] = None,
    ) -> TransformerInspectionResult:
        """
        综合巡视 (V3.0: 支持YOLOv8-ViT和热成像融合)

        Args:
            image: 可见光图像
            thermal_image: 热成像图像(可选)
            rois: ROI列表

        Returns:
            综合巡视结果
        """
        start_time = time.perf_counter()

        # V3.0: 缺陷检测 - 支持热成像融合
        defects = self.detect_defects(image, thermal_image=thermal_image)

        # 处理ROI
        oil_level = None
        silica_gel = None

        if rois:
            for roi in rois:
                roi_type = roi.get("type", "")
                roi_bbox = roi.get("bbox", None)

                if roi_type == "oil_level" and roi_bbox:
                    oil_level = self.detect_oil_level(image, roi_bbox)
                elif roi_type == "silica_gel" and roi_bbox:
                    silica_gel = self.recognize_silica_gel(image, roi_bbox)

        # 热成像分析
        thermal = None
        if thermal_image is not None:
            thermal = self.analyze_thermal(thermal_image, image)

        processing_time = (time.perf_counter() - start_time) * 1000

        # 计算综合置信度
        confidences = [d.confidence for d in defects]
        if oil_level:
            confidences.append(oil_level.confidence)
        if silica_gel:
            confidences.append(silica_gel.confidence)

        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return TransformerInspectionResult(
            defects=defects,
            oil_level=oil_level,
            silica_gel=silica_gel,
            thermal=thermal,
            confidence=avg_confidence,
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
    
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """非极大值抑制"""
        if not detections:
            return []
        
        # 按置信度排序
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

    # ==================== V3.5 新增方法 ====================

    def segment_components(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        SegFormer组件分割 (V3.5)

        分割变压器各个组件区域

        Args:
            image: 输入图像
            roi_bbox: ROI区域

        Returns:
            分割结果
        """
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)

        if self._segformer is None:
            return {
                "success": False,
                "error": "SegFormer未初始化",
                "masks": []
            }

        try:
            result = self._segformer.segment(image, return_probabilities=False)

            masks_data = []
            for mask in result.class_masks:
                masks_data.append({
                    "class_id": mask.class_id,
                    "class_name": mask.class_name,
                    "bbox": mask.bbox,
                    "area": mask.area,
                    "centroid": mask.centroid,
                    "confidence": mask.confidence
                })

            return {
                "success": True,
                "full_mask": result.full_mask,
                "masks": masks_data,
                "inference_time_ms": result.inference_time_ms,
                "model_version": result.model_version
            }

        except Exception as e:
            logger.warning("[TransformerDetector] 组件分割失败: %s", e)
            return {
                "success": False,
                "error": str(e),
                "masks": []
            }

    def analyze_texture(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None,
        return_response_maps: bool = False
    ) -> Dict[str, Any]:
        """
        Gabor纹理分析 (V3.5)

        检测表面纹理异常（锈蚀、油污、涂层剥落等）

        Args:
            image: 输入图像
            roi_bbox: ROI区域
            return_response_maps: 是否返回响应图

        Returns:
            纹理分析结果
        """
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)

        if self._gabor_analyzer is None:
            return {
                "success": False,
                "error": "Gabor分析器未初始化",
                "anomalies": []
            }

        try:
            result = self._gabor_analyzer.analyze(
                image,
                return_response_maps=return_response_maps
            )

            anomalies_data = []
            for anomaly in result.anomalies:
                anomalies_data.append({
                    "type": anomaly.anomaly_type.value,
                    "bbox": anomaly.bbox,
                    "confidence": anomaly.confidence,
                    "severity": anomaly.severity,
                    "description": anomaly.description,
                    "feature": {
                        "energy": anomaly.feature.energy,
                        "entropy": anomaly.feature.entropy,
                        "uniformity": anomaly.feature.uniformity
                    }
                })

            return {
                "success": True,
                "anomalies": anomalies_data,
                "global_features": {
                    "mean_response": result.global_features.mean_response,
                    "energy": result.global_features.energy,
                    "entropy": result.global_features.entropy,
                    "uniformity": result.global_features.uniformity,
                    "dominant_orientation": result.global_features.dominant_orientation
                },
                "processing_time_ms": result.processing_time_ms
            }

        except Exception as e:
            logger.warning("[TransformerDetector] 纹理分析失败: %s", e)
            return {
                "success": False,
                "error": str(e),
                "anomalies": []
            }

    def analyze_comprehensive(
        self,
        image: np.ndarray,
        thermal_image: Optional[np.ndarray] = None,
        roi_bbox: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        综合分析 (V3.5)

        融合检测、分割和纹理分析的综合结果

        Args:
            image: 可见光图像
            thermal_image: 红外热像 (可选)
            roi_bbox: ROI区域

        Returns:
            综合分析结果
        """
        start_time = time.perf_counter()

        result = {
            "defects": [],
            "segmentation": None,
            "texture_anomalies": [],
            "oil_level": None,
            "silica_gel": None,
            "thermal": None,
            "fused_assessment": {},
            "processing_time_ms": 0.0
        }

        # 1. 缺陷检测
        defects = self.detect_defects(image, roi_bbox, thermal_image)
        result["defects"] = [
            {
                "type": d.defect_type.value,
                "bbox": d.bbox,
                "confidence": d.confidence,
                "class_name": d.class_name
            }
            for d in defects
        ]

        # 2. 组件分割
        if self._segformer is not None:
            seg_result = self.segment_components(image, roi_bbox)
            result["segmentation"] = seg_result

        # 3. 纹理分析
        if self._gabor_analyzer is not None:
            texture_result = self.analyze_texture(image, roi_bbox)
            result["texture_anomalies"] = texture_result.get("anomalies", [])

        # 4. 油位检测
        try:
            oil_result = self.detect_oil_level(image, roi_bbox)
            result["oil_level"] = {
                "level_ratio": oil_result.level_ratio,
                "status": oil_result.level_status,
                "confidence": oil_result.confidence
            }
        except Exception as exc:
            logger.warning("[TransformerDetector] 油位综合分析失败: %s", exc)

        # 5. 硅胶状态
        try:
            silica_result = self.recognize_silica_gel(image, roi_bbox)
            result["silica_gel"] = {
                "state": silica_result.state.value,
                "confidence": silica_result.confidence,
                "color_rgb": silica_result.color_rgb
            }
        except Exception as exc:
            logger.warning("[TransformerDetector] 硅胶综合分析失败: %s", exc)

        # 6. 热成像分析
        if thermal_image is not None:
            try:
                visible_image = self._crop_roi(image, roi_bbox) if roi_bbox else image
                thermal_input = self._crop_roi(thermal_image, roi_bbox) if roi_bbox else thermal_image
                thermal_result = self.analyze_thermal(
                    thermal_input,
                    visible_image=visible_image,
                )
                result["thermal"] = {
                    "max_temperature": thermal_result.max_temperature,
                    "avg_temperature": thermal_result.avg_temperature,
                    "level": thermal_result.level.value,
                    "hotspot_count": thermal_result.hotspot_count
                }
            except Exception as exc:
                logger.warning("[TransformerDetector] 热成像综合分析失败: %s", exc)

        # 7. 融合评估
        result["fused_assessment"] = self._fuse_analysis_results(
            result["defects"],
            result.get("texture_anomalies", []),
            result.get("segmentation"),
            result.get("thermal")
        )

        result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
        return result

    def _fuse_analysis_results(
        self,
        defects: List[Dict],
        texture_anomalies: List[Dict],
        segmentation: Optional[Dict],
        thermal: Optional[Dict]
    ) -> Dict[str, Any]:
        """融合多种分析结果 (V3.5)"""
        assessment = {
            "overall_status": "normal",
            "risk_level": 0,
            "confidence": 0.0,
            "findings": [],
            "recommendations": []
        }

        findings = []
        risk_score = 0

        # 缺陷评估
        for defect in defects:
            if defect["confidence"] > 0.7:
                findings.append({
                    "source": "detection",
                    "type": defect["type"],
                    "severity": "high" if defect["confidence"] > 0.85 else "medium"
                })
                risk_score += 2 if defect["confidence"] > 0.85 else 1

        # 纹理异常评估
        for anomaly in texture_anomalies:
            if anomaly["confidence"] > 0.6:
                findings.append({
                    "source": "texture",
                    "type": anomaly["type"],
                    "severity": "high" if anomaly["severity"] > 0.7 else "medium"
                })
                risk_score += 1 if anomaly["severity"] > 0.7 else 0.5

        # 热成像评估
        if thermal is not None:
            if thermal.get("level") in ["alarm", "critical"]:
                findings.append({
                    "source": "thermal",
                    "type": "temperature_anomaly",
                    "severity": "high"
                })
                risk_score += 3

        # 总体状态判定
        if risk_score >= 5:
            assessment["overall_status"] = "critical"
            assessment["recommendations"].append("建议立即停机检查")
        elif risk_score >= 3:
            assessment["overall_status"] = "alarm"
            assessment["recommendations"].append("建议安排检修")
        elif risk_score >= 1:
            assessment["overall_status"] = "warning"
            assessment["recommendations"].append("建议加强监测")
        else:
            assessment["overall_status"] = "normal"

        assessment["risk_level"] = min(10, int(risk_score * 2))
        assessment["confidence"] = min(0.95, 0.5 + len(findings) * 0.1)
        assessment["findings"] = findings

        return assessment

    def get_segformer_info(self) -> Optional[Dict[str, Any]]:
        """获取SegFormer分割器信息 (V3.5)"""
        if self._segformer is None:
            return None
        return self._segformer.model_info

    def get_gabor_config(self) -> Optional[Dict[str, Any]]:
        """获取Gabor分析器配置 (V3.5)"""
        if self._gabor_analyzer is None:
            return None
        config = self._gabor_analyzer.filter_config
        return {
            "wavelengths": config.wavelengths,
            "num_orientations": config.num_orientations,
            "sigma_ratio": config.sigma_ratio,
            "window_size": self._gabor_analyzer.window_size,
            "window_stride": self._gabor_analyzer.window_stride
        }


# 便捷函数
def create_detector(config: Dict[str, Any], model_registry=None) -> TransformerDetectorEnhanced:
    """创建检测器实例"""
    detector = TransformerDetectorEnhanced(config, model_registry)
    detector.initialize()
    return detector
