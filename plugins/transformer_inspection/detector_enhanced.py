"""
主变自主巡视检测器 - 增强版
输变电激光监测平台 (A组) - 全自动AI巡检改造

增强功能:
- YOLOv8缺陷检测: 破损/锈蚀/油泄漏/异物
- U-Net油位分割: 精确油位标记检测
- CNN硅胶分类: 变色状态识别
- 热成像对齐: 可见光与热成像配准
- 多模型融合: 综合决策输出
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


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
            model_registry: 模型注册表实例
        """
        self.config = config
        self._model_registry = model_registry
        self._initialized = False
        
        # 配置参数
        self._confidence_threshold = config.get("confidence_threshold", 0.5)
        self._nms_threshold = config.get("nms_threshold", 0.4)
        self._use_deep_learning = config.get("use_deep_learning", True)
        
        # 版本信息
        self._model_version = "transformer_enhanced_v1.0"
        self._code_hash = self._calculate_code_hash()
    
    def _calculate_code_hash(self) -> str:
        """计算代码版本hash"""
        import inspect
        source = inspect.getsource(self.__class__)
        return f"sha256:{hashlib.sha256(source.encode()).hexdigest()[:12]}"
    
    def initialize(self) -> bool:
        """初始化检测器"""
        try:
            # 如果有模型注册表，预加载模型
            if self._model_registry and self._use_deep_learning:
                for model_key, model_id in self.MODEL_IDS.items():
                    try:
                        self._model_registry.load(model_id)
                    except Exception as e:
                        print(f"[TransformerDetector] 模型 {model_id} 加载失败: {e}")
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"[TransformerDetector] 初始化失败: {e}")
            return False
    
    def detect_defects(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None,
    ) -> List[Detection]:
        """
        缺陷检测
        
        Args:
            image: BGR图像
            roi_bbox: 可选的ROI区域
            
        Returns:
            检测结果列表
        """
        start_time = time.perf_counter()
        
        # 裁剪ROI
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)
        
        detections = []
        
        # 优先使用深度学习
        if self._use_deep_learning and self._model_registry:
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
    
    def _detect_by_deep_learning(self, image: np.ndarray) -> List[Detection]:
        """深度学习缺陷检测"""
        detections = []
        
        try:
            model_id = self.MODEL_IDS["defect"]
            result = self._model_registry.infer(model_id, image)  # type: ignore[union-attr]
            
            for det in result.detections:
                class_id = det.get("class_id", 0)
                defect_type = self.DEFECT_CLASSES.get(class_id, DefectType.DAMAGE)
                
                detections.append(Detection(
                    defect_type=defect_type,
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    class_name=det.get("class_name", defect_type.value),
                    metadata={
                        "source": "deep_learning",
                        "model_id": model_id,
                    }
                ))
        except Exception as e:
            print(f"[TransformerDetector] 深度学习检测失败: {e}")
        
        return detections
    
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
            print(f"[TransformerDetector] 深度学习油位检测失败: {e}")
        
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
            print(f"[TransformerDetector] 深度学习硅胶识别失败: {e}")
        
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
            print(f"[TransformerDetector] 图像对齐失败: {e}")
            return None
    
    def inspect(
        self,
        image: np.ndarray,
        thermal_image: Optional[np.ndarray] = None,
        rois: Optional[List[Dict[str, Any]]] = None,
    ) -> TransformerInspectionResult:
        """
        综合巡视
        
        Args:
            image: 可见光图像
            thermal_image: 热成像图像(可选)
            rois: ROI列表
            
        Returns:
            综合巡视结果
        """
        start_time = time.perf_counter()
        
        # 缺陷检测
        defects = self.detect_defects(image)
        
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


# 便捷函数
def create_detector(config: Dict[str, Any], model_registry=None) -> TransformerDetectorEnhanced:
    """创建检测器实例"""
    detector = TransformerDetectorEnhanced(config, model_registry)
    detector.initialize()
    return detector