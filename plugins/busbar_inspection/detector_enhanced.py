"""
母线自主巡视检测器 - 增强版
输变电激光监测平台 (C组) - 全自动AI巡检改造

增强功能:
- 4K图像切片处理: 重叠瓦片分解
- YOLOv8m小目标检测: 高精度远距检测
- 多尺度特征融合: 大小目标兼顾
- 质量门禁增强: 模糊/过曝/遮挡检测
- 智能变焦建议: 自动计算推荐倍数
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
    reason_code: str = ""
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
    
    # 缺陷类别映射
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
            model_registry: 模型注册表实例
        """
        self.config = config
        self._model_registry = model_registry
        self._initialized = False
        
        # 配置参数
        self._confidence_threshold = config.get("confidence_threshold", 0.4)
        self._nms_threshold = config.get("nms_threshold", 0.4)
        self._tile_size = config.get("tile_size", self.DEFAULT_TILE_SIZE)
        self._tile_overlap = config.get("tile_overlap", self.DEFAULT_OVERLAP)
        self._use_slicing = config.get("use_slicing", True)
        self._use_deep_learning = config.get("use_deep_learning", True)
        
        # 质量门禁阈值
        self._clarity_threshold = config.get("clarity_threshold", 0.5)
        self._brightness_range = config.get("brightness_range", (0.2, 0.8))
        self._contrast_threshold = config.get("contrast_threshold", 0.3)
        
        # 版本信息
        self._model_version = "busbar_enhanced_v1.0"
        self._code_hash = self._calculate_code_hash()
    
    def _calculate_code_hash(self) -> str:
        """计算代码版本hash"""
        import inspect
        source = inspect.getsource(self.__class__)
        return f"sha256:{hashlib.sha256(source.encode()).hexdigest()[:12]}"
    
    def initialize(self) -> bool:
        """初始化检测器"""
        try:
            if self._model_registry and self._use_deep_learning:
                for model_key, model_id in self.MODEL_IDS.items():
                    try:
                        self._model_registry.load(model_id)
                    except Exception as e:
                        print(f"[BusbarDetector] 模型 {model_id} 加载失败: {e}")
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"[BusbarDetector] 初始化失败: {e}")
            return False
    
    def detect_defects(
        self,
        image: np.ndarray,
        use_slicing: Optional[bool] = None,
        roi_bbox: Optional[Dict[str, float]] = None,
    ) -> List[BusbarDetection]:
        """
        缺陷检测
        
        Args:
            image: BGR图像(支持4K)
            use_slicing: 是否使用切片(默认根据图像大小自动决定)
            roi_bbox: 可选的ROI区域
            
        Returns:
            检测结果列表
        """
        start_time = time.perf_counter()
        
        # 裁剪ROI
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)
        
        h, w = image.shape[:2]
        
        # 自动决定是否切片
        if use_slicing is None:
            use_slicing = self._use_slicing and (w > 2000 or h > 2000)
        
        detections = []
        
        if use_slicing:
            # 切片检测
            detections = self._detect_with_slicing(image)
        else:
            # 整图检测
            detections = self._detect_single(image)
        
        # 过滤环境干扰
        detections = self._filter_environmental_noise(detections)
        
        # NMS合并
        detections = self._apply_global_nms(detections)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        for det in detections:
            det.metadata["processing_time_ms"] = processing_time
        
        return detections
    
    def _detect_with_slicing(self, image: np.ndarray) -> List[BusbarDetection]:
        """切片检测"""
        h, w = image.shape[:2]
        detections = []
        
        # 生成切片
        tiles = self._generate_tiles(w, h)
        
        for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
            # 裁剪切片
            tile_image = image[y1:y2, x1:x2]
            
            # 检测当前切片
            tile_detections = self._detect_single(tile_image)
            
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
    
    def _detect_single(self, image: np.ndarray) -> List[BusbarDetection]:
        """单图检测"""
        # 优先使用深度学习
        if self._use_deep_learning and self._model_registry:
            detections = self._detect_by_deep_learning(image)
            if detections:
                return detections
        
        # 回退到传统方法
        return self._detect_by_traditional(image)
    
    def _detect_by_deep_learning(self, image: np.ndarray) -> List[BusbarDetection]:
        """深度学习检测"""
        detections = []
        
        try:
            model_id = self.MODEL_IDS["detector"]
            result = self._model_registry.infer(model_id, image)  # type: ignore[union-attr]
            
            for det in result.detections:
                if det["confidence"] >= self._confidence_threshold:
                    class_id = det.get("class_id", 0)
                    defect_type = self.DEFECT_CLASSES.get(class_id, BusbarDefectType.CRACK)
                    
                    detections.append(BusbarDetection(
                        defect_type=defect_type,
                        bbox=det["bbox"],
                        confidence=det["confidence"],
                        class_name=det.get("class_name", defect_type.value),
                        metadata={"source": "deep_learning", "model_id": model_id}
                    ))
        except Exception as e:
            print(f"[BusbarDetector] 深度学习检测失败: {e}")
        
        return detections
    
    def _detect_by_traditional(self, image: np.ndarray) -> List[BusbarDetection]:
        """传统方法检测(回退方案)"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 1. 销钉缺失检测 - 圆形缺失
        pin_detections = self._detect_missing_pins(image)
        detections.extend(pin_detections)
        
        # 2. 裂纹检测 - 细长线条
        crack_detections = self._detect_cracks(image)
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
                        confidence = float(0.6 + (50 - mean_val) / 100)
                        
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
    
    def _detect_cracks(self, image: np.ndarray) -> List[BusbarDetection]:
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
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # 过滤短线
                if length > 50:
                    # 计算线条方向
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    
                    # 裂纹通常是细长的
                    aspect = length / max(abs(x2-x1), abs(y2-y1), 1)
                    
                    if aspect > 3:  # 细长线条
                        confidence = min(0.7, 0.4 + length / 200)
                        
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
                            metadata={"source": "traditional", "length": length, "angle": angle}
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
                    confidence = min(0.65, 0.35 + area / 5000)
                    
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
                clarity_score=1.0,
                brightness_score=0.5,
                contrast_score=0.5,
                occlusion_ratio=0.0
            )
        
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 清晰度评分 - 拉普拉斯方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        clarity_score = min(1.0, laplacian.var() / 1500)
        
        # 2. 亮度评分
        mean_brightness = float(np.mean(gray) / 255.0)
        brightness_score = float(1.0 - 2 * abs(mean_brightness - 0.5))
        
        # 3. 对比度评分
        p5, p95 = np.percentile(gray, [5, 95])
        contrast_score = (p95 - p5) / 255.0
        
        # 4. 遮挡检测
        occlusion_ratio = self._detect_occlusion(gray)
        
        # 判断状态
        status = QualityGateStatus.PASS
        reason_code = ""
        
        if clarity_score < self._clarity_threshold:
            status = QualityGateStatus.FAIL_BLUR
            reason_code = "1001"
        elif mean_brightness > self._brightness_range[1]:
            status = QualityGateStatus.FAIL_OVEREXPOSED
            reason_code = "1002"
        elif mean_brightness < self._brightness_range[0]:
            status = QualityGateStatus.FAIL_UNDEREXPOSED
            reason_code = "1003"
        elif contrast_score < self._contrast_threshold:
            status = QualityGateStatus.FAIL_LOW_CONTRAST
            reason_code = "1005"
        elif occlusion_ratio > 0.3:
            status = QualityGateStatus.FAIL_OCCLUDED
            reason_code = "1004"
        
        return QualityGateResult(
            status=status,
            clarity_score=clarity_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            occlusion_ratio=occlusion_ratio,
            reason_code=reason_code,
            metadata={
                "mean_brightness": mean_brightness,
                "laplacian_var": laplacian.var(),
            }
        )
    
    def _detect_occlusion(self, gray: np.ndarray) -> float:
        """检测遮挡比例"""
        if cv2 is None:
            return 0.0
        
        # 使用边缘密度检测
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 低边缘密度的大块区域可能是遮挡
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        local_density = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        low_density_ratio = np.sum(local_density == 0) / local_density.size
        
        return low_density_ratio
    
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
        
        # 如果质量不通过，跳过检测
        if quality_gate.status != QualityGateStatus.PASS:
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


# 便捷函数
def create_detector(config: Dict[str, Any], model_registry=None) -> BusbarDetectorEnhanced:
    """创建检测器实例"""
    detector = BusbarDetectorEnhanced(config, model_registry)
    detector.initialize()
    return detector