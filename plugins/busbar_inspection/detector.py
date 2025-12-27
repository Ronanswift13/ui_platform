"""
母线自主巡视插件 - 检测器实现 (C组)

核心功能:
1. 远距小目标缺陷检测（4K大视场）
2. 切片(Tiling) + 多尺度推理
3. 多ROI并发处理（Batch推理）
4. 环境干扰过滤 + 误报原因码
5. 建议变焦倍率输出
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import cv2
import hashlib
import time

# 尝试导入ONNX Runtime
if TYPE_CHECKING:
    # 仅用于类型检查
    try:
        import onnxruntime as ort
    except ImportError:
        pass

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None  # type: ignore
    HAS_ONNX = False


# 导入平台定义的 BBox 以确保类型一致
from platform_core.schema.models import BoundingBox as BBox

# ============================================================
# 数据类定义
# ============================================================

def bbox_to_pixel(bbox: BBox, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    将 BBox 转换为像素坐标 (x1, y1, x2, y2)

    Args:
        bbox: BoundingBox 对象
        img_w: 图像宽度
        img_h: 图像高度

    Returns:
        (x1, y1, x2, y2) 像素坐标
    """
    x1 = int(max(0, bbox.x * img_w))
    y1 = int(max(0, bbox.y * img_h))
    x2 = int(min(img_w, (bbox.x + bbox.width) * img_w))
    y2 = int(min(img_h, (bbox.y + bbox.height) * img_h))
    return x1, y1, x2, y2


@dataclass
class Detection:
    """检测结果"""
    bbox: BBox  # 使用平台的 BoundingBox
    label: str
    confidence: float
    class_id: int = 0


@dataclass
class QualityMetrics:
    """图像质量指标"""
    clarity_score: float = 0.0
    is_overexposed: bool = False
    is_low_contrast: bool = False
    is_occluded: bool = False
    edge_energy: float = 0.0


@dataclass
class ZoomSuggestion:
    """变焦建议"""
    suggested_zoom: float = 1.0
    suggested_action: str = "NONE"  # NONE, ZOOM_IN, REFOCUS, RECAPTURE, CHANGE_VIEW
    min_object_size_px: float = 0.0
    target_size_px: float = 90.0


@dataclass
class DetectionResult:
    """完整检测结果"""
    detections: List[Detection]
    quality: QualityMetrics
    zoom_suggestion: ZoomSuggestion
    reason_code: Optional[int] = None
    latency_ms: int = 0
    debug_info: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 图像质量评估器
# ============================================================

class QualityEvaluator:
    """图像质量评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 质量评估配置
        """
        self.blur_thr = config.get("blur_thr", 0.35)
        self.y_high = config.get("y_high", 245)
        self.overexp_ratio = config.get("overexp_ratio", 0.25)
        self.dr_min = config.get("dr_min", 35)
        self.edge_thr = config.get("edge_thr", 10)
    
    def evaluate_clarity(self, image: np.ndarray) -> float:
        """
        评估清晰度（Laplacian方差）
        
        Returns:
            清晰度分数 [0, 1]
        """
        if image.size == 0:
            return 0.0
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variance = lap.var()
        
        # 归一化到[0, 1]
        score = float(np.clip(variance / 1500.0, 0.0, 1.0))
        return score
    
    def is_overexposed(self, image: np.ndarray) -> bool:
        """检测是否过曝/逆光"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        p_high = float((gray > self.y_high).mean())
        return p_high >= self.overexp_ratio
    
    def is_low_contrast(self, image: np.ndarray) -> bool:
        """检测是否低对比度"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        p5 = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)
        dr = p95 - p5
        
        return dr < self.dr_min
    
    def evaluate_edge_energy(self, image: np.ndarray) -> float:
        """计算边缘能量（用于遮挡检测）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        energy = np.mean(np.abs(sobel_x) + np.abs(sobel_y))
        return float(energy)
    
    def is_occluded(self, image: np.ndarray) -> bool:
        """检测是否被遮挡"""
        energy = self.evaluate_edge_energy(image)
        return energy < self.edge_thr
    
    def evaluate(self, image: np.ndarray) -> QualityMetrics:
        """
        完整质量评估
        
        Returns:
            QualityMetrics
        """
        clarity = self.evaluate_clarity(image)
        overexposed = self.is_overexposed(image)
        low_contrast = self.is_low_contrast(image)
        edge_energy = self.evaluate_edge_energy(image)
        occluded = edge_energy < self.edge_thr
        
        return QualityMetrics(
            clarity_score=clarity,
            is_overexposed=overexposed,
            is_low_contrast=low_contrast,
            is_occluded=occluded,
            edge_energy=edge_energy
        )
    
    def get_reason_code(self, metrics: QualityMetrics) -> Optional[int]:
        """根据质量指标返回原因码"""
        if metrics.is_overexposed or metrics.is_low_contrast:
            return 101  # 逆光/过曝/低对比
        if metrics.is_occluded:
            return 102  # 遮挡/不可见
        if metrics.clarity_score < self.blur_thr:
            return 103  # 模糊/失焦
        return None
    
    def get_suggested_action(self, metrics: QualityMetrics) -> str:
        """根据质量指标返回建议动作"""
        if metrics.is_overexposed or metrics.is_low_contrast:
            return "ADJUST_EXPOSURE_OR_CHANGE_VIEW"
        if metrics.is_occluded:
            return "CHANGE_VIEW_OR_RECAPTURE"
        if metrics.clarity_score < self.blur_thr:
            return "REFOCUS_OR_RECAPTURE"
        return "NONE"


# ============================================================
# 切片(Tiling)处理器
# ============================================================

class TileProcessor:
    """切片处理器 - 用于大图小目标检测"""
    
    def __init__(self, tile_size: int = 640, overlap: int = 128):
        """
        Args:
            tile_size: 切片大小（像素）
            overlap: 重叠大小（像素）
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
    
    def generate_tiles(
        self,
        x0: int, y0: int, x1: int, y1: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        生成切片列表
        
        Args:
            x0, y0, x1, y1: ROI区域（像素坐标）
            
        Returns:
            切片列表 [(tx0, ty0, tx1, ty1), ...]
        """
        tiles = []
        
        for ty in range(y0, y1, self.stride):
            for tx in range(x0, x1, self.stride):
                tx1 = min(tx + self.tile_size, x1)
                ty1 = min(ty + self.tile_size, y1)
                tiles.append((tx, ty, tx1, ty1))
        
        return tiles
    
    def extract_tile(
        self,
        image: np.ndarray,
        tile_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """提取切片图像"""
        tx0, ty0, tx1, ty1 = tile_box
        return image[ty0:ty1, tx0:tx1].copy()
    
    def remap_detection(
        self,
        det: Detection,
        tile_box: Tuple[int, int, int, int],
        img_w: int,
        img_h: int
    ) -> Detection:
        """
        将切片内的检测结果映射回原图坐标

        Args:
            det: 切片内检测结果（归一化坐标）
            tile_box: 切片位置
            img_w, img_h: 原图尺寸

        Returns:
            原图坐标的检测结果
        """
        tx0, ty0, tx1, ty1 = tile_box
        tile_w = tx1 - tx0
        tile_h = ty1 - ty0

        # 转换到原图像素坐标
        abs_x = tx0 + det.bbox.x * tile_w
        abs_y = ty0 + det.bbox.y * tile_h
        abs_w = det.bbox.width * tile_w
        abs_h = det.bbox.height * tile_h

        # 归一化到原图
        return Detection(
            bbox=BBox(
                x=abs_x / img_w,
                y=abs_y / img_h,
                width=abs_w / img_w,
                height=abs_h / img_h
            ),
            label=det.label,
            confidence=det.confidence,
            class_id=det.class_id
        )


# ============================================================
# NMS（非极大值抑制）
# ============================================================

def compute_iou(box1: BBox, box2: BBox) -> float:
    """计算两个框的IoU"""
    # 计算交集
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x + box1.width, box2.x + box2.width)
    y2 = min(box1.y + box1.height, box2.y + box2.height)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 计算并集
    area1 = box1.width * box1.height
    area2 = box2.width * box2.height
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def nms(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """
    非极大值抑制
    
    Args:
        detections: 检测结果列表
        iou_threshold: IoU阈值
        
    Returns:
        过滤后的检测结果
    """
    if not detections:
        return []
    
    # 按置信度排序
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    
    keep = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        
        # 过滤掉与best重叠过大的框
        sorted_dets = [
            d for d in sorted_dets
            if compute_iou(best.bbox, d.bbox) < iou_threshold
        ]
    
    return keep


# ============================================================
# 变焦建议计算器
# ============================================================

class ZoomAdvisor:
    """变焦建议计算器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 变焦配置
        """
        self.target_px = config.get("target_px", 90)  # 目标像素尺寸
        self.min_obj_px = config.get("min_obj_px", 18)  # 最小可检测尺寸
        self.z_min = config.get("zmin", 1.0)
        self.z_max = config.get("zmax", 12.0)
    
    def compute_suggestion(
        self,
        detections: List[Detection],
        img_w: int,
        img_h: int
    ) -> ZoomSuggestion:
        """
        计算变焦建议
        
        Args:
            detections: 检测结果
            img_w, img_h: 图像尺寸
            
        Returns:
            ZoomSuggestion
        """
        if not detections:
            return ZoomSuggestion(
                suggested_zoom=1.0,
                suggested_action="NONE",
                min_object_size_px=0.0,
                target_size_px=self.target_px
            )
        
        # 找到最小的检测框
        min_size_px = float('inf')
        for det in detections:
            box_w_px = det.bbox.width * img_w
            box_h_px = det.bbox.height * img_h
            size_px = max(box_w_px, box_h_px)
            min_size_px = min(min_size_px, size_px)
        
        if min_size_px == float('inf'):
            min_size_px = 0.0
        
        # 计算建议变焦倍率
        if min_size_px < self.min_obj_px:
            # 目标太小，需要变焦
            suggested_zoom = self.target_px / max(min_size_px, 1.0)
            suggested_zoom = np.clip(suggested_zoom, self.z_min, self.z_max)
            action = "ZOOM_IN"
        elif min_size_px < self.target_px * 0.5:
            # 目标较小，建议变焦
            suggested_zoom = self.target_px / min_size_px
            suggested_zoom = np.clip(suggested_zoom, self.z_min, self.z_max)
            action = "ZOOM_IN" if suggested_zoom > 1.5 else "NONE"
        else:
            suggested_zoom = 1.0
            action = "NONE"
        
        return ZoomSuggestion(
            suggested_zoom=float(suggested_zoom),
            suggested_action=action,
            min_object_size_px=float(min_size_px),
            target_size_px=float(self.target_px)
        )


# ============================================================
# 传统视觉检测器（基于规则）
# ============================================================

class RuleBasedDetector:
    """基于规则的传统视觉检测器"""
    
    # 类别标签映射
    LABELS = {
        0: "ok",
        1: "pin_missing",
        2: "crack",
        3: "loose_fitting",
        4: "broken_part",
        5: "foreign_object"
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 检测配置
        """
        self.conf_threshold = config.get("conf_thr", 0.25)
        
        # 边缘检测参数
        self.canny_low = 50
        self.canny_high = 150
        
        # 轮廓过滤参数
        self.min_area = 100
        self.max_area = 50000
    
    def detect_edges_anomaly(self, image: np.ndarray) -> List[Detection]:
        """
        基于边缘检测的异常检测
        
        检测可能的缺陷：裂纹、破损等
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        detections = []
        
        # 边缘检测
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area < area < self.max_area:
                x, y, bw, bh = cv2.boundingRect(contour)
                
                # 计算置信度（基于面积和形状）
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # 长条形可能是裂纹
                aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
                
                if aspect_ratio > 5:
                    # 可能是裂纹
                    confidence = min(0.5 + area / 10000, 0.85)
                    label = "crack"
                    class_id = 2
                elif circularity < 0.3:
                    # 不规则形状，可能是破损
                    confidence = min(0.4 + area / 5000, 0.75)
                    label = "broken_part"
                    class_id = 4
                else:
                    continue
                
                if confidence >= self.conf_threshold:
                    detections.append(Detection(
                        bbox=BBox(x=x/w, y=y/h, width=bw/w, height=bh/h),
                        label=label,
                        confidence=confidence,
                        class_id=class_id
                    ))
        
        return detections
    
    def detect_foreign_object(self, image: np.ndarray) -> List[Detection]:
        """
        检测异物
        
        基于颜色异常和轮廓检测
        """
        if len(image.shape) != 3:
            return []
        
        h, w = image.shape[:2]
        detections = []
        
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测异常颜色区域（非背景色）
        # 假设背景主要是灰色/金属色
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 50, 200]))
        
        # 取反得到可能的异物区域
        foreign_mask = cv2.bitwise_not(gray_mask)
        
        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreign_mask = cv2.morphologyEx(foreign_mask, cv2.MORPH_OPEN, kernel)
        foreign_mask = cv2.morphologyEx(foreign_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(foreign_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if 200 < area < 30000:
                x, y, bw, bh = cv2.boundingRect(contour)
                
                # 计算置信度
                confidence = min(0.4 + area / 5000, 0.80)
                
                if confidence >= self.conf_threshold:
                    detections.append(Detection(
                        bbox=BBox(x=x/w, y=y/h, width=bw/w, height=bh/h),
                        label="foreign_object",
                        confidence=confidence,
                        class_id=5
                    ))
        
        return detections
    
    def detect_missing_pin(self, image: np.ndarray) -> List[Detection]:
        """
        检测销钉缺失
        
        基于小圆形结构检测
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        detections = []
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 圆形检测
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            
            for (cx, cy, r) in circles:
                # 检查圆形区域的特征
                mask = np.zeros_like(gray)
                cv2.circle(mask, (cx, cy), r, 255, -1)

                mean_val = float(cv2.mean(gray, mask=mask)[0])  # type: ignore[index]
                
                # 如果区域过暗或过亮，可能是缺失或异常
                if mean_val < 50 or mean_val > 200:
                    confidence = 0.6

                    detections.append(Detection(
                        bbox=BBox(
                            x=(cx-r)/w,
                            y=(cy-r)/h,
                            width=(2*r)/w,
                            height=(2*r)/h
                        ),
                        label="pin_missing",
                        confidence=confidence,
                        class_id=1
                    ))
        
        return detections
    
    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[Detection]:
        """
        执行完整检测

        Args:
            image: BGR图像
            conf_threshold: 置信度阈值（可选，如果不提供则使用初始化时的阈值）

        Returns:
            检测结果列表
        """
        # 使用传入的阈值或默认阈值
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        original_threshold = self.conf_threshold
        self.conf_threshold = threshold

        all_detections = []

        # 边缘异常检测
        all_detections.extend(self.detect_edges_anomaly(image))

        # 异物检测
        all_detections.extend(self.detect_foreign_object(image))

        # 销钉缺失检测
        all_detections.extend(self.detect_missing_pin(image))

        # 恢复原始阈值
        self.conf_threshold = original_threshold

        return all_detections


# ============================================================
# ONNX模型推理器
# ============================================================

class ONNXDetector:
    """ONNX模型检测器"""
    
    LABELS = {
        0: "ok",
        1: "pin_missing",
        2: "crack",
        3: "loose_fitting",
        4: "broken_part",
        5: "foreign_object"
    }
    
    def __init__(self, model_path: str, input_size: int = 640, providers: Optional[List[str]] = None):
        """
        Args:
            model_path: ONNX模型路径
            input_size: 输入尺寸
            providers: 执行提供者列表
        """
        if not HAS_ONNX:
            raise RuntimeError("onnxruntime未安装")
        
        self.model_path = model_path
        self.input_size = input_size
        
        providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)  # type: ignore
        self.input_name = self.session.get_inputs()[0].name
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        # BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.input_size, self.input_size))
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess(
        self,
        outputs: Any,
        conf_threshold: float
    ) -> List[Detection]:
        """后处理模型输出"""
        # 具体实现取决于模型输出格式
        # 这里提供通用YOLO格式处理
        detections = []
        
        # 假设输出格式为 [batch, num_detections, 6] (x, y, w, h, conf, class)
        det = outputs[0]
        if len(det.shape) == 3:
            det = det[0]  # 去掉batch维度
        
        for row in det:
            if len(row) >= 6:
                x, y, w, h, conf, cls = row[:6]
                
                if conf >= conf_threshold:
                    # 转换为归一化坐标
                    x_norm = (x - w/2) / self.input_size
                    y_norm = (y - h/2) / self.input_size
                    w_norm = w / self.input_size
                    h_norm = h / self.input_size
                    
                    class_id = int(cls)
                    label = self.LABELS.get(class_id, f"class_{class_id}")

                    detections.append(Detection(
                        bbox=BBox(x=x_norm, y=y_norm, width=w_norm, height=h_norm),
                        label=label,
                        confidence=float(conf),
                        class_id=class_id
                    ))
        
        return detections
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        """执行检测"""
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # 后处理
        detections = self.postprocess(outputs, conf_threshold)
        
        return detections


# ============================================================
# 主检测器
# ============================================================

class BusbarDetector:
    """母线检测器 - 集成所有检测功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 完整配置字典
        """
        self.config = config
        
        # 模型配置
        model_cfg = config.get("model", {})
        self.model_path = model_cfg.get("model_path", "")
        self.model_version = model_cfg.get("model_version", "busbar-det@1.0.0")
        self.input_size = model_cfg.get("input_size", 640)
        
        # 推理配置
        thresholds = config.get("thresholds", {})
        self.conf_threshold = thresholds.get("conf_thr", 0.25)
        self.nms_iou = thresholds.get("nms_iou", 0.50)
        
        # 切片配置
        tiling_cfg = config.get("tiling", {})
        tile_size = tiling_cfg.get("tile_size", 640)
        overlap = tiling_cfg.get("overlap", 128)
        self.tile_processor = TileProcessor(tile_size=tile_size, overlap=overlap)
        
        # 质量评估配置
        quality_cfg = config.get("quality", {})
        self.quality_evaluator = QualityEvaluator(quality_cfg)
        
        # 变焦建议配置
        zoom_cfg = config.get("zoom", {})
        self.zoom_advisor = ZoomAdvisor(zoom_cfg)
        
        # 初始化检测器
        self._init_detector()
    
    def _init_detector(self):
        """初始化检测器"""
        # 尝试加载ONNX模型
        if self.model_path and HAS_ONNX:
            try:
                runtime_cfg = self.config.get("runtime", {})
                providers = runtime_cfg.get("providers", ["CPUExecutionProvider"])
                self.detector = ONNXDetector(
                    self.model_path,
                    self.input_size,
                    providers
                )
                self.use_deep_model = True
            except Exception as e:
                print(f"[BusbarDetector] 加载ONNX模型失败: {e}，使用规则检测器")
                self.detector = RuleBasedDetector({"conf_thr": self.conf_threshold})
                self.use_deep_model = False
        else:
            # 使用规则检测器
            self.detector = RuleBasedDetector({"conf_thr": self.conf_threshold})
            self.use_deep_model = False
    
    def detect_roi(
        self,
        image: np.ndarray,
        use_tiling: bool = True
    ) -> DetectionResult:
        """
        检测单个ROI
        
        Args:
            image: ROI图像
            use_tiling: 是否使用切片
            
        Returns:
            DetectionResult
        """
        t0 = time.time()
        h, w = image.shape[:2]
        
        # 1. 质量评估
        quality = self.quality_evaluator.evaluate(image)
        reason_code = self.quality_evaluator.get_reason_code(quality)
        
        # 如果质量太差，直接返回
        if reason_code is not None:
            action = self.quality_evaluator.get_suggested_action(quality)
            return DetectionResult(
                detections=[],
                quality=quality,
                zoom_suggestion=ZoomSuggestion(
                    suggested_zoom=1.0,
                    suggested_action=action
                ),
                reason_code=reason_code,
                latency_ms=int((time.time() - t0) * 1000),
                debug_info={"quality_gate": "failed"}
            )
        
        # 2. 执行检测
        all_detections = []
        
        if use_tiling and max(h, w) > self.input_size * 1.5:
            # 使用切片检测
            tiles = self.tile_processor.generate_tiles(0, 0, w, h)
            
            for tile_box in tiles:
                tile_img = self.tile_processor.extract_tile(image, tile_box)
                
                if tile_img.size == 0:
                    continue
                
                # 检测
                tile_dets = self.detector.detect(tile_img, self.conf_threshold)
                
                # 映射回原图
                for det in tile_dets:
                    remapped = self.tile_processor.remap_detection(det, tile_box, w, h)
                    all_detections.append(remapped)
        else:
            # 直接检测
            all_detections = self.detector.detect(image, self.conf_threshold)
        
        # 3. NMS去重
        all_detections = nms(all_detections, self.nms_iou)
        
        # 4. 计算变焦建议
        zoom_suggestion = self.zoom_advisor.compute_suggestion(all_detections, w, h)
        
        # 如果目标过小，设置原因码
        if zoom_suggestion.suggested_action == "ZOOM_IN":
            reason_code = 201  # 目标过小
        
        latency_ms = int((time.time() - t0) * 1000)
        
        return DetectionResult(
            detections=all_detections,
            quality=quality,
            zoom_suggestion=zoom_suggestion,
            reason_code=reason_code,
            latency_ms=latency_ms,
            debug_info={
                "use_tiling": use_tiling,
                "num_tiles": len(self.tile_processor.generate_tiles(0, 0, w, h)) if use_tiling else 1,
                "use_deep_model": self.use_deep_model
            }
        )
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        use_tiling: bool = True
    ) -> List[DetectionResult]:
        """
        批量检测多个ROI
        
        Args:
            images: ROI图像列表
            use_tiling: 是否使用切片
            
        Returns:
            检测结果列表
        """
        results = []
        for img in images:
            result = self.detect_roi(img, use_tiling)
            results.append(result)
        return results
    
    @property
    def code_version(self) -> str:
        """计算代码版本hash"""
        h = hashlib.sha256()
        h.update(b"busbar_inspection_detector_v1")
        return f"sha256:{h.hexdigest()[:12]}"
