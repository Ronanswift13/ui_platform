"""
母线巡视增强检测器 V3.6
========================

实现母线巡视的AI检测功能:
- 绝缘子缺陷检测 (裂纹、污损、闪络痕迹)
- 销钉缺失检测
- 金具状态检测 (松动、锈蚀)
- 导线状态检测 (破损、断股)
- 异物悬挂检测
- 接触情况分析

支持功能:
- 4K高分辨率图像切片检测
- 图像质量评估与变焦建议
- 小目标增强检测
- NMS后处理
- 环境干扰过滤

版本: 3.6.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime
import time
import hashlib
import logging

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


# =============================================================================
# 缺陷类型定义
# =============================================================================
class BusbarDefectType(Enum):
    """母线缺陷类型"""
    INSULATOR_CRACK = "insulator_crack"          # 绝缘子裂纹
    INSULATOR_DIRTY = "insulator_dirty"          # 绝缘子污损
    INSULATOR_FLASHOVER = "insulator_flashover"  # 绝缘子闪络
    PIN_MISSING = "pin_missing"                  # 销钉缺失
    PIN_LOOSE = "pin_loose"                      # 销钉松动
    FITTING_RUST = "fitting_rust"                # 金具锈蚀
    FITTING_LOOSE = "fitting_loose"              # 金具松动
    WIRE_DAMAGE = "wire_damage"                  # 导线破损
    WIRE_BROKEN_STRAND = "wire_broken_strand"    # 导线断股
    FOREIGN_OBJECT = "foreign_object"            # 异物悬挂
    CONTACT_ABNORMAL = "contact_abnormal"        # 接触异常
    BIRD_NEST = "bird_nest"                      # 鸟巢
    UNKNOWN = "unknown"                          # 未知缺陷


# 缺陷类别映射 (YOLO类别ID -> 缺陷类型)
DEFECT_CLASS_MAP = {
    0: BusbarDefectType.INSULATOR_CRACK,
    1: BusbarDefectType.INSULATOR_DIRTY,
    2: BusbarDefectType.INSULATOR_FLASHOVER,
    3: BusbarDefectType.PIN_MISSING,
    4: BusbarDefectType.PIN_LOOSE,
    5: BusbarDefectType.FITTING_RUST,
    6: BusbarDefectType.FITTING_LOOSE,
    7: BusbarDefectType.WIRE_DAMAGE,
    8: BusbarDefectType.WIRE_BROKEN_STRAND,
    9: BusbarDefectType.FOREIGN_OBJECT,
    10: BusbarDefectType.CONTACT_ABNORMAL,
    11: BusbarDefectType.BIRD_NEST,
}

# 缺陷标签名称
DEFECT_LABELS = {
    BusbarDefectType.INSULATOR_CRACK: "绝缘子裂纹",
    BusbarDefectType.INSULATOR_DIRTY: "绝缘子污损",
    BusbarDefectType.INSULATOR_FLASHOVER: "绝缘子闪络",
    BusbarDefectType.PIN_MISSING: "销钉缺失",
    BusbarDefectType.PIN_LOOSE: "销钉松动",
    BusbarDefectType.FITTING_RUST: "金具锈蚀",
    BusbarDefectType.FITTING_LOOSE: "金具松动",
    BusbarDefectType.WIRE_DAMAGE: "导线破损",
    BusbarDefectType.WIRE_BROKEN_STRAND: "导线断股",
    BusbarDefectType.FOREIGN_OBJECT: "异物悬挂",
    BusbarDefectType.CONTACT_ABNORMAL: "接触异常",
    BusbarDefectType.BIRD_NEST: "鸟巢",
    BusbarDefectType.UNKNOWN: "未知缺陷",
}

# 告警级别映射
DEFECT_ALARM_LEVELS = {
    BusbarDefectType.INSULATOR_CRACK: "error",
    BusbarDefectType.INSULATOR_DIRTY: "warning",
    BusbarDefectType.INSULATOR_FLASHOVER: "critical",
    BusbarDefectType.PIN_MISSING: "error",
    BusbarDefectType.PIN_LOOSE: "warning",
    BusbarDefectType.FITTING_RUST: "warning",
    BusbarDefectType.FITTING_LOOSE: "warning",
    BusbarDefectType.WIRE_DAMAGE: "error",
    BusbarDefectType.WIRE_BROKEN_STRAND: "critical",
    BusbarDefectType.FOREIGN_OBJECT: "warning",
    BusbarDefectType.CONTACT_ABNORMAL: "error",
    BusbarDefectType.BIRD_NEST: "info",
}


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class BusbarDetection:
    """母线检测结果"""
    defect_type: BusbarDefectType
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x, y, w, h) normalized
    label: str = ""
    alarm_level: str = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.label:
            self.label = DEFECT_LABELS.get(self.defect_type, "未知")
        if self.alarm_level == "info":
            self.alarm_level = DEFECT_ALARM_LEVELS.get(self.defect_type, "info")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.defect_type.value,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "alarm_level": self.alarm_level,
            "metadata": self.metadata
        }


@dataclass
class QualityAssessment:
    """图像质量评估结果"""
    clarity_score: float      # 清晰度 0-1
    contrast_score: float     # 对比度 0-1
    exposure_score: float     # 曝光度 0-1
    overall_score: float      # 综合评分 0-1
    is_acceptable: bool       # 是否可接受
    reason_codes: List[str] = field(default_factory=list)  # 原因码
    zoom_suggestion: Optional[float] = None  # 变焦建议
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "clarity": round(self.clarity_score, 3),
            "contrast": round(self.contrast_score, 3),
            "exposure": round(self.exposure_score, 3),
            "overall": round(self.overall_score, 3),
            "acceptable": self.is_acceptable,
            "reasons": self.reason_codes,
            "zoom_suggestion": self.zoom_suggestion
        }


# =============================================================================
# 图像质量评估器
# =============================================================================
class QualityAssessor:
    """图像质量评估器"""
    
    def __init__(self, 
                 clarity_threshold: float = 0.3,
                 contrast_threshold: float = 0.2,
                 overexposure_ratio: float = 0.25,
                 underexposure_ratio: float = 0.25):
        self.clarity_threshold = clarity_threshold
        self.contrast_threshold = contrast_threshold
        self.overexposure_ratio = overexposure_ratio
        self.underexposure_ratio = underexposure_ratio
    
    def assess(self, image: np.ndarray) -> QualityAssessment:
        """评估图像质量"""
        if cv2 is None:
            return QualityAssessment(
                clarity_score=1.0, contrast_score=1.0, exposure_score=1.0,
                overall_score=1.0, is_acceptable=True
            )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 清晰度评估 (Laplacian方差)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        clarity = min(float(laplacian.var()) / 1500.0, 1.0)
        
        # 对比度评估 (动态范围)
        p5, p95 = np.percentile(gray, [5, 95])
        dynamic_range = (p95 - p5) / 255.0
        contrast = min(dynamic_range / 0.5, 1.0)
        
        # 曝光评估
        mean_brightness = gray.mean() / 255.0
        overexposed = float((gray > 245).mean())
        underexposed = float((gray < 10).mean())
        
        exposure = 1.0
        if mean_brightness > 0.7 or overexposed > self.overexposure_ratio:
            exposure = max(0, 1.0 - (overexposed / self.overexposure_ratio))
        elif mean_brightness < 0.3 or underexposed > self.underexposure_ratio:
            exposure = max(0, 1.0 - (underexposed / self.underexposure_ratio))
        
        # 综合评分
        overall = clarity * 0.4 + contrast * 0.3 + exposure * 0.3
        
        # 原因码
        reasons = []
        if clarity < self.clarity_threshold:
            reasons.append("BLUR")
        if contrast < self.contrast_threshold:
            reasons.append("LOW_CONTRAST")
        if overexposed > self.overexposure_ratio:
            reasons.append("OVEREXPOSED")
        if underexposed > self.underexposure_ratio:
            reasons.append("UNDEREXPOSED")
        
        # 变焦建议
        zoom_suggestion = None
        if clarity < 0.5:
            # 建议放大倍数
            zoom_suggestion = min(2.0, 1.0 / max(clarity, 0.3))
        
        return QualityAssessment(
            clarity_score=clarity,
            contrast_score=contrast,
            exposure_score=exposure,
            overall_score=overall,
            is_acceptable=len(reasons) == 0,
            reason_codes=reasons,
            zoom_suggestion=zoom_suggestion
        )


# =============================================================================
# 切片处理器
# =============================================================================
class TileProcessor:
    """切片处理器 - 用于4K大图检测"""
    
    def __init__(self, 
                 tile_size: int = 640,
                 overlap: int = 128,
                 min_tile_size: int = 320):
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_tile_size = min_tile_size
    
    def generate_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """生成切片列表
        
        Returns:
            List of (tile_image, (x1, y1, x2, y2))
        """
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap
        
        tiles = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x1 = min(x, max(0, w - self.tile_size))
                y1 = min(y, max(0, h - self.tile_size))
                x2 = min(x1 + self.tile_size, w)
                y2 = min(y1 + self.tile_size, h)
                
                # 确保切片大小足够
                if (x2 - x1) < self.min_tile_size or (y2 - y1) < self.min_tile_size:
                    continue
                
                tile = image[y1:y2, x1:x2]
                tiles.append((tile, (x1, y1, x2, y2)))
        
        return tiles
    
    def map_detection_to_original(self, 
                                   detection: BusbarDetection,
                                   tile_bbox: Tuple[int, int, int, int],
                                   original_size: Tuple[int, int]) -> BusbarDetection:
        """将切片检测结果映射回原图坐标"""
        x1, y1, x2, y2 = tile_bbox
        tile_w = x2 - x1
        tile_h = y2 - y1
        orig_w, orig_h = original_size
        
        det_x, det_y, det_w, det_h = detection.bbox
        
        # 转换坐标
        new_x = (x1 + det_x * tile_w) / orig_w
        new_y = (y1 + det_y * tile_h) / orig_h
        new_w = det_w * tile_w / orig_w
        new_h = det_h * tile_h / orig_h
        
        return BusbarDetection(
            defect_type=detection.defect_type,
            confidence=detection.confidence,
            bbox=(new_x, new_y, new_w, new_h),
            label=detection.label,
            alarm_level=detection.alarm_level,
            metadata={**detection.metadata, "source_tile": tile_bbox}
        )


# =============================================================================
# NMS后处理器
# =============================================================================
class NMSProcessor:
    """非极大值抑制处理器"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
    
    def process(self, detections: List[BusbarDetection]) -> List[BusbarDetection]:
        """执行NMS"""
        if len(detections) <= 1:
            return detections
        
        # 按置信度排序
        sorted_detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        kept = []
        for det in sorted_detections:
            should_keep = True
            
            for kept_det in kept:
                # 只对同类型检测进行NMS
                if det.defect_type != kept_det.defect_type:
                    continue
                
                iou = self._calculate_iou(det.bbox, kept_det.bbox)
                if iou > self.iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                kept.append(det)
        
        return kept
    
    def _calculate_iou(self, 
                       box1: Tuple[float, float, float, float],
                       box2: Tuple[float, float, float, float]) -> float:
        """计算IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0


# =============================================================================
# ONNX模型推理器
# =============================================================================
class ONNXInferencer:
    """ONNX模型推理器"""
    
    def __init__(self, 
                 model_path: str,
                 input_size: Tuple[int, int] = (640, 640),
                 providers: Optional[List[str]] = None):
        self.model_path = model_path
        self.input_size = input_size
        self.session = None
        self.providers = providers or ['CPUExecutionProvider']
        
        if ort is not None:
            self._load_model()
    
    def _load_model(self):
        """加载ONNX模型"""
        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            logger.info(f"[ONNX] 模型加载成功: {self.model_path}")
        except Exception as e:
            logger.error(f"[ONNX] 模型加载失败: {e}")
            self.session = None
    
    def infer(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[BusbarDetection]:
        """执行推理"""
        if self.session is None:
            return []
        
        # 预处理
        input_tensor = self._preprocess(image)
        
        # 推理
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # 后处理
        detections = self._postprocess(outputs, confidence_threshold)
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        if cv2 is None:
            return np.zeros((1, 3, *self.input_size), dtype=np.float32)
        
        # 调整大小
        resized = cv2.resize(image, self.input_size)
        
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化
        normalized = rgb.astype(np.float32) / 255.0
        
        # HWC -> CHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # 添加batch维度
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _postprocess(self, outputs: List[np.ndarray], 
                     confidence_threshold: float) -> List[BusbarDetection]:
        """后处理输出"""
        detections = []
        
        # 假设YOLOv8格式输出: [batch, num_boxes, 4+num_classes]
        if len(outputs) > 0:
            output = outputs[0]
            
            if len(output.shape) == 3:
                output = output[0]  # 移除batch维度
            
            # 遍历检测框
            for det in output:
                if len(det) < 5:
                    continue
                
                x, y, w, h = det[:4]
                class_scores = det[4:]
                
                if len(class_scores) > 0:
                    class_id = int(np.argmax(class_scores))
                    confidence = float(class_scores[class_id])
                    
                    if confidence >= confidence_threshold:
                        defect_type = DEFECT_CLASS_MAP.get(class_id, BusbarDefectType.UNKNOWN)
                        
                        # 转换为归一化坐标
                        norm_x = float(x) / self.input_size[0]
                        norm_y = float(y) / self.input_size[1]
                        norm_w = float(w) / self.input_size[0]
                        norm_h = float(h) / self.input_size[1]
                        
                        detections.append(BusbarDetection(
                            defect_type=defect_type,
                            confidence=confidence,
                            bbox=(norm_x, norm_y, norm_w, norm_h),
                            metadata={"class_id": class_id, "source": "onnx"}
                        ))
        
        return detections


# =============================================================================
# 母线增强检测器
# =============================================================================
class BusbarEnhancedDetector:
    """母线增强检测器
    
    集成功能:
    - 图像质量评估
    - 4K切片检测
    - ONNX模型推理
    - 传统方法回退
    - NMS后处理
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 input_size: Tuple[int, int] = (640, 640),
                 confidence_threshold: float = 0.5,
                 enable_tiling: bool = True,
                 tile_size: int = 640,
                 tile_overlap: int = 128):
        
        self.confidence_threshold = confidence_threshold
        self.enable_tiling = enable_tiling
        
        # 组件初始化
        self.quality_assessor = QualityAssessor()
        self.tile_processor = TileProcessor(tile_size, tile_overlap)
        self.nms_processor = NMSProcessor(iou_threshold=0.5)
        
        # ONNX推理器
        self.onnx_inferencer = None
        if model_path and ort is not None:
            self.onnx_inferencer = ONNXInferencer(model_path, input_size)
        
        self._initialized = True
        logger.info("[BusbarDetector] 增强检测器初始化完成")
    
    def detect(self, 
               image: np.ndarray,
               detection_types: Optional[List[str]] = None,
               confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """执行检测
        
        Args:
            image: BGR图像
            detection_types: 检测类型列表
            confidence_threshold: 置信度阈值
        
        Returns:
            {
                "success": bool,
                "quality": QualityAssessment,
                "detections": List[BusbarDetection],
                "inference_time_ms": float
            }
        """
        start_time = time.time()
        conf_thresh = confidence_threshold or self.confidence_threshold
        
        # 1. 图像质量评估
        quality = self.quality_assessor.assess(image)
        
        if not quality.is_acceptable and quality.overall_score < 0.2:
            return {
                "success": False,
                "quality": quality.to_dict(),
                "detections": [],
                "alarms": [],
                "inference_time_ms": (time.time() - start_time) * 1000,
                "message": f"图像质量不足: {', '.join(quality.reason_codes)}"
            }
        
        # 2. 执行检测
        h, w = image.shape[:2]
        all_detections = []
        
        # 判断是否需要切片检测
        use_tiling = self.enable_tiling and (w > 1920 or h > 1080)
        
        if use_tiling:
            # 切片检测
            tiles = self.tile_processor.generate_tiles(image)
            
            for tile_img, tile_bbox in tiles:
                tile_detections = self._detect_single(tile_img, conf_thresh)
                
                # 映射回原图坐标
                for det in tile_detections:
                    mapped_det = self.tile_processor.map_detection_to_original(
                        det, tile_bbox, (w, h)
                    )
                    all_detections.append(mapped_det)
        else:
            # 单图检测
            all_detections = self._detect_single(image, conf_thresh)
        
        # 3. NMS后处理
        final_detections = self.nms_processor.process(all_detections)
        
        # 4. 过滤检测类型
        if detection_types:
            final_detections = [
                d for d in final_detections 
                if d.defect_type.value in detection_types
            ]
        
        # 5. 生成告警
        alarms = self._generate_alarms(final_detections)
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "quality": quality.to_dict(),
            "detections": [d.to_dict() for d in final_detections],
            "alarms": alarms,
            "inference_time_ms": round(inference_time, 2),
            "metadata": {
                "image_size": [w, h],
                "tiling_used": use_tiling,
                "detection_count": len(final_detections)
            }
        }
    
    def _detect_single(self, image: np.ndarray, 
                       confidence_threshold: float) -> List[BusbarDetection]:
        """单图检测"""
        # 优先使用ONNX模型
        if self.onnx_inferencer and self.onnx_inferencer.session:
            return self.onnx_inferencer.infer(image, confidence_threshold)
        
        # 回退到传统方法
        return self._detect_traditional(image, confidence_threshold)
    
    def _detect_traditional(self, image: np.ndarray,
                           confidence_threshold: float) -> List[BusbarDetection]:
        """传统方法检测（模拟/回退）"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        # 1. 边缘检测寻找裂纹
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 过滤过小的区域
            if area < 100:
                continue
            
            # 获取边界框
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # 长宽比判断（裂纹通常是细长的）
            aspect_ratio = max(cw, ch) / max(min(cw, ch), 1)
            
            if aspect_ratio > 5:  # 可能是裂纹
                confidence = min(0.5 + (aspect_ratio - 5) * 0.05, 0.85)
                
                if confidence >= confidence_threshold:
                    detections.append(BusbarDetection(
                        defect_type=BusbarDefectType.INSULATOR_CRACK,
                        confidence=confidence,
                        bbox=(x/w, y/h, cw/w, ch/h),
                        metadata={"method": "edge_detection", "aspect_ratio": aspect_ratio}
                    ))
        
        # 2. 圆形检测寻找销钉缺失
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=5, maxRadius=30
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                # 分析圆形区域
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), r, 255, -1)
                mean_val = cv2.mean(gray, mask=mask)[0]
                
                # 如果圆形区域偏暗，可能是销钉缺失
                if mean_val < 100:
                    confidence = min(0.5 + (100 - mean_val) / 100, 0.8)
                    
                    if confidence >= confidence_threshold:
                        detections.append(BusbarDetection(
                            defect_type=BusbarDefectType.PIN_MISSING,
                            confidence=confidence,
                            bbox=(max(0, cx-r)/w, max(0, cy-r)/h, 2*r/w, 2*r/h),
                            metadata={"method": "circle_detection", "mean_intensity": mean_val}
                        ))
        
        return detections
    
    def _generate_alarms(self, detections: List[BusbarDetection]) -> List[Dict[str, Any]]:
        """生成告警"""
        alarms = []
        
        for det in detections:
            if det.alarm_level in ["warning", "error", "critical"]:
                alarms.append({
                    "type": det.defect_type.value,
                    "level": det.alarm_level,
                    "message": f"检测到{det.label}，置信度{det.confidence:.1%}",
                    "timestamp": datetime.now().isoformat(),
                    "detection_id": hashlib.md5(
                        f"{det.defect_type.value}{det.confidence}{time.time()}".encode()
                    ).hexdigest()[:8]
                })
        
        return alarms


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    "BusbarDefectType",
    "BusbarDetection",
    "QualityAssessment",
    "QualityAssessor",
    "TileProcessor",
    "NMSProcessor",
    "ONNXInferencer",
    "BusbarEnhancedDetector",
    "DEFECT_LABELS",
    "DEFECT_ALARM_LEVELS",
]
