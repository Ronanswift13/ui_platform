"""
母线自主巡视检测器 - 增强版
输变电激光监测平台 (C组) - 全自动AI巡检改造

增强功能:
- 4K图像切片+多尺度检测
- YOLOv8m/PP-YOLOE小目标检测
- 增强质量门禁
- 线缆弧垂检测(激光测距辅助)
- 智能变焦建议
"""

from __future__ import annotations
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class DefectCategory(Enum):
    """缺陷类别"""
    PIN_MISSING = "pin_missing"           # 销钉缺失
    CRACK = "crack"                       # 裂纹
    FOREIGN_OBJECT = "foreign_object"     # 异物
    CORROSION = "corrosion"               # 腐蚀
    FLASHOVER = "flashover"               # 闪络痕迹
    BROKEN_STRAND = "broken_strand"       # 断股


class QualityIssue(Enum):
    """质量问题"""
    BACKLIGHT = "backlight"               # 逆光
    BLUR = "blur"                         # 模糊
    OCCLUSION = "occlusion"               # 遮挡
    LOW_CONTRAST = "low_contrast"         # 低对比度
    OVEREXPOSURE = "overexposure"         # 过曝
    UNDEREXPOSURE = "underexposure"       # 欠曝


@dataclass
class QualityGate:
    """质量门禁结果"""
    passed: bool
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    reason_code: Optional[int] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class SliceResult:
    """切片检测结果"""
    slice_id: str
    detections: List[Dict]
    slice_bbox: Dict[str, int]           # 切片在原图中的位置
    scale_factor: float


@dataclass
class ZoomSuggestion:
    """变焦建议"""
    should_zoom: bool
    current_target_size: int
    suggested_zoom_factor: float
    target_bbox: Optional[Dict] = None
    reason: str = ""


class BusbarDetectorEnhanced:
    """
    母线巡视增强检测器
    
    专为远距离小目标检测优化
    """
    
    # 模型ID映射
    MODEL_IDS = {
        "small_object": "busbar_yolov8m_small",
        "defect_detector": "busbar_defect_ppyoloe",
        "insulator": "insulator_detector_yolov8",
    }
    
    # 缺陷类别映射
    DEFECT_CLASSES = {
        "pin_missing": DefectCategory.PIN_MISSING,
        "crack": DefectCategory.CRACK,
        "foreign": DefectCategory.FOREIGN_OBJECT,
        "corrosion": DefectCategory.CORROSION,
        "flashover": DefectCategory.FLASHOVER,
        "broken": DefectCategory.BROKEN_STRAND,
    }
    
    # 原因码定义
    REASON_CODES = {
        101: "逆光/过曝/低对比",
        102: "遮挡/目标不可见",
        103: "模糊/失焦",
        104: "欠曝/过暗",
        105: "天气干扰",
        201: "目标过小需变焦",
        202: "无有效目标",
        301: "检测结果不可靠",
    }
    
    def __init__(self, config: dict[str, Any], model_registry=None, ptz_controller=None):
        """初始化增强检测器"""
        self.config = config
        self._model_registry = model_registry
        self._ptz_controller = ptz_controller
        
        # 切片配置
        self.slicing_config = config.get("slicing", {})
        self.slice_enabled = self.slicing_config.get("enabled", True)
        self.slice_size = tuple(self.slicing_config.get("slice_size", [640, 640]))
        self.overlap = self.slicing_config.get("overlap", 0.2)
        self.scale_factors = self.slicing_config.get("scale_factors", [1.0, 0.5])
        
        # 质量门禁配置
        self.quality_config = config.get("quality_gate", {})
        
        # 变焦配置
        self.zoom_config = config.get("zoom_suggestion", {})
        self.min_target_size = self.zoom_config.get("min_target_size", 32)
        
        self.use_deep_learning = config.get("use_deep_learning", True)
    
    # ==================== 主检测入口 ====================
    
    def detect_defects(
        self,
        image: np.ndarray,
        roi_type: str = "busbar",
        use_slicing: bool = True,
    ) -> Dict:
        """
        检测母线缺陷
        
        流程:
        1. 质量门禁检查
        2. 多尺度切片检测
        3. 结果合并和NMS
        4. 变焦建议生成
        """
        h, w = image.shape[:2]
        
        # 1. 质量门禁
        quality = self.check_quality_gate(image)
        if not quality.passed:
            return {
                "detections": [],
                "quality_gate": quality,
                "reason_code": quality.reason_code,
                "suggestions": quality.suggestions,
            }
        
        # 2. 执行检测
        if use_slicing and self.slice_enabled and max(h, w) > 1920:
            detections = self._detect_with_slicing(image)
        else:
            detections = self._detect_single(image)
        
        # 3. NMS去重
        detections = self._apply_nms(detections, iou_threshold=0.4)
        
        # 4. 添加变焦建议
        zoom_suggestion = self._generate_zoom_suggestion(detections, (h, w))
        
        # 5. 添加原因码
        for det in detections:
            if det["confidence"] < 0.5:
                det["reason_code"] = 301
        
        return {
            "detections": detections,
            "quality_gate": quality,
            "zoom_suggestion": zoom_suggestion,
            "image_size": {"width": w, "height": h},
            "slicing_used": use_slicing and self.slice_enabled,
        }
    
    # ==================== 质量门禁 ====================
    
    def check_quality_gate(self, image: np.ndarray) -> QualityGate:
        """
        质量门禁检查
        
        检查:
        - 逆光/过曝
        - 模糊
        - 遮挡
        - 天气干扰
        """
        if cv2 is None:
            return QualityGate(passed=True)
        
        issues = []
        metrics = {}
        suggestions = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 亮度检查
        mean_brightness = np.mean(gray)
        metrics["brightness"] = mean_brightness
        
        brightness_high = self.quality_config.get("brightness_threshold_high", 200)
        brightness_low = self.quality_config.get("brightness_threshold_low", 50)
        
        if mean_brightness > brightness_high:
            issues.append(QualityIssue.OVEREXPOSURE)
            suggestions.append("减少曝光或等待光线变化")
        elif mean_brightness < brightness_low:
            issues.append(QualityIssue.UNDEREXPOSURE)
            suggestions.append("增加曝光或启用补光")
        
        # 2. 模糊检查
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics["laplacian_var"] = lap_var
        
        blur_threshold = self.quality_config.get("laplacian_threshold", 100)
        if lap_var < blur_threshold:
            issues.append(QualityIssue.BLUR)
            suggestions.append("调整焦距或稳定相机")
        
        # 3. 对比度检查
        contrast = np.std(gray)
        metrics["contrast"] = contrast
        
        contrast_threshold = self.quality_config.get("contrast_threshold", 30)
        if contrast < contrast_threshold:
            issues.append(QualityIssue.LOW_CONTRAST)
            suggestions.append("调整图像增强参数")
        
        # 4. 边缘密度检查(遮挡检测)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        metrics["edge_density"] = edge_density
        
        edge_min = self.quality_config.get("edge_density_min", 0.05)
        if edge_density < edge_min:
            issues.append(QualityIssue.OCCLUSION)
            suggestions.append("检查是否有遮挡物")
        
        # 判断是否通过
        passed = len(issues) == 0
        reason_code = None
        
        if not passed:
            if QualityIssue.OVEREXPOSURE in issues or QualityIssue.LOW_CONTRAST in issues:
                reason_code = 101
            elif QualityIssue.OCCLUSION in issues:
                reason_code = 102
            elif QualityIssue.BLUR in issues:
                reason_code = 103
            elif QualityIssue.UNDEREXPOSURE in issues:
                reason_code = 104
        
        return QualityGate(
            passed=passed,
            issues=issues,
            metrics=metrics,
            reason_code=reason_code,
            suggestions=suggestions,
        )
    
    # ==================== 切片检测 ====================
    
    def _detect_with_slicing(self, image: np.ndarray) -> List[Dict]:
        """
        4K图像切片检测
        
        将大图分割为重叠的小块分别检测，然后合并结果
        """
        if cv2 is None:
            return self._detect_single(image)

        h, w = image.shape[:2]
        all_detections = []
        
        for scale in self.scale_factors:
            # 缩放
            if scale != 1.0:
                scaled = cv2.resize(image, None, fx=scale, fy=scale)
            else:
                scaled = image
            
            sh, sw = scaled.shape[:2]
            slice_h, slice_w = self.slice_size
            
            # 计算步长
            step_h = int(slice_h * (1 - self.overlap))
            step_w = int(slice_w * (1 - self.overlap))
            
            slice_idx = 0
            for y in range(0, sh, step_h):
                for x in range(0, sw, step_w):
                    # 提取切片
                    y2 = min(y + slice_h, sh)
                    x2 = min(x + slice_w, sw)
                    slice_img = scaled[y:y2, x:x2]
                    
                    # 检测
                    slice_dets = self._detect_single(slice_img)
                    
                    # 转换坐标到原图
                    for det in slice_dets:
                        bbox = det["bbox"]
                        
                        # 切片坐标 -> 缩放图坐标 -> 原图坐标
                        det["bbox"] = {
                            "x": (x + bbox["x"] * (x2 - x)) / sw / scale,
                            "y": (y + bbox["y"] * (y2 - y)) / sh / scale,
                            "width": bbox["width"] * (x2 - x) / sw / scale,
                            "height": bbox["height"] * (y2 - y) / sh / scale,
                        }
                        det["slice_info"] = {
                            "slice_id": f"s{slice_idx}_scale{scale}",
                            "scale": scale,
                            "slice_bbox": {"x": x, "y": y, "width": x2-x, "height": y2-y},
                        }
                        all_detections.append(det)
                    
                    slice_idx += 1
        
        return all_detections
    
    def _detect_single(self, image: np.ndarray) -> List[Dict]:
        """单图检测"""
        if self.use_deep_learning and self._model_registry:
            try:
                return self._detect_with_model(image)
            except Exception as e:
                print(f"[BusbarDetector] DL检测失败: {e}")
        
        return self._detect_traditional(image)
    
    def _detect_with_model(self, image: np.ndarray) -> List[Dict]:
        """使用深度学习模型检测"""
        assert self._model_registry is not None
        model_id = self.MODEL_IDS["small_object"]
        result = self._model_registry.infer(model_id, image)
        
        detections = []
        for det in result.detections:
            defect_type = self.DEFECT_CLASSES.get(det["class_name"], DefectCategory.FOREIGN_OBJECT)
            
            detections.append({
                "label": defect_type.value,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "method": "deep_learning",
                "model_id": model_id,
            })
        
        return detections
    
    def _detect_traditional(self, image: np.ndarray) -> List[Dict]:
        """传统方法检测"""
        if cv2 is None:
            return []
        
        detections = []
        h, w = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 轮廓检测
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        config = self.config.get("defect_detection", {})
        min_area = config.get("min_area", 50)
        max_area = config.get("max_area", 10000)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # 简单分类
                aspect_ratio = cw / ch if ch > 0 else 1
                circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2 + 1e-6)
                
                if circularity > 0.7:
                    label = "foreign_object"
                elif aspect_ratio > 3:
                    label = "crack"
                else:
                    label = "pin_missing"
                
                detections.append({
                    "label": label,
                    "bbox": {"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                    "confidence": 0.5,
                    "method": "traditional",
                    "metadata": {"area": area, "circularity": circularity},
                })
        
        return detections
    
    # ==================== NMS去重 ====================
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """非极大值抑制"""
        if not detections:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                d for d in detections
                if self._iou(best["bbox"], d["bbox"]) < iou_threshold
            ]
        
        return keep
    
    def _iou(self, box1: Dict, box2: Dict) -> float:
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
    
    # ==================== 变焦建议 ====================
    
    def _generate_zoom_suggestion(
        self,
        detections: List[Dict],
        image_shape: Tuple[int, int],
    ) -> ZoomSuggestion:
        """生成变焦建议"""
        if not detections:
            return ZoomSuggestion(
                should_zoom=False,
                current_target_size=0,
                suggested_zoom_factor=1.0,
                reason="no_targets",
            )
        
        h, w = image_shape
        
        # 找最小的目标
        min_size = float('inf')
        smallest_det = None
        
        for det in detections:
            bbox = det["bbox"]
            size = min(bbox["width"] * w, bbox["height"] * h)
            if size < min_size:
                min_size = size
                smallest_det = det
        
        # 判断是否需要变焦
        should_zoom = min_size < self.min_target_size
        
        if should_zoom:
            # 计算建议变焦倍数
            suggested_zoom = self.min_target_size / min_size * self.zoom_config.get("suggested_zoom_factor", 2.0)
            suggested_zoom = min(suggested_zoom, self.zoom_config.get("max_zoom", 10.0))
            
            return ZoomSuggestion(
                should_zoom=True,
                current_target_size=int(min_size),
                suggested_zoom_factor=suggested_zoom,
                target_bbox=smallest_det["bbox"] if smallest_det else None,
                reason="target_too_small",
            )
        
        return ZoomSuggestion(
            should_zoom=False,
            current_target_size=int(min_size),
            suggested_zoom_factor=1.0,
            reason="size_acceptable",
        )
    
    # ==================== 线缆弧垂检测 ====================
    
    def detect_cable_sag(
        self,
        image: np.ndarray,
        distance_mm: Optional[float] = None,
    ) -> Dict:
        """
        检测线缆弧垂
        
        Args:
            image: 图像
            distance_mm: 激光测距结果(mm)
            
        Returns:
            弧垂检测结果
        """
        if cv2 is None:
            return {"sag_detected": False}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 边缘检测
        edges = cv2.Canny(gray, 30, 100)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                                minLineLength=w//4, maxLineGap=20)
        
        if lines is None:
            return {"sag_detected": False, "reason": "no_lines"}
        
        # 分析线条
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # 近水平的线
            if abs(angle) < 15 or abs(angle) > 165:
                horizontal_lines.append({
                    "points": (x1, y1, x2, y2),
                    "center_y": (y1 + y2) / 2,
                    "length": np.sqrt((x2-x1)**2 + (y2-y1)**2),
                })
        
        if not horizontal_lines:
            return {"sag_detected": False, "reason": "no_horizontal_lines"}
        
        # 计算弧垂(简化: 取y坐标的标准差)
        y_values = [l["center_y"] for l in horizontal_lines]
        sag_std = np.std(y_values)
        
        # 判断是否异常
        sag_threshold = self.config.get("cable_sag", {}).get("threshold_px", 20)
        sag_detected = sag_std > sag_threshold
        
        # 如果有距离信息,转换为实际尺寸
        sag_mm = None
        if distance_mm and sag_detected:
            # 简化计算: 假设像素到mm的转换
            pixel_size_mm = distance_mm / 1000 / w * 35  # 假设35mm焦距
            sag_mm = sag_std * pixel_size_mm
        
        return {
            "sag_detected": sag_detected,
            "sag_pixels": sag_std,
            "sag_mm": sag_mm,
            "line_count": len(horizontal_lines),
            "severity": "warning" if sag_detected else "normal",
        }
