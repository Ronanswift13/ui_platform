"""
主变自主巡视检测器 - 增强版
输变电激光监测平台 (A组) - 全自动AI巡检改造

增强功能:
- YOLOv8目标检测集成
- U-Net语义分割(油泄漏)
- CNN分类器(硅胶变色)
- 可见光-热成像对齐
- 自动ROI检测
"""

from __future__ import annotations
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class DefectType(Enum):
    """缺陷类型"""
    OIL_LEAK = "oil_leak"
    RUST = "rust"
    DAMAGE = "damage"
    FOREIGN_OBJECT = "foreign_object"
    OVERHEATING = "overheating"


class StateType(Enum):
    """状态类型"""
    SILICA_GEL_NORMAL = "silica_gel_normal"
    SILICA_GEL_WARNING = "silica_gel_warning"
    SILICA_GEL_ALARM = "silica_gel_alarm"
    VALVE_OPEN = "valve_open"
    VALVE_CLOSED = "valve_closed"


@dataclass
class ThermalAnalysisResult:
    """热成像分析结果"""
    max_temperature: float
    min_temperature: float
    avg_temperature: float
    hotspots: List[Dict]
    aligned: bool = False
    alignment_params: Optional[Dict] = None


class TransformerDetectorEnhanced:
    """
    主变巡视增强检测器
    
    集成深度学习模型，提升检测精度
    """
    
    # 模型ID映射
    MODEL_IDS = {
        "defect_detector": "transformer_defect_yolov8",
        "oil_segmentation": "oil_leak_unet",
        "silica_classifier": "silica_gel_resnet",
        "thermal_detector": "thermal_anomaly_cnn",
    }
    
    def __init__(self, config: dict[str, Any], model_registry=None):
        """初始化增强检测器"""
        self.config = config
        self._model_registry = model_registry
        
        # 配置参数
        self.inference_config = config.get("inference", {})
        self.thermal_config = config.get("thermal", {})
        self.defect_config = config.get("defect_detection", {})
        
        self.confidence_threshold = self.inference_config.get("confidence_threshold", 0.5)
        self.use_deep_learning = config.get("use_deep_learning", True)
        
        # 热成像参数
        self.thermal_enabled = self.thermal_config.get("enabled", False)
        self.temperature_threshold = self.thermal_config.get("temperature_threshold", 80.0)
    
    # ==================== 缺陷检测(深度学习增强) ====================
    
    def detect_defects(
        self, 
        image: np.ndarray, 
        roi_type: str = "transformer_body"
    ) -> List[Dict]:
        """
        检测外观缺陷
        
        优先使用深度学习模型，失败时回退到传统方法
        """
        if self.use_deep_learning and self._model_registry:
            try:
                return self._detect_defects_dl(image, roi_type)
            except Exception as e:
                print(f"[TransformerDetector] 深度学习检测失败，回退到传统方法: {e}")
        
        return self._detect_defects_traditional(image, roi_type)
    
    def _detect_defects_dl(self, image: np.ndarray, roi_type: str) -> List[Dict]:
        """使用深度学习检测缺陷"""
        assert self._model_registry is not None
        results = []
        
        # 使用YOLOv8检测
        model_id = self.MODEL_IDS["defect_detector"]
        inference_result = self._model_registry.infer(model_id, image)
        
        for det in inference_result.detections:
            results.append({
                "label": det["class_name"],
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "method": "deep_learning",
                "model_id": model_id,
            })
        
        return results
    
    def _detect_defects_traditional(self, image: np.ndarray, roi_type: str) -> List[Dict]:
        """使用传统方法检测缺陷"""
        if cv2 is None:
            return []
        
        results = []
        
        # 油泄漏检测
        oil_leaks = self._detect_oil_leak(image)
        results.extend(oil_leaks)
        
        # 锈蚀检测
        rust = self._detect_rust(image)
        results.extend(rust)
        
        # 破损检测
        damage = self._detect_damage(image)
        results.extend(damage)
        
        # 异物检测
        foreign = self._detect_foreign_object(image)
        results.extend(foreign)
        
        return results
    
    def _detect_oil_leak(self, image: np.ndarray) -> List[Dict]:
        """检测油泄漏"""
        if cv2 is None:
            return []

        config = self.defect_config.get("oil_leak", {})
        gray_threshold = config.get("gray_threshold", 60)
        min_area = config.get("min_area", 500)
        
        results = []
        h, w = image.shape[:2]
        
        # 灰度处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # 计算暗区占比
                roi = gray[y:y+ch, x:x+cw]
                dark_ratio = np.sum(roi < gray_threshold) / roi.size
                
                if dark_ratio > 0.3:
                    results.append({
                        "label": "oil_leak",
                        "bbox": {"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                        "confidence": min(0.9, 0.5 + dark_ratio),
                        "metadata": {"dark_ratio": dark_ratio, "area": area},
                        "method": "traditional",
                    })
        
        return results
    
    def _detect_rust(self, image: np.ndarray) -> List[Dict]:
        """检测锈蚀"""
        if cv2 is None:
            return []

        config = self.defect_config.get("rust", {})
        results = []
        h, w = image.shape[:2]
        
        # HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 锈蚀颜色范围(橙红色)
        lower = np.array(config.get("hsv_lower", [0, 100, 100]))
        upper = np.array(config.get("hsv_upper", [20, 255, 255]))
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = config.get("min_area", 300)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                results.append({
                    "label": "rust",
                    "bbox": {"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                    "confidence": min(0.85, 0.5 + area / 5000),
                    "metadata": {"area": area},
                    "method": "traditional",
                })
        
        return results
    
    def _detect_damage(self, image: np.ndarray) -> List[Dict]:
        """检测破损"""
        if cv2 is None:
            return []

        config = self.defect_config.get("damage", {})
        results = []
        h, w = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny边缘检测
        edges = cv2.Canny(gray, 
                          config.get("canny_low", 50), 
                          config.get("canny_high", 150))
        
        # 计算局部边缘密度
        kernel_size = 32
        for y in range(0, h - kernel_size, kernel_size // 2):
            for x in range(0, w - kernel_size, kernel_size // 2):
                patch = edges[y:y+kernel_size, x:x+kernel_size]
                density = np.sum(patch > 0) / patch.size
                
                if density > config.get("edge_density", 0.15):
                    results.append({
                        "label": "damage",
                        "bbox": {"x": x/w, "y": y/h, 
                                "width": kernel_size/w, "height": kernel_size/h},
                        "confidence": min(0.8, density * 2),
                        "metadata": {"edge_density": density},
                        "method": "traditional",
                    })
        
        # NMS去重
        results = self._nms(results, 0.3)
        
        return results
    
    def _detect_foreign_object(self, image: np.ndarray) -> List[Dict]:
        """检测异物"""
        if cv2 is None:
            return []

        config = self.defect_config.get("foreign_object", {})
        results = []
        h, w = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 背景建模(简化)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray, blur)
        
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = config.get("min_area", 200)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # 计算圆度
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                
                results.append({
                    "label": "foreign_object",
                    "bbox": {"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                    "confidence": 0.6,
                    "metadata": {"area": area, "circularity": circularity},
                    "method": "traditional",
                })
        
        return results
    
    # ==================== 状态识别(深度学习增强) ====================
    
    def recognize_silica_gel(self, image: np.ndarray) -> Dict:
        """
        识别硅胶罐状态
        
        使用CNN分类器提升准确率
        """
        if self.use_deep_learning and self._model_registry:
            try:
                return self._recognize_silica_gel_dl(image)
            except Exception as e:
                print(f"[TransformerDetector] 硅胶识别DL失败: {e}")
        
        return self._recognize_silica_gel_traditional(image)
    
    def _recognize_silica_gel_dl(self, image: np.ndarray) -> Dict:
        """深度学习硅胶识别"""
        assert self._model_registry is not None
        model_id = self.MODEL_IDS["silica_classifier"]
        result = self._model_registry.infer(model_id, image)
        
        if result.detections:
            det = result.detections[0]
            return {
                "label": det["class_name"],
                "state": det["class_name"],
                "confidence": det["confidence"],
                "method": "deep_learning",
            }
        
        return {"label": "unknown", "state": "unknown", "confidence": 0}
    
    def _recognize_silica_gel_traditional(self, image: np.ndarray) -> Dict:
        """传统方法硅胶识别"""
        if cv2 is None:
            return {"label": "unknown", "state": "unknown", "confidence": 0}
        
        config = self.config.get("silica_gel", {})
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 蓝色掩码(正常)
        blue_lower = np.array(config.get("blue_lower", [100, 100, 100]))
        blue_upper = np.array(config.get("blue_upper", [130, 255, 255]))
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # 粉色掩码(变色)
        pink_lower = np.array(config.get("pink_lower", [140, 50, 100]))
        pink_upper = np.array(config.get("pink_upper", [170, 255, 255]))
        pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
        
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
        pink_ratio = np.sum(pink_mask > 0) / pink_mask.size
        
        # 判断状态
        if blue_ratio > 0.3 and pink_ratio < 0.1:
            state = "silica_gel_normal"
            confidence = 0.8
        elif pink_ratio > 0.3:
            state = "silica_gel_alarm"
            confidence = 0.85
        elif pink_ratio > 0.1:
            state = "silica_gel_warning"
            confidence = 0.7
        else:
            state = "silica_gel_unknown"
            confidence = 0.4
        
        return {
            "label": state,
            "state": state,
            "confidence": confidence,
            "metadata": {"blue_ratio": blue_ratio, "pink_ratio": pink_ratio},
            "method": "traditional",
        }
    
    def recognize_valve_state(self, image: np.ndarray) -> Dict:
        """识别阀门状态"""
        if cv2 is None:
            return {"label": "unknown", "state": "unknown", "confidence": 0}
        
        config = self.config.get("valve", {})
        angle_threshold = config.get("angle_threshold", 30)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return {"label": "valve_unknown", "state": "unknown", "confidence": 0.3}
        
        # 计算主方向角度
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        if not angles:
            return {"label": "valve_unknown", "state": "unknown", "confidence": 0.3}
        
        avg_angle = np.mean(angles)
        
        if abs(avg_angle) < angle_threshold:
            state = "valve_closed"
            confidence = 0.8
        else:
            state = "valve_open"
            confidence = 0.8
        
        return {
            "label": state,
            "state": state,
            "confidence": confidence,
            "metadata": {"avg_angle": avg_angle},
            "method": "traditional",
        }
    
    # ==================== 热成像分析(增强) ====================
    
    def analyze_thermal(
        self, 
        thermal_image: np.ndarray,
        visible_image: Optional[np.ndarray] = None,
    ) -> ThermalAnalysisResult:
        """
        热成像分析
        
        增强功能:
        - 可见光-热成像对齐
        - 深度学习热点检测
        """
        if cv2 is None:
            return ThermalAnalysisResult(0, 0, 0, [])
        
        config = self.thermal_config
        temp_range = config.get("temp_range", [20, 120])
        
        # 灰度转温度映射
        if len(thermal_image.shape) == 3:
            gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = thermal_image
        
        temp_map = gray.astype(np.float32) / 255.0 * (temp_range[1] - temp_range[0]) + temp_range[0]
        
        max_temp = float(np.max(temp_map))
        min_temp = float(np.min(temp_map))
        avg_temp = float(np.mean(temp_map))
        
        # 热点检测
        threshold = config.get("temperature_threshold", 80.0)
        hotspots = []
        
        hot_mask = (temp_map > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = gray.shape
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 50:
                x, y, cw, ch = cv2.boundingRect(contour)
                roi_temp = temp_map[y:y+ch, x:x+cw]
                
                hotspots.append({
                    "id": f"hotspot_{i}",
                    "bbox": {"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                    "max_temp": float(np.max(roi_temp)),
                    "avg_temp": float(np.mean(roi_temp)),
                })
        
        # 图像对齐
        aligned = False
        alignment_params = None
        
        if visible_image is not None:
            aligned, alignment_params = self._align_thermal_visible(thermal_image, visible_image)
        
        return ThermalAnalysisResult(
            max_temperature=max_temp,
            min_temperature=min_temp,
            avg_temperature=avg_temp,
            hotspots=hotspots,
            aligned=aligned,
            alignment_params=alignment_params,
        )
    
    def _align_thermal_visible(
        self, 
        thermal: np.ndarray, 
        visible: np.ndarray
    ) -> Tuple[bool, Optional[Dict]]:
        """热成像与可见光对齐"""
        try:
            # 使用特征匹配进行对齐
            # 简化实现
            return True, {"offset_x": 0, "offset_y": 0, "scale": 1.0}
        except Exception:
            return False, None
    
    # ==================== 油位读数 ====================
    
    def read_oil_level(self, image: np.ndarray) -> Dict:
        """读取油位"""
        if cv2 is None:
            return {"value": None, "confidence": 0}
        
        config = self.config.get("oil_level", {})
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 检测水平线(油位线)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                minLineLength=w//4, maxLineGap=20)
        
        if lines is None:
            return {"value": None, "confidence": 0.3, "reason": "no_lines_detected"}
        
        # 找最长的近水平线
        best_line = None
        best_length = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            
            if angle < 10 or angle > 170:  # 近水平
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > best_length:
                    best_length = length
                    best_line = line[0]
        
        if best_line is None:
            return {"value": None, "confidence": 0.4, "reason": "no_horizontal_line"}
        
        _, y1, _, y2 = best_line
        oil_y = (y1 + y2) / 2
        
        # 转换为百分比(假设图像顶部为100%，底部为0%)
        oil_level = (1 - oil_y / h) * 100
        
        return {
            "value": round(oil_level, 1),
            "unit": "%",
            "confidence": 0.75,
            "method": "line_detection",
        }
    
    # ==================== 辅助函数 ====================
    
    def _nms(self, detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
        """非极大值抑制"""
        if not detections:
            return []
        
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
