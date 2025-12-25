"""
主变检测器 - 实际的检测逻辑实现

使用OpenCV进行基础的缺陷和状态检测
"""

from typing import Any, Optional
import numpy as np
import cv2


class TransformerDetector:
    """
    主变检测器
    
    实现基于OpenCV的缺陷检测和状态识别
    """
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get("inference", {}).get("confidence_threshold", 0.5)
        self.defect_types = config.get("recognition", {}).get("defect_types", [])
        self.state_types = config.get("recognition", {}).get("state_types", [])
        
    def detect_defects(self, roi_image: np.ndarray, roi_type: str) -> list[dict]:
        """
        检测缺陷
        
        Args:
            roi_image: ROI区域图像
            roi_type: ROI类型 (bushing/radiator/oil_level等)
            
        Returns:
            检测结果列表
        """
        results = []
        
        # 转换为灰度图
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # 检测油漏（深色斑点）
        oil_leak = self._detect_oil_leak(gray)
        if oil_leak:
            results.append({
                "type": "defect",
                "label": "oil_leak",
                "confidence": oil_leak["confidence"],
                "bbox": oil_leak["bbox"]
            })
        
        # 检测锈蚀（纹理变化）
        rust = self._detect_rust(roi_image)
        if rust:
            results.append({
                "type": "defect",
                "label": "rust",
                "confidence": rust["confidence"],
                "bbox": rust["bbox"]
            })
        
        # 检测破损（边缘检测）
        damage = self._detect_damage(gray)
        if damage:
            results.append({
                "type": "defect",
                "label": "damage",
                "confidence": damage["confidence"],
                "bbox": damage["bbox"]
            })
        
        # 检测异物（轮廓检测）
        foreign = self._detect_foreign_object(gray)
        if foreign:
            results.append({
                "type": "defect",
                "label": "foreign_object",
                "confidence": foreign["confidence"],
                "bbox": foreign["bbox"]
            })
        
        return results
    
    def detect_state(self, roi_image: np.ndarray, roi_type: str) -> Optional[dict]:
        """
        检测状态
        
        Args:
            roi_image: ROI区域图像
            roi_type: ROI类型
            
        Returns:
            状态检测结果
        """
        if roi_type == "breather":
            # 检测呼吸器硅胶颜色
            return self._detect_silica_gel_color(roi_image)
        elif roi_type == "valve":
            # 检测阀门开闭状态
            return self._detect_valve_state(roi_image)
        
        return None
    
    def _detect_oil_leak(self, gray: np.ndarray) -> Optional[dict]:
        """检测油漏（基于深色区域检测）"""
        # 阈值化找出深色区域
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 找最大的暗色区域
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            # 如果面积足够大，认为是油漏
            if area > 500:  # 像素阈值
                x, y, w, h = cv2.boundingRect(max_contour)
                h_img, w_img = gray.shape
                
                return {
                    "confidence": min(0.5 + area / 10000, 0.95),
                    "bbox": {
                        "x": x / w_img,
                        "y": y / h_img,
                        "width": w / w_img,
                        "height": h / h_img
                    }
                }
        
        return None
    
    def _detect_rust(self, image: np.ndarray) -> Optional[dict]:
        """检测锈蚀（基于颜色检测）"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义铁锈的颜色范围（橙红色）
        lower_rust = np.array([0, 50, 50])
        upper_rust = np.array([20, 255, 255])
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower_rust, upper_rust)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            if area > 300:
                x, y, w, h = cv2.boundingRect(max_contour)
                h_img, w_img = image.shape[:2]
                
                return {
                    "confidence": min(0.5 + area / 8000, 0.9),
                    "bbox": {
                        "x": x / w_img,
                        "y": y / h_img,
                        "width": w / w_img,
                        "height": h / h_img
                    }
                }
        
        return None
    
    def _detect_damage(self, gray: np.ndarray) -> Optional[dict]:
        """检测破损（基于边缘检测）"""
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 计算边缘密度
        edge_density = np.sum(edges > 0) / edges.size
        
        # 如果边缘密度异常高，可能是破损
        if edge_density > 0.15:
            # 查找边缘集中区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            dilated = cv2.dilate(edges, kernel)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                h_img, w_img = gray.shape
                
                return {
                    "confidence": min(0.4 + edge_density, 0.85),
                    "bbox": {
                        "x": x / w_img,
                        "y": y / h_img,
                        "width": w / w_img,
                        "height": h / h_img
                    }
                }
        
        return None
    
    def _detect_foreign_object(self, gray: np.ndarray) -> Optional[dict]:
        """检测异物（基于轮廓检测）"""
        # 边缘检测
        edges = cv2.Canny(gray, 30, 100)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤出可能的异物（小而独立的轮廓）
        foreign_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # 圆形度检测 (异物通常不规则)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # 面积适中且不太规则
                if 200 < area < 2000 and 0.2 < circularity < 0.7:
                    foreign_objects.append((contour, area))
        
        if foreign_objects:
            # 取最大的可疑物体
            max_contour, area = max(foreign_objects, key=lambda x: x[1])
            x, y, w, h = cv2.boundingRect(max_contour)
            h_img, w_img = gray.shape
            
            return {
                "confidence": min(0.4 + area / 5000, 0.8),
                "bbox": {
                    "x": x / w_img,
                    "y": y / h_img,
                    "width": w / w_img,
                    "height": h / h_img
                }
            }
        
        return None
    
    def _detect_silica_gel_color(self, image: np.ndarray) -> Optional[dict]:
        """检测呼吸器硅胶颜色"""
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测蓝色（正常）
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
        
        # 检测粉红色（变色）
        lower_pink = np.array([150, 50, 50])
        upper_pink = np.array([180, 255, 255])
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
        pink_ratio = np.sum(pink_mask > 0) / pink_mask.size
        
        if blue_ratio > 0.1:
            return {
                "type": "state",
                "label": "silica_gel_normal",
                "confidence": min(0.6 + blue_ratio, 0.95),
                "value": "normal",
                "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            }
        elif pink_ratio > 0.1:
            return {
                "type": "state",
                "label": "silica_gel_abnormal",
                "confidence": min(0.6 + pink_ratio, 0.95),
                "value": "abnormal",
                "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            }
        
        return None
    
    def _detect_valve_state(self, image: np.ndarray) -> Optional[dict]:
        """检测阀门开闭状态（基于轮廓方向）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            # 分析主要线条方向
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            avg_angle = np.mean(angles)
            
            # 如果接近水平，视为开启；接近垂直，视为关闭
            if abs(avg_angle) < 30 or abs(avg_angle) > 150:
                label = "valve_open"
                value = "open"
            else:
                label = "valve_closed"
                value = "closed"
            
            return {
                "type": "state",
                "label": label,
                "confidence": 0.75,
                "value": value,
                "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            }
        
        return None
