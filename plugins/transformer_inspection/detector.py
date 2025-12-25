"""
主变检测器 - 实际的检测逻辑实现

使用OpenCV进行基础的缺陷和状态检测
"""

from typing import Any, Optional, List
import numpy as np
import cv2


class TransformerDetector:
    """
    主变检测器
    
    实现基于OpenCV的缺陷检测和状态识别
    """
    
    def __init__(self, config: dict[str, Any]):
        self.update_config(config)

    def update_config(self, config: dict[str, Any]) -> None:
        """更新检测配置"""
        self.config = config or {}
        inference = self.config.get("inference", {})
        recognition = self.config.get("recognition", {})
        algorithms = self.config.get("algorithms", {})

        self.confidence_threshold = inference.get("confidence_threshold", 0.5)
        self.nms_threshold = inference.get("nms_threshold", 0.4)
        self.max_detections = inference.get("max_detections", 100)
        self.defect_types = recognition.get("defect_types", [])
        self.state_types = recognition.get("state_types", [])

        oil_leak_cfg = algorithms.get("oil_leak", {})
        rust_cfg = algorithms.get("rust", {})
        damage_cfg = algorithms.get("damage", {})
        foreign_cfg = algorithms.get("foreign_object", {})
        silica_cfg = algorithms.get("silica_gel", {})
        valve_cfg = algorithms.get("valve", {})

        self.oil_leak_gray_threshold = oil_leak_cfg.get("gray_threshold", 60)
        self.oil_leak_min_area = oil_leak_cfg.get("min_area", 500)

        self.rust_min_area = rust_cfg.get("min_area", 300)

        self.damage_canny_low = damage_cfg.get("canny_low", 50)
        self.damage_canny_high = damage_cfg.get("canny_high", 150)
        self.damage_edge_density_threshold = damage_cfg.get("edge_density_threshold", 0.15)

        self.foreign_canny_low = foreign_cfg.get("canny_low", 30)
        self.foreign_canny_high = foreign_cfg.get("canny_high", 100)
        self.foreign_min_area = foreign_cfg.get("min_area", 200)
        self.foreign_max_area = foreign_cfg.get("max_area", 2000)
        self.foreign_circularity_min = foreign_cfg.get("circularity_min", 0.2)
        self.foreign_circularity_max = foreign_cfg.get("circularity_max", 0.7)

        self.silica_blue_ratio_threshold = silica_cfg.get("blue_ratio_threshold", 0.1)
        self.silica_pink_ratio_threshold = silica_cfg.get("pink_ratio_threshold", 0.1)

        self.valve_angle_threshold = valve_cfg.get("angle_threshold", 30)
        self.valve_hough_threshold = valve_cfg.get("hough_threshold", 50)
        self.valve_min_line_length = valve_cfg.get("min_line_length", 30)
        self.valve_max_line_gap = valve_cfg.get("max_line_gap", 10)
        
    def detect_defects(self, roi_image: np.ndarray, roi_type: str, defect_filter: Optional[List[str]] = None) -> List[dict]:
        """
        检测缺陷

        Args:
            roi_image: ROI区域图像
            roi_type: ROI类型 (bushing/radiator/oil_level等)
            defect_filter: 只检测指定类型的缺陷，None表示检测全部

        Returns:
            检测结果列表
        """
        results = []

        # 转换为灰度图
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

        # 决定检测哪些类型
        check_types = defect_filter if defect_filter else self.defect_types

        # 检测油漏（深色斑点）
        if "oil_leak" in self.defect_types and "oil_leak" in check_types:
            oil_leak = self._detect_oil_leak(gray)
            if oil_leak and oil_leak["confidence"] >= self.confidence_threshold:
                results.append({
                    "type": "defect",
                    "label": "oil_leak",
                    "confidence": oil_leak["confidence"],
                    "bbox": oil_leak["bbox"]
                })
        
        # 检测锈蚀（纹理变化）
        if "rust" in self.defect_types and "rust" in check_types:
            rust = self._detect_rust(roi_image)
            if rust and rust["confidence"] >= self.confidence_threshold:
                results.append({
                    "type": "defect",
                    "label": "rust",
                    "confidence": rust["confidence"],
                    "bbox": rust["bbox"]
                })
        
        # 检测破损（边缘检测）
        if "damage" in self.defect_types and "damage" in check_types:
            damage = self._detect_damage(gray)
            if damage and damage["confidence"] >= self.confidence_threshold:
                results.append({
                    "type": "defect",
                    "label": "damage",
                    "confidence": damage["confidence"],
                    "bbox": damage["bbox"]
                })
        
        # 检测异物（轮廓检测）
        if "foreign_object" in self.defect_types and "foreign_object" in check_types:
            foreign = self._detect_foreign_object(gray)
            if foreign and foreign["confidence"] >= self.confidence_threshold:
                results.append({
                    "type": "defect",
                    "label": "foreign_object",
                    "confidence": foreign["confidence"],
                    "bbox": foreign["bbox"]
                })

        # 应用 NMS 非极大值抑制
        if len(results) > 1:
            results = self._apply_nms(results, self.nms_threshold)

        # 应用 max_detections 限制
        if len(results) > self.max_detections:
            results = sorted(results, key=lambda x: x["confidence"], reverse=True)[:self.max_detections]

        return results
    
    def detect_state(self, roi_image: np.ndarray, roi_type: str, state_filter: Optional[List[str]] = None) -> List[dict]:
        """
        检测状态

        Args:
            roi_image: ROI区域图像
            roi_type: ROI类型
            state_filter: 只检测指定类型的状态，None表示检测全部

        Returns:
            状态检测结果
        """
        results = []

        # 决定检测哪些状态
        check_silica = state_filter is None or "silica_gel" in state_filter
        check_valve = state_filter is None or "valve" in state_filter

        if roi_type in ["breather", "state"] and check_silica:
            if "silica_gel_normal" in self.state_types or "silica_gel_abnormal" in self.state_types:
                silica = self._detect_silica_gel_color(roi_image)
                if silica and silica["confidence"] >= self.confidence_threshold:
                    if silica["label"] in self.state_types:
                        results.append(silica)

        if roi_type in ["valve", "state"] and check_valve:
            if "valve_open" in self.state_types or "valve_closed" in self.state_types:
                valve = self._detect_valve_state(roi_image)
                if valve and valve["confidence"] >= self.confidence_threshold:
                    if valve["label"] in self.state_types:
                        results.append(valve)

        return results
    
    def _detect_oil_leak(self, gray: np.ndarray) -> Optional[dict]:
        """检测油漏（基于深色区域检测）"""
        # 阈值化找出深色区域
        _, binary = cv2.threshold(gray, self.oil_leak_gray_threshold, 255, cv2.THRESH_BINARY_INV)
        
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
            if area > self.oil_leak_min_area:  # 像素阈值
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
            
            if area > self.rust_min_area:
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
        edges = cv2.Canny(gray, self.damage_canny_low, self.damage_canny_high)
        
        # 计算边缘密度
        edge_density = np.sum(edges > 0) / edges.size
        
        # 如果边缘密度异常高，可能是破损
        if edge_density > self.damage_edge_density_threshold:
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
        edges = cv2.Canny(gray, self.foreign_canny_low, self.foreign_canny_high)
        
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
                if (
                    self.foreign_min_area < area < self.foreign_max_area
                    and self.foreign_circularity_min < circularity < self.foreign_circularity_max
                ):
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
        
        if blue_ratio > self.silica_blue_ratio_threshold:
            return {
                "type": "state",
                "label": "silica_gel_normal",
                "confidence": min(0.6 + blue_ratio, 0.95),
                "value": "normal",
                "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            }
        elif pink_ratio > self.silica_pink_ratio_threshold:
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
        
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            self.valve_hough_threshold,
            minLineLength=self.valve_min_line_length,
            maxLineGap=self.valve_max_line_gap,
        )
        
        if lines is not None and len(lines) > 0:
            # 分析主要线条方向
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            avg_angle = np.mean(angles)
            
            # 如果接近水平，视为开启；接近垂直，视为关闭
            if abs(avg_angle) < self.valve_angle_threshold or abs(avg_angle) > (180 - self.valve_angle_threshold):
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

    def extract_thermal_metrics(self, image: np.ndarray, thermal_config: Optional[dict] = None) -> dict:
        """提取热成像温度指标"""
        thermal_config = thermal_config or {}
        min_temp = float(thermal_config.get("min_temp", 20.0))
        max_temp = float(thermal_config.get("max_temp", 120.0))
        threshold = float(thermal_config.get("temperature_threshold", 80.0))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        temp_map = min_temp + (gray.astype(np.float32) / 255.0) * (max_temp - min_temp)

        min_value = float(np.min(temp_map))
        max_value = float(np.max(temp_map))
        avg_value = float(np.mean(temp_map))

        mask = temp_map >= threshold
        hotspot_ratio = float(np.sum(mask)) / float(mask.size) if mask.size else 0.0
        hotspot_bbox = None

        if mask.any():
            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                h_img, w_img = gray.shape
                hotspot_bbox = {
                    "x": x / w_img,
                    "y": y / h_img,
                    "width": w / w_img,
                    "height": h / h_img,
                }

        return {
            "min_temp": min_value,
            "max_temp": max_value,
            "avg_temp": avg_value,
            "threshold": threshold,
            "over_threshold": max_value >= threshold,
            "hotspot_ratio": hotspot_ratio,
            "hotspot_bbox": hotspot_bbox,
        }

    def _apply_nms(self, results: List[dict], threshold: float) -> List[dict]:
        """
        应用非极大值抑制 (NMS)

        Args:
            results: 检测结果列表
            threshold: IoU 阈值

        Returns:
            过滤后的结果列表
        """
        if not results:
            return results

        # 按置信度降序排序
        sorted_results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        keep = []

        while sorted_results:
            best = sorted_results.pop(0)
            keep.append(best)

            remaining = []
            for item in sorted_results:
                iou = self._compute_iou(best["bbox"], item["bbox"])
                if iou < threshold:
                    remaining.append(item)
            sorted_results = remaining

        return keep

    def _compute_iou(self, box1: dict, box2: dict) -> float:
        """
        计算两个边界框的 IoU

        Args:
            box1: 边界框1 {x, y, width, height}
            box2: 边界框2 {x, y, width, height}

        Returns:
            IoU 值
        """
        x1_min, y1_min = box1["x"], box1["y"]
        x1_max, y1_max = box1["x"] + box1["width"], box1["y"] + box1["height"]
        x2_min, y2_min = box2["x"], box2["y"]
        x2_max, y2_max = box2["x"] + box2["width"], box2["y"] + box2["height"]

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def read_oil_level(self, roi_image: np.ndarray) -> Optional[dict]:
        """
        读取油位计刻度

        基于边缘检测和水平线分析确定油位高度

        Args:
            roi_image: 油位计ROI区域图像

        Returns:
            油位读数结果，包含 value (0-100百分比) 和 confidence
        """
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 边缘检测
        edges = cv2.Canny(gray, 30, 100)

        # 使用霍夫直线检测水平线
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=w // 3,
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            # 备用方案：基于灰度梯度分析
            return self._read_oil_level_by_gradient(gray)

        # 分析水平线（油面线通常接近水平）
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # 接近水平的线（角度 < 15度）
            if angle < 15 or angle > 165:
                avg_y = (y1 + y2) / 2
                horizontal_lines.append(avg_y)

        if not horizontal_lines:
            return self._read_oil_level_by_gradient(gray)

        # 取中位数作为油面位置
        oil_surface_y = np.median(horizontal_lines)

        # 计算油位百分比（从底部向上）
        level_percent = (h - oil_surface_y) / h * 100
        level_percent = max(0, min(100, level_percent))

        return {
            "label": "oil_level_reading",
            "value": round(level_percent, 1),
            "confidence": 0.75,
            "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        }

    def _read_oil_level_by_gradient(self, gray: np.ndarray) -> Optional[dict]:
        """
        基于灰度梯度分析油位

        当霍夫直线检测失败时的备用方案
        """
        h, _ = gray.shape

        # 计算垂直方向的梯度
        gradient = np.abs(np.diff(gray.astype(np.float32), axis=0))

        # 对每行求和
        row_sums = np.sum(gradient, axis=1)

        # 找到梯度变化最大的行（可能是油面）
        if len(row_sums) == 0:
            return None

        max_gradient_row = np.argmax(row_sums)

        # 计算油位百分比
        level_percent = (h - max_gradient_row) / h * 100
        level_percent = max(0, min(100, level_percent))

        return {
            "label": "oil_level_reading",
            "value": round(level_percent, 1),
            "confidence": 0.55,  # 备用方案置信度较低
            "bbox": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        }
