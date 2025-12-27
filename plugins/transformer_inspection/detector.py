"""
主变自主巡视检测器 - 完整实现
输变电激光星芒破夜绘明监测平台 (A组)

功能范围:
- 外观缺陷识别: 破损、锈蚀、渗漏油、异物悬挂
- 状态识别: 呼吸器硅胶变色、阀门开闭状态
- 热成像集成: 红外图像温度提取
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class TransformerDetector:
    """
    主变压器检测器
    
    基于传统视觉算法实现缺陷检测和状态识别
    """
    
    def __init__(self, config: dict[str, Any]):
        """
        初始化检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.algorithms = config.get("algorithms", {})
        self.thermal_config = config.get("thermal", {})
        self.inference_config = config.get("inference", {})
        
        # 提取关键参数
        self.confidence_threshold = self.inference_config.get("confidence_threshold", 0.5)
        
    # ==================== 缺陷检测 ====================
    
    def detect_defects(self, image: np.ndarray, roi_type: str) -> list[dict[str, Any]]:
        """
        检测缺陷
        
        Args:
            image: BGR图像
            roi_type: ROI类型
            
        Returns:
            缺陷列表 [{"label": str, "bbox": dict, "confidence": float}, ...]
        """
        if cv2 is None:
            return []
            
        defects = []
        
        # 根据ROI类型选择检测方法
        if roi_type in ["radiator", "bushing", "body"]:
            # 渗漏油检测
            oil_leaks = self._detect_oil_leak(image)
            defects.extend(oil_leaks)
            
            # 锈蚀检测
            rust_defects = self._detect_rust(image)
            defects.extend(rust_defects)
            
        if roi_type in ["bushing", "terminal_box", "body"]:
            # 破损检测
            damage_defects = self._detect_damage(image)
            defects.extend(damage_defects)
            
        # 异物检测(所有ROI类型)
        foreign_objects = self._detect_foreign_object(image)
        defects.extend(foreign_objects)
        
        return defects
    
    def _detect_oil_leak(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        检测渗漏油 - 基于深色区域检测
        
        Args:
            image: BGR图像
            
        Returns:
            检测结果列表
        """
        params = self.algorithms.get("oil_leak", {})
        gray_threshold = params.get("gray_threshold", 60)
        min_area = params.get("min_area", 500)
        max_area = params.get("max_area", 50000)
        kernel_size = params.get("morphology_kernel", 5)
        
        results = []
        
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 阈值化检测深色区域
        _, binary = cv2.threshold(gray, gray_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学操作去噪
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # 计算置信度(基于面积和形状)
                aspect_ratio = float(cw) / ch if ch > 0 else 0
                confidence = min(0.9, 0.5 + (area / max_area) * 0.3 + (0.2 if 0.5 < aspect_ratio < 2.0 else 0))
                
                if confidence >= self.confidence_threshold:
                    results.append({
                        "label": "oil_leak",
                        "bbox": {
                            "x": x / w,
                            "y": y / h,
                            "width": cw / w,
                            "height": ch / h
                        },
                        "confidence": confidence,
                        "metadata": {"area": area, "aspect_ratio": aspect_ratio}
                    })
        
        return results
    
    def _detect_rust(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        检测锈蚀 - 基于HSV颜色空间检测橙红色区域
        
        Args:
            image: BGR图像
            
        Returns:
            检测结果列表
        """
        params = self.algorithms.get("rust", {})
        min_area = params.get("min_area", 300)
        max_area = params.get("max_area", 30000)
        hsv_lower = np.array(params.get("hsv_lower", [5, 50, 50]))
        hsv_upper = np.array(params.get("hsv_upper", [25, 255, 255]))
        
        results = []
        
        # 转HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 颜色掩码
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # 计算颜色纯度作为置信度
                roi_mask = mask[y:y+ch, x:x+cw]
                color_ratio = np.sum(roi_mask > 0) / (cw * ch) if cw * ch > 0 else 0
                confidence = min(0.9, 0.4 + color_ratio * 0.5)
                
                if confidence >= self.confidence_threshold:
                    results.append({
                        "label": "rust",
                        "bbox": {
                            "x": x / w,
                            "y": y / h,
                            "width": cw / w,
                            "height": ch / h
                        },
                        "confidence": confidence,
                        "metadata": {"area": area, "color_ratio": color_ratio}
                    })
        
        return results
    
    def _detect_damage(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        检测破损 - 基于边缘密度分析
        
        Args:
            image: BGR图像
            
        Returns:
            检测结果列表
        """
        params = self.algorithms.get("damage", {})
        canny_low = params.get("canny_low", 50)
        canny_high = params.get("canny_high", 150)
        density_threshold = params.get("edge_density_threshold", 0.15)
        block_size = params.get("block_size", 32)
        
        results = []
        
        # 转灰度并边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        h, w = image.shape[:2]
        
        # 分块分析边缘密度
        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = edges[y:y+block_size, x:x+block_size]
                density = np.sum(block > 0) / (block_size * block_size)
                
                if density > density_threshold:
                    confidence = min(0.85, 0.5 + (density - density_threshold) * 2)
                    
                    if confidence >= self.confidence_threshold:
                        results.append({
                            "label": "damage",
                            "bbox": {
                                "x": x / w,
                                "y": y / h,
                                "width": block_size / w,
                                "height": block_size / h
                            },
                            "confidence": confidence,
                            "metadata": {"edge_density": density}
                        })
        
        # 合并相邻检测结果
        results = self._merge_nearby_detections(results, iou_threshold=0.3)
        
        return results
    
    def _detect_foreign_object(self, image: np.ndarray) -> list[dict[str, Any]]:
        """
        检测异物 - 基于轮廓形状分析
        
        Args:
            image: BGR图像
            
        Returns:
            检测结果列表
        """
        params = self.algorithms.get("foreign_object", {})
        canny_low = params.get("canny_low", 30)
        canny_high = params.get("canny_high", 100)
        min_area = params.get("min_area", 200)
        max_area = params.get("max_area", 2000)
        circularity_min = params.get("circularity_min", 0.2)
        circularity_max = params.get("circularity_max", 0.7)
        
        results = []
        
        # 边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
        
        # 膨胀连接边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    if circularity_min < circularity < circularity_max:
                        x, y, cw, ch = cv2.boundingRect(contour)
                        
                        confidence = min(0.8, 0.5 + (1 - abs(circularity - 0.5)) * 0.3)
                        
                        if confidence >= self.confidence_threshold:
                            results.append({
                                "label": "foreign_object",
                                "bbox": {
                                    "x": x / w,
                                    "y": y / h,
                                    "width": cw / w,
                                    "height": ch / h
                                },
                                "confidence": confidence,
                                "metadata": {"area": area, "circularity": circularity}
                            })
        
        return results
    
    # ==================== 状态识别 ====================
    
    def recognize_state(self, image: np.ndarray, roi_type: str) -> dict[str, Any]:
        """
        识别设备状态
        
        Args:
            image: BGR图像
            roi_type: ROI类型
            
        Returns:
            状态识别结果
        """
        if cv2 is None:
            return {"label": "unknown", "confidence": 0.0}
            
        if roi_type == "breather":
            return self._recognize_silica_gel(image)
        elif roi_type == "valve":
            return self._recognize_valve_state(image)
        elif roi_type == "oil_level":
            return self._read_oil_level(image)
        else:
            return {"label": "normal", "confidence": 0.6}
    
    def _recognize_silica_gel(self, image: np.ndarray) -> dict[str, Any]:
        """
        识别呼吸器硅胶颜色状态
        
        正常: 蓝色
        异常: 粉红色(吸湿后变色)
        
        Args:
            image: BGR图像
            
        Returns:
            状态识别结果
        """
        params = self.algorithms.get("silica_gel", {})
        blue_threshold = params.get("blue_ratio_threshold", 0.1)
        pink_threshold = params.get("pink_ratio_threshold", 0.1)
        
        hsv_blue_lower = np.array(params.get("hsv_blue_lower", [100, 50, 50]))
        hsv_blue_upper = np.array(params.get("hsv_blue_upper", [130, 255, 255]))
        hsv_pink_lower = np.array(params.get("hsv_pink_lower", [140, 30, 100]))
        hsv_pink_upper = np.array(params.get("hsv_pink_upper", [170, 255, 255]))
        
        # 转HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 蓝色掩码
        blue_mask = cv2.inRange(hsv, hsv_blue_lower, hsv_blue_upper)
        blue_ratio = np.sum(blue_mask > 0) / (image.shape[0] * image.shape[1])
        
        # 粉红色掩码
        pink_mask = cv2.inRange(hsv, hsv_pink_lower, hsv_pink_upper)
        pink_ratio = np.sum(pink_mask > 0) / (image.shape[0] * image.shape[1])
        
        # 判断状态
        if blue_ratio > blue_threshold and blue_ratio > pink_ratio:
            return {
                "label": "silica_gel_normal",
                "confidence": min(0.95, 0.6 + blue_ratio),
                "metadata": {"blue_ratio": blue_ratio, "pink_ratio": pink_ratio}
            }
        elif pink_ratio > pink_threshold:
            return {
                "label": "silica_gel_abnormal",
                "confidence": min(0.95, 0.6 + pink_ratio),
                "metadata": {"blue_ratio": blue_ratio, "pink_ratio": pink_ratio}
            }
        else:
            return {
                "label": "silica_gel_unknown",
                "confidence": 0.4,
                "metadata": {"blue_ratio": blue_ratio, "pink_ratio": pink_ratio}
            }
    
    def _recognize_valve_state(self, image: np.ndarray) -> dict[str, Any]:
        """
        识别阀门开闭状态 - 基于霍夫直线检测
        
        Args:
            image: BGR图像
            
        Returns:
            状态识别结果
        """
        params = self.algorithms.get("valve", {})
        angle_threshold = params.get("angle_threshold", 30)
        hough_threshold = params.get("hough_threshold", 50)
        min_line_length = params.get("min_line_length", 30)
        max_line_gap = params.get("max_line_gap", 10)
        
        # 边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold,
                                minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        if lines is None or len(lines) == 0:
            return {"label": "valve_unknown", "confidence": 0.3}
        
        # 计算主要线段角度
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        # 取主要角度
        main_angle = np.median(angles)
        
        # 判断状态
        if abs(main_angle) < angle_threshold or abs(abs(main_angle) - 180) < angle_threshold:
            # 接近水平 - 开启
            return {
                "label": "valve_open",
                "confidence": min(0.9, 0.7 + (angle_threshold - abs(main_angle)) / angle_threshold * 0.2),
                "metadata": {"main_angle": main_angle}
            }
        elif abs(abs(main_angle) - 90) < angle_threshold:
            # 接近垂直 - 关闭
            return {
                "label": "valve_closed",
                "confidence": min(0.9, 0.7 + (angle_threshold - abs(abs(main_angle) - 90)) / angle_threshold * 0.2),
                "metadata": {"main_angle": main_angle}
            }
        else:
            return {
                "label": "valve_intermediate",
                "confidence": 0.5,
                "metadata": {"main_angle": main_angle}
            }
    
    def _read_oil_level(self, image: np.ndarray) -> dict[str, Any]:
        """
        读取油位计刻度
        
        Args:
            image: BGR图像
            
        Returns:
            读数结果
        """
        oil_level_config = self.config.get("oil_level", {})
        min_line_ratio = oil_level_config.get("min_line_length_ratio", 0.33)
        hough_threshold = oil_level_config.get("hough_threshold", 30)
        angle_tolerance = oil_level_config.get("angle_tolerance", 15)
        
        h, w = image.shape[:2]
        min_line_length = int(w * min_line_ratio)
        
        # 边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold,
                                minLineLength=min_line_length, maxLineGap=10)
        
        if lines is None:
            return {"label": "oil_level_reading", "value": None, "confidence": 0.3}
        
        # 查找水平线(油位线)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < angle_tolerance or angle > (180 - angle_tolerance):
                horizontal_lines.append((y1 + y2) / 2)
        
        if not horizontal_lines:
            return {"label": "oil_level_reading", "value": None, "confidence": 0.3}
        
        # 取最显著的水平线位置
        oil_level_y = np.median(horizontal_lines)
        
        # 转换为百分比(假设顶部为100%,底部为0%)
        level_percent = (1 - oil_level_y / h) * 100
        level_percent = max(0, min(100, level_percent))
        
        return {
            "label": "oil_level_reading",
            "value": round(level_percent, 1),
            "confidence": 0.7,
            "metadata": {"y_position": oil_level_y, "image_height": h}
        }
    
    # ==================== 热成像处理 ====================
    
    def analyze_thermal(self, thermal_image: np.ndarray) -> dict[str, Any]:
        """
        分析热成像图像
        
        Args:
            thermal_image: 热成像图像(灰度或伪彩色)
            
        Returns:
            温度分析结果
        """
        if cv2 is None:
            return {"enabled": False}
            
        if not self.thermal_config.get("enabled", False):
            return {"enabled": False}
        
        min_temp = self.thermal_config.get("min_temp", 20.0)
        max_temp = self.thermal_config.get("max_temp", 120.0)
        temp_threshold = self.thermal_config.get("temperature_threshold", 80.0)
        hotspot_threshold = self.thermal_config.get("hotspot_threshold", 0.8)
        
        # 转灰度
        if len(thermal_image.shape) == 3:
            gray = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = thermal_image
        
        # 灰度映射到温度
        gray_float = gray.astype(np.float32) / 255.0
        temp_map = min_temp + gray_float * (max_temp - min_temp)
        
        # 统计温度
        max_temp_value = float(np.max(temp_map))
        min_temp_value = float(np.min(temp_map))
        avg_temp_value = float(np.mean(temp_map))
        
        # 热点检测
        hotspot_mask = temp_map > (min_temp + (max_temp - min_temp) * hotspot_threshold)
        hotspot_ratio = np.sum(hotspot_mask) / hotspot_mask.size
        
        # 查找热点位置
        hotspots = []
        if hotspot_ratio > 0:
            contours, _ = cv2.findContours(hotspot_mask.astype(np.uint8) * 255,
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = gray.shape
            for contour in contours[:5]:  # 最多5个热点
                x, y, cw, ch = cv2.boundingRect(contour)
                hotspots.append({
                    "x": x / w,
                    "y": y / h,
                    "width": cw / w,
                    "height": ch / h,
                    "max_temp": float(np.max(temp_map[y:y+ch, x:x+cw]))
                })
        
        # 判断是否超温告警
        is_overtemp = max_temp_value > temp_threshold
        
        return {
            "enabled": True,
            "max_temp": max_temp_value,
            "min_temp": min_temp_value,
            "avg_temp": avg_temp_value,
            "is_overtemp": is_overtemp,
            "threshold": temp_threshold,
            "hotspots": hotspots,
            "hotspot_ratio": hotspot_ratio
        }
    
    # ==================== 辅助函数 ====================
    
    def _merge_nearby_detections(self, detections: list[dict], iou_threshold: float = 0.3) -> list[dict]:
        """
        合并相邻的检测结果
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            合并后的检测结果
        """
        if len(detections) <= 1:
            return detections
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        merged = []
        used = set()
        
        for i, det in enumerate(detections):
            if i in used:
                continue
            
            # 查找可合并的检测
            group = [det]
            for j, other in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                if self._compute_iou(det["bbox"], other["bbox"]) > iou_threshold:
                    group.append(other)
                    used.add(j)
            
            # 合并组内检测
            if len(group) > 1:
                merged_bbox = self._merge_bboxes([d["bbox"] for d in group])
                merged_conf = max(d["confidence"] for d in group)
                det = {
                    "label": det["label"],
                    "bbox": merged_bbox,
                    "confidence": merged_conf,
                    "metadata": {"merged_count": len(group)}
                }
            
            merged.append(det)
            used.add(i)
        
        return merged
    
    def _compute_iou(self, bbox1: dict, bbox2: dict) -> float:
        """计算两个bbox的IoU"""
        x1 = max(bbox1["x"], bbox2["x"])
        y1 = max(bbox1["y"], bbox2["y"])
        x2 = min(bbox1["x"] + bbox1["width"], bbox2["x"] + bbox2["width"])
        y2 = min(bbox1["y"] + bbox1["height"], bbox2["y"] + bbox2["height"])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1["width"] * bbox1["height"]
        area2 = bbox2["width"] * bbox2["height"]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_bboxes(self, bboxes: list[dict]) -> dict:
        """合并多个bbox"""
        x_min = min(b["x"] for b in bboxes)
        y_min = min(b["y"] for b in bboxes)
        x_max = max(b["x"] + b["width"] for b in bboxes)
        y_max = max(b["y"] + b["height"] for b in bboxes)
        
        return {
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min
        }