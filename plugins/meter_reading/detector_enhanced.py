"""
表计读数检测器 - 增强版
输变电激光监测平台 (E组) - 全自动AI巡检改造

增强功能:
- HRNet关键点检测: 表盘圆心/指针/刻度精确定位
- 透视矫正: 基于关键点的图像变换
- 增强指针检测: 改进霍夫变换
- CRNN数字OCR: 数字表和七段码识别
- 自动量程识别: 文本OCR识别表盘标记
- 失败兜底策略: 多次重试+人工复核标记
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import time
import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class MeterType(Enum):
    """表计类型"""
    PRESSURE_GAUGE = "pressure_gauge"       # 压强表
    TEMPERATURE_GAUGE = "temperature_gauge" # 温度表
    OIL_LEVEL_GAUGE = "oil_level_gauge"     # 油位表
    SF6_DENSITY_GAUGE = "sf6_density_gauge" # SF6密度表
    DIGITAL_DISPLAY = "digital_display"     # 数字显示屏
    LED_INDICATOR = "led_indicator"         # LED指示灯
    AMMETER = "ammeter"                     # 电流表
    VOLTMETER = "voltmeter"                 # 电压表
    SEVEN_SEGMENT = "seven_segment"         # 七段码显示


class ReadingStatus(Enum):
    """读数状态"""
    SUCCESS = "success"                     # 成功
    LOW_CONFIDENCE = "low_confidence"       # 低置信度
    FAILED = "failed"                       # 失败
    NEED_MANUAL_REVIEW = "need_manual_review"  # 需人工复核


@dataclass
class Keypoint:
    """关键点"""
    name: str
    x: float                                # 归一化x坐标
    y: float                                # 归一化y坐标
    confidence: float
    visible: bool = True


@dataclass
class MeterKeypoints:
    """表计关键点集合"""
    center: Optional[Keypoint] = None       # 圆心
    pointer_tip: Optional[Keypoint] = None  # 指针尖端
    pointer_base: Optional[Keypoint] = None # 指针基部
    scale_min: Optional[Keypoint] = None    # 最小刻度
    scale_max: Optional[Keypoint] = None    # 最大刻度
    dial_corners: List[Keypoint] = field(default_factory=list)  # 表盘四角


@dataclass
class MeterReading:
    """表计读数结果"""
    meter_type: MeterType
    value: Optional[float] = None           # 读数值
    unit: str = ""                          # 单位
    confidence: float = 0.0
    status: ReadingStatus = ReadingStatus.FAILED
    keypoints: Optional[MeterKeypoints] = None
    pointer_angle: Optional[float] = None   # 指针角度
    range_min: Optional[float] = None       # 量程最小值
    range_max: Optional[float] = None       # 量程最大值
    need_manual_review: bool = False
    failure_reason: str = ""
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DigitalReading:
    """数字读数结果"""
    text: str
    value: Optional[float] = None
    confidence: float = 0.0
    bbox: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeterInspectionResult:
    """表计巡视综合结果"""
    readings: List[MeterReading] = field(default_factory=list)
    digital_readings: List[DigitalReading] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_version: str = ""
    code_hash: str = ""


class MeterReadingDetectorEnhanced:
    """
    表计读数增强检测器
    
    支持多种表计类型的精确读数
    """
    
    # 模型ID映射
    MODEL_IDS = {
        "keypoint": "hrnet_meter_keypoint",     # HRNet关键点
        "ocr": "crnn_meter_ocr",                # CRNN OCR
        "classifier": "meter_type_classifier",  # 表计分类
    }
    
    # 默认配置
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    DEFAULT_MANUAL_REVIEW_THRESHOLD = 0.5
    DEFAULT_MAX_ROTATION = 45               # 最大旋转角度
    DEFAULT_MAX_RETRY = 3
    
    # 表计量程预设
    METER_RANGES = {
        MeterType.PRESSURE_GAUGE: {"min": 0, "max": 1.6, "unit": "MPa"},
        MeterType.TEMPERATURE_GAUGE: {"min": -20, "max": 100, "unit": "°C"},
        MeterType.OIL_LEVEL_GAUGE: {"min": 0, "max": 100, "unit": "%"},
        MeterType.SF6_DENSITY_GAUGE: {"min": 0, "max": 0.8, "unit": "MPa"},
        MeterType.AMMETER: {"min": 0, "max": 100, "unit": "A"},
        MeterType.VOLTMETER: {"min": 0, "max": 500, "unit": "V"},
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
        self._confidence_threshold = config.get("confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
        self._manual_review_threshold = config.get("manual_review_threshold", self.DEFAULT_MANUAL_REVIEW_THRESHOLD)
        self._max_rotation = config.get("max_rotation_angle", self.DEFAULT_MAX_ROTATION)
        self._max_retry = config.get("max_retry", self.DEFAULT_MAX_RETRY)
        self._use_deep_learning = config.get("use_deep_learning", True)
        self._perspective_correction = config.get("perspective_correction", True)
        
        # 版本信息
        self._model_version = "meter_enhanced_v1.0"
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
                        print(f"[MeterDetector] 模型 {model_id} 加载失败: {e}")
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"[MeterDetector] 初始化失败: {e}")
            return False
    
    def read_meter(
        self,
        image: np.ndarray,
        meter_type: MeterType,
        roi_bbox: Optional[Dict[str, float]] = None,
        roi_id: str = "",
    ) -> MeterReading:
        """
        读取表计
        
        Args:
            image: BGR图像
            meter_type: 表计类型
            roi_bbox: ROI区域
            roi_id: ROI ID
            
        Returns:
            读数结果
        """
        # 裁剪ROI
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)
        
        # 根据类型选择方法
        if meter_type in [MeterType.DIGITAL_DISPLAY, MeterType.SEVEN_SEGMENT]:
            return self._read_digital_meter(image, meter_type, roi_id)
        elif meter_type == MeterType.LED_INDICATOR:
            return self._read_led_indicator(image, roi_id)
        else:
            return self._read_analog_meter(image, meter_type, roi_id)
    
    def _read_analog_meter(
        self,
        image: np.ndarray,
        meter_type: MeterType,
        roi_id: str = "",
    ) -> MeterReading:
        """读取模拟表计"""
        result = MeterReading(
            meter_type=meter_type,
            status=ReadingStatus.FAILED,
        )
        
        retry_count = 0
        while retry_count < self._max_retry:
            try:
                # 1. 检测关键点
                keypoints = self._detect_keypoints(image)
                result.keypoints = keypoints
                
                # 2. 透视矫正
                corrected_image = image
                if self._perspective_correction and keypoints.dial_corners:
                    corrected_image = self._apply_perspective_correction(image, keypoints)
                    # 重新检测关键点
                    keypoints = self._detect_keypoints(corrected_image)
                    result.keypoints = keypoints
                
                # 3. 检测指针角度
                pointer_angle = self._detect_pointer_angle(corrected_image, keypoints)
                result.pointer_angle = pointer_angle
                
                if pointer_angle is None:
                    retry_count += 1
                    continue
                
                # 4. 识别量程
                range_info = self._recognize_range(corrected_image, meter_type)
                result.range_min = range_info.get("min")
                result.range_max = range_info.get("max")
                result.unit = range_info.get("unit", "")
                
                # 5. 计算读数
                value, confidence = self._calculate_reading(
                    pointer_angle,
                    result.range_min,
                    result.range_max,
                )
                
                result.value = value
                result.confidence = confidence
                result.retry_count = retry_count
                
                # 判断状态
                if confidence >= self._confidence_threshold:
                    result.status = ReadingStatus.SUCCESS
                elif confidence >= self._manual_review_threshold:
                    result.status = ReadingStatus.LOW_CONFIDENCE
                    result.need_manual_review = True
                else:
                    result.status = ReadingStatus.NEED_MANUAL_REVIEW
                    result.need_manual_review = True
                
                break
                
            except Exception as e:
                result.failure_reason = str(e)
                retry_count += 1
        
        result.retry_count = retry_count
        result.metadata["roi_id"] = roi_id
        
        return result
    
    def _detect_keypoints(self, image: np.ndarray) -> MeterKeypoints:
        """检测关键点"""
        keypoints = MeterKeypoints()
        
        # 优先使用深度学习
        if self._use_deep_learning and self._model_registry:
            dl_keypoints = self._detect_keypoints_dl(image)
            if dl_keypoints:
                return dl_keypoints
        
        # 回退到传统方法
        return self._detect_keypoints_traditional(image)
    
    def _detect_keypoints_dl(self, image: np.ndarray) -> Optional[MeterKeypoints]:
        """深度学习关键点检测"""
        try:
            model_id = self.MODEL_IDS["keypoint"]
            result = self._model_registry.infer(model_id, image)  # type: ignore[union-attr]
            
            if result.raw_outputs:
                # 解析HRNet输出
                heatmaps = result.raw_outputs.get("heatmaps")
                if heatmaps is not None:
                    return self._parse_heatmaps(heatmaps)
            
            # 如果有检测结果
            if result.detections:
                keypoints = MeterKeypoints()
                for det in result.detections:
                    kp_type = det.get("class_name", "")
                    x, y = det.get("x", 0.5), det.get("y", 0.5)
                    conf = det.get("confidence", 0)
                    
                    kp = Keypoint(name=kp_type, x=x, y=y, confidence=conf)
                    
                    if kp_type == "center":
                        keypoints.center = kp
                    elif kp_type == "pointer_tip":
                        keypoints.pointer_tip = kp
                    elif kp_type == "pointer_base":
                        keypoints.pointer_base = kp
                    elif kp_type == "scale_min":
                        keypoints.scale_min = kp
                    elif kp_type == "scale_max":
                        keypoints.scale_max = kp
                
                return keypoints
        except Exception as e:
            print(f"[MeterDetector] 深度学习关键点检测失败: {e}")
        
        return None
    
    def _parse_heatmaps(self, heatmaps: np.ndarray) -> MeterKeypoints:
        """解析热力图"""
        keypoints = MeterKeypoints()
        
        # 假设热力图顺序: center, pointer_tip, pointer_base, scale_min, scale_max
        kp_names = ["center", "pointer_tip", "pointer_base", "scale_min", "scale_max"]
        
        for i, name in enumerate(kp_names):
            if i < heatmaps.shape[0]:
                hm = heatmaps[i]
                
                # 找到最大值位置
                max_idx = np.unravel_index(np.argmax(hm), hm.shape)
                y, x = max_idx
                h, w = hm.shape
                
                kp = Keypoint(
                    name=name,
                    x=x / w,
                    y=y / h,
                    confidence=float(hm[max_idx]),
                )
                
                if name == "center":
                    keypoints.center = kp
                elif name == "pointer_tip":
                    keypoints.pointer_tip = kp
                elif name == "pointer_base":
                    keypoints.pointer_base = kp
                elif name == "scale_min":
                    keypoints.scale_min = kp
                elif name == "scale_max":
                    keypoints.scale_max = kp
        
        return keypoints
    
    def _detect_keypoints_traditional(self, image: np.ndarray) -> MeterKeypoints:
        """传统方法关键点检测"""
        if cv2 is None:
            return MeterKeypoints()
        
        keypoints = MeterKeypoints()
        h, w = image.shape[:2]
        
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 检测圆形表盘
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 50,
            param1=50, param2=30, minRadius=50, maxRadius=min(h, w) // 2
        )
        
        if circles is not None:
            # 取最大的圆作为表盘
            circles_rounded = np.uint16(np.around(circles))
            circles_array = circles_rounded[0]  # type: ignore[index]
            best_circle = max(circles_array, key=lambda c: c[2])
            cx, cy, r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])
            
            keypoints.center = Keypoint(
                name="center",
                x=cx / w,
                y=cy / h,
                confidence=0.8,
            )
            
            # 假设表盘边界
            keypoints.dial_corners = [
                Keypoint("corner_tl", max(0, cx-r)/w, max(0, cy-r)/h, 0.7),
                Keypoint("corner_tr", min(w, cx+r)/w, max(0, cy-r)/h, 0.7),
                Keypoint("corner_br", min(w, cx+r)/w, min(h, cy+r)/h, 0.7),
                Keypoint("corner_bl", max(0, cx-r)/w, min(h, cy+r)/h, 0.7),
            ]
        
        return keypoints
    
    def _apply_perspective_correction(
        self,
        image: np.ndarray,
        keypoints: MeterKeypoints
    ) -> np.ndarray:
        """应用透视矫正"""
        if cv2 is None or len(keypoints.dial_corners) < 4:
            return image
        
        h, w = image.shape[:2]
        
        # 源点
        src_points = np.array([
            [keypoints.dial_corners[0].x * w, keypoints.dial_corners[0].y * h],
            [keypoints.dial_corners[1].x * w, keypoints.dial_corners[1].y * h],
            [keypoints.dial_corners[2].x * w, keypoints.dial_corners[2].y * h],
            [keypoints.dial_corners[3].x * w, keypoints.dial_corners[3].y * h],
        ], dtype=np.float32)

        # 目标点(正方形)
        size = min(h, w)
        dst_points = np.array([
            [0, 0],
            [size, 0],
            [size, size],
            [0, size],
        ], dtype=np.float32)
        
        # 计算变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用变换
        corrected = cv2.warpPerspective(image, M, (size, size))
        
        return corrected
    
    def _detect_pointer_angle(
        self,
        image: np.ndarray,
        keypoints: MeterKeypoints
    ) -> Optional[float]:
        """检测指针角度"""
        # 如果有关键点
        if keypoints.center and keypoints.pointer_tip:
            cx = keypoints.center.x
            cy = keypoints.center.y
            px = keypoints.pointer_tip.x
            py = keypoints.pointer_tip.y
            
            # 计算角度
            angle = math.atan2(py - cy, px - cx) * 180 / math.pi
            return angle
        
        # 回退到霍夫线检测
        return self._detect_pointer_by_hough(image, keypoints)
    
    def _detect_pointer_by_hough(
        self,
        image: np.ndarray,
        keypoints: MeterKeypoints
    ) -> Optional[float]:
        """霍夫变换检测指针"""
        if cv2 is None:
            return None
        
        h, w = image.shape[:2]
        
        # 获取圆心
        if keypoints.center:
            cx = int(keypoints.center.x * w)
            cy = int(keypoints.center.y * h)
        else:
            cx, cy = w // 2, h // 2
        
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return None
        
        # 找穿过圆心的线
        best_line = None
        min_dist = float('inf')
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算线到圆心的距离
            dist = abs((y2-y1)*cx - (x2-x1)*cy + x2*y1 - y2*x1) / (
                math.sqrt((y2-y1)**2 + (x2-x1)**2) + 1e-6
            )
            
            if dist < min_dist and dist < 20:  # 距离阈值
                min_dist = dist
                best_line = (x1, y1, x2, y2)
        
        if best_line is None:
            return None
        
        x1, y1, x2, y2 = best_line
        
        # 确定指针方向(远离圆心的端点)
        d1 = math.sqrt((x1-cx)**2 + (y1-cy)**2)
        d2 = math.sqrt((x2-cx)**2 + (y2-cy)**2)
        
        if d1 > d2:
            px, py = x1, y1
        else:
            px, py = x2, y2
        
        # 计算角度
        angle = math.atan2(py - cy, px - cx) * 180 / math.pi
        
        return angle
    
    def _recognize_range(
        self,
        image: np.ndarray,
        meter_type: MeterType
    ) -> Dict[str, Any]:
        """识别量程"""
        # 使用预设量程
        if meter_type in self.METER_RANGES:
            return self.METER_RANGES[meter_type].copy()
        
        # 尝试OCR识别
        if self._use_deep_learning and self._model_registry:
            ocr_result = self._ocr_range_text(image)
            if ocr_result:
                return ocr_result
        
        # 默认值
        return {"min": 0, "max": 100, "unit": ""}
    
    def _ocr_range_text(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """OCR识别量程文字"""
        try:
            model_id = self.MODEL_IDS["ocr"]
            result = self._model_registry.infer(model_id, image)  # type: ignore[union-attr]
            
            if result.detections:
                texts = [d.get("text", "") for d in result.detections]
                
                # 解析数字
                numbers = []
                unit = ""
                
                for text in texts:
                    # 提取数字
                    import re
                    nums = re.findall(r'[-+]?\d*\.?\d+', text)
                    numbers.extend([float(n) for n in nums])
                    
                    # 提取单位
                    if "MPa" in text or "mpa" in text:
                        unit = "MPa"
                    elif "°C" in text or "℃" in text:
                        unit = "°C"
                    elif "%" in text:
                        unit = "%"
                
                if len(numbers) >= 2:
                    return {
                        "min": min(numbers),
                        "max": max(numbers),
                        "unit": unit
                    }
        except Exception as e:
            print(f"[MeterDetector] OCR量程识别失败: {e}")
        
        return None
    
    def _calculate_reading(
        self,
        angle: float,
        range_min: Optional[float],
        range_max: Optional[float],
    ) -> Tuple[float, float]:
        """计算读数值"""
        if range_min is None:
            range_min = 0
        if range_max is None:
            range_max = 100
        
        # 假设表盘范围: -135° 到 135° (通用指针表)
        min_angle = -135
        max_angle = 135
        
        # 归一化角度到[-180, 180]
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        
        # 计算比例
        if max_angle != min_angle:
            ratio = (angle - min_angle) / (max_angle - min_angle)
        else:
            ratio = 0.5
        
        ratio = max(0, min(1, ratio))
        
        # 计算读数
        value = range_min + ratio * (range_max - range_min)
        
        # 置信度基于角度的有效性
        confidence = 0.9 if min_angle <= angle <= max_angle else 0.6
        
        return value, confidence
    
    def _read_digital_meter(
        self,
        image: np.ndarray,
        meter_type: MeterType,
        roi_id: str = "",
    ) -> MeterReading:
        """读取数字表计"""
        result = MeterReading(
            meter_type=meter_type,
            status=ReadingStatus.FAILED,
        )
        
        # OCR识别
        text, confidence = self._ocr_digital(image)
        
        if text:
            # 解析数值
            import re
            nums = re.findall(r'[-+]?\d*\.?\d+', text)
            
            if nums:
                result.value = float(nums[0])
                result.confidence = confidence
                
                if confidence >= self._confidence_threshold:
                    result.status = ReadingStatus.SUCCESS
                elif confidence >= self._manual_review_threshold:
                    result.status = ReadingStatus.LOW_CONFIDENCE
                    result.need_manual_review = True
                else:
                    result.status = ReadingStatus.NEED_MANUAL_REVIEW
                    result.need_manual_review = True
        
        result.metadata["roi_id"] = roi_id
        result.metadata["raw_text"] = text
        
        return result
    
    def _ocr_digital(self, image: np.ndarray) -> Tuple[str, float]:
        """OCR识别数字"""
        # 优先使用深度学习
        if self._use_deep_learning and self._model_registry:
            try:
                model_id = self.MODEL_IDS["ocr"]
                result = self._model_registry.infer(model_id, image)  # type: ignore[union-attr]
                
                if result.detections:
                    texts = [d.get("text", "") for d in result.detections]
                    confidences = [d.get("confidence", 0) for d in result.detections]
                    
                    combined_text = "".join(texts)
                    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

                    return combined_text, avg_confidence
            except Exception as e:
                print(f"[MeterDetector] OCR识别失败: {e}")
        
        # 回退到传统方法
        return self._ocr_traditional(image)
    
    def _ocr_traditional(self, image: np.ndarray) -> Tuple[str, float]:
        """传统OCR方法"""
        if cv2 is None:
            return "", 0.0
        
        # 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 这里应该调用OCR引擎
        # 简化返回空
        return "", 0.0
    
    def _read_led_indicator(
        self,
        image: np.ndarray,
        roi_id: str = "",
    ) -> MeterReading:
        """读取LED指示灯"""
        result = MeterReading(
            meter_type=MeterType.LED_INDICATOR,
            status=ReadingStatus.FAILED,
        )
        
        if cv2 is None:
            return result
        
        # 分析颜色
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测亮度
        brightness = np.mean(hsv[:, :, 2])
        
        # 检测主要颜色
        hue = np.mean(hsv[:, :, 0])
        
        # 判断状态
        if brightness > 100:  # 亮
            if 0 <= hue < 15 or hue > 165:  # 红色
                result.value = 1  # 红灯亮
                result.unit = "red"
            elif 35 < hue < 85:  # 绿色
                result.value = 2  # 绿灯亮
                result.unit = "green"
            elif 15 < hue < 35:  # 黄色
                result.value = 3  # 黄灯亮
                result.unit = "yellow"
            else:
                result.value = 4  # 其他颜色亮
                result.unit = "other"

            result.confidence = float(min(0.9, brightness / 200))
            result.status = ReadingStatus.SUCCESS
        else:
            result.value = 0  # 灯灭
            result.unit = "off"
            result.confidence = 0.8
            result.status = ReadingStatus.SUCCESS
        
        result.metadata["roi_id"] = roi_id
        result.metadata["brightness"] = brightness
        result.metadata["hue"] = hue
        
        return result
    
    def read_batch(
        self,
        image: np.ndarray,
        rois: List[Dict[str, Any]],
    ) -> List[MeterReading]:
        """
        批量读取
        
        Args:
            image: BGR图像
            rois: ROI列表
            
        Returns:
            读数结果列表
        """
        results = []
        
        for roi in rois:
            roi_id = roi.get("id", "")
            roi_type = roi.get("type", "pressure_gauge")
            bbox = roi.get("bbox")
            
            # 解析表计类型
            try:
                meter_type = MeterType(roi_type)
            except ValueError:
                meter_type = MeterType.PRESSURE_GAUGE
            
            # 读取
            reading = self.read_meter(image, meter_type, bbox, roi_id)
            results.append(reading)
        
        return results
    
    def inspect(
        self,
        image: np.ndarray,
        rois: Optional[List[Dict[str, Any]]] = None,
    ) -> MeterInspectionResult:
        """
        综合巡视
        
        Args:
            image: BGR图像
            rois: ROI列表
            
        Returns:
            综合巡视结果
        """
        start_time = time.perf_counter()
        
        readings = []
        digital_readings = []
        
        if rois:
            readings = self.read_batch(image, rois)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return MeterInspectionResult(
            readings=readings,
            digital_readings=digital_readings,
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
def create_detector(config: Dict[str, Any], model_registry=None) -> MeterReadingDetectorEnhanced:
    """创建检测器实例"""
    detector = MeterReadingDetectorEnhanced(config, model_registry)
    detector.initialize()
    return detector