"""
开关间隔自主巡视插件 - 检测器实现 (B组)

核心功能:
1. 分合位状态识别 (断路器/隔离开关/接地开关)
2. 清晰度评价 (Laplacian方差)
3. 多证据融合 (OCR + 颜色 + 角度)
4. 互锁逻辑校验
5. SF6表计读数 (升级功能)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Protocol
import numpy as np
import cv2
import hashlib
import math

# ============================================================
# 数据类定义
# ============================================================

@dataclass
class StateEvidence:
    """状态识别证据"""
    ocr_text: Optional[str] = None
    ocr_confidence: float = 0.0
    red_ratio: float = 0.0
    green_ratio: float = 0.0
    angle_deg: Optional[float] = None
    clarity_score: float = 0.0


@dataclass
class StateResult:
    """状态识别结果"""
    state: str  # open/closed/intermediate/unknown
    confidence: float
    evidence: StateEvidence
    reason_code: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GaugeResult:
    """表计读数结果"""
    value: Optional[float] = None
    unit: str = ""
    confidence: float = 0.0
    reason_code: Optional[int] = None


# ============================================================
# 清晰度评价
# ============================================================

class ClarityEvaluator:
    """图像清晰度评价器"""
    
    def __init__(self, mu: float = 120.0, tau: float = 30.0, min_score: float = 0.70):
        """
        Args:
            mu: 可接受清晰度阈值（Laplacian方差基准值）
            tau: Sigmoid陡峭度参数
            min_score: 最低清晰度分数阈值
        """
        self.mu = mu
        self.tau = tau
        self.min_score = min_score
    
    def evaluate_laplacian(self, image: np.ndarray) -> float:
        """
        使用Laplacian方差评估清晰度
        
        公式: score = sigmoid((var - mu) / tau)
        
        Args:
            image: BGR图像
            
        Returns:
            清晰度分数 [0, 1]
        """
        if image.size == 0:
            return 0.0
        
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 计算Laplacian方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Sigmoid映射到[0, 1]
        score = 1.0 / (1.0 + np.exp(-(variance - self.mu) / max(self.tau, 1e-6)))
        
        return float(np.clip(score, 0.0, 1.0))
    
    def evaluate_tenengrad(self, image: np.ndarray) -> float:
        """
        使用Tenengrad梯度方法评估清晰度
        
        Args:
            image: BGR图像
            
        Returns:
            清晰度分数 [0, 1]
        """
        if image.size == 0:
            return 0.0
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Sobel梯度
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Tenengrad = sqrt(gx^2 + gy^2)
        gradient = np.sqrt(gx**2 + gy**2)
        tenengrad = gradient.mean()
        
        # 归一化到[0, 1]
        score = 1.0 / (1.0 + np.exp(-(tenengrad - 30) / 15))
        
        return float(np.clip(score, 0.0, 1.0))
    
    def is_clear_enough(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        判断图像是否足够清晰
        
        Returns:
            (是否清晰, 清晰度分数)
        """
        score = self.evaluate_laplacian(image)
        return score >= self.min_score, score


# ============================================================
# OCR文字识别
# ============================================================

class OCRReader(Protocol):
    """OCR引擎最小接口"""

    def readtext(self, image: np.ndarray) -> List[Tuple[Any, str, float]]:
        ...


class OCRRecognizer:
    """OCR文字识别器"""
    
    def __init__(self, config: Optional[Dict[str, Any]]):
        """
        Args:
            config: OCR配置，包含keywords_open, keywords_closed等
        """
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.engine = config.get("engine", "easyocr")
        self.languages = config.get("languages", ["ch_sim", "en"])
        if self.languages is None:
            self.languages = ["ch_sim", "en"]
        keywords_open = config.get("keywords_open", ["分", "OPEN", "OFF", "断开", "拉开"])
        if keywords_open is None:
            keywords_open = ["分", "OPEN", "OFF", "断开", "拉开"]
        keywords_closed = config.get("keywords_closed", ["合", "CLOSE", "ON", "闭合", "合上"])
        if keywords_closed is None:
            keywords_closed = ["合", "CLOSE", "ON", "闭合", "合上"]
        self.keywords_open = [k.upper() for k in keywords_open]
        self.keywords_closed = [k.upper() for k in keywords_closed]
        self._reader: Optional[OCRReader] = None
        self._reader_unavailable = False
    
    def _get_reader(self) -> Optional["OCRReader"]:
        """延迟加载OCR引擎"""
        if not self.enabled or self._reader_unavailable:
            return None
        if self._reader is None:
            try:
                import importlib
                easyocr = importlib.import_module("easyocr")
                self._reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
            except Exception:
                self._reader_unavailable = True
                self._reader = None
        return self._reader
    
    def recognize(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        识别图像中的文字
        
        Args:
            image: BGR图像
            
        Returns:
            (识别文本, 置信度)
        """
        if not self.enabled or image is None or image.size == 0:
            return None, 0.0
        
        reader = self._get_reader()
        if reader is None:
            return None, 0.0
        
        try:
            results = reader.readtext(image)
            if not results:
                return None, 0.0
            
            # 取置信度最高的结果
            bbox, text, conf = max(results, key=lambda x: x[2])
            return text.strip(), float(conf)
        except Exception:
            return None, 0.0
    
    def match_state(self, text: Optional[str], confidence: float) -> Tuple[str, float, float]:
        """
        匹配文字到状态
        
        Args:
            text: OCR识别的文本
            confidence: OCR置信度
            
        Returns:
            (状态, open分数, closed分数)
        """
        if text is None:
            return "unknown", 0.0, 0.0
        
        text_upper = text.upper()
        
        # 计算匹配分数
        open_score = 0.0
        closed_score = 0.0
        
        for kw in self.keywords_open:
            if kw in text_upper:
                open_score = confidence
                break
        
        for kw in self.keywords_closed:
            if kw in text_upper:
                closed_score = confidence
                break
        
        if open_score > closed_score:
            return "open", open_score, closed_score
        elif closed_score > open_score:
            return "closed", open_score, closed_score
        else:
            return "unknown", open_score, closed_score


# ============================================================
# 颜色提示分析
# ============================================================

class ColorHintAnalyzer:
    """颜色提示分析器（红/绿指示灯）"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 颜色配置，包含HSV范围
        """
        self.enabled = config.get("enabled", True)
        
        # HSV颜色范围
        self.hsv_green = config.get("hsv_green", [[35, 40, 40], [85, 255, 255]])
        self.hsv_red1 = config.get("hsv_red1", [[0, 60, 60], [10, 255, 255]])
        self.hsv_red2 = config.get("hsv_red2", [[170, 60, 60], [180, 255, 255]])
    
    def _hsv_ratio(self, image: np.ndarray, lower: List[int], upper: List[int]) -> float:
        """计算指定HSV范围的像素占比"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        return float(mask.mean() / 255.0)
    
    def analyze(self, image: np.ndarray) -> Tuple[float, float]:
        """
        分析图像中的红绿颜色占比
        
        Args:
            image: BGR图像
            
        Returns:
            (red_ratio, green_ratio)
        """
        if not self.enabled or image.size == 0:
            return 0.0, 0.0
        
        # 绿色占比
        green_ratio = self._hsv_ratio(image, self.hsv_green[0], self.hsv_green[1])
        
        # 红色占比（两个范围取并集）
        red_ratio1 = self._hsv_ratio(image, self.hsv_red1[0], self.hsv_red1[1])
        red_ratio2 = self._hsv_ratio(image, self.hsv_red2[0], self.hsv_red2[1])
        red_ratio = max(red_ratio1, red_ratio2)
        
        return red_ratio, green_ratio
    
    def match_state(self, red_ratio: float, green_ratio: float) -> Tuple[str, float, float]:
        """
        根据颜色占比匹配状态
        
        变电站惯例：
        - 红色 = 分闸 (open)
        - 绿色 = 合闸 (closed)
        
        Returns:
            (状态, open分数, closed分数)
        """
        if red_ratio > green_ratio:
            return "open", red_ratio, green_ratio
        elif green_ratio > red_ratio:
            return "closed", red_ratio, green_ratio
        else:
            return "unknown", red_ratio, green_ratio


# ============================================================
# 连杆角度检测
# ============================================================

class LinkageAngleDetector:
    """连杆角度检测器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 角度配置，包含各类开关的标定角度
        """
        self.angle_reference = config.get("angle_reference", {
            "breaker": {"open_deg": -60, "closed_deg": 30},
            "isolator": {"open_deg": -70, "closed_deg": 20},
            "grounding": {"open_deg": -80, "closed_deg": 10}
        })
        self.sigma_deg = config.get("angle_sigma_deg", 18)
    
    def detect_dominant_angle(self, image: np.ndarray) -> Optional[float]:
        """
        检测图像中的主要线段角度
        
        使用Canny边缘检测 + HoughLinesP
        
        Args:
            image: BGR图像
            
        Returns:
            角度（度），或None表示检测失败
        """
        if image.size == 0:
            return None
        
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 边缘检测
        edges = cv2.Canny(gray, 80, 160)
        
        # Hough直线检测
        min_line_length = int(0.25 * min(gray.shape[:2]))
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=60,
            minLineLength=max(min_line_length, 20),
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            return None
        
        # 找到最长的线段
        def line_length(line):
            x1, y1, x2, y2 = line[0]
            return (x2 - x1)**2 + (y2 - y1)**2
        
        longest_line = max(lines, key=line_length)
        x1, y1, x2, y2 = longest_line[0]
        
        # 计算角度
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        return float(angle)
    
    def _gaussian_score(self, angle: float, reference: float) -> float:
        """计算角度的高斯得分"""
        d = abs(angle - reference)
        return float(np.exp(-(d**2) / (2 * self.sigma_deg**2)))
    
    def match_state(self, angle: Optional[float], device_type: str) -> Tuple[str, float, float]:
        """
        根据角度匹配状态
        
        Args:
            angle: 检测到的角度
            device_type: 设备类型 (breaker/isolator/grounding)
            
        Returns:
            (状态, open分数, closed分数)
        """
        if angle is None:
            return "unknown", 0.0, 0.0
        
        ref = self.angle_reference.get(device_type, {
            "open_deg": -60,
            "closed_deg": 30
        })
        
        open_ref = ref.get("open_deg", -60)
        closed_ref = ref.get("closed_deg", 30)
        
        open_score = self._gaussian_score(angle, open_ref)
        closed_score = self._gaussian_score(angle, closed_ref)
        
        if open_score > closed_score:
            return "open", open_score, closed_score
        elif closed_score > open_score:
            return "closed", open_score, closed_score
        else:
            return "unknown", open_score, closed_score


# ============================================================
# SF6表计读数器
# ============================================================

class GaugeReader:
    """SF6压强/密度表读数器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 表计配置
        """
        self.enabled = config.get("enabled", False)
        pointer_cfg = config.get("pointer", {})
        self.theta_min_deg = pointer_cfg.get("theta_min_deg", -120)
        self.theta_max_deg = pointer_cfg.get("theta_max_deg", 120)
        self.value_min = pointer_cfg.get("value_min", 0.0)
        self.value_max = pointer_cfg.get("value_max", 1.0)
        self.unit = pointer_cfg.get("unit", "MPa")
    
    def _detect_circle(self, gray: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """检测表盘圆形"""
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=gray.shape[0] // 4,
            param1=50,
            param2=30,
            minRadius=gray.shape[0] // 6,
            maxRadius=gray.shape[0] // 2
        )
        
        if circles is not None and len(circles) > 0:
            circle = circles[0][0]
            return int(circle[0]), int(circle[1]), int(circle[2])
        return None
    
    def _detect_pointer_angle(self, gray: np.ndarray, center: Tuple[int, int]) -> Optional[float]:
        """检测指针角度"""
        cx, cy = center
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 检测直线
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=20,
            maxLineGap=5
        )
        
        if lines is None:
            return None
        
        # 找到经过圆心附近的最长线段
        best_line = None
        best_length = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算线段到圆心的距离
            line_vec = np.array([x2 - x1, y2 - y1])
            point_vec = np.array([cx - x1, cy - y1])
            line_len = np.linalg.norm(line_vec)
            
            if line_len < 1:
                continue
            
            # 投影
            proj = np.dot(point_vec, line_vec) / (line_len ** 2)
            closest_point = np.array([x1, y1]) + proj * line_vec
            dist = np.linalg.norm(np.array([cx, cy]) - closest_point)
            
            # 线段经过圆心附近
            if dist < 15 and line_len > best_length:
                best_length = line_len
                best_line = line[0]
        
        if best_line is None:
            return None
        
        x1, y1, x2, y2 = best_line
        # 计算角度（以圆心为原点）
        angle = math.degrees(math.atan2(cy - (y1 + y2) / 2, (x1 + x2) / 2 - cx))
        
        return angle
    
    def read_pointer_gauge(self, image: np.ndarray) -> GaugeResult:
        """
        读取指针式表计
        
        Args:
            image: BGR图像
            
        Returns:
            GaugeResult
        """
        if not self.enabled or image.size == 0:
            return GaugeResult(reason_code=1004)
        
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 检测圆形表盘
        circle = self._detect_circle(gray)
        if circle is None:
            return GaugeResult(reason_code=1004)
        
        cx, cy, r = circle
        
        # 检测指针角度
        angle = self._detect_pointer_angle(gray, (cx, cy))
        if angle is None:
            return GaugeResult(reason_code=1004)
        
        # 角度映射到读数
        # 将角度归一化到[0, 1]范围
        theta_range = self.theta_max_deg - self.theta_min_deg
        normalized = (angle - self.theta_min_deg) / theta_range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # 映射到值域
        value = self.value_min + normalized * (self.value_max - self.value_min)
        
        return GaugeResult(
            value=float(value),
            unit=self.unit,
            confidence=0.75,
            reason_code=None
        )
    
    def read_digital_gauge(self, image: np.ndarray, ocr_recognizer: OCRRecognizer) -> GaugeResult:
        """
        读取数字式表计
        
        Args:
            image: BGR图像
            ocr_recognizer: OCR识别器
            
        Returns:
            GaugeResult
        """
        if not self.enabled:
            return GaugeResult(reason_code=1004)
        
        text, conf = ocr_recognizer.recognize(image)
        if text is None:
            return GaugeResult(reason_code=1002)
        
        # 尝试提取数字
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if not numbers:
            return GaugeResult(reason_code=1002)
        
        try:
            value = float(numbers[0])
            return GaugeResult(
                value=value,
                unit=self.unit,
                confidence=conf,
                reason_code=None
            )
        except ValueError:
            return GaugeResult(reason_code=1002)


# ============================================================
# 状态融合器
# ============================================================

class StateFusion:
    """多证据状态融合器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 融合配置
        """
        weights = config.get("fusion_weights", {"text": 0.5, "color": 0.3, "angle": 0.2})
        self.w_text = weights.get("text", 0.5)
        self.w_color = weights.get("color", 0.3)
        self.w_angle = weights.get("angle", 0.2)
        self.min_state_score = config.get("min_state_score", 0.55)
    
    def fuse(
        self,
        text_scores: Tuple[float, float],
        color_scores: Tuple[float, float],
        angle_scores: Tuple[float, float]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        融合多证据得出最终状态
        
        Args:
            text_scores: (open_score, closed_score) from OCR
            color_scores: (open_score, closed_score) from color
            angle_scores: (open_score, closed_score) from angle
            
        Returns:
            (state, confidence, debug_info)
        """
        # 加权求和
        s_open = (
            self.w_text * text_scores[0] +
            self.w_color * color_scores[0] +
            self.w_angle * angle_scores[0]
        )
        
        s_closed = (
            self.w_text * text_scores[1] +
            self.w_color * color_scores[1] +
            self.w_angle * angle_scores[1]
        )
        
        debug_info = {
            "S_open": float(s_open),
            "S_closed": float(s_closed),
            "text_scores": text_scores,
            "color_scores": color_scores,
            "angle_scores": angle_scores
        }
        
        # 判决
        max_score = max(s_open, s_closed)
        if max_score < self.min_state_score:
            return "unknown", float(max_score), debug_info
        
        if s_open > s_closed:
            return "open", float(s_open), debug_info
        else:
            return "closed", float(s_closed), debug_info


# ============================================================
# 互锁逻辑校验器
# ============================================================

class InterlockValidator:
    """五防互锁逻辑校验器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 互锁规则配置
        """
        self.enabled = config.get("enabled", True)
        self.rules = config.get("rules", [])
    
    def validate(
        self,
        breaker_state: str,
        isolator_state: str,
        grounding_state: str
    ) -> List[Dict[str, Any]]:
        """
        校验互锁逻辑
        
        Args:
            breaker_state: 断路器状态
            isolator_state: 隔离开关状态
            grounding_state: 接地开关状态
            
        Returns:
            违规告警列表
        """
        if not self.enabled:
            return []
        
        violations = []
        
        for rule in self.rules:
            condition = rule.get("condition", "")
            
            # 规则1：防止带电合接地刀
            # 隔离开关未断开时，禁止合接地开关
            if condition == "isolator_open_before_grounding":
                if isolator_state != "open" and grounding_state == "closed":
                    violations.append({
                        "rule_name": rule.get("name", "互锁逻辑异常"),
                        "rule_id": rule.get("id", "interlock_grounding"),
                        "severity": rule.get("severity", "error"),
                        "reason_code": rule.get("reason_code", 2001),
                        "description": rule.get("description", "隔离开关未断开时，禁止合接地开关"),
                        "states": {
                            "isolator": isolator_state,
                            "grounding": grounding_state
                        }
                    })
            
            # 规则2：异常工况提示
            # 断路器合闸但隔离开关断开
            elif condition == "breaker_closed_while_isolator_open":
                if breaker_state == "closed" and isolator_state == "open":
                    violations.append({
                        "rule_name": rule.get("name", "异常工况提示"),
                        "rule_id": rule.get("id", "breaker_isolator_mismatch"),
                        "severity": rule.get("severity", "warning"),
                        "reason_code": rule.get("reason_code", 2002),
                        "description": rule.get("description", "断路器合闸但隔离开关断开"),
                        "states": {
                            "breaker": breaker_state,
                            "isolator": isolator_state
                        }
                    })
        
        return violations


# ============================================================
# 主检测器
# ============================================================

class SwitchDetector:
    """开关间隔检测器 - 集成所有检测功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 完整配置字典
        """
        self.config = config
        
        # 初始化各子模块
        state_cfg = config.get("state_recognition", {})
        
        # 清晰度评价器
        quality_cfg = config.get("image_quality", {})
        self.clarity_evaluator = ClarityEvaluator(
            mu=quality_cfg.get("mu", 120),
            tau=quality_cfg.get("tau", 30),
            min_score=quality_cfg.get("min_clarity_score", 0.70)
        )
        
        # OCR识别器
        self.ocr_recognizer = OCRRecognizer(state_cfg.get("ocr", {}))
        
        # 颜色分析器
        self.color_analyzer = ColorHintAnalyzer(state_cfg.get("color_hint", {}))
        
        # 角度检测器
        self.angle_detector = LinkageAngleDetector(state_cfg)
        
        # 状态融合器
        self.state_fusion = StateFusion(state_cfg)
        
        # 互锁校验器
        self.interlock_validator = InterlockValidator(config.get("logic_validation", {}))
        
        # 表计读数器
        self.gauge_reader = GaugeReader(config.get("gauge_reading", {}))
        
        # 置信度阈值
        inference_cfg = config.get("inference", {})
        self.confidence_threshold = inference_cfg.get("confidence_threshold", 0.6)
    
    def evaluate_clarity(self, image: np.ndarray) -> Tuple[bool, float]:
        """评估图像清晰度"""
        return self.clarity_evaluator.is_clear_enough(image)
    
    def recognize_indicator_state(
        self,
        image: np.ndarray,
        device_type: str
    ) -> StateResult:
        """
        识别指示牌状态（融合OCR + 颜色）
        
        Args:
            image: ROI图像
            device_type: 设备类型
            
        Returns:
            StateResult
        """
        evidence = StateEvidence()
        
        # 清晰度评估
        is_clear, clarity_score = self.evaluate_clarity(image)
        evidence.clarity_score = clarity_score
        
        if not is_clear:
            return StateResult(
                state="unknown",
                confidence=0.0,
                evidence=evidence,
                reason_code=1001  # 清晰度过低
            )
        
        # OCR识别
        text, ocr_conf = self.ocr_recognizer.recognize(image)
        evidence.ocr_text = text
        evidence.ocr_confidence = ocr_conf
        _, text_open, text_closed = self.ocr_recognizer.match_state(text, ocr_conf)
        
        # 颜色分析
        red_ratio, green_ratio = self.color_analyzer.analyze(image)
        evidence.red_ratio = red_ratio
        evidence.green_ratio = green_ratio
        _, color_open, color_closed = self.color_analyzer.match_state(red_ratio, green_ratio)
        
        # 融合（指示牌不使用角度）
        state, confidence, debug_info = self.state_fusion.fuse(
            text_scores=(text_open, text_closed),
            color_scores=(color_open, color_closed),
            angle_scores=(0.0, 0.0)  # 指示牌不使用角度
        )
        
        reason_code = None
        if text is None and ocr_conf == 0:
            reason_code = 1002  # OCR失败
        
        return StateResult(
            state=state,
            confidence=confidence,
            evidence=evidence,
            reason_code=reason_code,
            extra=debug_info
        )
    
    def recognize_linkage_state(
        self,
        image: np.ndarray,
        device_type: str
    ) -> StateResult:
        """
        识别连杆状态（使用角度检测）
        
        Args:
            image: ROI图像
            device_type: 设备类型
            
        Returns:
            StateResult
        """
        evidence = StateEvidence()
        
        # 清晰度评估
        is_clear, clarity_score = self.evaluate_clarity(image)
        evidence.clarity_score = clarity_score
        
        if not is_clear:
            return StateResult(
                state="unknown",
                confidence=0.0,
                evidence=evidence,
                reason_code=1001
            )
        
        # 角度检测
        angle = self.angle_detector.detect_dominant_angle(image)
        evidence.angle_deg = angle
        
        if angle is None:
            return StateResult(
                state="unknown",
                confidence=0.0,
                evidence=evidence,
                reason_code=1003  # 未检测到有效角度
            )
        
        # 角度匹配
        _, angle_open, angle_closed = self.angle_detector.match_state(angle, device_type)
        
        # 融合（连杆主要依赖角度）
        state, confidence, debug_info = self.state_fusion.fuse(
            text_scores=(0.0, 0.0),
            color_scores=(0.0, 0.0),
            angle_scores=(angle_open, angle_closed)
        )
        
        return StateResult(
            state=state,
            confidence=confidence,
            evidence=evidence,
            reason_code=None,
            extra=debug_info
        )
    
    def recognize_state_full(
        self,
        indicator_image: Optional[np.ndarray],
        linkage_image: Optional[np.ndarray],
        device_type: str
    ) -> StateResult:
        """
        完整状态识别（融合指示牌 + 连杆）
        
        Args:
            indicator_image: 指示牌ROI图像
            linkage_image: 连杆ROI图像
            device_type: 设备类型
            
        Returns:
            StateResult
        """
        evidence = StateEvidence()
        text_scores = (0.0, 0.0)
        color_scores = (0.0, 0.0)
        angle_scores = (0.0, 0.0)
        reason_code = None
        
        # 处理指示牌
        if indicator_image is not None and indicator_image.size > 0:
            is_clear, clarity_score = self.evaluate_clarity(indicator_image)
            evidence.clarity_score = max(evidence.clarity_score, clarity_score)
            
            if is_clear:
                # OCR
                text, ocr_conf = self.ocr_recognizer.recognize(indicator_image)
                evidence.ocr_text = text
                evidence.ocr_confidence = ocr_conf
                _, text_open, text_closed = self.ocr_recognizer.match_state(text, ocr_conf)
                text_scores = (text_open, text_closed)
                
                # 颜色
                red_ratio, green_ratio = self.color_analyzer.analyze(indicator_image)
                evidence.red_ratio = red_ratio
                evidence.green_ratio = green_ratio
                _, color_open, color_closed = self.color_analyzer.match_state(red_ratio, green_ratio)
                color_scores = (color_open, color_closed)
        
        # 处理连杆
        if linkage_image is not None and linkage_image.size > 0:
            is_clear, clarity_score = self.evaluate_clarity(linkage_image)
            evidence.clarity_score = max(evidence.clarity_score, clarity_score)
            
            if is_clear:
                angle = self.angle_detector.detect_dominant_angle(linkage_image)
                evidence.angle_deg = angle
                
                if angle is not None:
                    _, angle_open, angle_closed = self.angle_detector.match_state(angle, device_type)
                    angle_scores = (angle_open, angle_closed)
                else:
                    reason_code = 1003
        
        # 融合
        state, confidence, debug_info = self.state_fusion.fuse(
            text_scores=text_scores,
            color_scores=color_scores,
            angle_scores=angle_scores
        )
        
        # 检查是否有足够证据
        if evidence.clarity_score < self.clarity_evaluator.min_score:
            reason_code = 1001
        
        return StateResult(
            state=state,
            confidence=confidence,
            evidence=evidence,
            reason_code=reason_code,
            extra=debug_info
        )
    
    def validate_interlock(
        self,
        breaker_state: str,
        isolator_state: str,
        grounding_state: str
    ) -> List[Dict[str, Any]]:
        """校验互锁逻辑"""
        return self.interlock_validator.validate(
            breaker_state, isolator_state, grounding_state
        )
    
    def read_gauge(self, image: np.ndarray, gauge_type: str = "pointer") -> GaugeResult:
        """
        读取表计
        
        Args:
            image: 表计ROI图像
            gauge_type: 表计类型 (pointer/digital)
            
        Returns:
            GaugeResult
        """
        if gauge_type == "digital":
            return self.gauge_reader.read_digital_gauge(image, self.ocr_recognizer)
        else:
            return self.gauge_reader.read_pointer_gauge(image)
