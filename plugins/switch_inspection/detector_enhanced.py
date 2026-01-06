"""
开关间隔巡视检测器 - 增强版
输变电激光监测平台 (B组) - 全自动AI巡检改造

增强功能:
- 多任务模型同时识别(状态+指示灯+读数)
- CRNN/Transformer OCR文字识别
- 深度学习证据融合
- 规则引擎集成
- 增强清晰度评价
"""

from __future__ import annotations
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class SwitchType(Enum):
    """开关类型"""
    BREAKER = "breaker"           # 断路器
    ISOLATOR = "isolator"         # 隔离开关
    GROUNDING = "grounding"       # 接地开关


class SwitchState(Enum):
    """开关状态"""
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"
    INTERMEDIATE = "intermediate"  # 中间位置


class ClarityLevel(Enum):
    """清晰度级别"""
    EXCELLENT = "excellent"       # 优秀 > 0.9
    GOOD = "good"                 # 良好 0.7-0.9
    ACCEPTABLE = "acceptable"     # 可接受 0.5-0.7
    POOR = "poor"                 # 差 < 0.5
    RESHOOT_REQUIRED = "reshoot_required"  # 需要复拍


@dataclass
class StateEvidence:
    """状态证据"""
    source: str                   # 来源: ocr/color/angle/dl
    state: SwitchState
    confidence: float
    raw_value: Any = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class ClarityResult:
    """清晰度评估结果"""
    score: float
    level: ClarityLevel
    method: str
    metrics: Dict[str, float] = field(default_factory=dict)
    reshoot_suggestion: Optional[Dict] = None


class SwitchDetectorEnhanced:
    """
    开关间隔增强检测器
    
    集成多任务深度学习模型和规则引擎
    """
    
    # 模型ID映射
    MODEL_IDS = {
        "multi_task": "switch_multitask_yolov8",      # 多任务模型
        "ocr": "switch_ocr_crnn",                      # OCR模型
        "state_classifier": "switch_state_resnet",    # 状态分类器
    }
    
    # OCR关键词
    OCR_KEYWORDS = {
        "open": ["分", "开", "OPEN", "OFF", "断"],
        "closed": ["合", "关", "CLOSE", "ON", "通"],
    }
    
    # 角度参考值
    ANGLE_REFERENCES = {
        SwitchType.BREAKER: {"open": -60, "closed": 30},
        SwitchType.ISOLATOR: {"open": -70, "closed": 20},
        SwitchType.GROUNDING: {"open": -80, "closed": 10},
    }
    
    def __init__(self, config: dict[str, Any], model_registry=None, fusion_engine=None):
        """初始化增强检测器"""
        self.config = config
        self._model_registry = model_registry
        self._fusion_engine = fusion_engine
        
        # 配置参数
        self.state_config = config.get("state_recognition", {})
        self.clarity_config = config.get("clarity_evaluation", {})
        self.logic_config = config.get("logic_validation", {})
        
        # 融合权重
        weights = self.state_config.get("fusion_weights", {})
        self.weights = {
            "ocr": weights.get("text", 0.5),
            "color": weights.get("color", 0.3),
            "angle": weights.get("angle", 0.2),
            "dl": weights.get("deep_learning", 0.6),
        }
        
        self.use_deep_learning = config.get("use_deep_learning", True)
        self.min_state_score = self.state_config.get("min_state_score", 0.55)
    
    # ==================== 多任务状态识别 ====================
    
    def recognize_switch_state(
        self,
        image: np.ndarray,
        switch_type: SwitchType = SwitchType.BREAKER,
        use_fusion: bool = True,
    ) -> Dict:
        """
        识别开关状态
        
        多证据融合: OCR + 颜色 + 角度 + 深度学习
        """
        evidences: List[StateEvidence] = []
        
        # 1. 深度学习多任务识别
        if self.use_deep_learning and self._model_registry:
            dl_evidence = self._recognize_by_deep_learning(image, switch_type)
            if dl_evidence:
                evidences.append(dl_evidence)
        
        # 2. OCR文字识别
        ocr_evidence = self._recognize_by_ocr(image)
        if ocr_evidence:
            evidences.append(ocr_evidence)
        
        # 3. 颜色检测
        color_evidence = self._recognize_by_color(image)
        if color_evidence:
            evidences.append(color_evidence)
        
        # 4. 角度检测
        angle_evidence = self._recognize_by_angle(image, switch_type)
        if angle_evidence:
            evidences.append(angle_evidence)
        
        # 5. 证据融合
        if use_fusion and self._fusion_engine:
            return self._fuse_with_engine(evidences)
        else:
            return self._fuse_evidences(evidences)
    
    def _recognize_by_deep_learning(
        self, 
        image: np.ndarray, 
        switch_type: SwitchType
    ) -> Optional[StateEvidence]:
        """深度学习多任务识别"""
        try:
            assert self._model_registry is not None
            model_id = self.MODEL_IDS["multi_task"]
            result = self._model_registry.infer(model_id, image)
            
            if result.detections:
                det = result.detections[0]
                state = SwitchState.OPEN if "open" in det["class_name"].lower() else SwitchState.CLOSED
                
                return StateEvidence(
                    source="deep_learning",
                    state=state,
                    confidence=det["confidence"],
                    raw_value=det,
                    metadata={"model_id": model_id},
                )
        except Exception as e:
            print(f"[SwitchDetector] DL识别失败: {e}")
        
        return None
    
    def _recognize_by_ocr(self, image: np.ndarray) -> Optional[StateEvidence]:
        """OCR文字识别"""
        if cv2 is None:
            return None
        
        # 尝试使用深度学习OCR
        if self._model_registry:
            try:
                model_id = self.MODEL_IDS["ocr"]
                result = self._model_registry.infer(model_id, image)
                if result.detections:
                    text = result.detections[0].get("text", "")
                    return self._parse_ocr_result(text)
            except Exception:
                pass
        
        # 回退: 使用传统OCR预处理
        return self._ocr_traditional(image)
    
    def _ocr_traditional(self, image: np.ndarray) -> Optional[StateEvidence]:
        """传统OCR预处理"""
        if cv2 is None:
            return None

        # 简化实现：基于颜色区域提取
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 这里应该调用OCR引擎(如PaddleOCR/EasyOCR)
        # 简化为返回None
        return None
    
    def _parse_ocr_result(self, text: str) -> Optional[StateEvidence]:
        """解析OCR结果"""
        text_upper = text.upper()
        
        open_score = sum(1 for kw in self.OCR_KEYWORDS["open"] if kw in text_upper)
        closed_score = sum(1 for kw in self.OCR_KEYWORDS["closed"] if kw in text_upper)
        
        if open_score > closed_score:
            state = SwitchState.OPEN
            confidence = min(0.9, 0.6 + open_score * 0.15)
        elif closed_score > open_score:
            state = SwitchState.CLOSED
            confidence = min(0.9, 0.6 + closed_score * 0.15)
        else:
            return None
        
        return StateEvidence(
            source="ocr",
            state=state,
            confidence=confidence,
            raw_value=text,
            metadata={"open_score": open_score, "closed_score": closed_score},
        )
    
    def _recognize_by_color(self, image: np.ndarray) -> Optional[StateEvidence]:
        """颜色检测"""
        if cv2 is None:
            return None
        
        config = self.state_config.get("color_hints", {})
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 绿色(合位)
        green_lower = np.array(config.get("green_lower", [35, 100, 100]))
        green_upper = np.array(config.get("green_upper", [85, 255, 255]))
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = float(np.sum(green_mask > 0) / green_mask.size)
        
        # 红色(分位)
        red_lower1 = np.array(config.get("red_lower1", [0, 100, 100]))
        red_upper1 = np.array(config.get("red_upper1", [10, 255, 255]))
        red_lower2 = np.array(config.get("red_lower2", [160, 100, 100]))
        red_upper2 = np.array(config.get("red_upper2", [180, 255, 255]))
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_ratio = float(np.sum(red_mask > 0) / red_mask.size)
        
        min_ratio = config.get("min_color_ratio", 0.01)
        
        if green_ratio > red_ratio and green_ratio > min_ratio:
            state = SwitchState.CLOSED
            confidence = min(0.85, 0.5 + green_ratio * 10)
        elif red_ratio > green_ratio and red_ratio > min_ratio:
            state = SwitchState.OPEN
            confidence = min(0.85, 0.5 + red_ratio * 10)
        else:
            return None
        
        return StateEvidence(
            source="color",
            state=state,
            confidence=confidence,
            raw_value={"green": green_ratio, "red": red_ratio},
            metadata={"green_ratio": green_ratio, "red_ratio": red_ratio},
        )
    
    def _recognize_by_angle(
        self, 
        image: np.ndarray, 
        switch_type: SwitchType
    ) -> Optional[StateEvidence]:
        """角度检测"""
        if cv2 is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return None
        
        # 计算主方向角度
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        if not angles:
            return None
        
        avg_angle = float(np.mean(angles))
        
        # 与参考角度比较
        ref = self.ANGLE_REFERENCES.get(switch_type, self.ANGLE_REFERENCES[SwitchType.BREAKER])
        
        diff_open = abs(avg_angle - ref["open"])
        diff_closed = abs(avg_angle - ref["closed"])
        
        angle_threshold = self.state_config.get("angle_threshold", 20)
        
        if diff_open < diff_closed and diff_open < angle_threshold:
            state = SwitchState.OPEN
            confidence = max(0.5, 1 - diff_open / 90)
        elif diff_closed < diff_open and diff_closed < angle_threshold:
            state = SwitchState.CLOSED
            confidence = max(0.5, 1 - diff_closed / 90)
        else:
            state = SwitchState.INTERMEDIATE
            confidence = 0.4
        
        return StateEvidence(
            source="angle",
            state=state,
            confidence=confidence,
            raw_value=avg_angle,
            metadata={"avg_angle": avg_angle, "diff_open": diff_open, "diff_closed": diff_closed},
        )
    
    def _fuse_evidences(self, evidences: List[StateEvidence]) -> Dict:
        """融合证据(内置方法)"""
        if not evidences:
            return {
                "state": SwitchState.UNKNOWN.value,
                "confidence": 0,
                "evidences": [],
                "reason_code": 1001,
            }
        
        # 按状态累加得分
        scores = {SwitchState.OPEN: 0, SwitchState.CLOSED: 0, SwitchState.INTERMEDIATE: 0}
        
        for ev in evidences:
            weight = self.weights.get(ev.source, 1.0)
            scores[ev.state] += ev.confidence * weight
        
        # 归一化
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        # 选择最高得分
        best_state, best_score = max(scores.items(), key=lambda item: item[1])
        
        # 检查是否满足最小阈值
        if best_score < self.min_state_score:
            return {
                "state": SwitchState.UNKNOWN.value,
                "confidence": best_score,
                "evidences": [self._evidence_to_dict(e) for e in evidences],
                "reason_code": 1002,
                "scores": {k.value: v for k, v in scores.items()},
            }
        
        return {
            "state": best_state.value,
            "confidence": best_score,
            "evidences": [self._evidence_to_dict(e) for e in evidences],
            "scores": {k.value: v for k, v in scores.items()},
        }
    
    def _fuse_with_engine(self, evidences: List[StateEvidence]) -> Dict:
        """使用融合引擎"""
        from platform_core.fusion_engine import Evidence, EvidenceType
        
        assert self._fusion_engine is not None
        fusion_evidences = []
        for ev in evidences:
            etype = {
                "ocr": EvidenceType.OCR_TEXT,
                "color": EvidenceType.COLOR_DETECTION,
                "angle": EvidenceType.ANGLE_DETECTION,
                "deep_learning": EvidenceType.DEEP_LEARNING,
            }.get(ev.source, EvidenceType.RULE_BASED)
            
            fusion_evidences.append(Evidence(
                evidence_id=f"{ev.source}_{id(ev)}",
                evidence_type=etype,
                source=ev.source,
                value=ev.state.value,
                confidence=ev.confidence,
                weight=self.weights.get(ev.source, 1.0),
            ))
        
        result = self._fusion_engine.fuse(fusion_evidences)
        
        return {
            "state": result.final_value,
            "confidence": result.final_confidence,
            "evidences": [self._evidence_to_dict(e) for e in evidences],
            "fusion_method": result.fusion_method,
            "conflict_detected": result.conflict_detected,
        }
    
    def _evidence_to_dict(self, ev: StateEvidence) -> Dict:
        """证据转字典"""
        return {
            "source": ev.source,
            "state": ev.state.value,
            "confidence": ev.confidence,
            "metadata": ev.metadata,
        }
    
    # ==================== 逻辑校验 ====================
    
    def validate_logic(
        self,
        bay_states: Dict[str, str],
        device_id: str,
        new_state: str,
    ) -> Dict:
        """
        五防逻辑校验
        
        Args:
            bay_states: 间隔内各设备当前状态
            device_id: 待校验设备ID
            new_state: 新状态
            
        Returns:
            校验结果
        """
        # 预留规则配置入口
        _ = self.logic_config.get("interlock_rules", {})
        violations = []
        
        # 规则1: 断路器合闸时，两侧隔离开关必须在合位
        if "breaker" in device_id and new_state == "closed":
            for dev_id, state in bay_states.items():
                if "isolator" in dev_id and state != "closed":
                    violations.append({
                        "rule": "breaker_close_requires_isolator_closed",
                        "device": dev_id,
                        "expected": "closed",
                        "actual": state,
                    })
        
        # 规则2: 隔离开关操作时，断路器必须在分位
        if "isolator" in device_id:
            breaker_state = bay_states.get("breaker", "unknown")
            if breaker_state != "open":
                violations.append({
                    "rule": "isolator_operation_requires_breaker_open",
                    "device": "breaker",
                    "expected": "open",
                    "actual": breaker_state,
                })
        
        # 规则3: 接地开关合闸时，断路器和隔离开关必须在分位
        if "grounding" in device_id and new_state == "closed":
            for dev_id, state in bay_states.items():
                if ("breaker" in dev_id or "isolator" in dev_id) and state != "open":
                    violations.append({
                        "rule": "grounding_close_requires_all_open",
                        "device": dev_id,
                        "expected": "open",
                        "actual": state,
                    })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "reason_code": 2001 if violations else None,
        }
    
    # ==================== 清晰度评价 ====================
    
    def evaluate_clarity(
        self,
        image: np.ndarray,
        method: str = "combined",
    ) -> ClarityResult:
        """
        图像清晰度评价
        
        增强方法:
        - combined: 组合多种方法
        - laplacian: 拉普拉斯方差
        - sobel: Sobel梯度
        - fft: 频域分析
        """
        if cv2 is None:
            return ClarityResult(0, ClarityLevel.POOR, "none")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        metrics = {}
        
        # 拉普拉斯方差
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics["laplacian"] = lap_var
        
        # Sobel梯度
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0).var()
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1).var()
        metrics["sobel"] = (sobel_x + sobel_y) / 2
        
        # 频域分析
        if method in ["fft", "combined"]:
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            high_freq_ratio = np.sum(magnitude > np.mean(magnitude)) / magnitude.size
            metrics["fft"] = high_freq_ratio
        
        # 综合得分
        config = self.clarity_config
        
        if method == "combined":
            lap_norm = min(1.0, lap_var / config.get("laplacian_max", 500))
            sobel_norm = min(1.0, metrics["sobel"] / config.get("sobel_max", 1000))
            fft_norm = metrics.get("fft", 0.5)
            
            score = 0.5 * lap_norm + 0.3 * sobel_norm + 0.2 * fft_norm
        elif method == "laplacian":
            score = min(1.0, lap_var / config.get("laplacian_max", 500))
        else:
            score = min(1.0, metrics["sobel"] / config.get("sobel_max", 1000))
        
        # 确定级别
        if score > 0.9:
            level = ClarityLevel.EXCELLENT
        elif score > 0.7:
            level = ClarityLevel.GOOD
        elif score > 0.5:
            level = ClarityLevel.ACCEPTABLE
        else:
            level = ClarityLevel.POOR
        
        # 复拍建议
        reshoot = None
        min_score = config.get("min_clarity_score", 0.7)
        if score < min_score:
            level = ClarityLevel.RESHOOT_REQUIRED
            reshoot = {
                "reason": "clarity_below_threshold",
                "current_score": score,
                "threshold": min_score,
                "suggestions": [
                    "increase_zoom" if score < 0.5 else "adjust_focus",
                    "stabilize_camera",
                ]
            }
        
        return ClarityResult(
            score=score,
            level=level,
            method=method,
            metrics=metrics,
            reshoot_suggestion=reshoot,
        )
    
    # ==================== SF6密度表读数 ====================
    
    def read_sf6_gauge(self, image: np.ndarray) -> Dict:
        """SF6密度表读数"""
        if cv2 is None:
            return {"value": None, "confidence": 0}
        
        config = self.config.get("sf6_gauge", {})
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 检测圆形表盘
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                    param1=100, param2=30, 
                                    minRadius=30, maxRadius=min(h, w)//2)
        
        if circles is None:
            return {"value": None, "confidence": 0.3, "reason": "no_dial_detected"}
        
        # 取最大的圆
        circles = np.around(circles).astype(np.uint16)
        best = max(circles[0], key=lambda c: c[2])
        cx, cy, r = int(best[0]), int(best[1]), int(best[2])
        
        # 创建掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        cv2.circle(mask, (cx, cy), r//4, 0, -1)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, mask)
        
        # 霍夫直线检测指针
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30,
                                minLineLength=r//3, maxLineGap=10)
        
        if lines is None:
            return {"value": None, "confidence": 0.4, "reason": "no_pointer_detected"}
        
        # 找经过中心的最长线段
        best_line = None
        best_length = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算到中心的距离
            dist = self._point_to_line_dist(cx, cy, x1, y1, x2, y2)
            
            if dist < r * 0.2:
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > best_length:
                    best_length = length
                    best_line = line[0]
        
        if best_line is None:
            return {"value": None, "confidence": 0.4, "reason": "no_valid_pointer"}
        
        x1, y1, x2, y2 = best_line
        
        # 确定指针方向
        d1 = np.sqrt((x1-cx)**2 + (y1-cy)**2)
        d2 = np.sqrt((x2-cx)**2 + (y2-cy)**2)
        px, py = (x1, y1) if d1 > d2 else (x2, y2)
        
        # 计算角度
        angle = np.degrees(np.arctan2(px - cx, cy - py))
        
        # 转换为读数(假设范围0-0.8MPa, 角度范围-135到135)
        scale_min = config.get("scale_min", 0)
        scale_max = config.get("scale_max", 0.8)
        angle_min = config.get("angle_min", -135)
        angle_max = config.get("angle_max", 135)
        
        angle = max(angle_min, min(angle_max, angle))
        ratio = (angle - angle_min) / (angle_max - angle_min)
        value = scale_min + ratio * (scale_max - scale_min)
        
        return {
            "value": round(value, 3),
            "unit": "MPa",
            "confidence": 0.75,
            "metadata": {"angle": angle, "center": (cx, cy), "radius": r},
        }
    
    def _point_to_line_dist(self, px, py, x1, y1, x2, y2) -> float:
        """点到线段距离"""
        line_len = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if line_len == 0:
            return np.sqrt((px-x1)**2 + (py-y1)**2)
        
        t = max(0, min(1, ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / (line_len**2)))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        return np.sqrt((px-proj_x)**2 + (py-proj_y)**2)
