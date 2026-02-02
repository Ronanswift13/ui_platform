"""
消防监测检测器 - 核心引擎
===========================

输变电激光星芒破夜绘明监测平台 - 室内消防监测

功能:
1. YOLOv8 火焰/烟雾实时检测
2. DeepSORT 火势跟踪与扩散分析
3. 热成像异常检测
4. 多传感器融合火灾置信度
5. 火灾等级评估
6. 主动灭火联动控制

作者: 室内监测组
版本: 1.0.0
"""

from __future__ import annotations
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
# 数据结构
# =============================================================================

class FireType(str, Enum):
    """火灾检测类型"""
    FIRE = "fire"
    SMOKE = "smoke"
    SPARK = "spark"
    EMBER = "ember"
    HOT_SPOT = "hot_spot"


class FireLevel(str, Enum):
    """火灾等级"""
    NONE = "none"
    SMOLDERING = "smoldering"      # 阴燃
    SMALL = "small"                # 小火
    MEDIUM = "medium"              # 中火
    LARGE = "large"                # 大火
    CRITICAL = "critical"          # 紧急


class ZonePriority(str, Enum):
    """区域优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FireDetection:
    """单次火灾检测结果"""
    fire_type: FireType
    bbox: Dict[str, float]
    confidence: float
    area_ratio: float = 0.0        # 面积占比
    zone_id: str = ""
    zone_name: str = ""
    track_id: int = -1
    spread_rate: float = 0.0       # 扩散速率 (像素/帧)
    temperature: float = 0.0       # 关联温度
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorReading:
    """传感器读数"""
    smoke_concentration: float = 0.0    # 烟雾浓度 (%)
    temperature: float = 25.0           # 温度 (°C)
    co_concentration: float = 0.0       # CO浓度 (ppm)
    humidity: float = 50.0              # 湿度 (%)
    timestamp: float = 0.0


@dataclass
class FireAssessment:
    """火灾综合评估"""
    fire_level: FireLevel
    visual_confidence: float
    sensor_confidence: float
    fusion_confidence: float
    detections: List[FireDetection]
    sensor_reading: Optional[SensorReading]
    active_zones: List[str]
    spread_trend: str = "stable"    # decreasing | stable | increasing | rapid
    suppression_actions: List[Dict[str, Any]] = field(default_factory=list)
    evacuation_needed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 火灾跟踪器
# =============================================================================

class FireTracker:
    """
    火势跟踪器 - 基于IoU的简化DeepSORT
    
    跟踪火焰/烟雾的位置、面积和扩散趋势
    """
    
    def __init__(self, max_age: int = 30, min_confirm: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_confirm = min_confirm
        self.iou_threshold = iou_threshold
        
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1
        self._frame_count = 0
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        更新跟踪状态
        
        Args:
            detections: 检测结果 [{"bbox": {x,y,w,h}, "type": str, "confidence": float}, ...]
            
        Returns:
            带track_id的检测结果
        """
        self._frame_count += 1
        tracked = []
        matched_tracks = set()
        matched_dets = set()
        
        # IoU匹配
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_tid = None
            
            for tid, track in self._tracks.items():
                if tid in matched_tracks:
                    continue
                iou = self._compute_iou(det["bbox"], track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_tid = tid
            
            if best_tid is not None:
                matched_tracks.add(best_tid)
                matched_dets.add(det_idx)
                self._update_track(best_tid, det)
                det["track_id"] = best_tid
                det["confirmed"] = self._tracks[best_tid]["hit_count"] >= self.min_confirm
                det["spread_rate"] = self._tracks[best_tid].get("spread_rate", 0.0)
                tracked.append(det)
            else:
                # 新轨迹
                tid = self._create_track(det)
                det["track_id"] = tid
                det["confirmed"] = False
                det["spread_rate"] = 0.0
                tracked.append(det)
        
        # 清理过期轨迹
        expired = [
            tid for tid, t in self._tracks.items()
            if (self._frame_count - t["last_seen"]) > self.max_age
        ]
        for tid in expired:
            del self._tracks[tid]
        
        return tracked
    
    def _create_track(self, detection: Dict) -> int:
        tid = self._next_id
        self._next_id += 1
        
        area = detection["bbox"].get("width", 0) * detection["bbox"].get("height", 0)
        self._tracks[tid] = {
            "bbox": detection["bbox"].copy(),
            "type": detection.get("type", "unknown"),
            "confidence": detection.get("confidence", 0),
            "first_seen": self._frame_count,
            "last_seen": self._frame_count,
            "hit_count": 1,
            "area_history": deque([area], maxlen=30),
            "spread_rate": 0.0,
        }
        return tid
    
    def _update_track(self, tid: int, detection: Dict):
        track = self._tracks[tid]
        track["bbox"] = detection["bbox"].copy()
        track["confidence"] = max(track["confidence"], detection.get("confidence", 0))
        track["last_seen"] = self._frame_count
        track["hit_count"] += 1
        
        area = detection["bbox"].get("width", 0) * detection["bbox"].get("height", 0)
        track["area_history"].append(area)
        
        # 计算扩散速率
        if len(track["area_history"]) >= 3:
            recent = list(track["area_history"])
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            track["spread_rate"] = np.mean(diffs[-5:]) if len(diffs) >= 5 else np.mean(diffs)
    
    @staticmethod
    def _compute_iou(box1: Dict, box2: Dict) -> float:
        x1 = max(box1["x"], box2["x"])
        y1 = max(box1["y"], box2["y"])
        x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
        y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def get_spread_trend(self) -> str:
        """获取整体火势扩散趋势"""
        if not self._tracks:
            return "stable"
        
        rates = [t["spread_rate"] for t in self._tracks.values() if t["hit_count"] >= self.min_confirm]
        if not rates:
            return "stable"
        
        avg_rate = np.mean(rates)
        if avg_rate > 0.01:
            return "rapid" if avg_rate > 0.05 else "increasing"
        elif avg_rate < -0.005:
            return "decreasing"
        return "stable"
    
    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0


# =============================================================================
# 多传感器融合引擎
# =============================================================================

class FireFusionEngine:
    """
    多传感器融合 - 火灾置信度计算
    
    融合来源:
    - 视觉检测 (YOLOv8)
    - 烟雾传感器
    - 温度传感器
    - CO传感器
    - 热成像
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.visual_weight = config.get("visual_weight", 0.4)
        self.smoke_sensor_weight = config.get("smoke_sensor_weight", 0.3)
        self.thermal_sensor_weight = config.get("thermal_sensor_weight", 0.3)
        self.co_alarm_ppm = config.get("co_alarm_ppm", 50)
        self.temp_alarm = config.get("temperature_alarm_celsius", 70)
        self.temp_warning = config.get("temperature_warning_celsius", 55)
        self.fusion_method = config.get("fusion_method", "weighted_ds")
    
    def fuse(
        self,
        visual_confidence: float,
        sensor_reading: Optional[SensorReading],
        thermal_confidence: float = 0.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        融合多源数据计算火灾综合置信度
        
        Returns:
            (fusion_confidence, details_dict)
        """
        evidences = {"visual": visual_confidence}
        
        # 传感器证据
        sensor_conf = 0.0
        if sensor_reading:
            smoke_evidence = min(1.0, sensor_reading.smoke_concentration / 100.0)
            temp_evidence = 0.0
            if sensor_reading.temperature > self.temp_alarm:
                temp_evidence = 1.0
            elif sensor_reading.temperature > self.temp_warning:
                temp_evidence = (sensor_reading.temperature - self.temp_warning) / (self.temp_alarm - self.temp_warning)
            
            co_evidence = min(1.0, sensor_reading.co_concentration / self.co_alarm_ppm) if self.co_alarm_ppm > 0 else 0.0
            
            sensor_conf = max(smoke_evidence, temp_evidence, co_evidence)
            evidences["smoke_sensor"] = smoke_evidence
            evidences["temperature"] = temp_evidence
            evidences["co_sensor"] = co_evidence
        
        # 热成像证据
        evidences["thermal"] = thermal_confidence
        
        # DS理论加权融合
        if self.fusion_method == "weighted_ds":
            fusion = self._dempster_shafer_fusion(evidences)
        else:
            # 简单加权平均
            weights = {
                "visual": self.visual_weight,
                "smoke_sensor": self.smoke_sensor_weight * 0.5,
                "temperature": self.thermal_sensor_weight * 0.3,
                "co_sensor": self.smoke_sensor_weight * 0.5,
                "thermal": self.thermal_sensor_weight * 0.7,
            }
            total_weight = sum(weights.get(k, 0) for k in evidences)
            if total_weight > 0:
                fusion = sum(evidences[k] * weights.get(k, 0) for k in evidences) / total_weight
            else:
                fusion = visual_confidence
        
        details = {
            "method": self.fusion_method,
            "evidences": evidences,
            "fusion_confidence": round(fusion, 4),
        }
        
        return round(fusion, 4), details
    
    def _dempster_shafer_fusion(self, evidences: Dict[str, float]) -> float:
        """Dempster-Shafer理论融合"""
        combined = 0.0
        total_weight = 0.0
        
        weight_map = {
            "visual": self.visual_weight,
            "smoke_sensor": self.smoke_sensor_weight,
            "temperature": self.thermal_sensor_weight * 0.5,
            "co_sensor": self.smoke_sensor_weight * 0.5,
            "thermal": self.thermal_sensor_weight,
        }
        
        for key, value in evidences.items():
            w = weight_map.get(key, 0.1)
            # DS mass assignment
            m_fire = value * w
            combined += m_fire
            total_weight += w
        
        if total_weight > 0:
            combined /= total_weight
        
        # 多源增强: 多个源同时报警时提升置信度
        active_sources = sum(1 for v in evidences.values() if v > 0.3)
        if active_sources >= 3:
            combined = min(1.0, combined * 1.3)
        elif active_sources >= 2:
            combined = min(1.0, combined * 1.15)
        
        return min(1.0, combined)


# =============================================================================
# 消防检测器
# =============================================================================

class FireDetector:
    """
    消防监测核心检测器
    
    集成视觉检测、传感器融合、火势跟踪和灭火联动
    """
    
    # 检测类别
    CLASSES = ["fire", "smoke", "spark", "ember"]
    
    # 火灾等级映射
    LEVEL_THRESHOLDS = {
        FireLevel.SMOLDERING: {"min_conf": 0.3, "max_area": 0.02, "smoke_only": True},
        FireLevel.SMALL: {"min_conf": 0.5, "max_area": 0.05},
        FireLevel.MEDIUM: {"min_conf": 0.6, "max_area": 0.15},
        FireLevel.LARGE: {"min_conf": 0.7, "max_area": 0.30},
        FireLevel.CRITICAL: {"min_conf": 0.8, "min_area": 0.30},
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 模型配置
        model_cfg = config.get("model", {})
        self.model_path = model_cfg.get("path", "models/fire/fire_smoke_yolov8n.onnx")
        self.device = model_cfg.get("device", "cpu")
        self.input_size = tuple(model_cfg.get("input_size", [640, 640]))
        
        # 检测参数
        det_cfg = config.get("detection", {})
        self.confidence_threshold = det_cfg.get("confidence_threshold", 0.45)
        self.nms_threshold = det_cfg.get("nms_threshold", 0.5)
        self.fire_area_alarm = det_cfg.get("fire_area_alarm_ratio", 0.05)
        self.smoke_density_alarm = det_cfg.get("smoke_density_alarm", 0.3)
        self.min_confirm_frames = det_cfg.get("min_confirm_frames", 3)
        
        # 区域配置
        self.zones = self._parse_zones(config.get("zones", []))
        
        # 灭火联动配置
        self.suppression_config = config.get("suppression", {})
        
        # 应急疏散配置
        self.evacuation_config = config.get("evacuation", {})
        
        # 火灾等级配置覆盖
        level_cfg = config.get("fire_level", {})
        if level_cfg:
            self._update_level_thresholds(level_cfg)
        
        # 核心组件
        self._session: Optional[Any] = None
        self._tracker = FireTracker(
            max_age=det_cfg.get("max_track_age", 30),
            min_confirm=self.min_confirm_frames,
        )
        self._fusion = FireFusionEngine(config.get("sensor_fusion", {}))
        
        # 历史状态
        self._event_history: deque = deque(maxlen=config.get("history", {}).get("max_events", 10000))
        self._consecutive_alarm_frames = 0
        self._last_fire_level = FireLevel.NONE
        
        # 统计
        self._inference_count = 0
        self._alarm_count = 0
        self._initialized = False
    
    def initialize(self) -> bool:
        """初始化检测器，加载模型"""
        try:
            model_path = Path(self.model_path)
            if model_path.exists() and ort is not None:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(str(model_path), providers=providers)
                logger.info(f"[FireDetector] ONNX模型加载成功: {model_path}")
            else:
                logger.warning(f"[FireDetector] 模型不存在或onnxruntime不可用，使用模拟模式")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"[FireDetector] 初始化失败: {e}")
            self._initialized = False
            return False
    
    def detect(
        self,
        frame: np.ndarray,
        thermal_frame: Optional[np.ndarray] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
    ) -> FireAssessment:
        """
        执行完整火灾检测流程
        
        Args:
            frame: 可见光图像 (BGR, HxWxC)
            thermal_frame: 热成像帧 (可选)
            sensor_data: 传感器数据 (可选)
            
        Returns:
            FireAssessment 综合评估结果
        """
        start_time = time.time()
        self._inference_count += 1
        
        # 1. 视觉检测
        raw_detections = self._visual_detect(frame)
        
        # 2. 热成像检测
        thermal_conf = 0.0
        if thermal_frame is not None:
            thermal_conf = self._thermal_detect(thermal_frame)
        
        # 3. 跟踪与确认
        tracked = self._tracker.update(raw_detections)
        confirmed = [d for d in tracked if d.get("confirmed", False)]
        
        # 4. 构建FireDetection列表
        fire_detections = self._build_fire_detections(confirmed, frame.shape)
        
        # 5. 传感器融合
        sensor_reading = self._parse_sensor_data(sensor_data)
        visual_conf = max((d.confidence for d in fire_detections), default=0.0)
        fusion_conf, fusion_details = self._fusion.fuse(visual_conf, sensor_reading, thermal_conf)
        
        # 6. 火灾等级评估
        fire_level = self._assess_fire_level(fire_detections, fusion_conf, sensor_reading)
        
        # 7. 扩散趋势
        spread_trend = self._tracker.get_spread_trend()
        
        # 8. 灭火联动
        suppression_actions = self._determine_suppression(fire_level, fire_detections)
        
        # 9. 疏散评估
        evacuation_needed = fire_level in (FireLevel.LARGE, FireLevel.CRITICAL)
        
        # 10. 更新历史
        self._update_state(fire_level)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        assessment = FireAssessment(
            fire_level=fire_level,
            visual_confidence=visual_conf,
            sensor_confidence=sensor_reading.smoke_concentration / 100.0 if sensor_reading else 0.0,
            fusion_confidence=fusion_conf,
            detections=fire_detections,
            sensor_reading=sensor_reading,
            active_zones=[d.zone_id for d in fire_detections if d.zone_id],
            spread_trend=spread_trend,
            suppression_actions=suppression_actions,
            evacuation_needed=evacuation_needed,
            metadata={
                "inference_time_ms": round(elapsed_ms, 2),
                "inference_count": self._inference_count,
                "fusion_details": fusion_details,
                "tracker_active": len(self._tracker._tracks),
                "model_loaded": self._session is not None,
            },
        )
        
        return assessment
    
    # =========================================================================
    # 视觉检测
    # =========================================================================
    
    def _visual_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """YOLOv8视觉检测"""
        if self._session is not None:
            return self._yolo_inference(frame)
        else:
            return self._simulate_detection(frame)
    
    def _yolo_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """使用ONNX Runtime执行YOLOv8推理"""
        try:
            h, w = frame.shape[:2]
            # 预处理
            blob = self._preprocess(frame)
            
            # 推理
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: blob})
            
            # 后处理
            detections = self._postprocess(outputs[0], w, h)
            return detections
            
        except Exception as e:
            logger.error(f"[FireDetector] YOLO推理失败: {e}")
            return []
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """图像预处理"""
        if cv2 is None:
            return np.random.randn(1, 3, *self.input_size).astype(np.float32)
        
        img = cv2.resize(frame, self.input_size)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def _postprocess(self, output: np.ndarray, orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
        """YOLOv8后处理"""
        detections = []
        
        if output.ndim == 3:
            output = output[0]
        
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        for row in output:
            if len(row) < 4 + len(self.CLASSES):
                continue
            
            cx, cy, w, h = row[:4]
            class_scores = row[4:4 + len(self.CLASSES)]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            if confidence < self.confidence_threshold:
                continue
            
            # 归一化坐标
            x = float((cx - w / 2) / self.input_size[0])
            y = float((cy - h / 2) / self.input_size[1])
            nw = float(w / self.input_size[0])
            nh = float(h / self.input_size[1])
            
            detections.append({
                "bbox": {"x": max(0, x), "y": max(0, y), "width": min(1.0, nw), "height": min(1.0, nh)},
                "type": self.CLASSES[class_id] if class_id < len(self.CLASSES) else "unknown",
                "confidence": confidence,
                "class_id": class_id,
            })
        
        # NMS
        if detections:
            detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """非极大值抑制"""
        if len(detections) <= 1:
            return detections
        
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept = []
        
        while sorted_dets:
            best = sorted_dets.pop(0)
            kept.append(best)
            sorted_dets = [
                d for d in sorted_dets
                if FireTracker._compute_iou(best["bbox"], d["bbox"]) < self.nms_threshold
            ]
        
        return kept
    
    def _simulate_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """模拟检测（无模型时使用颜色特征）"""
        detections = []
        
        if cv2 is None or frame is None:
            return detections
        
        h, w = frame.shape[:2]
        
        # 火焰颜色检测 (HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 火焰: 低色调、高饱和度、高亮度
        lower_fire = np.array([0, 80, 180])
        upper_fire = np.array([30, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        
        # 烟雾: 低饱和度、中等亮度（灰色）
        lower_smoke = np.array([0, 0, 100])
        upper_smoke = np.array([180, 60, 200])
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)
        
        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        
        for mask, fire_type in [(fire_mask, "fire"), (smoke_mask, "smoke")]:
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                min_area = (w * h) * 0.001  # 最小面积阈值
                
                if area < min_area:
                    continue
                
                x, y, bw, bh = cv2.boundingRect(contour)
                confidence = min(0.9, 0.3 + (area / (w * h)) * 5)
                
                detections.append({
                    "bbox": {
                        "x": float(x / w),
                        "y": float(y / h),
                        "width": float(bw / w),
                        "height": float(bh / h),
                    },
                    "type": fire_type,
                    "confidence": confidence,
                    "class_id": self.CLASSES.index(fire_type) if fire_type in self.CLASSES else 0,
                })
        
        return detections
    
    # =========================================================================
    # 热成像检测
    # =========================================================================
    
    def _thermal_detect(self, thermal_frame: np.ndarray) -> float:
        """热成像异常检测"""
        if thermal_frame is None:
            return 0.0
        
        try:
            # 简化: 基于温度阈值
            if thermal_frame.ndim == 2:
                max_temp = float(np.max(thermal_frame))
                avg_temp = float(np.mean(thermal_frame))
            else:
                gray = thermal_frame[:, :, 0] if thermal_frame.shape[2] >= 1 else thermal_frame
                max_temp = float(np.max(gray))
                avg_temp = float(np.mean(gray))
            
            # 归一化置信度
            if max_temp > self._fusion.temp_alarm:
                return min(1.0, 0.7 + (max_temp - self._fusion.temp_alarm) / 100)
            elif max_temp > self._fusion.temp_warning:
                return (max_temp - self._fusion.temp_warning) / (self._fusion.temp_alarm - self._fusion.temp_warning) * 0.7
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"[FireDetector] 热成像检测异常: {e}")
            return 0.0
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _build_fire_detections(self, tracked: List[Dict], frame_shape: Tuple) -> List[FireDetection]:
        """构建FireDetection列表"""
        h, w = frame_shape[:2]
        frame_area = w * h if w * h > 0 else 1
        
        results = []
        for t in tracked:
            bbox = t["bbox"]
            area_ratio = bbox.get("width", 0) * bbox.get("height", 0)
            
            # 匹配区域
            zone_id, zone_name = self._match_zone(bbox)
            
            results.append(FireDetection(
                fire_type=FireType(t.get("type", "fire")) if t.get("type") in [e.value for e in FireType] else FireType.FIRE,
                bbox=bbox,
                confidence=t.get("confidence", 0),
                area_ratio=area_ratio,
                zone_id=zone_id,
                zone_name=zone_name,
                track_id=t.get("track_id", -1),
                spread_rate=t.get("spread_rate", 0.0),
            ))
        
        return results
    
    def _match_zone(self, bbox: Dict[str, float]) -> Tuple[str, str]:
        """匹配检测框所在区域"""
        center_x = bbox.get("x", 0) + bbox.get("width", 0) / 2
        center_y = bbox.get("y", 0) + bbox.get("height", 0) / 2
        
        for zone in self.zones:
            polygon = zone.get("polygon", [])
            if self._point_in_polygon(center_x, center_y, polygon):
                return zone["zone_id"], zone.get("zone_name", "")
        
        return "", ""
    
    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(polygon)
        if n < 3:
            return False
        
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _parse_sensor_data(self, data: Optional[Dict]) -> Optional[SensorReading]:
        if data is None:
            return None
        return SensorReading(
            smoke_concentration=data.get("smoke_concentration", 0),
            temperature=data.get("temperature", 25),
            co_concentration=data.get("co_concentration", 0),
            humidity=data.get("humidity", 50),
            timestamp=data.get("timestamp", time.time()),
        )
    
    def _parse_zones(self, zones_list: List[Dict]) -> List[Dict]:
        parsed = []
        for z in zones_list:
            parsed.append({
                "zone_id": z.get("zone_id", f"zone_{len(parsed)}"),
                "zone_name": z.get("zone_name", ""),
                "polygon": z.get("polygon", []),
                "priority": z.get("priority", "medium"),
            })
        return parsed
    
    def _update_level_thresholds(self, cfg: Dict):
        for level_name, params in cfg.items():
            try:
                level = FireLevel(level_name)
                if level in self.LEVEL_THRESHOLDS and isinstance(params, dict):
                    self.LEVEL_THRESHOLDS[level].update(params)
            except ValueError:
                pass
    
    def _assess_fire_level(
        self,
        detections: List[FireDetection],
        fusion_conf: float,
        sensor: Optional[SensorReading],
    ) -> FireLevel:
        """综合评估火灾等级"""
        if not detections and fusion_conf < 0.2:
            return FireLevel.NONE
        
        max_area = max((d.area_ratio for d in detections), default=0)
        has_fire = any(d.fire_type == FireType.FIRE for d in detections)
        has_smoke = any(d.fire_type == FireType.SMOKE for d in detections)
        
        # 从高到低判定
        if fusion_conf >= 0.8 and max_area >= 0.30:
            return FireLevel.CRITICAL
        if fusion_conf >= 0.7 and max_area >= 0.15:
            return FireLevel.LARGE
        if fusion_conf >= 0.6 and max_area >= 0.05:
            return FireLevel.MEDIUM
        if fusion_conf >= 0.5 and has_fire:
            return FireLevel.SMALL
        if fusion_conf >= 0.3 and has_smoke and not has_fire:
            return FireLevel.SMOLDERING
        
        # 传感器单独触发
        if sensor:
            if sensor.temperature > self._fusion.temp_alarm or sensor.co_concentration > self._fusion.co_alarm_ppm:
                return FireLevel.SMALL
            if sensor.smoke_concentration > 30:
                return FireLevel.SMOLDERING
        
        return FireLevel.NONE
    
    def _determine_suppression(self, level: FireLevel, detections: List[FireDetection]) -> List[Dict]:
        """确定灭火联动动作"""
        actions = []
        trigger = self.suppression_config.get("trigger_level", "alarm")
        
        level_order = [FireLevel.NONE, FireLevel.SMOLDERING, FireLevel.SMALL, FireLevel.MEDIUM, FireLevel.LARGE, FireLevel.CRITICAL]
        trigger_map = {"warning": FireLevel.SMOLDERING, "alarm": FireLevel.SMALL, "critical": FireLevel.LARGE}
        min_trigger = trigger_map.get(trigger, FireLevel.SMALL)
        
        if level_order.index(level) < level_order.index(min_trigger):
            return actions
        
        # 声光报警
        if self.suppression_config.get("alarm_sound_enabled", True):
            actions.append({"action": "activate_alarm_sound", "level": level.value, "zones": [d.zone_id for d in detections if d.zone_id]})
        
        if self.suppression_config.get("alarm_light_enabled", True):
            actions.append({"action": "activate_alarm_light", "level": level.value, "color": "red"})
        
        # 自动喷淋
        if self.suppression_config.get("auto_sprinkler_enabled", False) and level in (FireLevel.MEDIUM, FireLevel.LARGE, FireLevel.CRITICAL):
            zones = [d.zone_id for d in detections if d.zone_id]
            actions.append({"action": "activate_sprinkler", "zones": zones})
        
        # 自动断电
        if self.suppression_config.get("auto_power_cutoff", False) and level in (FireLevel.LARGE, FireLevel.CRITICAL):
            actions.append({"action": "power_cutoff", "zones": [d.zone_id for d in detections if d.zone_id]})
        
        return actions
    
    def _update_state(self, level: FireLevel):
        """更新内部状态"""
        if level != FireLevel.NONE:
            self._consecutive_alarm_frames += 1
            self._alarm_count += 1
        else:
            self._consecutive_alarm_frames = 0
        
        self._last_fire_level = level
        
        self._event_history.append({
            "timestamp": time.time(),
            "level": level.value,
            "consecutive": self._consecutive_alarm_frames,
        })
    
    def get_evacuation_routes(self) -> List[Dict]:
        """获取疏散路线"""
        routes = self.evacuation_config.get("routes", [])
        assembly = self.evacuation_config.get("assembly_point", "")
        return [{"routes": routes, "assembly_point": assembly}]
    
    def reset(self):
        """重置检测器状态"""
        self._tracker.reset()
        self._consecutive_alarm_frames = 0
        self._last_fire_level = FireLevel.NONE
        self._event_history.clear()
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def stats(self) -> Dict:
        return {
            "inference_count": self._inference_count,
            "alarm_count": self._alarm_count,
            "model_loaded": self._session is not None,
            "active_tracks": len(self._tracker._tracks),
            "last_fire_level": self._last_fire_level.value,
        }
