"""
电容器自主巡视检测器 - 增强版
输变电激光监测平台 (D组) - 全自动AI巡检改造

增强功能:
- YOLOv8电容器检测: 精确定位电容器单元
- 姿态估计倾斜分析: 基于几何的倾斜角度计算
- RT-DETR入侵检测: 人/车/动物实时检测
- 时序入侵确认: 防止瞬时误报
- 三相排列校验: 电容器组完整性验证
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import time
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class CapacitorDefectType(Enum):
    """电容器缺陷类型"""
    TILT_WARNING = "tilt_warning"       # 倾斜警告
    TILT_ERROR = "tilt_error"           # 倾斜严重
    COLLAPSE = "collapse"               # 倒塌
    MISSING_UNIT = "missing_unit"       # 单元缺失
    DEFORMATION = "deformation"         # 变形
    INSULATOR_DAMAGE = "insulator_damage"  # 绝缘子损坏


class IntrusionType(Enum):
    """入侵类型"""
    PERSON = "person"                   # 人员
    VEHICLE = "vehicle"                 # 车辆
    ANIMAL = "animal"                   # 动物
    UNKNOWN = "unknown"                 # 未知


class ZoneType(Enum):
    """区域类型"""
    RESTRICTED = "restricted"           # 禁入区
    WARNING = "warning"                 # 警告区
    EQUIPMENT = "equipment"             # 设备区


@dataclass
class CapacitorDetection:
    """电容器检测结果"""
    defect_type: CapacitorDefectType
    bbox: Dict[str, float]
    confidence: float
    class_name: str
    tilt_angle: Optional[float] = None  # 倾斜角度
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntrusionDetection:
    """入侵检测结果"""
    intrusion_type: IntrusionType
    bbox: Dict[str, float]
    confidence: float
    zone: ZoneType
    track_id: Optional[int] = None      # 跟踪ID
    duration_sec: float = 0.0           # 持续时间
    confirmed: bool = False             # 是否确认
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapacitorBankStatus:
    """电容器组状态"""
    total_units: int                    # 总单元数
    detected_units: int                 # 检测到的单元数
    missing_positions: List[Tuple[int, int]] = field(default_factory=list)  # 缺失位置
    tilted_units: List[Dict] = field(default_factory=list)  # 倾斜单元
    alignment_score: float = 1.0        # 排列整齐度
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapacitorInspectionResult:
    """电容器巡视综合结果"""
    structural_defects: List[CapacitorDetection] = field(default_factory=list)
    intrusions: List[IntrusionDetection] = field(default_factory=list)
    bank_status: Optional[CapacitorBankStatus] = None
    processing_time_ms: float = 0.0
    model_version: str = ""
    code_hash: str = ""


class IntrusionTracker:
    """入侵目标跟踪器"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age          # 最大丢失帧数
        self.min_hits = min_hits        # 最小命中次数
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
    
    def update(self, detections: List[Dict], timestamp: float) -> List[Dict]:
        """更新跟踪"""
        updated_tracks = []
        
        # 匹配检测和跟踪
        matched, unmatched_dets, unmatched_tracks = self._match(detections)
        
        # 更新匹配的跟踪
        for det_idx, track_id in matched:
            det = detections[det_idx]
            self.tracks[track_id]["bbox"] = det["bbox"]
            self.tracks[track_id]["confidence"] = det["confidence"]
            self.tracks[track_id]["hits"] += 1
            self.tracks[track_id]["age"] = 0
            self.tracks[track_id]["last_seen"] = timestamp
            
            track = self.tracks[track_id]
            if track["hits"] >= self.min_hits:
                track["confirmed"] = True
                track["duration"] = timestamp - track["first_seen"]
            
            updated_tracks.append({**det, "track_id": track_id, **track})
        
        # 创建新跟踪
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self.tracks[self.next_id] = {
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "type": det.get("type", "unknown"),
                "hits": 1,
                "age": 0,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "confirmed": False,
                "duration": 0.0,
            }
            updated_tracks.append({**det, "track_id": self.next_id, **self.tracks[self.next_id]})
            self.next_id += 1
        
        # 老化未匹配的跟踪
        for track_id in unmatched_tracks:
            self.tracks[track_id]["age"] += 1
            if self.tracks[track_id]["age"] > self.max_age:
                del self.tracks[track_id]
        
        return updated_tracks
    
    def _match(self, detections: List[Dict]) -> Tuple[List, List, List]:
        """匹配检测和跟踪"""
        if not detections or not self.tracks:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        # 计算IoU矩阵
        det_bboxes = [d["bbox"] for d in detections]
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]["bbox"] for tid in track_ids]
        
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        for i, db in enumerate(det_bboxes):
            for j, tb in enumerate(track_bboxes):
                iou_matrix[i, j] = self._iou(db, tb)
        
        # 贪婪匹配
        matched = []
        matched_dets = set()
        matched_tracks = set()
        
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                break
            
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matched.append((i, track_ids[j]))
            matched_dets.add(i)
            matched_tracks.add(track_ids[j])
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]
        
        return matched, unmatched_dets, unmatched_tracks
    
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


class CapacitorDetectorEnhanced:
    """
    电容器巡视增强检测器
    
    集成深度学习进行结构缺陷检测和入侵检测
    """
    
    # 模型ID映射
    MODEL_IDS = {
        "capacitor": "capacitor_yolov8",        # 电容器检测
        "intrusion": "rtdetr_intrusion",        # 入侵检测
    }
    
    # 入侵类别映射
    INTRUSION_CLASSES = {
        0: IntrusionType.PERSON,
        1: IntrusionType.VEHICLE,
        2: IntrusionType.ANIMAL,
    }
    
    # 默认配置
    DEFAULT_TILT_WARNING = 3.0      # 倾斜警告角度
    DEFAULT_TILT_ERROR = 5.0        # 倾斜严重角度
    DEFAULT_ALERT_DELAY = 2.0       # 入侵告警延迟
    
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
        self._confidence_threshold = config.get("confidence_threshold", 0.55)
        self._nms_threshold = config.get("nms_threshold", 0.4)
        self._use_deep_learning = config.get("use_deep_learning", True)
        
        # 倾斜检测
        tilt_config = config.get("tilt_detection", {})
        self._tilt_warning = tilt_config.get("warning_angle", self.DEFAULT_TILT_WARNING)
        self._tilt_error = tilt_config.get("max_tilt_angle", self.DEFAULT_TILT_ERROR)
        
        # 入侵检测
        intrusion_config = config.get("intrusion_detection", {})
        self._intrusion_enabled = intrusion_config.get("enabled", True)
        self._alert_delay = intrusion_config.get("alert_delay", self.DEFAULT_ALERT_DELAY)
        
        # 电容器组配置
        bank_config = config.get("capacitor_bank", {})
        self._expected_rows = bank_config.get("rows", 3)
        self._expected_cols = bank_config.get("columns", 4)
        
        # 入侵跟踪器
        self._intrusion_tracker = IntrusionTracker()
        
        # 版本信息
        self._model_version = "capacitor_enhanced_v1.0"
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
                        print(f"[CapacitorDetector] 模型 {model_id} 加载失败: {e}")
            
            self._initialized = True
            return True
        except Exception as e:
            print(f"[CapacitorDetector] 初始化失败: {e}")
            return False
    
    def detect_structural_defects(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None,
    ) -> List[CapacitorDetection]:
        """
        结构缺陷检测
        
        Args:
            image: BGR图像
            roi_bbox: ROI区域
            
        Returns:
            结构缺陷列表
        """
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)
        
        defects = []
        
        # 检测电容器单元
        units = self._detect_capacitor_units(image)
        
        # 分析每个单元的倾斜
        for unit in units:
            tilt_angle = self._calculate_tilt_angle(image, unit["bbox"])
            
            if abs(tilt_angle) >= self._tilt_error:
                defects.append(CapacitorDetection(
                    defect_type=CapacitorDefectType.TILT_ERROR,
                    bbox=unit["bbox"],
                    confidence=unit["confidence"],
                    class_name="电容器倾斜(严重)",
                    tilt_angle=tilt_angle,
                    metadata={"source": "tilt_analysis"}
                ))
            elif abs(tilt_angle) >= self._tilt_warning:
                defects.append(CapacitorDetection(
                    defect_type=CapacitorDefectType.TILT_WARNING,
                    bbox=unit["bbox"],
                    confidence=unit["confidence"],
                    class_name="电容器倾斜(警告)",
                    tilt_angle=tilt_angle,
                    metadata={"source": "tilt_analysis"}
                ))
        
        # 检测倒塌
        collapse_defects = self._detect_collapse(image, units)
        defects.extend(collapse_defects)
        
        # 检测缺失
        missing_defects = self._detect_missing_units(image, units)
        defects.extend(missing_defects)
        
        return defects
    
    def _detect_capacitor_units(self, image: np.ndarray) -> List[Dict]:
        """检测电容器单元"""
        units = []
        
        # 优先使用深度学习
        if self._use_deep_learning and self._model_registry:
            try:
                model_id = self.MODEL_IDS["capacitor"]
                result = self._model_registry.infer(model_id, image)
                
                for det in result.detections:
                    if det["confidence"] >= self._confidence_threshold:
                        units.append({
                            "bbox": det["bbox"],
                            "confidence": det["confidence"],
                            "class_name": det.get("class_name", "capacitor"),
                            "source": "deep_learning"
                        })
                
                return units
            except Exception as e:
                print(f"[CapacitorDetector] 深度学习检测失败: {e}")
        
        # 回退到传统方法
        return self._detect_units_traditional(image)
    
    def _detect_units_traditional(self, image: np.ndarray) -> List[Dict]:
        """传统方法检测电容器单元"""
        if cv2 is None:
            return []
        
        units = []
        h, w = image.shape[:2]
        
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = cw / (ch + 1e-6)
                
                # 电容器通常是垂直的矩形
                if 0.2 < aspect_ratio < 0.8:
                    units.append({
                        "bbox": {"x": x/w, "y": y/h, "width": cw/w, "height": ch/h},
                        "confidence": 0.7,
                        "class_name": "capacitor",
                        "source": "traditional"
                    })
        
        return units
    
    def _calculate_tilt_angle(self, image: np.ndarray, bbox: Dict[str, float]) -> float:
        """计算倾斜角度"""
        if cv2 is None:
            return 0.0
        
        h, w = image.shape[:2]
        x = int(bbox["x"] * w)
        y = int(bbox["y"] * h)
        bw = int(bbox["width"] * w)
        bh = int(bbox["height"] * h)
        
        # 裁剪区域
        roi = image[y:y+bh, x:x+bw]
        if roi.size == 0:
            return 0.0
        
        # 转换为灰度
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 霍夫线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return 0.0
        
        # 计算主要方向
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # 只考虑接近垂直的线
            if 60 < abs(angle) < 120:
                angles.append(90 - abs(angle))
        
        if not angles:
            return 0.0
        
        # 返回平均倾斜角
        return float(np.mean(angles))
    
    def _detect_collapse(
        self,
        image: np.ndarray,
        units: List[Dict]
    ) -> List[CapacitorDetection]:
        """检测倒塌"""
        if cv2 is None or not units:
            return []
        
        defects = []
        h, w = image.shape[:2]
        
        # 计算平均高度
        heights = [u["bbox"]["height"] for u in units]
        avg_height = np.mean(heights) if heights else 0
        
        for unit in units:
            bbox = unit["bbox"]
            unit_height = bbox["height"]
            
            # 高度明显低于平均值可能是倒塌
            if unit_height < avg_height * 0.5:
                defects.append(CapacitorDetection(
                    defect_type=CapacitorDefectType.COLLAPSE,
                    bbox=bbox,
                    confidence=0.8,
                    class_name="电容器倒塌",
                    metadata={
                        "height_ratio": unit_height / avg_height,
                        "source": "height_analysis"
                    }
                ))
        
        return defects
    
    def _detect_missing_units(
        self,
        image: np.ndarray,
        units: List[Dict]
    ) -> List[CapacitorDetection]:
        """检测缺失单元"""
        defects = []
        
        if len(units) < 2:
            return defects
        
        # 预期单元数
        expected_count = self._expected_rows * self._expected_cols
        detected_count = len(units)
        
        if detected_count < expected_count:
            # 分析位置找出缺失
            missing_positions = self._find_missing_positions(units)
            
            for pos in missing_positions:
                defects.append(CapacitorDetection(
                    defect_type=CapacitorDefectType.MISSING_UNIT,
                    bbox=pos["bbox"],
                    confidence=0.75,
                    class_name="电容器单元缺失",
                    metadata={
                        "expected_position": pos.get("position"),
                        "source": "grid_analysis"
                    }
                ))
        
        return defects
    
    def _find_missing_positions(self, units: List[Dict]) -> List[Dict]:
        """找出缺失位置"""
        if len(units) < 2:
            return []
        
        # 提取中心点
        centers = []
        for unit in units:
            bbox = unit["bbox"]
            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2
            centers.append((cx, cy))
        
        # 计算行列间距
        x_coords = sorted([c[0] for c in centers])
        y_coords = sorted([c[1] for c in centers])
        
        x_gaps = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        y_gaps = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        
        avg_x_gap = np.median(x_gaps) if x_gaps else 0.1
        avg_y_gap = np.median(y_gaps) if y_gaps else 0.2
        
        # 检测间隙异常
        missing = []
        for i, gap in enumerate(x_gaps):
            if gap > avg_x_gap * 1.5:  # 间隙过大
                missing.append({
                    "bbox": {
                        "x": x_coords[i] + avg_x_gap / 2,
                        "y": y_coords[0],
                        "width": avg_x_gap,
                        "height": avg_y_gap
                    },
                    "position": (i+1, 0)
                })
        
        return missing
    
    def detect_intrusion(
        self,
        image: np.ndarray,
        timestamp: float,
        zone_mask: Optional[np.ndarray] = None,
    ) -> List[IntrusionDetection]:
        """
        入侵检测
        
        Args:
            image: BGR图像
            timestamp: 时间戳
            zone_mask: 区域掩码
            
        Returns:
            入侵检测结果
        """
        if not self._intrusion_enabled:
            return []
        
        # 检测目标
        raw_detections = self._detect_intrusion_targets(image)
        
        # 更新跟踪
        tracked = self._intrusion_tracker.update(raw_detections, timestamp)
        
        # 转换为结果
        intrusions = []
        for track in tracked:
            # 确定区域
            zone = self._determine_zone(track["bbox"], zone_mask)
            
            # 检查是否确认
            confirmed = track.get("confirmed", False) and track.get("duration", 0) >= self._alert_delay
            
            intrusion_type = track.get("type", IntrusionType.UNKNOWN)
            if isinstance(intrusion_type, str):
                intrusion_type = IntrusionType(intrusion_type) if intrusion_type in [e.value for e in IntrusionType] else IntrusionType.UNKNOWN
            
            intrusions.append(IntrusionDetection(
                intrusion_type=intrusion_type,
                bbox=track["bbox"],
                confidence=track["confidence"],
                zone=zone,
                track_id=track.get("track_id"),
                duration_sec=track.get("duration", 0),
                confirmed=confirmed,
                metadata={
                    "hits": track.get("hits", 0),
                    "source": "deep_learning" if self._use_deep_learning else "traditional"
                }
            ))
        
        return intrusions
    
    def _detect_intrusion_targets(self, image: np.ndarray) -> List[Dict]:
        """检测入侵目标"""
        detections = []
        
        # 优先使用深度学习
        if self._use_deep_learning and self._model_registry:
            try:
                model_id = self.MODEL_IDS["intrusion"]
                result = self._model_registry.infer(model_id, image)
                
                for det in result.detections:
                    if det["confidence"] >= self._confidence_threshold:
                        class_id = det.get("class_id", 0)
                        intrusion_type = self.INTRUSION_CLASSES.get(class_id, IntrusionType.UNKNOWN)
                        
                        detections.append({
                            "bbox": det["bbox"],
                            "confidence": det["confidence"],
                            "type": intrusion_type.value,
                            "class_name": det.get("class_name", intrusion_type.value),
                        })
                
                return detections
            except Exception as e:
                print(f"[CapacitorDetector] 入侵检测失败: {e}")
        
        # 回退到传统方法(运动检测)
        return self._detect_intrusion_traditional(image)
    
    def _detect_intrusion_traditional(self, image: np.ndarray) -> List[Dict]:
        """传统方法入侵检测"""
        if cv2 is None:
            return []
        
        # 简化实现: 基于背景减除
        # 实际应用中应使用更复杂的方法
        return []
    
    def _determine_zone(
        self,
        bbox: Dict[str, float],
        zone_mask: Optional[np.ndarray] = None
    ) -> ZoneType:
        """确定所在区域"""
        if zone_mask is None:
            return ZoneType.EQUIPMENT
        
        # 计算中心点
        cx = int((bbox["x"] + bbox["width"] / 2) * zone_mask.shape[1])
        cy = int((bbox["y"] + bbox["height"] / 2) * zone_mask.shape[0])
        
        cx = max(0, min(cx, zone_mask.shape[1] - 1))
        cy = max(0, min(cy, zone_mask.shape[0] - 1))
        
        zone_value = zone_mask[cy, cx]
        
        zone_map = {
            0: ZoneType.EQUIPMENT,
            1: ZoneType.WARNING,
            2: ZoneType.RESTRICTED,
        }
        
        return zone_map.get(int(zone_value), ZoneType.EQUIPMENT)
    
    def analyze_bank_status(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None,
    ) -> CapacitorBankStatus:
        """
        分析电容器组状态
        
        Args:
            image: BGR图像
            roi_bbox: ROI区域
            
        Returns:
            电容器组状态
        """
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)
        
        # 检测所有单元
        units = self._detect_capacitor_units(image)
        
        # 计算排列整齐度
        alignment_score = self._calculate_alignment_score(units)
        
        # 找出倾斜单元
        tilted_units = []
        for unit in units:
            tilt = self._calculate_tilt_angle(image, unit["bbox"])
            if abs(tilt) >= self._tilt_warning:
                tilted_units.append({
                    "bbox": unit["bbox"],
                    "tilt_angle": tilt
                })
        
        # 找出缺失位置
        missing = self._find_missing_positions(units)
        missing_positions = [m.get("position", (0, 0)) for m in missing]
        
        return CapacitorBankStatus(
            total_units=self._expected_rows * self._expected_cols,
            detected_units=len(units),
            missing_positions=missing_positions,
            tilted_units=tilted_units,
            alignment_score=alignment_score,
            metadata={
                "expected_rows": self._expected_rows,
                "expected_cols": self._expected_cols,
            }
        )
    
    def _calculate_alignment_score(self, units: List[Dict]) -> float:
        """计算排列整齐度"""
        if len(units) < 2:
            return 1.0
        
        # 提取中心点
        centers = []
        for unit in units:
            bbox = unit["bbox"]
            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2
            centers.append((cx, cy))
        
        # 计算行方向标准差
        y_coords = [c[1] for c in centers]
        y_std = np.std(y_coords) if len(y_coords) > 1 else 0
        
        # 计算列方向标准差
        x_coords = [c[0] for c in centers]
        x_std = np.std(x_coords) if len(x_coords) > 1 else 0
        
        # 标准差越小，整齐度越高
        alignment = 1.0 - min(1.0, (y_std + x_std) * 10)

        return float(max(0.0, alignment))
    
    def inspect(
        self,
        image: np.ndarray,
        timestamp: Optional[float] = None,
        roi_bbox: Optional[Dict[str, float]] = None,
        zone_mask: Optional[np.ndarray] = None,
    ) -> CapacitorInspectionResult:
        """
        综合巡视
        
        Args:
            image: BGR图像
            timestamp: 时间戳
            roi_bbox: ROI区域
            zone_mask: 区域掩码
            
        Returns:
            综合巡视结果
        """
        start_time = time.perf_counter()
        
        if timestamp is None:
            timestamp = time.time()
        
        # 结构缺陷检测
        structural_defects = self.detect_structural_defects(image, roi_bbox)
        
        # 入侵检测
        intrusions = self.detect_intrusion(image, timestamp, zone_mask)
        
        # 电容器组状态
        bank_status = self.analyze_bank_status(image, roi_bbox)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return CapacitorInspectionResult(
            structural_defects=structural_defects,
            intrusions=intrusions,
            bank_status=bank_status,
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
def create_detector(config: Dict[str, Any], model_registry=None) -> CapacitorDetectorEnhanced:
    """创建检测器实例"""
    detector = CapacitorDetectorEnhanced(config, model_registry)
    detector.initialize()
    return detector