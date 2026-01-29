"""
电容器自主巡视检测器 - 增强版 V3.5
输变电激光监测平台 (D组) - 全自动AI巡检改造

增强功能:
- YOLOv8-ViT电容器检测 (UPDATE.md短期计划)
- 姿态估计倾斜分析: 基于几何的倾斜角度计算
- RT-DETR入侵检测: 人/车/动物实时检测
- 时序入侵确认: 防止瞬时误报
- 三相排列校验: 电容器组完整性验证

V3.0更新:
- 集成YOLOv8-ViT深度学习模型
- 增强鼓包/渗漏检测能力
- 支持多类型缺陷统一检测

V3.5更新 (室外监测迭代):
- 红外-可见光图像配准: 多模态融合检测
- YOLOv8-OBB旋转框检测: 倾斜电容器精确定位
- 热点异常检测: 自动识别过热区域
- 多模态证据融合: 可见光+热成像联合判定
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

# V3.0: 导入YOLOv8-ViT深度学习模型
try:
    from ai_models.deep_learning.yolov8_vit import (
        YOLOv8ViTDetector, YOLOv8ViTConfig, DetectionTask
    )
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    YOLOv8ViTDetector = None
    YOLOv8ViTConfig = None
    DetectionTask = None

# V3.5: 导入红外-可见光配准模块
try:
    from ai_models.deep_learning.thermal_visible_registration import (
        ThermalVisibleRegistration,
        RegistrationConfig,
        FusionConfig,
        FusionMethod
    )
    THERMAL_REGISTRATION_AVAILABLE = True
except ImportError:
    THERMAL_REGISTRATION_AVAILABLE = False
    ThermalVisibleRegistration = None
    RegistrationConfig = None
    FusionConfig = None
    FusionMethod = None

# V3.5: 导入YOLOv8-OBB旋转框检测模块
try:
    from ai_models.deep_learning.yolov8_obb import (
        YOLOv8OBBDetector,
        OBBConfig,
        OBBDetectionTask,
        OrientedBox
    )
    OBB_AVAILABLE = True
except ImportError:
    OBB_AVAILABLE = False
    YOLOv8OBBDetector = None
    OBBConfig = None
    OBBDetectionTask = None
    OrientedBox = None


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
            model_registry: 模型注册表实例 (已废弃，保留兼容性)
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

        # V3.0: YOLOv8-ViT深度学习检测器
        self._yolov8_vit_detector: Optional[YOLOv8ViTDetector] = None
        self._dl_initialized = False

        # V3.5: 红外-可见光配准模块
        self._thermal_registration: Optional[Any] = None
        self._thermal_fusion_enabled = config.get("thermal_fusion", {}).get("enabled", True)

        # V3.5: YOLOv8-OBB旋转框检测器
        self._obb_detector: Optional[Any] = None
        self._obb_enabled = config.get("obb_detection", {}).get("enabled", True)

        # 版本信息 (V3.5更新)
        self._model_version = "capacitor_enhanced_v3.5"
        self._code_hash = self._calculate_code_hash()
    
    def _calculate_code_hash(self) -> str:
        """计算代码版本hash"""
        import inspect
        source = inspect.getsource(self.__class__)
        return f"sha256:{hashlib.sha256(source.encode()).hexdigest()[:12]}"
    
    def initialize(self) -> bool:
        """初始化检测器"""
        try:
            # V3.0: 优先初始化YOLOv8-ViT深度学习检测器
            if self._use_deep_learning and DL_AVAILABLE:
                self._init_yolov8_vit()

            # V3.5: 初始化红外-可见光配准模块
            if self._thermal_fusion_enabled and THERMAL_REGISTRATION_AVAILABLE:
                self._init_thermal_registration()

            # V3.5: 初始化YOLOv8-OBB旋转框检测器
            if self._obb_enabled and OBB_AVAILABLE:
                self._init_obb_detector()

            # 兼容旧版：如果有模型注册表，预加载模型
            if self._model_registry and self._use_deep_learning and not self._dl_initialized:
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

    def _init_thermal_registration(self) -> bool:
        """初始化红外-可见光配准模块 (V3.5)"""
        if not THERMAL_REGISTRATION_AVAILABLE or ThermalVisibleRegistration is None:
            print("[CapacitorDetector] 红外配准模块不可用")
            return False

        try:
            thermal_config = self.config.get("thermal_fusion", {})

            reg_config = RegistrationConfig(
                min_match_count=thermal_config.get("min_match_count", 10),
                ransac_reproj_threshold=thermal_config.get("ransac_threshold", 5.0),
                max_features=thermal_config.get("max_features", 1000)
            )

            fusion_config = FusionConfig(
                method=FusionMethod.LAPLACIAN_PYRAMID,
                thermal_weight=thermal_config.get("thermal_weight", 0.4),
                visible_weight=thermal_config.get("visible_weight", 0.6),
                enhance_thermal=True
            )

            self._thermal_registration = ThermalVisibleRegistration(
                registration_config=reg_config,
                fusion_config=fusion_config
            )

            print("[CapacitorDetector] 红外-可见光配准模块初始化成功 (V3.5)")
            return True

        except Exception as e:
            print(f"[CapacitorDetector] 红外配准初始化失败: {e}")
            return False

    def _init_obb_detector(self) -> bool:
        """初始化YOLOv8-OBB旋转框检测器 (V3.5)"""
        if not OBB_AVAILABLE or YOLOv8OBBDetector is None:
            print("[CapacitorDetector] YOLOv8-OBB模块不可用")
            return False

        try:
            obb_config = self.config.get("obb_detection", {})

            config = OBBConfig(
                model_path=obb_config.get("model_path"),
                task=OBBDetectionTask.CAPACITOR_DEFECT,
                confidence_threshold=self._confidence_threshold,
                nms_threshold=self._nms_threshold,
                tilt_warning_threshold=self._tilt_warning,
                tilt_critical_threshold=self._tilt_error,
                device=self.config.get("device", "cpu")
            )

            self._obb_detector = YOLOv8OBBDetector(config)
            self._obb_detector.load()

            print("[CapacitorDetector] YOLOv8-OBB旋转框检测器初始化成功 (V3.5)")
            return True

        except Exception as e:
            print(f"[CapacitorDetector] OBB检测器初始化失败: {e}")
            return False

    def _init_yolov8_vit(self) -> bool:
        """初始化YOLOv8-ViT检测器 (V3.0)"""
        if not DL_AVAILABLE:
            print("[CapacitorDetector] YOLOv8-ViT模块不可用")
            return False

        try:
            model_path = self.config.get("yolov8_model_path", None)
            device = self.config.get("device", "cpu")

            config = YOLOv8ViTConfig(
                model_path=model_path,
                task=DetectionTask.CAPACITOR_DEFECT,
                confidence_threshold=self._confidence_threshold,
                nms_threshold=self._nms_threshold,
                device=device,
                use_vit_backbone=True,
                use_se_attention=True,
                use_faster_block=True,
            )

            self._yolov8_vit_detector = YOLOv8ViTDetector(config)
            self._yolov8_vit_detector.load()
            self._dl_initialized = True

            print(f"[CapacitorDetector] YOLOv8-ViT检测器初始化成功 (V3.0)")
            return True

        except Exception as e:
            print(f"[CapacitorDetector] YOLOv8-ViT初始化失败: {e}")
            self._dl_initialized = False
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
        """检测电容器单元 (V3.0: 优先使用YOLOv8-ViT)"""
        units = []

        # V3.0: 优先使用YOLOv8-ViT深度学习
        if self._use_deep_learning and self._dl_initialized and self._yolov8_vit_detector is not None:
            try:
                result = self._yolov8_vit_detector.detect(image)

                for det in result.detections:
                    if det.confidence >= self._confidence_threshold:
                        units.append({
                            "bbox": det.bbox,
                            "confidence": det.confidence,
                            "class_name": det.class_name,
                            "class_id": det.class_id,
                            "source": "yolov8_vit_v3",
                            "inference_time_ms": result.inference_time_ms,
                        })

                if units:
                    return units

            except Exception as e:
                print(f"[CapacitorDetector] YOLOv8-ViT检测失败: {e}")

        # 兼容旧版：使用model_registry
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
                            "source": "model_registry_legacy"
                        })

                return units
            except Exception as e:
                print(f"[CapacitorDetector] model_registry检测失败: {e}")

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

    # ==================== V3.5 新增方法 ====================

    def detect_with_thermal_fusion(
        self,
        visible_image: np.ndarray,
        thermal_image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        红外-可见光融合检测 (V3.5)

        Args:
            visible_image: 可见光图像 (BGR)
            thermal_image: 红外热像
            roi_bbox: ROI区域

        Returns:
            融合检测结果
        """
        start_time = time.perf_counter()

        if roi_bbox:
            visible_image = self._crop_roi(visible_image, roi_bbox)
            thermal_image = self._crop_roi(thermal_image, roi_bbox)

        result = {
            "structural_defects": [],
            "hotspots": [],
            "fused_image": None,
            "registration_success": False,
            "processing_time_ms": 0.0
        }

        # 红外-可见光配准
        if self._thermal_registration is not None:
            try:
                # 执行配准
                reg_result = self._thermal_registration.register(
                    thermal_image, visible_image
                )
                result["registration_success"] = reg_result.success
                result["matched_points"] = reg_result.matched_points
                result["reprojection_error"] = reg_result.reprojection_error

                # 图像融合
                fusion_result = self._thermal_registration.fuse(
                    thermal_image, visible_image,
                    reg_result.transform_matrix if reg_result.success else None
                )

                result["fused_image"] = fusion_result.fused_image
                result["hotspots"] = fusion_result.hotspots
                result["thermal_aligned"] = fusion_result.thermal_aligned

                # 使用融合图像进行检测
                if fusion_result.fused_image is not None:
                    defects = self.detect_structural_defects(fusion_result.fused_image)
                    result["structural_defects"] = defects

                    # 关联热点和缺陷
                    result["defect_thermal_correlation"] = self._correlate_defects_with_hotspots(
                        defects, fusion_result.hotspots
                    )

            except Exception as e:
                print(f"[CapacitorDetector] 热融合检测失败: {e}")
                # 回退到单独检测
                result["structural_defects"] = self.detect_structural_defects(visible_image)
        else:
            # 无配准模块，单独检测
            result["structural_defects"] = self.detect_structural_defects(visible_image)

        result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
        return result

    def detect_with_obb(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        OBB旋转框检测 (V3.5)

        用于精确检测倾斜的电容器

        Args:
            image: 输入图像
            roi_bbox: ROI区域

        Returns:
            OBB检测结果列表
        """
        if roi_bbox:
            image = self._crop_roi(image, roi_bbox)

        results = []

        if self._obb_detector is not None:
            try:
                obb_result = self._obb_detector.detect(image)

                for det in obb_result.detections:
                    results.append({
                        "bbox": det.obb.to_axis_aligned_bbox(),
                        "obb": {
                            "cx": det.obb.cx,
                            "cy": det.obb.cy,
                            "width": det.obb.width,
                            "height": det.obb.height,
                            "angle": det.obb.angle
                        },
                        "corners": det.obb.to_corners().tolist(),
                        "confidence": det.confidence,
                        "class_name": det.class_name,
                        "tilt_status": det.tilt_status,
                        "source": "yolov8_obb_v3.5"
                    })

            except Exception as e:
                print(f"[CapacitorDetector] OBB检测失败: {e}")
                # 回退到普通检测
                units = self._detect_capacitor_units(image)
                for unit in units:
                    tilt_angle = self._calculate_tilt_angle(image, unit["bbox"])
                    results.append({
                        "bbox": unit["bbox"],
                        "obb": None,
                        "confidence": unit["confidence"],
                        "class_name": unit.get("class_name", "capacitor"),
                        "tilt_status": {
                            "status": "warning" if abs(tilt_angle) >= self._tilt_warning else "normal",
                            "tilt_angle": tilt_angle
                        },
                        "source": "fallback"
                    })
        else:
            # 无OBB检测器，使用普通检测
            units = self._detect_capacitor_units(image)
            for unit in units:
                tilt_angle = self._calculate_tilt_angle(image, unit["bbox"])
                results.append({
                    "bbox": unit["bbox"],
                    "obb": None,
                    "confidence": unit["confidence"],
                    "class_name": unit.get("class_name", "capacitor"),
                    "tilt_status": {
                        "status": "warning" if abs(tilt_angle) >= self._tilt_warning else "normal",
                        "tilt_angle": tilt_angle
                    },
                    "source": "traditional"
                })

        return results

    def detect_multimodal(
        self,
        visible_image: np.ndarray,
        thermal_image: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        roi_bbox: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        多模态综合检测 (V3.5)

        融合可见光、红外和OBB检测结果

        Args:
            visible_image: 可见光图像
            thermal_image: 红外热像 (可选)
            timestamp: 时间戳
            roi_bbox: ROI区域

        Returns:
            多模态检测结果
        """
        start_time = time.perf_counter()
        timestamp = timestamp or time.time()

        result = {
            "visible_detections": [],
            "obb_detections": [],
            "thermal_fusion": None,
            "intrusions": [],
            "bank_status": None,
            "combined_defects": [],
            "confidence_boost": [],
            "processing_time_ms": 0.0
        }

        # 1. OBB旋转框检测
        obb_results = self.detect_with_obb(visible_image, roi_bbox)
        result["obb_detections"] = obb_results

        # 2. 常规结构缺陷检测
        visible_defects = self.detect_structural_defects(visible_image, roi_bbox)
        result["visible_detections"] = [
            {
                "defect_type": d.defect_type.value,
                "bbox": d.bbox,
                "confidence": d.confidence,
                "class_name": d.class_name,
                "tilt_angle": d.tilt_angle
            }
            for d in visible_defects
        ]

        # 3. 红外融合检测 (如果有热像)
        if thermal_image is not None:
            fusion_result = self.detect_with_thermal_fusion(
                visible_image, thermal_image, roi_bbox
            )
            result["thermal_fusion"] = fusion_result

        # 4. 入侵检测
        intrusions = self.detect_intrusion(visible_image, timestamp)
        result["intrusions"] = [
            {
                "type": i.intrusion_type.value,
                "bbox": i.bbox,
                "confidence": i.confidence,
                "zone": i.zone.value,
                "confirmed": i.confirmed,
                "duration_sec": i.duration_sec
            }
            for i in intrusions
        ]

        # 5. 电容器组状态分析
        bank_status = self.analyze_bank_status(visible_image, roi_bbox)
        result["bank_status"] = {
            "total_units": bank_status.total_units,
            "detected_units": bank_status.detected_units,
            "missing_positions": bank_status.missing_positions,
            "tilted_count": len(bank_status.tilted_units),
            "alignment_score": bank_status.alignment_score
        }

        # 6. 多模态证据融合
        result["combined_defects"] = self._fuse_multimodal_evidence(
            visible_defects,
            obb_results,
            result.get("thermal_fusion", {}).get("hotspots", [])
        )

        result["processing_time_ms"] = (time.perf_counter() - start_time) * 1000
        return result

    def _correlate_defects_with_hotspots(
        self,
        defects: List[CapacitorDetection],
        hotspots: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """关联缺陷和热点 (V3.5)"""
        correlations = []

        for defect in defects:
            best_hotspot = None
            best_overlap = 0.0

            for hotspot in hotspots:
                overlap = self._calculate_bbox_overlap(
                    defect.bbox, hotspot.get("bbox", {})
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_hotspot = hotspot

            if best_hotspot and best_overlap > 0.3:
                correlations.append({
                    "defect_type": defect.defect_type.value,
                    "defect_bbox": defect.bbox,
                    "hotspot_bbox": best_hotspot.get("bbox"),
                    "max_temperature": best_hotspot.get("max_intensity"),
                    "overlap_ratio": best_overlap,
                    "thermal_anomaly": best_hotspot.get("max_intensity", 0) > 200
                })

        return correlations

    def _calculate_bbox_overlap(self, bbox1: Dict, bbox2: Dict) -> float:
        """计算边界框重叠率"""
        x1 = max(bbox1.get("x", 0), bbox2.get("x", 0))
        y1 = max(bbox1.get("y", 0), bbox2.get("y", 0))
        x2 = min(
            bbox1.get("x", 0) + bbox1.get("width", 0),
            bbox2.get("x", 0) + bbox2.get("width", 0)
        )
        y2 = min(
            bbox1.get("y", 0) + bbox1.get("height", 0),
            bbox2.get("y", 0) + bbox2.get("height", 0)
        )

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = bbox1.get("width", 0) * bbox1.get("height", 0)

        return inter / (area1 + 1e-8)

    def _fuse_multimodal_evidence(
        self,
        visible_defects: List[CapacitorDetection],
        obb_results: List[Dict],
        hotspots: List[Dict]
    ) -> List[Dict[str, Any]]:
        """融合多模态检测证据 (V3.5)"""
        fused_defects = []

        # 以可见光检测为基础
        for defect in visible_defects:
            fused = {
                "defect_type": defect.defect_type.value,
                "bbox": defect.bbox,
                "confidence": defect.confidence,
                "tilt_angle": defect.tilt_angle,
                "sources": ["visible"],
                "evidence_count": 1
            }

            # 匹配OBB结果
            for obb in obb_results:
                if self._calculate_bbox_overlap(defect.bbox, obb.get("bbox", {})) > 0.5:
                    fused["obb_angle"] = obb.get("obb", {}).get("angle") if obb.get("obb") else None
                    fused["obb_tilt_status"] = obb.get("tilt_status")
                    fused["sources"].append("obb")
                    fused["evidence_count"] += 1

                    # 置信度提升
                    if obb.get("confidence", 0) > 0.5:
                        fused["confidence"] = min(0.99, fused["confidence"] * 1.2)
                    break

            # 匹配热点
            for hotspot in hotspots:
                if self._calculate_bbox_overlap(defect.bbox, hotspot.get("bbox", {})) > 0.3:
                    fused["thermal_anomaly"] = True
                    fused["max_temperature"] = hotspot.get("max_intensity")
                    fused["sources"].append("thermal")
                    fused["evidence_count"] += 1

                    # 热异常提升置信度
                    if hotspot.get("max_intensity", 0) > 200:
                        fused["confidence"] = min(0.99, fused["confidence"] * 1.15)
                    break

            fused_defects.append(fused)

        return fused_defects

    def get_thermal_registration_info(self) -> Optional[Dict[str, Any]]:
        """获取红外配准模块信息 (V3.5)"""
        if self._thermal_registration is None:
            return None
        return {
            "available": True,
            "cached_transform": self._thermal_registration.get_cached_transform() is not None
        }

    def get_obb_detector_info(self) -> Optional[Dict[str, Any]]:
        """获取OBB检测器信息 (V3.5)"""
        if self._obb_detector is None:
            return None
        return self._obb_detector.model_info


# 便捷函数
def create_detector(config: Dict[str, Any], model_registry=None) -> CapacitorDetectorEnhanced:
    """创建检测器实例"""
    detector = CapacitorDetectorEnhanced(config, model_registry)
    detector.initialize()
    return detector