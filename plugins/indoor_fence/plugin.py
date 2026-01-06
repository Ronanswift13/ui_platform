"""
室内电子围栏插件 - 完整实现
输变电激光星芒破夜绘明监测平台 (G组)

功能范围:
- 2D激光雷达电子围栏: 基于扫描雷达的精确测距
- 多人操作目标识别: 视觉-雷达融合多目标跟踪
- 黄线越界检测: 实时监测人员与警戒线距离
- 机柜授权校验: 验证人员操作授权状态

性能指标:
- 检测延迟: <100ms (融合延迟)
- 多目标容量: 10人/过道
- 定位精度: ±0.1m
- 告警响应: <50ms

基于《变电站多目标操作安全监测系统》方案实现
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import importlib.util
import sys
import numpy as np
import yaml
import threading
from collections import deque

from platform_core.plugin_manager.base import (
    BasePlugin,
    HealthStatus,
    PluginContext,
    PluginManifest,
    PluginStatus,
)
from platform_core.schema.models import (
    Alarm,
    AlarmLevel,
    AlarmRule,
    RecognitionResult,
    ROI,
    BoundingBox,
)


# =============================================================================
# 状态定义 (基于文档方案)
# =============================================================================
class PersonState(str, Enum):
    """人员状态枚举"""
    SAFE_AUTH = "SAFE_AUTH"           # 授权且安全
    SAFE_UNAUTH = "SAFE_UNAUTH"       # 未授权但几何安全
    DANGER_LINE = "DANGER_LINE"       # 越线危险
    VISION_ONLY = "VISION_ONLY"       # 仅视觉模式
    LIDAR_ONLY = "LIDAR_ONLY"         # 仅雷达模式


@dataclass
class PersonTracking:
    """人员跟踪数据"""
    person_id: str
    name: str = ""
    
    # 视觉数据
    visual_bbox: Optional[Dict[str, float]] = None
    visual_confidence: float = 0.0
    zone_visual: int = 0  # 视觉识别的机柜编号
    
    # 雷达数据
    lidar_distance: float = 0.0
    lidar_angle: float = 0.0
    zone_lidar: int = 0  # 雷达推算的机柜区间
    
    # 融合数据
    fused_position: Optional[Dict[str, float]] = None
    distance_to_line: float = 0.0
    cabinet_id: int = 0
    
    # 状态
    state: PersonState = PersonState.SAFE_AUTH
    authorized: bool = False
    visual_valid: bool = True
    lidar_valid: bool = True
    
    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)


@dataclass 
class CabinetInfo:
    """机柜信息"""
    id: int
    name: str
    x_start: float  # 归一化起始X坐标
    x_end: float    # 归一化结束X坐标
    authorized: bool = False
    status: str = "idle"  # idle, occupied, alert
    occupants: List[str] = field(default_factory=list)


@dataclass
class LidarScan:
    """雷达扫描数据"""
    timestamp: datetime
    angles: np.ndarray  # 角度数组 (度)
    distances: np.ndarray  # 距离数组 (米)
    scan_rate_hz: float = 10.0


def _load_detector_class():
    """动态加载检测器类"""
    detector_path = Path(__file__).parent / "detector_enhanced.py"
    if not detector_path.exists():
        detector_path = Path(__file__).parent / "detector.py"

    if not detector_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("indoor_detector", detector_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["indoor_detector"] = module
    spec.loader.exec_module(module)

    if hasattr(module, 'IndoorDetectorEnhanced'):
        return module.IndoorDetectorEnhanced
    if hasattr(module, 'IndoorDetector'):
        return module.IndoorDetector
    return None


_IndoorDetector = None

def get_detector_class():
    global _IndoorDetector
    if _IndoorDetector is None:
        _IndoorDetector = _load_detector_class()
    return _IndoorDetector


class IndoorFencePlugin(BasePlugin):
    """
    室内电子围栏插件
    
    实现基于视觉+2D激光雷达融合的安全监测系统
    """
    
    # 告警级别映射
    ALARM_LEVELS = {
        PersonState.SAFE_AUTH: AlarmLevel.INFO,
        PersonState.SAFE_UNAUTH: AlarmLevel.WARNING,
        PersonState.DANGER_LINE: AlarmLevel.CRITICAL,
        PersonState.VISION_ONLY: AlarmLevel.WARNING,
        PersonState.LIDAR_ONLY: AlarmLevel.WARNING,
    }
    
    # 标签名称
    LABEL_NAMES = {
        "line_cross": "越线告警",
        "unauthorized": "未授权操作",
        "multi_person": "同柜多人",
        "safe": "安全状态",
        "degraded": "降级模式",
    }
    
    def __init__(self, manifest: PluginManifest, plugin_dir: Path):
        """初始化插件"""
        super().__init__(manifest, plugin_dir)
        self._config: Dict[str, Any] = {}
        self._initialized = False
        self._detector = None
        
        # 人员跟踪
        self._tracked_persons: Dict[str, PersonTracking] = {}
        self._next_person_id = 1
        
        # 机柜配置
        self._cabinets: Dict[int, CabinetInfo] = {}
        self._allow_list: List[int] = []
        
        # 雷达数据缓冲
        self._lidar_buffer: deque = deque(maxlen=10)
        self._lidar_lock = threading.Lock()
        
        # 安全参数
        self._yellow_line_distance = 0.5  # 黄线安全距离(米)
        self._warning_distance = 0.3      # 警告距离
        self._danger_distance = 0.1       # 危险距离
        
        # 融合参数
        self._max_time_diff_ms = 50       # 最大时间差
        self._angle_threshold = 5.0       # 角度匹配阈值(度)
        
        # 统计
        self._inference_count = 0
        self._alert_count = 0
        self._last_inference_time: Optional[datetime] = None
        
        # 代码哈希
        self._code_hash = self._calculate_code_hash()
    
    def _calculate_code_hash(self) -> str:
        """计算代码版本hash"""
        h = hashlib.sha256()
        files_to_hash = ["plugin.py", "detector.py", "detector_enhanced.py"]
        for fname in files_to_hash:
            fpath = self.plugin_dir / fname
            if fpath.exists():
                h.update(fpath.read_bytes())
        return f"sha256:{h.hexdigest()[:12]}"
    
    @property
    def code_hash(self) -> str:
        """返回代码版本hash"""
        return self._code_hash
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        config_path = self.plugin_dir / "configs" / "default.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _merge_config(self, base: Dict, override: Dict) -> Dict:
        """递归合并配置"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def init(self, config: dict[str, Any] | None = None) -> bool:
        """
        初始化插件
        
        Args:
            config: 运行时配置
            
        Returns:
            是否成功初始化
        """
        config = config or {}
        
        try:
            # 加载配置
            default_config = self._load_default_config()
            self._config = self._merge_config(default_config, config)
            
            # 加载安全参数
            safety_config = self._config.get("safety_zone", {})
            self._yellow_line_distance = safety_config.get("yellow_line_distance_m", 0.5)
            self._warning_distance = safety_config.get("warning_distance_m", 0.3)
            self._danger_distance = safety_config.get("danger_distance_m", 0.1)
            
            # 加载授权配置
            auth_config = self._config.get("cabinet_authorization", {})
            self._allow_list = auth_config.get("allow_list", [])
            
            # 初始化机柜
            self._init_cabinets()
            
            # 加载检测器
            detector_class = get_detector_class()
            if detector_class:
                self._detector = detector_class(self._config)
                print(f"[{self.id}] 检测器加载成功")
            else:
                print(f"[{self.id}] 检测器未找到，使用模拟模式")
            
            # 加载融合参数
            fusion_config = self._config.get("fusion", {})
            self._max_time_diff_ms = fusion_config.get("max_time_diff_ms", 50)
            self._angle_threshold = fusion_config.get("angle_match_threshold_deg", 5.0)
            
            self._initialized = True
            self.status = PluginStatus.READY
            
            print(f"[{self.id}] 插件初始化成功")
            print(f"[{self.id}] 机柜数量: {len(self._cabinets)}")
            print(f"[{self.id}] 授权机柜: {self._allow_list}")
            print(f"[{self.id}] 黄线距离: {self._yellow_line_distance}m")
            
            return True
            
        except Exception as e:
            self.status = PluginStatus.ERROR
            self._last_error = str(e)
            print(f"[{self.id}] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _init_cabinets(self):
        """初始化机柜配置"""
        cabinet_config = self._config.get("cabinets", {})
        num_cabinets = cabinet_config.get("count", 6)
        
        for i in range(1, num_cabinets + 1):
            x_start = (i - 1) / num_cabinets
            x_end = i / num_cabinets
            
            self._cabinets[i] = CabinetInfo(
                id=i,
                name=f"机柜-{i:02d}",
                x_start=x_start,
                x_end=x_end,
                authorized=(i in self._allow_list),
            )
    
    def infer(
        self,
        frame: np.ndarray,
        rois: list[ROI],
        context: PluginContext,
    ) -> list[RecognitionResult]:
        """
        执行推理
        
        Args:
            frame: 输入图像帧 (BGR格式)
            rois: 识别区域列表
            context: 运行上下文
            
        Returns:
            识别结果列表
        """
        if not self._initialized:
            return [RecognitionResult(
                task_id=context.task_id,
                site_id=context.site_id,
                device_id=context.device_id,
                component_id=context.component_id,
                roi_id=roi.id,
                bbox=roi.bbox,
                label="error",
                confidence=0.0,
                model_version=self.version,
                code_version=self.code_hash,
                failure_reason="9000",
                metadata={"error": "插件未初始化"}
            ) for roi in rois]
        
        self.status = PluginStatus.RUNNING
        self._last_inference_time = datetime.now()
        self._inference_count += 1
        
        results: list[RecognitionResult] = []
        
        # 1. 视觉检测
        visual_detections = self._detect_persons_visual(frame)
        
        # 2. 获取雷达数据
        lidar_scan = self._get_latest_lidar_scan()
        lidar_clusters = self._cluster_lidar_points(lidar_scan) if lidar_scan else []
        
        # 3. 视觉-雷达融合
        fused_persons = self._fuse_visual_lidar(visual_detections, lidar_clusters)
        
        # 4. 更新跟踪和状态
        for person in fused_persons:
            # 确定机柜区域
            person.cabinet_id = self._determine_cabinet(person)
            person.authorized = person.cabinet_id in self._allow_list
            
            # 计算与黄线距离
            person.distance_to_line = self._calculate_line_distance(person)
            
            # 评估状态
            person.state = self._evaluate_state(person)
            
            # 更新跟踪
            self._tracked_persons[person.person_id] = person
            
            # 更新机柜占用状态
            self._update_cabinet_status(person)
        
        # 5. 检查同柜多人
        default_bbox = rois[0].bbox if rois else BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        multi_person_alerts = self._check_multi_person(context, default_bbox)
        
        # 6. 生成结果
        for roi in rois:
            for person in fused_persons:
                result = self._create_result(roi, person, context)
                results.append(result)
            
            # 添加多人告警
            for alert in multi_person_alerts:
                results.append(alert)
            
            # 如果没有检测到人
            if not fused_persons:
                results.append(RecognitionResult(
                    task_id=context.task_id,
                    site_id=context.site_id,
                    device_id=context.device_id,
                    component_id=context.component_id,
                    roi_id=roi.id,
                    bbox=roi.bbox,
                    label="no_person",
                    confidence=1.0,
                    value="区域空闲",
                    model_version=self.version,
                    code_version=self.code_hash,
                    metadata={"message": "未检测到人员"}
                ))
        
        self.status = PluginStatus.READY
        return results
    
    def _detect_persons_visual(self, frame: np.ndarray) -> List[Dict]:
        """视觉检测人员"""
        if self._detector:
            return self._detector.detect_persons(frame)
        return []
    
    def _get_latest_lidar_scan(self) -> Optional[LidarScan]:
        """获取最新雷达扫描数据"""
        with self._lidar_lock:
            if self._lidar_buffer:
                return self._lidar_buffer[-1]
        return None
    
    def _cluster_lidar_points(self, scan: LidarScan) -> List[Dict]:
        """
        雷达点云聚类
        
        基于文档方案: 识别距离突变点，聚类为独立目标
        """
        clusters = []
        
        if scan is None or len(scan.distances) == 0:
            return clusters
        
        # 背景距离估计 (取远距离的中位数)
        bg_distance = np.median(scan.distances[scan.distances > 3.0]) if np.any(scan.distances > 3.0) else 10.0
        
        # 前景提取: 距离明显小于背景的点
        fg_mask = scan.distances < (bg_distance - 0.5)
        
        if not np.any(fg_mask):
            return clusters
        
        # 角度连续性聚类
        fg_indices = np.where(fg_mask)[0]
        
        current_cluster = []
        for i, idx in enumerate(fg_indices):
            if not current_cluster:
                current_cluster.append(idx)
            else:
                # 检查角度连续性
                if idx - current_cluster[-1] <= 3:  # 允许3个点的间隔
                    current_cluster.append(idx)
                else:
                    # 保存当前簇
                    if len(current_cluster) >= 3:  # 至少3个点
                        clusters.append(self._create_cluster(scan, current_cluster))
                    current_cluster = [idx]
        
        # 处理最后一个簇
        if len(current_cluster) >= 3:
            clusters.append(self._create_cluster(scan, current_cluster))
        
        return clusters
    
    def _create_cluster(self, scan: LidarScan, indices: List[int]) -> Dict:
        """创建点云簇"""
        angles = scan.angles[indices]
        distances = scan.distances[indices]
        
        return {
            "center_angle": float(np.mean(angles)),
            "center_distance": float(np.mean(distances)),
            "min_distance": float(np.min(distances)),
            "angle_span": float(np.max(angles) - np.min(angles)),
            "point_count": len(indices),
        }
    
    def _fuse_visual_lidar(
        self, 
        visual_detections: List[Dict], 
        lidar_clusters: List[Dict]
    ) -> List[PersonTracking]:
        """
        视觉-雷达数据融合
        
        基于文档方案: 角度匹配融合
        """
        fused_persons = []
        used_clusters = set()
        
        # 相机参数
        camera_cx = self._config.get("camera", {}).get("cx", 320)
        camera_fx = self._config.get("camera", {}).get("fx", 500)
        
        for vis_det in visual_detections:
            # 计算视觉目标的方位角
            bbox = vis_det.get("bbox", {})
            center_x = bbox.get("x", 0) + bbox.get("width", 0) / 2
            alpha_img = np.degrees(np.arctan2(center_x - camera_cx / 640, camera_fx / 640))
            
            # 在雷达数据中寻找最佳匹配
            best_match = None
            min_angle_diff = self._angle_threshold
            
            for i, cluster in enumerate(lidar_clusters):
                if i in used_clusters:
                    continue
                
                angle_diff = abs(alpha_img - cluster["center_angle"])
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    best_match = (i, cluster)
            
            # 创建融合跟踪数据
            person = PersonTracking(
                person_id=f"P{self._next_person_id:03d}",
                visual_bbox=bbox,
                visual_confidence=vis_det.get("confidence", 0.0),
                zone_visual=self._get_zone_from_bbox(bbox),
                visual_valid=True,
            )
            self._next_person_id += 1
            
            if best_match:
                idx, cluster = best_match
                used_clusters.add(idx)
                
                person.lidar_distance = cluster["center_distance"]
                person.lidar_angle = cluster["center_angle"]
                person.zone_lidar = self._get_zone_from_angle(cluster["center_angle"])
                person.lidar_valid = True
                
                # 融合位置 (使用雷达距离 + 视觉横向位置)
                person.fused_position = {
                    "x": center_x,
                    "y": bbox.get("y", 0) + bbox.get("height", 0),
                    "distance": cluster["center_distance"],
                }
            else:
                person.lidar_valid = False
            
            fused_persons.append(person)
        
        return fused_persons
    
    def _get_zone_from_bbox(self, bbox: Dict) -> int:
        """从边界框获取机柜区域"""
        center_x = bbox.get("x", 0) + bbox.get("width", 0) / 2
        
        for cab_id, cab in self._cabinets.items():
            if cab.x_start <= center_x < cab.x_end:
                return cab_id
        
        return 0
    
    def _get_zone_from_angle(self, angle: float) -> int:
        """从雷达角度获取机柜区域"""
        # 简化映射: 假设角度范围-45到45度对应机柜1-6
        normalized = (angle + 45) / 90  # 归一化到0-1
        cab_id = int(normalized * len(self._cabinets)) + 1
        return max(1, min(len(self._cabinets), cab_id))
    
    def _determine_cabinet(self, person: PersonTracking) -> int:
        """确定人员所在机柜"""
        if person.zone_visual == person.zone_lidar and person.zone_visual > 0:
            return person.zone_visual
        
        # 优先使用视觉结果
        if person.visual_valid and person.zone_visual > 0:
            return person.zone_visual
        
        # 退化使用雷达结果
        if person.lidar_valid and person.zone_lidar > 0:
            return person.zone_lidar
        
        return 0
    
    def _calculate_line_distance(self, person: PersonTracking) -> float:
        """计算人员与黄线的距离"""
        if person.lidar_valid:
            # 使用雷达测量的距离
            # 假设黄线位置在距离雷达2米处
            yellow_line_position = self._config.get("yellow_line_position_m", 2.0)
            return abs(person.lidar_distance - yellow_line_position)
        
        # 无雷达数据时返回默认安全距离
        return self._yellow_line_distance + 0.1
    
    def _evaluate_state(self, person: PersonTracking) -> PersonState:
        """
        评估人员状态
        
        基于文档方案的状态定义
        """
        # 降级模式检查
        if not person.visual_valid:
            return PersonState.LIDAR_ONLY
        if not person.lidar_valid:
            return PersonState.VISION_ONLY
        
        # 越线检查 (最高优先级)
        if person.distance_to_line < self._danger_distance:
            return PersonState.DANGER_LINE
        
        # 授权检查
        if person.distance_to_line >= self._yellow_line_distance:
            if person.authorized:
                return PersonState.SAFE_AUTH
            else:
                return PersonState.SAFE_UNAUTH
        else:
            # 在警告区域
            if person.authorized:
                return PersonState.SAFE_AUTH
            else:
                return PersonState.SAFE_UNAUTH
    
    def _update_cabinet_status(self, person: PersonTracking):
        """更新机柜状态"""
        cab_id = person.cabinet_id
        if cab_id in self._cabinets:
            cab = self._cabinets[cab_id]
            
            if person.person_id not in cab.occupants:
                cab.occupants.append(person.person_id)
            
            if person.state == PersonState.DANGER_LINE:
                cab.status = "alert"
            elif len(cab.occupants) > 0:
                cab.status = "occupied"
    
    def _check_multi_person(
        self,
        context: PluginContext,
        default_bbox: BoundingBox,
    ) -> List[RecognitionResult]:
        """检查同柜多人情况"""
        alerts = []
        max_per_cab = self._config.get("cabinet_authorization", {}).get("max_persons_per_cabinet", 1)
        
        for cab_id, cab in self._cabinets.items():
            if len(cab.occupants) > max_per_cab:
                self._alert_count += 1
                alerts.append(RecognitionResult(
                    task_id=context.task_id,
                    site_id=context.site_id,
                    device_id=context.device_id,
                    component_id=context.component_id,
                    roi_id=f"cab_{cab_id}",
                    bbox=default_bbox,
                    label="multi_person",
                    confidence=1.0,
                    value=f"机柜{cab_id}前有{len(cab.occupants)}人",
                    model_version=self.version,
                    code_version=self.code_hash,
                    metadata={
                        "cabinet_id": cab_id,
                        "person_count": len(cab.occupants),
                        "persons": cab.occupants,
                    }
                ))
        
        return alerts
    
    def _create_result(
        self,
        roi: ROI,
        person: PersonTracking,
        context: PluginContext,
    ) -> RecognitionResult:
        """创建识别结果"""
        if person.state == PersonState.DANGER_LINE:
            self._alert_count += 1

        bbox = roi.bbox
        if person.visual_bbox:
            try:
                bbox = BoundingBox(**person.visual_bbox)
            except Exception:
                bbox = roi.bbox
        
        # 状态标签
        label_map = {
            PersonState.SAFE_AUTH: "safe_auth",
            PersonState.SAFE_UNAUTH: "unauthorized",
            PersonState.DANGER_LINE: "line_cross",
            PersonState.VISION_ONLY: "degraded",
            PersonState.LIDAR_ONLY: "degraded",
        }
        
        label_name_map = {
            PersonState.SAFE_AUTH: "授权安全",
            PersonState.SAFE_UNAUTH: "未授权操作",
            PersonState.DANGER_LINE: "越线危险",
            PersonState.VISION_ONLY: "降级模式(仅视觉)",
            PersonState.LIDAR_ONLY: "降级模式(仅雷达)",
        }
        
        return RecognitionResult(
            task_id=context.task_id,
            site_id=context.site_id,
            device_id=context.device_id,
            component_id=context.component_id,
            roi_id=roi.id,
            label=label_map.get(person.state, "unknown"),
            confidence=person.visual_confidence,
            value=f"机柜{person.cabinet_id} · {person.distance_to_line:.2f}m",
            bbox=bbox,
            model_version=self.version,
            code_version=self.code_hash,
            metadata={
                "person_id": person.person_id,
                "state": person.state.value,
                "label_name": label_name_map.get(person.state, "未知"),
                "cabinet_id": person.cabinet_id,
                "authorized": person.authorized,
                "distance_to_line_m": person.distance_to_line,
                "lidar_distance_m": person.lidar_distance,
                "zone_visual": person.zone_visual,
                "zone_lidar": person.zone_lidar,
                "visual_valid": person.visual_valid,
                "lidar_valid": person.lidar_valid,
            }
        )
    
    def postprocess(
        self,
        results: list[RecognitionResult],
        rules: list[AlarmRule],
    ) -> list[Alarm]:
        """后处理和告警生成"""
        alarms: list[Alarm] = []

        for result in results:
            if result.label == "line_cross":
                person_id = result.metadata.get("person_id", "人员")
                distance = result.metadata.get("distance_to_line_m")
                cabinet_id = result.metadata.get("cabinet_id")
                distance_text = f"，距离{distance:.2f}m" if distance is not None else ""
                cabinet_text = f"机柜{cabinet_id}" if cabinet_id is not None else "警戒区域"
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.CRITICAL,
                    title=self.LABEL_NAMES.get("line_cross", "越线告警"),
                    message=f"{person_id}已越过{cabinet_text}黄色警戒线{distance_text}",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))
                continue

            if result.label == "unauthorized":
                person_id = result.metadata.get("person_id", "人员")
                cabinet_id = result.metadata.get("cabinet_id")
                cabinet_text = f"机柜{cabinet_id}" if cabinet_id is not None else "未授权区域"
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title=self.LABEL_NAMES.get("unauthorized", "未授权操作"),
                    message=f"{person_id}在{cabinet_text}进行操作",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))
                continue

            if result.label == "multi_person":
                cabinet_id = result.metadata.get("cabinet_id")
                person_count = result.metadata.get("person_count")
                cabinet_text = f"机柜{cabinet_id}" if cabinet_id is not None else "机柜区域"
                count_text = f"{person_count}人" if person_count is not None else "多人"
                alarms.append(Alarm(
                    task_id=result.task_id,
                    result_id=None,
                    level=AlarmLevel.WARNING,
                    title=self.LABEL_NAMES.get("multi_person", "同柜多人"),
                    message=f"{cabinet_text}前检测到{count_text}同时操作",
                    site_id=result.site_id,
                    device_id=result.device_id,
                    component_id=result.component_id,
                ))

        return alarms

    def healthcheck(self) -> HealthStatus:
        """健康检查"""
        if not self._initialized:
            return HealthStatus(
                healthy=False,
                message="插件未初始化",
                details={"status": "not_initialized"},
            )
        
        lidar_ok = len(self._lidar_buffer) > 0
        
        return HealthStatus(
            healthy=True,
            message=f"室内监控就绪，{len(self._cabinets)}个机柜，{len(self._tracked_persons)}人跟踪中",
            details={
                "status": "ready",
                "inference_count": self._inference_count,
                "alert_count": self._alert_count,
                "tracked_persons": len(self._tracked_persons),
                "cabinet_count": len(self._cabinets),
                "allow_list": self._allow_list,
                "lidar_connected": lidar_ok,
                "last_inference": self._last_inference_time.isoformat() if self._last_inference_time else None,
            }
        )
    
    def cleanup(self) -> None:
        """清理资源"""
        self._tracked_persons.clear()
        self._lidar_buffer.clear()
        for cab in self._cabinets.values():
            cab.occupants.clear()
            cab.status = "idle"
        
        if self._detector and hasattr(self._detector, 'cleanup'):
            self._detector.cleanup()
        
        self._initialized = False
        self.status = PluginStatus.UNLOADED
        print(f"[{self.id}] 插件已清理")
    
    # ==================== 雷达数据接口 ====================
    
    def feed_lidar_scan(self, scan_data: Dict):
        """
        接收雷达扫描数据
        
        Args:
            scan_data: {
                "angles": List[float],
                "distances": List[float],
                "timestamp": datetime or str
            }
        """
        scan = LidarScan(
            timestamp=datetime.fromisoformat(scan_data["timestamp"]) if isinstance(scan_data["timestamp"], str) else scan_data["timestamp"],
            angles=np.array(scan_data["angles"]),
            distances=np.array(scan_data["distances"]),
        )
        
        with self._lidar_lock:
            self._lidar_buffer.append(scan)
    
    # ==================== 授权管理接口 ====================
    
    def update_allow_list(self, cabinet_ids: List[int]):
        """更新授权机柜列表"""
        self._allow_list = cabinet_ids
        for cab_id, cab in self._cabinets.items():
            cab.authorized = cab_id in self._allow_list
        print(f"[{self.id}] 授权列表已更新: {self._allow_list}")
    
    def get_cabinet_status(self) -> Dict[int, Dict]:
        """获取所有机柜状态"""
        return {
            cab_id: {
                "id": cab.id,
                "name": cab.name,
                "authorized": cab.authorized,
                "status": cab.status,
                "occupants": cab.occupants.copy(),
            }
            for cab_id, cab in self._cabinets.items()
        }
    
    def get_tracked_persons(self) -> List[Dict]:
        """获取跟踪人员列表"""
        return [
            {
                "person_id": p.person_id,
                "state": p.state.value,
                "cabinet_id": p.cabinet_id,
                "authorized": p.authorized,
                "distance_to_line_m": p.distance_to_line,
                "timestamp": p.timestamp.isoformat(),
            }
            for p in self._tracked_persons.values()
        ]
    
    def get_ui_config(self) -> Dict[str, Any]:
        """获取UI配置"""
        return {
            "detection_types": [
                {"id": "person_detection", "name": "人员检测", "default": True},
                {"id": "line_check", "name": "越线检测", "default": True},
                {"id": "auth_check", "name": "授权校验", "default": True},
                {"id": "multi_person", "name": "多人检测", "default": True},
            ],
            "cabinet_count": len(self._cabinets),
            "allow_list": self._allow_list,
            "yellow_line_distance_m": self._yellow_line_distance,
            "lidar_enabled": self._config.get("lidar", {}).get("enabled", False),
        }
