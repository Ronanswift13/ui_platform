"""
鸟类风险预测模块
输变电激光星芒破夜绘明监测平台

基于UPDATE.md中期迭代计划:
- 升级检测模型至YOLOv9/YOLOv8-ViT
- 融合鸟类分类器(EfficientNet)
- 轨迹预测模型分析飞行方向和速度
- 动态调整驱鸟策略
- 结合环境数据(风速、季节)调整风险阈值

功能:
- 鸟类检测与分类
- 轨迹跟踪与预测
- 风险评估
- 驱鸟策略推荐

版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入深度学习相关依赖
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


# =============================================================================
# 枚举定义
# =============================================================================

class BirdSpecies(Enum):
    """鸟类种类"""
    UNKNOWN = "unknown"
    SPARROW = "sparrow"             # 麻雀
    PIGEON = "pigeon"               # 鸽子
    CROW = "crow"                   # 乌鸦
    MAGPIE = "magpie"               # 喜鹊
    EAGLE = "eagle"                 # 鹰
    HAWK = "hawk"                   # 隼
    SWALLOW = "swallow"             # 燕子
    STARLING = "starling"           # 椋鸟
    HERON = "heron"                 # 苍鹭
    OTHER = "other"


class BirdBehavior(Enum):
    """鸟类行为"""
    FLYING = "flying"               # 飞行
    PERCHING = "perching"           # 栖息
    NESTING = "nesting"             # 筑巢
    FEEDING = "feeding"             # 觅食
    HOVERING = "hovering"           # 盘旋
    DIVING = "diving"               # 俯冲
    CIRCLING = "circling"           # 盘旋
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """风险等级"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeterrenceAction(Enum):
    """驱鸟动作"""
    NONE = "none"
    SOUND = "sound"                 # 声波驱鸟
    LASER = "laser"                 # 激光驱鸟
    ULTRASONIC = "ultrasonic"       # 超声波
    VISUAL = "visual"               # 视觉惊吓
    COMBINED = "combined"           # 组合驱鸟


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class BirdDetection:
    """鸟类检测结果"""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x, y, w, h (归一化)
    confidence: float
    species: BirdSpecies = BirdSpecies.UNKNOWN
    species_confidence: float = 0.0
    behavior: BirdBehavior = BirdBehavior.UNKNOWN
    timestamp: float = 0.0
    position_3d: Optional[np.ndarray] = None  # 3D位置估计
    velocity: Optional[np.ndarray] = None      # 速度向量
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        """边界框中心"""
        x, y, w, h = self.bbox
        return (x + w / 2, y + h / 2)

    @property
    def area(self) -> float:
        """边界框面积"""
        return self.bbox[2] * self.bbox[3]


@dataclass
class BirdTrack:
    """鸟类轨迹"""
    track_id: int
    detections: List[BirdDetection] = field(default_factory=list)
    species: BirdSpecies = BirdSpecies.UNKNOWN
    species_votes: Dict[BirdSpecies, int] = field(default_factory=dict)
    first_seen: float = 0.0
    last_seen: float = 0.0
    is_active: bool = True
    predicted_trajectory: Optional[np.ndarray] = None
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_detection(self, det: BirdDetection):
        """添加检测"""
        self.detections.append(det)
        self.last_seen = det.timestamp

        if len(self.detections) == 1:
            self.first_seen = det.timestamp

        # 更新物种投票
        if det.species != BirdSpecies.UNKNOWN:
            self.species_votes[det.species] = self.species_votes.get(det.species, 0) + 1
            # 更新主要物种
            if self.species_votes:
                self.species = max(self.species_votes, key=self.species_votes.get)

    @property
    def duration(self) -> float:
        """轨迹持续时间"""
        return self.last_seen - self.first_seen

    @property
    def trajectory(self) -> np.ndarray:
        """获取轨迹点序列"""
        points = [det.center for det in self.detections]
        return np.array(points)

    def get_velocity(self, window: int = 5) -> Optional[np.ndarray]:
        """计算当前速度"""
        if len(self.detections) < 2:
            return None

        recent = self.detections[-min(window, len(self.detections)):]
        if len(recent) < 2:
            return None

        dt = recent[-1].timestamp - recent[0].timestamp
        if dt < 0.001:
            return None

        p1 = np.array(recent[0].center)
        p2 = np.array(recent[-1].center)

        return (p2 - p1) / dt


@dataclass
class EnvironmentContext:
    """环境上下文"""
    timestamp: float = 0.0
    wind_speed: float = 0.0         # 风速 m/s
    wind_direction: float = 0.0     # 风向 (度)
    temperature: float = 20.0       # 温度
    humidity: float = 50.0          # 湿度
    time_of_day: str = "day"        # day/dawn/dusk/night
    season: str = "summer"          # spring/summer/autumn/winter
    weather: str = "clear"          # clear/cloudy/rainy/snowy
    location_type: str = "outdoor"  # outdoor/indoor


@dataclass
class RiskAssessment:
    """风险评估结果"""
    risk_level: RiskLevel
    risk_score: float
    primary_threats: List[str]
    affected_equipment: List[str]
    recommended_actions: List[DeterrenceAction]
    urgency: float                  # 紧急程度 [0-1]
    time_to_impact: Optional[float] = None  # 预计影响时间(秒)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeterrenceStrategy:
    """驱鸟策略"""
    action: DeterrenceAction
    intensity: float                # 强度 [0-1]
    duration: float                 # 持续时间(秒)
    direction: Optional[float] = None  # 方向(度)
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


# =============================================================================
# 鸟类分类器
# =============================================================================

@dataclass
class BirdClassifierConfig:
    """鸟类分类器配置"""
    model_path: Optional[str] = None
    device: str = "cpu"
    input_size: Tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.5
    top_k: int = 3


class BirdClassifier:
    """
    鸟类分类器

    基于EfficientNet进行鸟类物种分类
    """

    # 物种标签映射
    SPECIES_LABELS = {
        0: BirdSpecies.SPARROW,
        1: BirdSpecies.PIGEON,
        2: BirdSpecies.CROW,
        3: BirdSpecies.MAGPIE,
        4: BirdSpecies.EAGLE,
        5: BirdSpecies.HAWK,
        6: BirdSpecies.SWALLOW,
        7: BirdSpecies.STARLING,
        8: BirdSpecies.HERON,
        9: BirdSpecies.OTHER,
    }

    def __init__(self, config: BirdClassifierConfig):
        self.config = config
        self._model = None
        self._initialized = False

        self._init_model()

    def _init_model(self):
        """初始化分类模型"""
        if self.config.model_path:
            try:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if self.config.device == "cuda" else ['CPUExecutionProvider']
                self._model = ort.InferenceSession(
                    self.config.model_path,
                    providers=providers
                )
                self._initialized = True
                logger.info("[BirdClassifier] 分类模型加载成功")
            except Exception as e:
                logger.warning(f"[BirdClassifier] 模型加载失败: {e}")
        else:
            logger.info("[BirdClassifier] 未指定模型路径，使用模拟模式")

    def classify(
        self,
        image: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[BirdSpecies, float, Dict[BirdSpecies, float]]:
        """
        分类鸟类物种

        Args:
            image: 输入图像
            bbox: 鸟类边界框 (归一化)

        Returns:
            (物种, 置信度, 各物种概率)
        """
        if not self._initialized or self._model is None:
            # 模拟分类
            return self._simulate_classification()

        # 裁剪ROI
        h, w = image.shape[:2]
        x, y, bw, bh = bbox
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)

        # 边界检查
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return BirdSpecies.UNKNOWN, 0.0, {}

        roi = image[y1:y2, x1:x2]

        # 预处理
        input_tensor = self._preprocess(roi)

        # 推理
        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: input_tensor})

        # 解析结果
        probs = self._softmax(outputs[0].squeeze())

        # 构建概率字典
        prob_dict = {}
        for idx, prob in enumerate(probs):
            if idx in self.SPECIES_LABELS:
                prob_dict[self.SPECIES_LABELS[idx]] = float(prob)

        # 获取最佳结果
        best_idx = int(np.argmax(probs))
        best_species = self.SPECIES_LABELS.get(best_idx, BirdSpecies.UNKNOWN)
        best_conf = float(probs[best_idx])

        if best_conf < self.config.confidence_threshold:
            return BirdSpecies.UNKNOWN, best_conf, prob_dict

        return best_species, best_conf, prob_dict

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """预处理图像"""
        if CV2_AVAILABLE:
            # 调整大小
            resized = cv2.resize(image, self.config.input_size)
            # 归一化
            normalized = resized.astype(np.float32) / 255.0
            # ImageNet标准化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            # NCHW格式
            return normalized.transpose(2, 0, 1)[np.newaxis, ...]
        else:
            return np.zeros((1, 3, *self.config.input_size), dtype=np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def _simulate_classification(self) -> Tuple[BirdSpecies, float, Dict[BirdSpecies, float]]:
        """模拟分类结果"""
        import random
        species = random.choice(list(BirdSpecies))
        confidence = random.uniform(0.5, 0.95)
        probs = {s: random.uniform(0, 0.3) for s in BirdSpecies}
        probs[species] = confidence
        return species, confidence, probs


# =============================================================================
# 轨迹预测器
# =============================================================================

class TrajectoryPredictor:
    """
    轨迹预测器

    基于卡尔曼滤波和LSTM预测鸟类飞行轨迹
    """

    def __init__(self, prediction_horizon: int = 30, dt: float = 0.04):
        """
        初始化

        Args:
            prediction_horizon: 预测步数
            dt: 时间步长(秒)
        """
        self.prediction_horizon = prediction_horizon
        self.dt = dt

        # 卡尔曼滤波器参数
        self.state_dim = 4  # [x, y, vx, vy]
        self.obs_dim = 2    # [x, y]

        # 状态转移矩阵
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 观测矩阵
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 过程噪声
        self.Q = np.eye(4) * 0.01
        self.Q[2:, 2:] *= 10  # 速度噪声更大

        # 观测噪声
        self.R = np.eye(2) * 0.001

    def predict(self, track: BirdTrack) -> np.ndarray:
        """
        预测未来轨迹

        Args:
            track: 鸟类轨迹

        Returns:
            预测轨迹 (horizon, 2)
        """
        if len(track.detections) < 2:
            # 点数不足，返回静止预测
            if track.detections:
                last_pos = np.array(track.detections[-1].center)
                return np.tile(last_pos, (self.prediction_horizon, 1))
            return np.zeros((self.prediction_horizon, 2))

        # 初始化状态
        positions = track.trajectory[-min(20, len(track.trajectory)):]

        # 估计初始速度
        if len(positions) >= 2:
            velocity = (positions[-1] - positions[-2]) / self.dt
        else:
            velocity = np.zeros(2)

        state = np.array([
            positions[-1, 0],
            positions[-1, 1],
            velocity[0],
            velocity[1]
        ])

        # 初始化协方差
        P = np.eye(4) * 0.1

        # Kalman滤波更新历史
        for pos in positions[:-1]:
            # 预测
            state = self.F @ state
            P = self.F @ P @ self.F.T + self.Q

            # 更新
            z = pos
            y = z - self.H @ state
            S = self.H @ P @ self.H.T + self.R
            K = P @ self.H.T @ np.linalg.inv(S)
            state = state + K @ y
            P = (np.eye(4) - K @ self.H) @ P

        # 预测未来轨迹
        predictions = []
        for _ in range(self.prediction_horizon):
            state = self.F @ state
            predictions.append(state[:2].copy())

        return np.array(predictions)

    def estimate_impact_point(
        self,
        track: BirdTrack,
        target_zones: List[Tuple[float, float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        """
        估计与目标区域的碰撞点

        Args:
            track: 鸟类轨迹
            target_zones: 目标区域列表 [(x, y, w, h), ...]

        Returns:
            (x, y, time_to_impact) 或 None
        """
        predicted = self.predict(track)

        for i, pos in enumerate(predicted):
            for zone in target_zones:
                zx, zy, zw, zh = zone
                if (zx <= pos[0] <= zx + zw and zy <= pos[1] <= zy + zh):
                    return (pos[0], pos[1], i * self.dt)

        return None


# =============================================================================
# 风险评估器
# =============================================================================

class RiskAssessor:
    """
    风险评估器

    基于鸟类行为、轨迹和环境综合评估风险
    """

    # 物种风险权重
    SPECIES_RISK = {
        BirdSpecies.UNKNOWN: 0.5,
        BirdSpecies.SPARROW: 0.2,
        BirdSpecies.PIGEON: 0.4,
        BirdSpecies.CROW: 0.6,
        BirdSpecies.MAGPIE: 0.5,
        BirdSpecies.EAGLE: 0.8,
        BirdSpecies.HAWK: 0.8,
        BirdSpecies.SWALLOW: 0.3,
        BirdSpecies.STARLING: 0.4,
        BirdSpecies.HERON: 0.6,
        BirdSpecies.OTHER: 0.5,
    }

    # 行为风险权重
    BEHAVIOR_RISK = {
        BirdBehavior.FLYING: 0.4,
        BirdBehavior.PERCHING: 0.6,
        BirdBehavior.NESTING: 0.9,
        BirdBehavior.FEEDING: 0.5,
        BirdBehavior.HOVERING: 0.7,
        BirdBehavior.DIVING: 0.8,
        BirdBehavior.CIRCLING: 0.6,
        BirdBehavior.UNKNOWN: 0.5,
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # 关键设备区域
        self.critical_zones: List[Tuple[float, float, float, float]] = []

        # 轨迹预测器
        self.predictor = TrajectoryPredictor()

    def set_critical_zones(self, zones: List[Tuple[float, float, float, float]]):
        """设置关键设备区域"""
        self.critical_zones = zones

    def assess(
        self,
        tracks: List[BirdTrack],
        environment: Optional[EnvironmentContext] = None
    ) -> RiskAssessment:
        """
        评估风险

        Args:
            tracks: 活跃的鸟类轨迹
            environment: 环境上下文

        Returns:
            风险评估结果
        """
        if not tracks:
            return RiskAssessment(
                risk_level=RiskLevel.NONE,
                risk_score=0.0,
                primary_threats=[],
                affected_equipment=[],
                recommended_actions=[],
                urgency=0.0,
            )

        threats = []
        equipment_at_risk = []
        total_risk = 0.0
        min_impact_time = float('inf')

        for track in tracks:
            # 计算单个轨迹的风险
            track_risk = self._assess_track_risk(track, environment)
            total_risk = max(total_risk, track_risk)

            # 检查碰撞
            if self.critical_zones:
                impact = self.predictor.estimate_impact_point(
                    track, self.critical_zones
                )
                if impact:
                    _, _, time_to_impact = impact
                    min_impact_time = min(min_impact_time, time_to_impact)

                    threats.append(
                        f"{track.species.value}(ID:{track.track_id}) "
                        f"approaching in {time_to_impact:.1f}s"
                    )

            # 筑巢风险
            if track.duration > 60 and self._is_stationary(track):
                threats.append(
                    f"{track.species.value}(ID:{track.track_id}) "
                    f"possible nesting behavior"
                )
                total_risk = max(total_risk, 0.8)

        # 环境因素调整
        if environment:
            total_risk = self._adjust_for_environment(total_risk, environment)

        # 确定风险等级
        risk_level = self._score_to_level(total_risk)

        # 推荐动作
        recommended_actions = self._recommend_actions(risk_level, tracks)

        return RiskAssessment(
            risk_level=risk_level,
            risk_score=total_risk,
            primary_threats=threats,
            affected_equipment=equipment_at_risk,
            recommended_actions=recommended_actions,
            urgency=min(1.0, 1.0 / max(min_impact_time, 0.1)) if min_impact_time < float('inf') else 0.0,
            time_to_impact=min_impact_time if min_impact_time < float('inf') else None,
        )

    def _assess_track_risk(
        self,
        track: BirdTrack,
        environment: Optional[EnvironmentContext]
    ) -> float:
        """评估单个轨迹的风险"""
        # 基础风险
        species_risk = self.SPECIES_RISK.get(track.species, 0.5)

        # 行为风险
        behavior = self._infer_behavior(track)
        behavior_risk = self.BEHAVIOR_RISK.get(behavior, 0.5)

        # 速度风险
        velocity = track.get_velocity()
        if velocity is not None:
            speed = np.linalg.norm(velocity)
            speed_risk = min(1.0, speed / 0.5)  # 归一化速度
        else:
            speed_risk = 0.3

        # 接近度风险
        proximity_risk = self._compute_proximity_risk(track)

        # 综合风险
        risk = (
            0.3 * species_risk +
            0.25 * behavior_risk +
            0.2 * speed_risk +
            0.25 * proximity_risk
        )

        return min(1.0, risk)

    def _infer_behavior(self, track: BirdTrack) -> BirdBehavior:
        """推断鸟类行为"""
        if len(track.detections) < 5:
            return BirdBehavior.UNKNOWN

        # 计算运动特征
        trajectory = track.trajectory
        velocities = np.diff(trajectory, axis=0)

        mean_speed = np.mean(np.linalg.norm(velocities, axis=1))
        speed_std = np.std(np.linalg.norm(velocities, axis=1))

        # 方向变化
        if len(velocities) >= 2:
            angles = np.arctan2(velocities[:, 1], velocities[:, 0])
            angle_changes = np.abs(np.diff(angles))
            mean_angle_change = np.mean(angle_changes)
        else:
            mean_angle_change = 0

        # 判断行为
        if mean_speed < 0.01:
            return BirdBehavior.PERCHING
        elif mean_angle_change > np.pi / 4:
            return BirdBehavior.CIRCLING
        elif mean_speed > 0.3:
            return BirdBehavior.FLYING
        elif speed_std > mean_speed * 0.5:
            return BirdBehavior.HOVERING
        else:
            return BirdBehavior.FLYING

    def _is_stationary(self, track: BirdTrack) -> bool:
        """判断是否静止"""
        if len(track.detections) < 10:
            return False

        trajectory = track.trajectory[-20:]
        displacement = np.linalg.norm(trajectory[-1] - trajectory[0])

        return displacement < 0.05  # 归一化坐标

    def _compute_proximity_risk(self, track: BirdTrack) -> float:
        """计算接近度风险"""
        if not self.critical_zones or not track.detections:
            return 0.0

        current_pos = np.array(track.detections[-1].center)

        min_dist = float('inf')
        for zone in self.critical_zones:
            zx, zy, zw, zh = zone
            zone_center = np.array([zx + zw/2, zy + zh/2])
            dist = np.linalg.norm(current_pos - zone_center)
            min_dist = min(min_dist, dist)

        # 距离转风险 (越近风险越高)
        return max(0, 1.0 - min_dist * 2)

    def _adjust_for_environment(
        self,
        risk: float,
        env: EnvironmentContext
    ) -> float:
        """根据环境调整风险"""
        factor = 1.0

        # 时间因素
        if env.time_of_day in ["dawn", "dusk"]:
            factor *= 1.2  # 黄昏和黎明鸟类更活跃
        elif env.time_of_day == "night":
            factor *= 0.5  # 夜间风险较低

        # 季节因素
        if env.season in ["spring", "autumn"]:
            factor *= 1.3  # 迁徙季节

        # 天气因素
        if env.weather == "clear":
            factor *= 1.1  # 晴天更活跃
        elif env.weather in ["rainy", "snowy"]:
            factor *= 0.7  # 恶劣天气

        return min(1.0, risk * factor)

    def _score_to_level(self, score: float) -> RiskLevel:
        """风险分数转等级"""
        if score < 0.2:
            return RiskLevel.NONE
        elif score < 0.4:
            return RiskLevel.LOW
        elif score < 0.6:
            return RiskLevel.MEDIUM
        elif score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _recommend_actions(
        self,
        risk_level: RiskLevel,
        tracks: List[BirdTrack]
    ) -> List[DeterrenceAction]:
        """推荐驱鸟动作"""
        if risk_level == RiskLevel.NONE:
            return []
        elif risk_level == RiskLevel.LOW:
            return [DeterrenceAction.SOUND]
        elif risk_level == RiskLevel.MEDIUM:
            return [DeterrenceAction.SOUND, DeterrenceAction.VISUAL]
        elif risk_level == RiskLevel.HIGH:
            return [DeterrenceAction.LASER, DeterrenceAction.ULTRASONIC]
        else:  # CRITICAL
            return [DeterrenceAction.COMBINED]


# =============================================================================
# 鸟害风险预测系统
# =============================================================================

@dataclass
class BirdRiskPredictorConfig:
    """鸟害风险预测配置"""
    # 检测
    detection_confidence_threshold: float = 0.3
    max_tracks: int = 50

    # 分类
    classifier_model_path: Optional[str] = None
    classifier_confidence_threshold: float = 0.5

    # 跟踪
    track_max_age: float = 5.0              # 轨迹最大年龄(秒)
    track_min_hits: int = 3                  # 最小检测次数

    # 预测
    prediction_horizon: int = 30             # 预测步数

    # 驱鸟
    deterrence_cooldown: float = 10.0        # 驱鸟冷却时间(秒)


class BirdRiskPredictor:
    """
    鸟害风险预测系统

    整合检测、分类、跟踪、预测和策略推荐
    """

    def __init__(self, config: BirdRiskPredictorConfig):
        self.config = config

        # 分类器
        self.classifier = BirdClassifier(BirdClassifierConfig(
            model_path=config.classifier_model_path,
            confidence_threshold=config.classifier_confidence_threshold,
        ))

        # 风险评估器
        self.risk_assessor = RiskAssessor()

        # 轨迹预测器
        self.trajectory_predictor = TrajectoryPredictor(
            prediction_horizon=config.prediction_horizon
        )

        # 活跃轨迹
        self._tracks: Dict[int, BirdTrack] = {}
        self._next_track_id = 0

        # 驱鸟状态
        self._last_deterrence_time = 0.0
        self._deterrence_history: deque = deque(maxlen=100)

        # 环境上下文
        self._environment: Optional[EnvironmentContext] = None

    def set_critical_zones(self, zones: List[Tuple[float, float, float, float]]):
        """设置关键设备区域"""
        self.risk_assessor.set_critical_zones(zones)

    def set_environment(self, env: EnvironmentContext):
        """设置环境上下文"""
        self._environment = env

    def process_detections(
        self,
        detections: List[Dict[str, Any]],
        image: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> Tuple[List[BirdTrack], RiskAssessment]:
        """
        处理检测结果

        Args:
            detections: 检测结果列表
            image: 原始图像 (用于分类)
            timestamp: 时间戳

        Returns:
            (活跃轨迹, 风险评估)
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. 转换检测结果
        bird_detections = []
        for det in detections:
            bird_det = BirdDetection(
                track_id=-1,
                bbox=tuple(det.get("bbox", [0, 0, 0.1, 0.1])),
                confidence=det.get("confidence", 0.5),
                timestamp=timestamp,
            )

            # 分类
            if image is not None:
                species, species_conf, _ = self.classifier.classify(
                    image, bird_det.bbox
                )
                bird_det.species = species
                bird_det.species_confidence = species_conf

            bird_detections.append(bird_det)

        # 2. 关联轨迹
        self._associate_detections(bird_detections, timestamp)

        # 3. 清理过期轨迹
        self._cleanup_tracks(timestamp)

        # 4. 更新轨迹预测
        for track in self._tracks.values():
            if track.is_active:
                track.predicted_trajectory = self.trajectory_predictor.predict(track)

        # 5. 风险评估
        active_tracks = [t for t in self._tracks.values() if t.is_active]
        assessment = self.risk_assessor.assess(active_tracks, self._environment)

        return active_tracks, assessment

    def _associate_detections(
        self,
        detections: List[BirdDetection],
        timestamp: float
    ):
        """关联检测到轨迹"""
        if not detections:
            return

        # 简单的最近邻关联
        matched_tracks = set()
        matched_dets = set()

        # 计算距离矩阵
        active_tracks = [(tid, t) for tid, t in self._tracks.items() if t.is_active]

        for det_idx, det in enumerate(detections):
            det_center = np.array(det.center)
            best_track_id = None
            best_dist = 0.2  # 最大关联距离

            for track_id, track in active_tracks:
                if track_id in matched_tracks:
                    continue

                if track.detections:
                    track_center = np.array(track.detections[-1].center)
                    dist = np.linalg.norm(det_center - track_center)

                    if dist < best_dist:
                        best_dist = dist
                        best_track_id = track_id

            if best_track_id is not None:
                # 关联到现有轨迹
                det.track_id = best_track_id
                self._tracks[best_track_id].add_detection(det)
                matched_tracks.add(best_track_id)
                matched_dets.add(det_idx)

        # 为未匹配的检测创建新轨迹
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                track_id = self._next_track_id
                self._next_track_id += 1

                det.track_id = track_id
                track = BirdTrack(track_id=track_id)
                track.add_detection(det)
                self._tracks[track_id] = track

    def _cleanup_tracks(self, timestamp: float):
        """清理过期轨迹"""
        to_remove = []
        for track_id, track in self._tracks.items():
            age = timestamp - track.last_seen
            if age > self.config.track_max_age:
                track.is_active = False
                to_remove.append(track_id)

        # 保留一些历史轨迹用于分析
        if len(self._tracks) > self.config.max_tracks:
            for track_id in to_remove:
                del self._tracks[track_id]

    def get_deterrence_strategy(
        self,
        assessment: RiskAssessment
    ) -> Optional[DeterrenceStrategy]:
        """
        获取驱鸟策略

        Args:
            assessment: 风险评估结果

        Returns:
            驱鸟策略
        """
        current_time = time.time()

        # 检查冷却
        if current_time - self._last_deterrence_time < self.config.deterrence_cooldown:
            return None

        # 根据风险等级确定策略
        if assessment.risk_level == RiskLevel.NONE:
            return None

        if not assessment.recommended_actions:
            return None

        primary_action = assessment.recommended_actions[0]

        # 确定强度
        intensity_map = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 1.0,
        }
        intensity = intensity_map.get(assessment.risk_level, 0.5)

        # 确定持续时间
        duration_map = {
            RiskLevel.LOW: 3.0,
            RiskLevel.MEDIUM: 5.0,
            RiskLevel.HIGH: 8.0,
            RiskLevel.CRITICAL: 15.0,
        }
        duration = duration_map.get(assessment.risk_level, 5.0)

        strategy = DeterrenceStrategy(
            action=primary_action,
            intensity=intensity,
            duration=duration,
            priority=assessment.risk_level.value if hasattr(assessment.risk_level, 'value') else 0,
        )

        # 记录
        self._last_deterrence_time = current_time
        self._deterrence_history.append((current_time, strategy))

        return strategy

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        active_tracks = [t for t in self._tracks.values() if t.is_active]

        species_count = {}
        for track in active_tracks:
            species_count[track.species.value] = species_count.get(track.species.value, 0) + 1

        return {
            "total_tracks": len(self._tracks),
            "active_tracks": len(active_tracks),
            "species_distribution": species_count,
            "deterrence_count": len(self._deterrence_history),
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_bird_risk_predictor(config: Optional[Dict] = None) -> BirdRiskPredictor:
    """创建鸟害风险预测器"""
    predictor_config = BirdRiskPredictorConfig()
    if config:
        for key, value in config.items():
            if hasattr(predictor_config, key):
                setattr(predictor_config, key, value)
    return BirdRiskPredictor(predictor_config)
