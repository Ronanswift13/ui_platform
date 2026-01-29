#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室内电子围栏 V3.0 - 增强版多目标跟踪器
========================================

基于迭代方案实现:
- 集成DeepSORT深度特征关联
- 姿态估计和行为分析
- 权限校验与行为意图判定
- 遮挡恢复和长期重识别

作者: G组 | 版本: 3.0.0
"""

from __future__ import annotations
import logging
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 行为类型定义
# =============================================================================
class BehaviorType(Enum):
    """人员行为类型"""
    NORMAL = "normal"               # 正常站立
    WALKING = "walking"             # 行走
    RUNNING = "running"             # 跑动
    CROUCHING = "crouching"         # 蹲下
    BENDING = "bending"             # 弯腰
    FALLING = "falling"             # 跌倒
    LYING = "lying"                 # 躺卧
    REACHING = "reaching"           # 伸手操作
    CARRYING = "carrying"           # 搬运物品
    UNKNOWN = "unknown"


class RiskIntent(Enum):
    """风险意图类型"""
    SAFE = "safe"                           # 安全
    UNAUTHORIZED_ACCESS = "unauthorized_access"  # 未授权访问
    UNAUTHORIZED_OPERATION = "unauthorized_operation"  # 未授权操作
    DANGER_ZONE_ENTRY = "danger_zone_entry"     # 闯入高危区
    FALL_RISK = "fall_risk"                     # 跌倒风险
    PROLONGED_STAY = "prolonged_stay"           # 长时间滞留
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"  # 可疑行为


# =============================================================================
# 姿态关键点定义 (兼容 MoveNet/MediaPipe)
# =============================================================================
@dataclass
class Keypoint:
    """人体关键点"""
    x: float            # 归一化x坐标 [0, 1]
    y: float            # 归一化y坐标 [0, 1]
    confidence: float   # 置信度 [0, 1]
    name: str = ""      # 关键点名称


@dataclass
class PoseEstimation:
    """姿态估计结果"""
    keypoints: Dict[str, Keypoint]  # 关键点字典
    bbox: Dict[str, float]          # 边界框 {x, y, w, h}
    confidence: float               # 整体置信度
    timestamp: float = field(default_factory=time.time)
    
    # 计算的姿态特征
    torso_angle: float = 0.0        # 躯干倾斜角度
    head_height: float = 0.0        # 头部相对高度
    arm_extension: float = 0.0      # 手臂伸展程度
    leg_spread: float = 0.0         # 腿部分开程度
    
    # 派生姿态状态
    is_standing: bool = True
    is_crouching: bool = False
    is_lying: bool = False
    is_reaching: bool = False
    
    # MoveNet 17个关键点索引
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def compute_features(self):
        """计算姿态特征"""
        kps = self.keypoints
        
        # 躯干角度 (肩膀到髋部的角度)
        if all(k in kps for k in ["left_shoulder", "left_hip"]):
            ls, lh = kps["left_shoulder"], kps["left_hip"]
            if ls.confidence > 0.5 and lh.confidence > 0.5:
                dx = ls.x - lh.x
                dy = ls.y - lh.y
                self.torso_angle = np.degrees(np.arctan2(dx, dy))
        
        # 头部相对高度 (相对于髋部)
        if "nose" in kps and "left_hip" in kps:
            nose, hip = kps["nose"], kps["left_hip"]
            if nose.confidence > 0.5 and hip.confidence > 0.5:
                self.head_height = hip.y - nose.y  # y向下，所以站立时是正值
        
        # 手臂伸展 (手腕距离肩膀的距离)
        if all(k in kps for k in ["left_wrist", "left_shoulder"]):
            lw, ls = kps["left_wrist"], kps["left_shoulder"]
            if lw.confidence > 0.5 and ls.confidence > 0.5:
                self.arm_extension = np.sqrt((lw.x - ls.x)**2 + (lw.y - ls.y)**2)
        
        # 腿部分开程度
        if all(k in kps for k in ["left_ankle", "right_ankle"]):
            la, ra = kps["left_ankle"], kps["right_ankle"]
            if la.confidence > 0.5 and ra.confidence > 0.5:
                self.leg_spread = abs(la.x - ra.x)
        
        # 派生状态判断
        self._infer_pose_state()
    
    def _infer_pose_state(self):
        """推断姿态状态"""
        # 跌倒/躺卧检测: 头部低于髋部
        if self.head_height < 0.1:  # 头部几乎和髋部平齐或更低
            self.is_lying = True
            self.is_standing = False
        
        # 蹲下检测: 躯干角度大
        if abs(self.torso_angle) > 45:
            self.is_crouching = True
            self.is_standing = False
        
        # 伸手操作检测
        if self.arm_extension > 0.4:  # 手臂伸展超过40%
            self.is_reaching = True


# =============================================================================
# 深度特征提取器
# =============================================================================
class ReIDFeatureExtractor:
    """
    ReID特征提取器
    
    用于从人员图像中提取外观特征向量,
    支持遮挡后的身份恢复
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_dim: int = 128,
        use_gpu: bool = False
    ):
        self.model_path = model_path
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu
        self._model = None
        self._initialized = False
        
        self._load_model()
    
    def _load_model(self):
        """加载ReID模型"""
        try:
            if self.model_path:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if self.use_gpu else ['CPUExecutionProvider']
                self._model = ort.InferenceSession(self.model_path, providers=providers)
                self._initialized = True
                logger.info(f"[ReID] 模型加载成功: {self.model_path}")
            else:
                # 使用简单的颜色直方图特征作为备选
                logger.info("[ReID] 未指定模型，使用颜色直方图特征")
                self._initialized = True
        except Exception as e:
            logger.warning(f"[ReID] 模型加载失败: {e}, 使用备选特征")
            self._initialized = True
    
    def extract(self, image: np.ndarray, bbox: Dict[str, float]) -> np.ndarray:
        """
        提取ReID特征
        
        Args:
            image: 原始图像 (H, W, C)
            bbox: 边界框 {x, y, w, h}
            
        Returns:
            feature: 特征向量 (feature_dim,)
        """
        if not self._initialized:
            return np.zeros(self.feature_dim)
        
        try:
            # 裁剪人员区域
            x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
            x = max(0, x)
            y = max(0, y)
            crop = image[y:y+h, x:x+w]
            
            if crop.size == 0:
                return np.zeros(self.feature_dim)
            
            if self._model is not None:
                # 使用深度模型
                return self._extract_deep_feature(crop)
            else:
                # 使用颜色直方图
                return self._extract_color_histogram(crop)
        except Exception as e:
            logger.warning(f"[ReID] 特征提取失败: {e}")
            return np.zeros(self.feature_dim)
    
    def _extract_deep_feature(self, crop: np.ndarray) -> np.ndarray:
        """使用深度模型提取特征"""
        import cv2
        
        # 预处理
        img = cv2.resize(crop, (128, 256))
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = img.transpose(2, 0, 1)[np.newaxis, ...]
        
        # 推理
        inputs = {self._model.get_inputs()[0].name: img}
        outputs = self._model.run(None, inputs)
        feature = outputs[0].flatten()
        
        # L2归一化
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        
        return feature[:self.feature_dim]
    
    def _extract_color_histogram(self, crop: np.ndarray) -> np.ndarray:
        """使用颜色直方图提取特征"""
        import cv2
        
        # 转换到HSV空间
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # 计算直方图
        h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
        
        # 归一化
        h_hist = h_hist / (h_hist.sum() + 1e-6)
        s_hist = s_hist / (s_hist.sum() + 1e-6)
        v_hist = v_hist / (v_hist.sum() + 1e-6)
        
        # 拼接特征
        feature = np.concatenate([h_hist, s_hist, v_hist])
        
        # 填充或截断到目标维度
        if len(feature) < self.feature_dim:
            feature = np.pad(feature, (0, self.feature_dim - len(feature)))
        else:
            feature = feature[:self.feature_dim]
        
        return feature


# =============================================================================
# 姿态估计器
# =============================================================================
class PoseEstimator:
    """
    姿态估计器
    
    支持 MoveNet 或 MediaPipe Pose 模型
    用于检测人体关键点和行为分析
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "movenet",  # movenet, mediapipe
        use_gpu: bool = False
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.use_gpu = use_gpu
        self._model = None
        self._initialized = False
        
        self._load_model()
    
    def _load_model(self):
        """加载姿态模型"""
        try:
            if self.model_path:
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                    if self.use_gpu else ['CPUExecutionProvider']
                self._model = ort.InferenceSession(self.model_path, providers=providers)
                self._initialized = True
                logger.info(f"[Pose] 模型加载成功: {self.model_path}")
            else:
                logger.info("[Pose] 未指定模型，使用模拟姿态估计")
                self._initialized = True
        except Exception as e:
            logger.warning(f"[Pose] 模型加载失败: {e}")
            self._initialized = True
    
    def estimate(self, image: np.ndarray, bbox: Dict[str, float]) -> Optional[PoseEstimation]:
        """
        估计人体姿态
        
        Args:
            image: 原始图像
            bbox: 人员边界框
            
        Returns:
            PoseEstimation 或 None
        """
        if not self._initialized:
            return None
        
        try:
            if self._model is not None:
                return self._estimate_with_model(image, bbox)
            else:
                return self._estimate_simulated(bbox)
        except Exception as e:
            logger.warning(f"[Pose] 姿态估计失败: {e}")
            return None
    
    def _estimate_with_model(self, image: np.ndarray, bbox: Dict[str, float]) -> PoseEstimation:
        """使用模型估计姿态"""
        import cv2
        
        # 裁剪并预处理
        x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
        crop = image[max(0,y):y+h, max(0,x):x+w]
        
        # MoveNet输入大小
        input_size = 192  # MoveNet Lightning
        img = cv2.resize(crop, (input_size, input_size))
        img = img.astype(np.int32)[np.newaxis, ...]
        
        # 推理
        inputs = {self._model.get_inputs()[0].name: img}
        outputs = self._model.run(None, inputs)
        keypoints_raw = outputs[0][0, 0, :, :]  # (17, 3)
        
        # 解析关键点
        keypoints = {}
        for i, name in enumerate(PoseEstimation.KEYPOINT_NAMES):
            ky, kx, kc = keypoints_raw[i]
            keypoints[name] = Keypoint(
                x=kx * w / input_size + x,
                y=ky * h / input_size + y,
                confidence=float(kc),
                name=name
            )
        
        pose = PoseEstimation(
            keypoints=keypoints,
            bbox=bbox,
            confidence=np.mean([kp.confidence for kp in keypoints.values()])
        )
        pose.compute_features()
        
        return pose
    
    def _estimate_simulated(self, bbox: Dict[str, float]) -> PoseEstimation:
        """模拟姿态估计 (用于测试)"""
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        
        # 生成合理的关键点位置
        keypoints = {}
        # 头部
        keypoints["nose"] = Keypoint(x + w*0.5, y + h*0.1, 0.9, "nose")
        keypoints["left_eye"] = Keypoint(x + w*0.4, y + h*0.08, 0.9, "left_eye")
        keypoints["right_eye"] = Keypoint(x + w*0.6, y + h*0.08, 0.9, "right_eye")
        keypoints["left_ear"] = Keypoint(x + w*0.3, y + h*0.1, 0.8, "left_ear")
        keypoints["right_ear"] = Keypoint(x + w*0.7, y + h*0.1, 0.8, "right_ear")
        # 肩膀
        keypoints["left_shoulder"] = Keypoint(x + w*0.3, y + h*0.25, 0.9, "left_shoulder")
        keypoints["right_shoulder"] = Keypoint(x + w*0.7, y + h*0.25, 0.9, "right_shoulder")
        # 手肘
        keypoints["left_elbow"] = Keypoint(x + w*0.2, y + h*0.4, 0.8, "left_elbow")
        keypoints["right_elbow"] = Keypoint(x + w*0.8, y + h*0.4, 0.8, "right_elbow")
        # 手腕
        keypoints["left_wrist"] = Keypoint(x + w*0.15, y + h*0.55, 0.8, "left_wrist")
        keypoints["right_wrist"] = Keypoint(x + w*0.85, y + h*0.55, 0.8, "right_wrist")
        # 髋部
        keypoints["left_hip"] = Keypoint(x + w*0.4, y + h*0.55, 0.9, "left_hip")
        keypoints["right_hip"] = Keypoint(x + w*0.6, y + h*0.55, 0.9, "right_hip")
        # 膝盖
        keypoints["left_knee"] = Keypoint(x + w*0.35, y + h*0.75, 0.8, "left_knee")
        keypoints["right_knee"] = Keypoint(x + w*0.65, y + h*0.75, 0.8, "right_knee")
        # 脚踝
        keypoints["left_ankle"] = Keypoint(x + w*0.35, y + h*0.95, 0.8, "left_ankle")
        keypoints["right_ankle"] = Keypoint(x + w*0.65, y + h*0.95, 0.8, "right_ankle")
        
        pose = PoseEstimation(
            keypoints=keypoints,
            bbox=bbox,
            confidence=0.85
        )
        pose.compute_features()
        
        return pose


# =============================================================================
# 行为分析器
# =============================================================================
class BehaviorAnalyzer:
    """
    行为分析器
    
    基于姿态序列分析人员行为:
    - 跌倒检测
    - 低位操作检测
    - 可疑行为识别
    """
    
    def __init__(
        self,
        fall_detection_threshold: float = 2.0,  # 跌倒判定时间阈值(秒)
        low_operation_threshold: float = 0.5,   # 低位操作高度阈值(相对)
        stay_time_threshold: float = 300.0,     # 滞留时间阈值(秒)
    ):
        self.fall_detection_threshold = fall_detection_threshold
        self.low_operation_threshold = low_operation_threshold
        self.stay_time_threshold = stay_time_threshold
        
        # 姿态历史 {track_id: deque of PoseEstimation}
        self._pose_history: Dict[str, deque] = {}
        
        # 行为状态 {track_id: (behavior, start_time)}
        self._behavior_states: Dict[str, Tuple[BehaviorType, float]] = {}
    
    def update(
        self,
        track_id: str,
        pose: Optional[PoseEstimation],
        velocity: Tuple[float, float] = (0, 0)
    ) -> Tuple[BehaviorType, float]:
        """
        更新行为分析
        
        Args:
            track_id: 跟踪ID
            pose: 姿态估计结果
            velocity: 移动速度 (vx, vy) m/s
            
        Returns:
            (behavior_type, confidence)
        """
        current_time = time.time()
        
        # 初始化历史
        if track_id not in self._pose_history:
            self._pose_history[track_id] = deque(maxlen=30)  # 保存最近30帧
        
        # 添加姿态到历史
        if pose is not None:
            self._pose_history[track_id].append(pose)
        
        # 分析行为
        behavior, confidence = self._analyze_behavior(track_id, velocity)
        
        # 更新状态
        if track_id not in self._behavior_states:
            self._behavior_states[track_id] = (behavior, current_time)
        elif self._behavior_states[track_id][0] != behavior:
            self._behavior_states[track_id] = (behavior, current_time)
        
        return behavior, confidence
    
    def _analyze_behavior(
        self,
        track_id: str,
        velocity: Tuple[float, float]
    ) -> Tuple[BehaviorType, float]:
        """分析行为类型"""
        history = self._pose_history.get(track_id, deque())
        
        if len(history) == 0:
            return BehaviorType.UNKNOWN, 0.0
        
        # 最近姿态
        recent_pose = history[-1]
        
        # 1. 跌倒检测
        if self._check_falling(history):
            return BehaviorType.FALLING, 0.95
        
        # 2. 躺卧检测
        if recent_pose.is_lying:
            return BehaviorType.LYING, 0.9
        
        # 3. 蹲下检测
        if recent_pose.is_crouching:
            return BehaviorType.CROUCHING, 0.85
        
        # 4. 弯腰检测
        if abs(recent_pose.torso_angle) > 30 and not recent_pose.is_crouching:
            return BehaviorType.BENDING, 0.8
        
        # 5. 伸手操作
        if recent_pose.is_reaching:
            return BehaviorType.REACHING, 0.8
        
        # 6. 跑动检测 (基于速度)
        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        if speed > 2.0:  # 超过2m/s
            return BehaviorType.RUNNING, 0.85
        
        # 7. 行走检测
        if speed > 0.3:  # 超过0.3m/s
            return BehaviorType.WALKING, 0.8
        
        # 8. 正常站立
        if recent_pose.is_standing:
            return BehaviorType.NORMAL, 0.9
        
        return BehaviorType.UNKNOWN, 0.5
    
    def _check_falling(self, history: deque) -> bool:
        """检测跌倒事件"""
        if len(history) < 5:
            return False
        
        # 检查最近几帧的头部高度变化
        recent = list(history)[-5:]
        head_heights = [p.head_height for p in recent if p is not None]
        
        if len(head_heights) < 3:
            return False
        
        # 头部高度急剧下降
        height_diff = head_heights[0] - head_heights[-1]
        if height_diff > 0.3:  # 头部下降超过30%
            # 检查是否持续处于低位
            low_count = sum(1 for h in head_heights if h < 0.2)
            if low_count >= 3:
                return True
        
        return False
    
    def get_behavior_duration(self, track_id: str) -> float:
        """获取当前行为持续时间"""
        if track_id not in self._behavior_states:
            return 0.0
        return time.time() - self._behavior_states[track_id][1]
    
    def clear_track(self, track_id: str):
        """清除跟踪数据"""
        self._pose_history.pop(track_id, None)
        self._behavior_states.pop(track_id, None)


# =============================================================================
# 风险意图分析器
# =============================================================================
class RiskIntentAnalyzer:
    """
    风险意图分析器
    
    综合位置、行为、授权状态分析人员意图风险
    """
    
    def __init__(
        self,
        unauthorized_stay_threshold: float = 10.0,  # 未授权滞留阈值(秒)
        danger_zone_threshold: float = 5.0,         # 危险区停留阈值(秒)
    ):
        self.unauthorized_stay_threshold = unauthorized_stay_threshold
        self.danger_zone_threshold = danger_zone_threshold
        
        # 跟踪状态 {track_id: {zone_id: enter_time}}
        self._zone_history: Dict[str, Dict[str, float]] = {}
    
    def analyze(
        self,
        track_id: str,
        position: Tuple[float, float],
        behavior: BehaviorType,
        is_authorized: bool,
        current_zone: Optional[str] = None,
        zone_type: Optional[str] = None,  # normal, restricted, danger
        nearby_cabinet: Optional[str] = None,
        is_cabinet_authorized: bool = True,
    ) -> Tuple[RiskIntent, float, str]:
        """
        分析风险意图
        
        Returns:
            (risk_intent, risk_score, reason)
        """
        current_time = time.time()
        
        # 初始化区域历史
        if track_id not in self._zone_history:
            self._zone_history[track_id] = {}
        
        # 更新区域进入时间
        if current_zone and current_zone not in self._zone_history[track_id]:
            self._zone_history[track_id][current_zone] = current_time
        
        risks = []
        
        # 1. 跌倒风险
        if behavior in [BehaviorType.FALLING, BehaviorType.LYING]:
            risks.append((RiskIntent.FALL_RISK, 0.95, "检测到人员跌倒或躺卧"))
        
        # 2. 危险区闯入
        if zone_type == "danger":
            stay_time = current_time - self._zone_history[track_id].get(current_zone, current_time)
            if stay_time > self.danger_zone_threshold:
                risks.append((RiskIntent.DANGER_ZONE_ENTRY, 0.9, 
                             f"在危险区域停留{stay_time:.1f}秒"))
        
        # 3. 未授权访问
        if not is_authorized and zone_type in ["restricted", "danger"]:
            stay_time = current_time - self._zone_history[track_id].get(current_zone, current_time)
            if stay_time > self.unauthorized_stay_threshold:
                risks.append((RiskIntent.UNAUTHORIZED_ACCESS, 0.85,
                             f"未授权人员在限制区域停留{stay_time:.1f}秒"))
        
        # 4. 未授权操作
        if nearby_cabinet and not is_cabinet_authorized:
            if behavior in [BehaviorType.REACHING, BehaviorType.BENDING]:
                risks.append((RiskIntent.UNAUTHORIZED_OPERATION, 0.9,
                             f"未授权操作机柜{nearby_cabinet}"))
        
        # 5. 可疑行为
        if behavior == BehaviorType.CROUCHING and zone_type == "restricted":
            risks.append((RiskIntent.SUSPICIOUS_BEHAVIOR, 0.7,
                         "在限制区域出现蹲下行为"))
        
        # 6. 长时间滞留
        if current_zone:
            total_stay = current_time - self._zone_history[track_id].get(current_zone, current_time)
            if total_stay > 300:  # 5分钟
                risks.append((RiskIntent.PROLONGED_STAY, 0.6,
                             f"在区域{current_zone}停留{total_stay/60:.1f}分钟"))
        
        # 返回最高风险
        if risks:
            risks.sort(key=lambda x: x[1], reverse=True)
            return risks[0]
        
        return RiskIntent.SAFE, 0.0, "正常活动"
    
    def clear_track(self, track_id: str):
        """清除跟踪数据"""
        self._zone_history.pop(track_id, None)


# =============================================================================
# V3.0 增强版跟踪结果
# =============================================================================
@dataclass
class EnhancedTrackResult:
    """V3.0增强版跟踪结果"""
    track_id: str                           # 稳定的跟踪ID
    person_id: Optional[str] = None         # 识别出的人员ID
    
    # 位置信息
    position: Tuple[float, float] = (0, 0)  # 地面坐标 (x, y)
    velocity: Tuple[float, float] = (0, 0)  # 速度 (vx, vy) m/s
    bbox: Dict[str, float] = field(default_factory=dict)  # 图像边界框
    
    # 跟踪状态
    confidence: float = 0.0                 # 跟踪置信度
    is_confirmed: bool = False              # 是否已确认
    age: int = 0                            # 跟踪帧数
    
    # 姿态和行为
    pose: Optional[PoseEstimation] = None   # 姿态估计
    behavior: BehaviorType = BehaviorType.UNKNOWN  # 行为类型
    behavior_confidence: float = 0.0        # 行为置信度
    behavior_duration: float = 0.0          # 行为持续时间
    
    # 风险评估
    risk_intent: RiskIntent = RiskIntent.SAFE  # 风险意图
    risk_score: float = 0.0                 # 风险分数
    risk_reason: str = ""                   # 风险原因
    
    # 授权状态
    is_authorized: bool = True              # 是否授权
    current_zone: Optional[str] = None      # 当前区域
    zone_type: Optional[str] = None         # 区域类型
    nearby_cabinet: Optional[str] = None    # 附近机柜
    
    # 时间戳
    timestamp: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    
    # 特征
    feature: Optional[np.ndarray] = None    # ReID特征向量
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "track_id": self.track_id,
            "person_id": self.person_id,
            "position": {"x": self.position[0], "y": self.position[1]},
            "velocity": {"vx": self.velocity[0], "vy": self.velocity[1]},
            "bbox": self.bbox,
            "confidence": self.confidence,
            "is_confirmed": self.is_confirmed,
            "age": self.age,
            "behavior": self.behavior.value,
            "behavior_confidence": self.behavior_confidence,
            "behavior_duration": self.behavior_duration,
            "risk_intent": self.risk_intent.value,
            "risk_score": self.risk_score,
            "risk_reason": self.risk_reason,
            "is_authorized": self.is_authorized,
            "current_zone": self.current_zone,
            "zone_type": self.zone_type,
            "nearby_cabinet": self.nearby_cabinet,
            "timestamp": self.timestamp,
        }


# =============================================================================
# V3.0 增强版融合跟踪器
# =============================================================================
class FusedTrackerV3:
    """
    V3.0 增强版融合跟踪器
    
    功能:
    1. DeepSORT深度特征关联
    2. 姿态估计和行为分析
    3. 风险意图判定
    4. 遮挡恢复和长期重识别
    """
    
    def __init__(
        self,
        # 跟踪参数
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        distance_threshold: float = 1.0,
        feature_threshold: float = 0.5,
        # 模型路径
        reid_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        # 其他配置
        use_gpu: bool = False,
        enable_behavior_analysis: bool = True,
        enable_risk_analysis: bool = True,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.feature_threshold = feature_threshold
        self.use_gpu = use_gpu
        self.enable_behavior_analysis = enable_behavior_analysis
        self.enable_risk_analysis = enable_risk_analysis
        
        # 跟踪状态
        self._tracks: Dict[str, EnhancedTrackResult] = {}
        self._next_id = 1
        
        # 特征历史 (用于长期重识别)
        self._feature_gallery: Dict[str, List[np.ndarray]] = {}
        
        # 初始化组件
        self._reid_extractor = ReIDFeatureExtractor(
            model_path=reid_model_path,
            use_gpu=use_gpu
        )
        self._pose_estimator = PoseEstimator(
            model_path=pose_model_path,
            use_gpu=use_gpu
        )
        self._behavior_analyzer = BehaviorAnalyzer()
        self._risk_analyzer = RiskIntentAnalyzer()
        
        # 卡尔曼滤波状态
        self._kf_states: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        
        logger.info("[FusedTrackerV3] 初始化完成")
    
    def _generate_track_id(self) -> str:
        """生成新的跟踪ID"""
        track_id = f"T{self._next_id:05d}"
        self._next_id += 1
        return track_id
    
    def update(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        zone_info: Optional[Dict[str, Any]] = None,
        authorization_info: Optional[Dict[str, bool]] = None,
    ) -> List[EnhancedTrackResult]:
        """
        更新跟踪
        
        Args:
            image: 当前帧图像
            detections: 检测结果列表 [{bbox, confidence, ...}, ...]
            zone_info: 区域信息 {zone_id: {type, polygon}}
            authorization_info: 授权信息 {person_id: authorized}
            
        Returns:
            已确认的跟踪结果列表
        """
        current_time = time.time()
        
        # 1. 提取特征
        features = []
        poses = []
        for det in detections:
            bbox = det.get('bbox', {})
            # ReID特征
            feature = self._reid_extractor.extract(image, bbox)
            features.append(feature)
            # 姿态估计
            pose = self._pose_estimator.estimate(image, bbox) if self.enable_behavior_analysis else None
            poses.append(pose)
        
        # 2. 预测现有跟踪
        for track_id, track in self._tracks.items():
            self._predict_track(track)
        
        # 3. 关联检测和跟踪
        matches, unmatched_dets, unmatched_tracks = self._associate(
            detections, features
        )
        
        # 4. 更新匹配的跟踪
        for track_id, det_idx in matches:
            self._update_track(
                track_id,
                detections[det_idx],
                features[det_idx],
                poses[det_idx],
                zone_info,
                authorization_info
            )
        
        # 5. 处理未匹配的跟踪
        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.age += 1
        
        # 6. 创建新跟踪
        for det_idx in unmatched_dets:
            self._create_track(
                detections[det_idx],
                features[det_idx],
                poses[det_idx],
                zone_info,
                authorization_info
            )
        
        # 7. 删除过期跟踪
        self._remove_stale_tracks()
        
        # 8. 返回确认的跟踪
        return [t for t in self._tracks.values() if t.is_confirmed]
    
    def _predict_track(self, track: EnhancedTrackResult):
        """卡尔曼滤波预测"""
        if track.track_id not in self._kf_states:
            return
        
        state, cov = self._kf_states[track.track_id]
        
        # 简单的恒速模型预测
        dt = 0.1
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        Q = np.eye(4) * 0.1
        
        state = F @ state
        cov = F @ cov @ F.T + Q
        
        self._kf_states[track.track_id] = (state, cov)
        track.position = (float(state[0]), float(state[1]))
        track.velocity = (float(state[2]), float(state[3]))
    
    def _associate(
        self,
        detections: List[Dict],
        features: List[np.ndarray]
    ) -> Tuple[List[Tuple[str, int]], List[int], List[str]]:
        """
        关联检测和跟踪
        
        使用级联匹配:
        1. 首先基于特征匹配最近确认的跟踪
        2. 然后基于位置匹配剩余跟踪
        """
        if len(detections) == 0:
            return [], [], list(self._tracks.keys())
        
        if len(self._tracks) == 0:
            return [], list(range(len(detections))), []
        
        # 计算代价矩阵
        track_ids = list(self._tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            track_feature = track.feature
            
            for j, det in enumerate(detections):
                # 位置距离
                det_pos = self._get_detection_position(det)
                pos_dist = np.sqrt(
                    (track.position[0] - det_pos[0])**2 +
                    (track.position[1] - det_pos[1])**2
                )
                
                # 特征距离
                if track_feature is not None and features[j] is not None:
                    feat_dist = 1 - np.dot(track_feature, features[j])
                else:
                    feat_dist = 1.0
                
                # 综合代价
                cost = 0.6 * pos_dist + 0.4 * feat_dist * self.distance_threshold
                cost_matrix[i, j] = cost
        
        # 匈牙利匹配
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(track_ids)
        
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < self.distance_threshold:
                track_id = track_ids[i]
                matches.append((track_id, j))
                unmatched_dets.discard(j)
                unmatched_tracks.discard(track_id)
        
        return matches, list(unmatched_dets), list(unmatched_tracks)
    
    def _get_detection_position(self, det: Dict) -> Tuple[float, float]:
        """从检测结果获取地面位置"""
        if 'ground_position' in det:
            return det['ground_position']
        
        # 从边界框估计
        bbox = det.get('bbox', {})
        x = bbox.get('x', 0) + bbox.get('w', 0) / 2
        y = bbox.get('y', 0) + bbox.get('h', 0)
        return (x / 100, y / 100)  # 简化的像素到米转换
    
    def _update_track(
        self,
        track_id: str,
        detection: Dict,
        feature: np.ndarray,
        pose: Optional[PoseEstimation],
        zone_info: Optional[Dict],
        auth_info: Optional[Dict]
    ):
        """更新跟踪"""
        track = self._tracks[track_id]
        current_time = time.time()
        
        # 更新位置
        pos = self._get_detection_position(detection)
        track.position = pos
        track.bbox = detection.get('bbox', {})
        track.confidence = detection.get('confidence', 0.9)
        track.timestamp = current_time
        track.age += 1
        
        # 更新卡尔曼滤波
        if track_id in self._kf_states:
            state, cov = self._kf_states[track_id]
            # 观测更新
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            R = np.eye(2) * 0.1
            z = np.array([pos[0], pos[1]])
            
            y = z - H @ state
            S = H @ cov @ H.T + R
            K = cov @ H.T @ np.linalg.inv(S)
            
            state = state + K @ y
            cov = (np.eye(4) - K @ H) @ cov
            
            self._kf_states[track_id] = (state, cov)
            track.velocity = (float(state[2]), float(state[3]))
        
        # 更新特征
        if feature is not None:
            track.feature = feature
            # 添加到特征库
            if track_id not in self._feature_gallery:
                self._feature_gallery[track_id] = []
            self._feature_gallery[track_id].append(feature)
            if len(self._feature_gallery[track_id]) > 10:
                self._feature_gallery[track_id].pop(0)
        
        # 更新姿态
        track.pose = pose
        
        # 行为分析
        if self.enable_behavior_analysis:
            behavior, conf = self._behavior_analyzer.update(
                track_id, pose, track.velocity
            )
            track.behavior = behavior
            track.behavior_confidence = conf
            track.behavior_duration = self._behavior_analyzer.get_behavior_duration(track_id)
        
        # 区域和授权状态
        if zone_info:
            track.current_zone, track.zone_type = self._determine_zone(pos, zone_info)
        
        track.is_authorized = auth_info.get(track.person_id, True) if auth_info else True
        
        # 风险分析
        if self.enable_risk_analysis:
            risk, score, reason = self._risk_analyzer.analyze(
                track_id=track_id,
                position=pos,
                behavior=track.behavior,
                is_authorized=track.is_authorized,
                current_zone=track.current_zone,
                zone_type=track.zone_type,
                nearby_cabinet=track.nearby_cabinet,
                is_cabinet_authorized=track.is_authorized
            )
            track.risk_intent = risk
            track.risk_score = score
            track.risk_reason = reason
        
        # 确认跟踪
        if not track.is_confirmed and track.age >= self.min_hits:
            track.is_confirmed = True
    
    def _create_track(
        self,
        detection: Dict,
        feature: np.ndarray,
        pose: Optional[PoseEstimation],
        zone_info: Optional[Dict],
        auth_info: Optional[Dict]
    ):
        """创建新跟踪"""
        track_id = self._generate_track_id()
        pos = self._get_detection_position(detection)
        current_time = time.time()
        
        track = EnhancedTrackResult(
            track_id=track_id,
            position=pos,
            bbox=detection.get('bbox', {}),
            confidence=detection.get('confidence', 0.9),
            is_confirmed=False,
            age=1,
            pose=pose,
            feature=feature,
            timestamp=current_time,
            first_seen=current_time
        )
        
        # 初始化卡尔曼滤波
        state = np.array([pos[0], pos[1], 0, 0])
        cov = np.eye(4) * 1.0
        self._kf_states[track_id] = (state, cov)
        
        # 区域
        if zone_info:
            track.current_zone, track.zone_type = self._determine_zone(pos, zone_info)
        
        self._tracks[track_id] = track
        self._feature_gallery[track_id] = [feature] if feature is not None else []
    
    def _determine_zone(
        self,
        pos: Tuple[float, float],
        zone_info: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """确定所在区域"""
        for zone_id, info in zone_info.items():
            polygon = info.get('polygon', [])
            if self._point_in_polygon(pos, polygon):
                return zone_id, info.get('type', 'normal')
        return None, None
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List) -> bool:
        """点是否在多边形内"""
        if len(polygon) < 3:
            return False
        
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _remove_stale_tracks(self):
        """移除过期跟踪"""
        stale_ids = []
        for track_id, track in self._tracks.items():
            if track.age > self.max_age and not track.is_confirmed:
                stale_ids.append(track_id)
            elif not self._is_track_active(track):
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            self._tracks.pop(track_id, None)
            self._kf_states.pop(track_id, None)
            self._feature_gallery.pop(track_id, None)
            self._behavior_analyzer.clear_track(track_id)
            self._risk_analyzer.clear_track(track_id)
    
    def _is_track_active(self, track: EnhancedTrackResult) -> bool:
        """检查跟踪是否仍然活跃"""
        inactive_time = time.time() - track.timestamp
        return inactive_time < self.max_age * 0.1  # 假设0.1秒/帧
    
    def get_all_tracks(self) -> List[EnhancedTrackResult]:
        """获取所有跟踪"""
        return list(self._tracks.values())
    
    def get_track(self, track_id: str) -> Optional[EnhancedTrackResult]:
        """获取指定跟踪"""
        return self._tracks.get(track_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        confirmed = sum(1 for t in self._tracks.values() if t.is_confirmed)
        return {
            "total_tracks": len(self._tracks),
            "confirmed_tracks": confirmed,
            "next_id": self._next_id,
            "feature_gallery_size": sum(len(v) for v in self._feature_gallery.values())
        }


# =============================================================================
# 导出
# =============================================================================
__all__ = [
    # 类型
    'BehaviorType',
    'RiskIntent',
    # 数据类
    'Keypoint',
    'PoseEstimation',
    'EnhancedTrackResult',
    # 组件
    'ReIDFeatureExtractor',
    'PoseEstimator',
    'BehaviorAnalyzer',
    'RiskIntentAnalyzer',
    # 跟踪器
    'FusedTrackerV3',
]
