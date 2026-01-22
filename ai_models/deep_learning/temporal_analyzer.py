"""
时序分析模块 - 多帧状态判断
输变电激光星芒破夜绘明监测平台

基于UPDATE.md中期迭代计划:
- 开关状态多帧判断
- LSTM/Transformer时序模型
- 手柄姿态序列分析
- SCADA数据交叉验证
- 异常开合过程检测

支持:
- 多帧特征融合
- 时序一致性验证
- 状态转换检测
- 异常模式识别

版本: 1.0.0
"""

from __future__ import annotations
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入深度学习框架
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# =============================================================================
# 枚举定义
# =============================================================================

class SwitchState(Enum):
    """开关状态"""
    UNKNOWN = "unknown"
    OPEN = "open"
    CLOSED = "closed"
    INTERMEDIATE = "intermediate"
    TRANSITIONING = "transitioning"


class TransitionType(Enum):
    """状态转换类型"""
    NONE = "none"
    OPENING = "opening"         # 正在分闸
    CLOSING = "closing"         # 正在合闸
    STUCK = "stuck"             # 卡滞
    BOUNCING = "bouncing"       # 弹跳
    ABNORMAL = "abnormal"       # 异常


class AnomalyType(Enum):
    """异常类型"""
    NONE = "none"
    SLOW_OPERATION = "slow_operation"       # 操作过慢
    FAST_OPERATION = "fast_operation"       # 操作过快
    INCOMPLETE_OPERATION = "incomplete"     # 操作不完整
    STATE_OSCILLATION = "oscillation"       # 状态振荡
    UNEXPECTED_TRANSITION = "unexpected"    # 意外转换


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class FrameObservation:
    """单帧观测"""
    timestamp: float
    state: SwitchState
    confidence: float
    angle: Optional[float] = None           # 手柄角度
    position: Optional[np.ndarray] = None   # 关键点位置
    features: Optional[np.ndarray] = None   # 特征向量
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalContext:
    """时序上下文"""
    observations: List[FrameObservation]
    window_size: int
    fps: float = 25.0

    @property
    def duration(self) -> float:
        """序列时长(秒)"""
        if len(self.observations) < 2:
            return 0.0
        return self.observations[-1].timestamp - self.observations[0].timestamp

    @property
    def state_sequence(self) -> List[SwitchState]:
        """状态序列"""
        return [obs.state for obs in self.observations]

    @property
    def confidence_sequence(self) -> np.ndarray:
        """置信度序列"""
        return np.array([obs.confidence for obs in self.observations])

    @property
    def angle_sequence(self) -> Optional[np.ndarray]:
        """角度序列"""
        angles = [obs.angle for obs in self.observations if obs.angle is not None]
        return np.array(angles) if angles else None


@dataclass
class TemporalDecision:
    """时序决策结果"""
    final_state: SwitchState
    confidence: float
    transition_type: TransitionType
    anomaly_type: AnomalyType
    anomaly_score: float = 0.0
    state_duration: float = 0.0             # 当前状态持续时间
    transition_progress: float = 0.0        # 转换进度 [0-1]
    evidences: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SCADACommand:
    """SCADA命令"""
    timestamp: float
    device_id: str
    command: str                            # "open" / "close"
    source: str = "scada"                   # scada / manual / auto


# =============================================================================
# 时序分析器配置
# =============================================================================

@dataclass
class TemporalAnalyzerConfig:
    """时序分析器配置"""
    # 时间窗口
    window_size: int = 30                   # 帧数
    min_observations: int = 5               # 最小观测数
    fps: float = 25.0                       # 帧率

    # 状态判断
    state_stability_threshold: float = 0.8  # 状态稳定阈值
    confidence_threshold: float = 0.6       # 置信度阈值
    vote_threshold: float = 0.7             # 投票阈值

    # 转换检测
    transition_time_min: float = 0.1        # 最小转换时间(秒)
    transition_time_max: float = 3.0        # 最大转换时间(秒)
    transition_time_normal: float = 0.5     # 正常转换时间(秒)

    # 角度分析
    angle_open: float = -60.0               # 分闸角度
    angle_closed: float = 30.0              # 合闸角度
    angle_tolerance: float = 15.0           # 角度容差

    # 异常检测
    oscillation_threshold: int = 3          # 振荡检测阈值
    stuck_timeout: float = 5.0              # 卡滞超时(秒)

    # 模型
    use_lstm: bool = True                   # 使用LSTM模型
    lstm_hidden_size: int = 64              # LSTM隐藏层大小
    lstm_num_layers: int = 2                # LSTM层数
    model_path: Optional[str] = None        # 模型路径


# =============================================================================
# LSTM时序模型
# =============================================================================

if TORCH_AVAILABLE:
    class SwitchStateLSTM(nn.Module):
        """
        开关状态LSTM模型

        输入: (batch, seq_len, features)
        输出: (batch, num_states)
        """

        def __init__(
            self,
            input_size: int = 4,            # [angle, confidence, state_onehot]
            hidden_size: int = 64,
            num_layers: int = 2,
            num_states: int = 4,            # unknown, open, closed, intermediate
            dropout: float = 0.1,
        ):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False,
            )

            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_states),
            )

            self.anomaly_detector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            x: torch.Tensor,
            lengths: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            前向传播

            Args:
                x: (batch, seq_len, features)
                lengths: (batch,) 实际序列长度

            Returns:
                (state_logits, anomaly_score)
            """
            # LSTM
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

            # Attention
            attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden)

            # 状态分类
            state_logits = self.classifier(context)  # (batch, num_states)

            # 异常检测
            anomaly_score = self.anomaly_detector(context)  # (batch, 1)

            return state_logits, anomaly_score.squeeze(-1)

    class SwitchStateTransformer(nn.Module):
        """
        开关状态Transformer模型
        """

        def __init__(
            self,
            input_size: int = 4,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            num_states: int = 4,
            dropout: float = 0.1,
            max_seq_len: int = 100,
        ):
            super().__init__()

            self.input_projection = nn.Linear(input_size, d_model)

            # 位置编码
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_seq_len, d_model) * 0.1
            )

            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

            # CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)

            # 分类头
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_states),
            )

            self.anomaly_detector = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size, seq_len, _ = x.shape

            # 投影
            x = self.input_projection(x)

            # 添加位置编码
            x = x + self.pos_encoding[:, :seq_len, :]

            # 添加CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Transformer
            x = self.transformer(x, src_key_padding_mask=mask)

            # CLS token输出
            cls_output = x[:, 0, :]

            # 分类
            state_logits = self.classifier(cls_output)
            anomaly_score = self.anomaly_detector(cls_output).squeeze(-1)

            return state_logits, anomaly_score


# =============================================================================
# 时序分析器
# =============================================================================

class TemporalAnalyzer:
    """
    时序分析器

    功能:
    - 多帧状态融合
    - 状态转换检测
    - 异常操作识别
    - SCADA命令验证
    """

    STATE_TO_IDX = {
        SwitchState.UNKNOWN: 0,
        SwitchState.OPEN: 1,
        SwitchState.CLOSED: 2,
        SwitchState.INTERMEDIATE: 3,
    }

    IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

    def __init__(self, config: TemporalAnalyzerConfig):
        self.config = config

        # 观测缓冲区
        self._observations: deque = deque(maxlen=config.window_size)

        # 状态历史
        self._state_history: deque = deque(maxlen=100)
        self._last_stable_state: Optional[SwitchState] = None
        self._state_start_time: float = 0.0

        # SCADA命令
        self._pending_command: Optional[SCADACommand] = None
        self._command_history: deque = deque(maxlen=50)

        # 深度学习模型
        self._model = None
        self._model_type = "lstm"  # "lstm" or "transformer"
        if config.use_lstm and TORCH_AVAILABLE:
            self._init_model()

    def _init_model(self):
        """初始化深度学习模型"""
        try:
            if self._model_type == "lstm":
                self._model = SwitchStateLSTM(
                    input_size=5,  # angle + confidence + 3 state onehot
                    hidden_size=self.config.lstm_hidden_size,
                    num_layers=self.config.lstm_num_layers,
                )
            else:
                self._model = SwitchStateTransformer(
                    input_size=5,
                    d_model=self.config.lstm_hidden_size,
                )

            # 加载预训练权重
            if self.config.model_path:
                state_dict = torch.load(self.config.model_path, map_location="cpu")
                self._model.load_state_dict(state_dict)

            self._model.eval()
            logger.info(f"[TemporalAnalyzer] {self._model_type.upper()}模型初始化成功")

        except Exception as e:
            logger.warning(f"[TemporalAnalyzer] 模型初始化失败: {e}")
            self._model = None

    def add_observation(self, obs: FrameObservation):
        """添加观测"""
        self._observations.append(obs)

    def add_scada_command(self, command: SCADACommand):
        """添加SCADA命令"""
        self._pending_command = command
        self._command_history.append(command)

    def analyze(self) -> TemporalDecision:
        """
        分析时序上下文并做出决策

        Returns:
            时序决策结果
        """
        if len(self._observations) < self.config.min_observations:
            return TemporalDecision(
                final_state=SwitchState.UNKNOWN,
                confidence=0.0,
                transition_type=TransitionType.NONE,
                anomaly_type=AnomalyType.NONE,
            )

        observations = list(self._observations)
        context = TemporalContext(
            observations=observations,
            window_size=self.config.window_size,
            fps=self.config.fps,
        )

        # 1. 基于规则的分析
        rule_result = self._analyze_by_rules(context)

        # 2. 基于深度学习的分析 (如果可用)
        dl_result = None
        if self._model is not None:
            dl_result = self._analyze_by_model(context)

        # 3. 融合决策
        final_result = self._fuse_decisions(rule_result, dl_result)

        # 4. SCADA验证
        final_result = self._validate_with_scada(final_result)

        # 5. 更新状态历史
        self._update_history(final_result)

        return final_result

    def _analyze_by_rules(self, context: TemporalContext) -> TemporalDecision:
        """基于规则的分析"""
        evidences = []

        # 1. 状态投票
        state_votes = self._vote_states(context)
        voted_state, vote_confidence = self._get_voted_state(state_votes)
        evidences.append({
            "type": "vote",
            "state": voted_state.value,
            "confidence": vote_confidence,
            "votes": {k.value: v for k, v in state_votes.items()},
        })

        # 2. 角度分析
        angle_state, angle_confidence = self._analyze_angles(context)
        if angle_state != SwitchState.UNKNOWN:
            evidences.append({
                "type": "angle",
                "state": angle_state.value,
                "confidence": angle_confidence,
            })

        # 3. 转换检测
        transition_type, transition_progress = self._detect_transition(context)
        if transition_type != TransitionType.NONE:
            evidences.append({
                "type": "transition",
                "transition": transition_type.value,
                "progress": transition_progress,
            })

        # 4. 异常检测
        anomaly_type, anomaly_score = self._detect_anomaly(context)
        if anomaly_type != AnomalyType.NONE:
            evidences.append({
                "type": "anomaly",
                "anomaly": anomaly_type.value,
                "score": anomaly_score,
            })

        # 5. 综合决策
        final_state = voted_state
        if angle_state != SwitchState.UNKNOWN and angle_confidence > vote_confidence:
            final_state = angle_state

        # 计算综合置信度
        confidence = vote_confidence
        if angle_confidence > 0:
            confidence = 0.6 * vote_confidence + 0.4 * angle_confidence

        # 如果正在转换，状态为TRANSITIONING
        if transition_type in [TransitionType.OPENING, TransitionType.CLOSING]:
            final_state = SwitchState.TRANSITIONING

        return TemporalDecision(
            final_state=final_state,
            confidence=confidence,
            transition_type=transition_type,
            anomaly_type=anomaly_type,
            anomaly_score=anomaly_score,
            transition_progress=transition_progress,
            evidences=evidences,
        )

    def _vote_states(self, context: TemporalContext) -> Dict[SwitchState, float]:
        """状态投票"""
        votes = {state: 0.0 for state in SwitchState}

        for obs in context.observations:
            # 加权投票 (越新的观测权重越高)
            weight = obs.confidence
            votes[obs.state] += weight

        # 归一化
        total = sum(votes.values())
        if total > 0:
            votes = {k: v / total for k, v in votes.items()}

        return votes

    def _get_voted_state(
        self,
        votes: Dict[SwitchState, float]
    ) -> Tuple[SwitchState, float]:
        """获取投票结果"""
        # 排除UNKNOWN
        filtered_votes = {k: v for k, v in votes.items() if k != SwitchState.UNKNOWN}

        if not filtered_votes:
            return SwitchState.UNKNOWN, 0.0

        best_state = max(filtered_votes, key=filtered_votes.get)
        confidence = filtered_votes[best_state]

        if confidence < self.config.vote_threshold:
            return SwitchState.UNKNOWN, confidence

        return best_state, confidence

    def _analyze_angles(
        self,
        context: TemporalContext
    ) -> Tuple[SwitchState, float]:
        """分析角度序列"""
        angles = context.angle_sequence
        if angles is None or len(angles) == 0:
            return SwitchState.UNKNOWN, 0.0

        # 使用最近的角度
        recent_angles = angles[-min(10, len(angles)):]
        mean_angle = np.mean(recent_angles)
        std_angle = np.std(recent_angles)

        # 判断状态
        angle_open = self.config.angle_open
        angle_closed = self.config.angle_closed
        tolerance = self.config.angle_tolerance

        if abs(mean_angle - angle_open) < tolerance and std_angle < tolerance / 2:
            confidence = 1.0 - abs(mean_angle - angle_open) / tolerance
            return SwitchState.OPEN, confidence

        if abs(mean_angle - angle_closed) < tolerance and std_angle < tolerance / 2:
            confidence = 1.0 - abs(mean_angle - angle_closed) / tolerance
            return SwitchState.CLOSED, confidence

        # 中间状态
        if std_angle < tolerance:
            return SwitchState.INTERMEDIATE, 0.5

        return SwitchState.UNKNOWN, 0.0

    def _detect_transition(
        self,
        context: TemporalContext
    ) -> Tuple[TransitionType, float]:
        """检测状态转换"""
        if len(context.observations) < 5:
            return TransitionType.NONE, 0.0

        states = context.state_sequence
        angles = context.angle_sequence

        # 检查状态变化
        unique_states = set(states)
        unique_states.discard(SwitchState.UNKNOWN)

        if len(unique_states) <= 1:
            return TransitionType.NONE, 0.0

        # 分析角度趋势
        if angles is not None and len(angles) >= 5:
            # 线性拟合
            x = np.arange(len(angles))
            coeffs = np.polyfit(x, angles, 1)
            slope = coeffs[0]

            # 正在分闸 (角度减小)
            if slope < -1.0:
                progress = self._estimate_transition_progress(
                    angles, self.config.angle_closed, self.config.angle_open
                )
                return TransitionType.OPENING, progress

            # 正在合闸 (角度增加)
            if slope > 1.0:
                progress = self._estimate_transition_progress(
                    angles, self.config.angle_open, self.config.angle_closed
                )
                return TransitionType.CLOSING, progress

        return TransitionType.NONE, 0.0

    def _estimate_transition_progress(
        self,
        angles: np.ndarray,
        start_angle: float,
        end_angle: float
    ) -> float:
        """估计转换进度"""
        current_angle = angles[-1]
        total_range = abs(end_angle - start_angle)

        if total_range == 0:
            return 0.0

        progress = abs(current_angle - start_angle) / total_range
        return max(0.0, min(1.0, progress))

    def _detect_anomaly(
        self,
        context: TemporalContext
    ) -> Tuple[AnomalyType, float]:
        """检测异常"""
        states = context.state_sequence
        duration = context.duration

        # 1. 状态振荡检测
        state_changes = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
        if state_changes >= self.config.oscillation_threshold:
            return AnomalyType.STATE_OSCILLATION, state_changes / len(states)

        # 2. 操作时间异常
        if self._pending_command and duration > 0:
            if duration > self.config.transition_time_max:
                return AnomalyType.SLOW_OPERATION, duration / self.config.transition_time_max
            if duration < self.config.transition_time_min:
                return AnomalyType.FAST_OPERATION, 1.0 - duration / self.config.transition_time_min

        return AnomalyType.NONE, 0.0

    def _analyze_by_model(self, context: TemporalContext) -> Optional[TemporalDecision]:
        """基于深度学习模型的分析"""
        if not TORCH_AVAILABLE or self._model is None:
            return None

        try:
            # 准备输入
            features = self._prepare_model_input(context)
            with torch.no_grad():
                state_logits, anomaly_score = self._model(features)

            # 解析输出
            probs = torch.softmax(state_logits, dim=-1).squeeze().numpy()
            state_idx = int(np.argmax(probs))
            confidence = float(probs[state_idx])
            anomaly = float(anomaly_score.item())

            state = self.IDX_TO_STATE.get(state_idx, SwitchState.UNKNOWN)

            return TemporalDecision(
                final_state=state,
                confidence=confidence,
                transition_type=TransitionType.NONE,
                anomaly_type=AnomalyType.ABNORMAL if anomaly > 0.5 else AnomalyType.NONE,
                anomaly_score=anomaly,
                metadata={"source": "model", "probs": probs.tolist()},
            )

        except Exception as e:
            logger.warning(f"[TemporalAnalyzer] 模型推理失败: {e}")
            return None

    def _prepare_model_input(self, context: TemporalContext) -> torch.Tensor:
        """准备模型输入"""
        features = []

        for obs in context.observations:
            # 角度 (归一化)
            angle = obs.angle if obs.angle is not None else 0.0
            angle_norm = (angle + 90) / 180  # [-90, 90] -> [0, 1]

            # 置信度
            conf = obs.confidence

            # 状态one-hot
            state_onehot = [0.0, 0.0, 0.0]
            if obs.state == SwitchState.OPEN:
                state_onehot[0] = 1.0
            elif obs.state == SwitchState.CLOSED:
                state_onehot[1] = 1.0
            elif obs.state == SwitchState.INTERMEDIATE:
                state_onehot[2] = 1.0

            features.append([angle_norm, conf] + state_onehot)

        features = np.array(features, dtype=np.float32)
        return torch.from_numpy(features).unsqueeze(0)

    def _fuse_decisions(
        self,
        rule_result: TemporalDecision,
        dl_result: Optional[TemporalDecision]
    ) -> TemporalDecision:
        """融合规则和深度学习决策"""
        if dl_result is None:
            return rule_result

        # 加权融合
        rule_weight = 0.4
        dl_weight = 0.6

        # 如果两者一致，提高置信度
        if rule_result.final_state == dl_result.final_state:
            confidence = max(rule_result.confidence, dl_result.confidence)
            confidence = min(1.0, confidence + 0.1)
        else:
            # 选择置信度更高的
            if dl_result.confidence * dl_weight > rule_result.confidence * rule_weight:
                return TemporalDecision(
                    final_state=dl_result.final_state,
                    confidence=dl_result.confidence,
                    transition_type=rule_result.transition_type,
                    anomaly_type=dl_result.anomaly_type if dl_result.anomaly_score > rule_result.anomaly_score else rule_result.anomaly_type,
                    anomaly_score=max(dl_result.anomaly_score, rule_result.anomaly_score),
                    transition_progress=rule_result.transition_progress,
                    evidences=rule_result.evidences + [{"source": "model", "state": dl_result.final_state.value}],
                )
            confidence = rule_result.confidence

        return TemporalDecision(
            final_state=rule_result.final_state,
            confidence=confidence,
            transition_type=rule_result.transition_type,
            anomaly_type=rule_result.anomaly_type,
            anomaly_score=max(rule_result.anomaly_score, dl_result.anomaly_score if dl_result else 0),
            transition_progress=rule_result.transition_progress,
            evidences=rule_result.evidences,
        )

    def _validate_with_scada(self, result: TemporalDecision) -> TemporalDecision:
        """与SCADA命令交叉验证"""
        if self._pending_command is None:
            return result

        command = self._pending_command
        expected_state = SwitchState.OPEN if command.command == "open" else SwitchState.CLOSED

        # 检查是否符合预期
        if result.final_state == expected_state:
            # 命令执行完成
            self._pending_command = None
            result.metadata["scada_validated"] = True
        elif result.transition_type != TransitionType.NONE:
            # 正在执行
            result.metadata["scada_command"] = command.command
        else:
            # 可能异常
            result.anomaly_type = AnomalyType.UNEXPECTED_TRANSITION
            result.metadata["scada_mismatch"] = True

        return result

    def _update_history(self, result: TemporalDecision):
        """更新状态历史"""
        current_time = time.time()

        if result.final_state != self._last_stable_state:
            self._last_stable_state = result.final_state
            self._state_start_time = current_time

        result.state_duration = current_time - self._state_start_time
        self._state_history.append((current_time, result))

    def reset(self):
        """重置分析器"""
        self._observations.clear()
        self._state_history.clear()
        self._last_stable_state = None
        self._pending_command = None

    def get_state_history(self, duration: float = 60.0) -> List[Tuple[float, SwitchState]]:
        """获取状态历史"""
        current_time = time.time()
        cutoff = current_time - duration

        history = []
        for ts, result in self._state_history:
            if ts >= cutoff:
                history.append((ts, result.final_state))

        return history


# =============================================================================
# 便捷函数
# =============================================================================

def create_analyzer(config: Optional[Dict] = None) -> TemporalAnalyzer:
    """创建时序分析器"""
    analyzer_config = TemporalAnalyzerConfig()
    if config:
        for key, value in config.items():
            if hasattr(analyzer_config, key):
                setattr(analyzer_config, key, value)
    return TemporalAnalyzer(analyzer_config)


def analyze_switch_sequence(
    observations: List[Dict[str, Any]],
    config: Optional[Dict] = None
) -> TemporalDecision:
    """
    分析开关状态序列

    Args:
        observations: 观测列表，每个包含 timestamp, state, confidence, angle
        config: 配置字典

    Returns:
        时序决策结果
    """
    analyzer = create_analyzer(config)

    for obs_dict in observations:
        state_str = obs_dict.get("state", "unknown")
        state = SwitchState(state_str) if state_str in [s.value for s in SwitchState] else SwitchState.UNKNOWN

        obs = FrameObservation(
            timestamp=obs_dict.get("timestamp", time.time()),
            state=state,
            confidence=obs_dict.get("confidence", 0.5),
            angle=obs_dict.get("angle"),
        )
        analyzer.add_observation(obs)

    return analyzer.analyze()
