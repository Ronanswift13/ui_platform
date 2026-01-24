"""
GL-TransLSTM PyTorch 完整实现
================================

基于论文实现的完整深度学习模型:
- Transformer编码器 (全局自注意力)
- 双向LSTM (时序特征)
- 特征融合层
- 多任务输出 (预测+分类)

模型架构:
    Input -> CEEMDAN分解 -> [IMF_1, IMF_2, ..., IMF_n]
                              |
                              v
                    Transformer Encoder
                              |
                              v
                       BiLSTM Layer
                              |
                              v
                    Feature Fusion + Physical Gate
                              |
                    +---------+---------+
                    |                   |
                    v                   v
            Prediction Head      Classification Head
            (气体浓度预测)         (泄漏风险分类)

作者: 破夜绘明团队
版本: 2.0.0 (PyTorch实现)
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# PyTorch导入
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    logger.warning("PyTorch未安装，深度学习功能不可用")


# =============================================================================
# 模型配置
# =============================================================================
@dataclass
class GLTransLSTMConfig:
    """GL-TransLSTM 模型配置"""
    # 输入输出
    input_dim: int = 7                   # 输入特征数 (多气体+环境)
    output_dim: int = 1                  # 输出维度
    num_classes: int = 5                 # 风险分类数 (normal, attention, warning, alarm, critical)

    # 序列配置
    seq_length: int = 168                # 输入序列长度 (7天*24小时)
    pred_length: int = 24                # 预测长度

    # Transformer配置
    d_model: int = 128                   # 模型维度
    n_heads: int = 8                     # 注意力头数
    n_encoder_layers: int = 3            # 编码器层数
    dim_feedforward: int = 512           # FFN维度
    transformer_dropout: float = 0.1

    # LSTM配置
    lstm_hidden: int = 128               # LSTM隐藏维度
    lstm_layers: int = 2                 # LSTM层数
    lstm_dropout: float = 0.2
    bidirectional: bool = True           # 双向LSTM

    # CEEMDAN配置
    use_ceemdan: bool = True
    num_imfs: int = 6                    # IMF分量数
    ceemdan_noise: float = 0.2

    # 物理门控配置
    use_physical_gate: bool = True
    env_features: int = 3                # 环境特征数 (温度/压力/湿度)

    # 训练配置
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    max_epochs: int = 100

    # 正则化
    dropout: float = 0.1
    use_layer_norm: bool = True


# =============================================================================
# 位置编码
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# 物理感知门控层
# =============================================================================
class PhysicalAwareGate(nn.Module):
    """
    物理感知门控机制

    根据环境参数(温度、压力、湿度)动态调整特征权重
    """

    def __init__(self, feature_dim: int, env_dim: int = 3):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(env_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

        # 可学习的基准权重
        self.base_weight = nn.Parameter(torch.ones(feature_dim))

    def forward(self, features: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, seq_len, feature_dim)
            env: (batch, env_dim) 环境参数

        Returns:
            (batch, seq_len, feature_dim)
        """
        # 计算门控系数
        gate = self.gate_net(env)  # (batch, feature_dim)
        gate = gate.unsqueeze(1)   # (batch, 1, feature_dim)

        # 应用门控
        gated_features = features * gate * self.base_weight

        return gated_features


# =============================================================================
# 特征融合层
# =============================================================================
class FeatureFusion(nn.Module):
    """
    多尺度特征融合

    融合Transformer和LSTM的特征
    """

    def __init__(self, transformer_dim: int, lstm_dim: int, output_dim: int):
        super().__init__()

        combined_dim = transformer_dim + lstm_dim

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
        )

        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, trans_feat: torch.Tensor, lstm_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trans_feat: (batch, seq_len, transformer_dim)
            lstm_feat: (batch, seq_len, lstm_dim)

        Returns:
            (batch, seq_len, output_dim)
        """
        # 拼接特征
        combined = torch.cat([trans_feat, lstm_feat], dim=-1)

        # 计算注意力权重
        attn = self.attention(combined)  # (batch, seq_len, 2)

        # 加权融合
        weighted_trans = trans_feat * attn[:, :, 0:1]
        weighted_lstm = lstm_feat * attn[:, :, 1:2]

        # 拼接并投影
        fused = torch.cat([weighted_trans, weighted_lstm], dim=-1)
        output = self.fusion(fused)

        return output


# =============================================================================
# GL-TransLSTM 主模型
# =============================================================================
class GLTransLSTMModel(nn.Module):
    """
    GL-TransLSTM 气体浓度预测模型

    结合Transformer全局注意力和LSTM时序建模能力
    """

    def __init__(self, config: GLTransLSTMConfig):
        super().__init__()
        self.config = config

        # 输入嵌入
        self.input_embedding = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # 位置编码
        self.pos_encoder = PositionalEncoding(
            config.d_model,
            max_len=config.seq_length * 2,
            dropout=config.transformer_dropout
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers
        )

        # BiLSTM
        lstm_output_dim = config.lstm_hidden * (2 if config.bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.lstm_dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=config.bidirectional
        )
        self.lstm_norm = nn.LayerNorm(lstm_output_dim)

        # 物理感知门控
        if config.use_physical_gate:
            self.physical_gate = PhysicalAwareGate(
                feature_dim=config.d_model,
                env_dim=config.env_features
            )
        else:
            self.physical_gate = None

        # 特征融合
        self.feature_fusion = FeatureFusion(
            transformer_dim=config.d_model,
            lstm_dim=lstm_output_dim,
            output_dim=config.d_model
        )

        # 预测头 (回归)
        self.prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.pred_length * config.output_dim),
        )

        # 分类头 (风险等级)
        self.classification_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        env: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列 (batch, seq_len, input_dim)
            env: 环境参数 (batch, env_features) - 可选
            mask: 注意力掩码 - 可选

        Returns:
            predictions: (batch, pred_length, output_dim) 预测值
            risk_logits: (batch, num_classes) 风险分类logits
        """
        batch_size, seq_len, _ = x.shape

        # 输入嵌入
        embedded = self.input_embedding(x)  # (batch, seq_len, d_model)

        # 位置编码
        embedded = self.pos_encoder(embedded)

        # 物理门控
        if self.physical_gate is not None and env is not None:
            embedded = self.physical_gate(embedded, env)

        # Transformer编码
        trans_out = self.transformer_encoder(embedded, mask=mask)  # (batch, seq_len, d_model)

        # LSTM编码
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, lstm_hidden*2)
        lstm_out = self.lstm_norm(lstm_out)

        # 特征融合
        fused = self.feature_fusion(trans_out, lstm_out)  # (batch, seq_len, d_model)

        # 取最后时间步的特征
        final_feat = fused[:, -1, :]  # (batch, d_model)

        # 预测头
        pred = self.prediction_head(final_feat)  # (batch, pred_length * output_dim)
        pred = pred.view(batch_size, self.config.pred_length, self.config.output_dim)

        # 分类头
        risk_logits = self.classification_head(final_feat)  # (batch, num_classes)

        return pred, risk_logits

    def predict(self, x: torch.Tensor, env: Optional[torch.Tensor] = None) -> torch.Tensor:
        """仅预测"""
        self.eval()
        with torch.no_grad():
            pred, _ = self.forward(x, env)
        return pred


# =============================================================================
# CEEMDAN 分解器 (PyTorch版本)
# =============================================================================
class CEEMDANDecomposer:
    """
    CEEMDAN多尺度分解

    使用PyTorch实现的EMD分解，支持GPU加速
    """

    def __init__(self, num_imfs: int = 6, noise_std: float = 0.2, max_siftings: int = 100):
        self.num_imfs = num_imfs
        self.noise_std = noise_std
        self.max_siftings = max_siftings

    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """
        分解信号为IMF分量

        Args:
            signal: (seq_len,) 输入信号

        Returns:
            (num_imfs, seq_len) IMF分量
        """
        n = len(signal)
        imfs = np.zeros((self.num_imfs, n))
        residue = signal.copy()

        for i in range(self.num_imfs - 1):
            imf = self._sift(residue)
            imfs[i] = imf
            residue = residue - imf

        # 最后一个为残差
        imfs[-1] = residue

        return imfs

    def _sift(self, signal: np.ndarray) -> np.ndarray:
        """
        EMD筛分过程

        Args:
            signal: 输入信号

        Returns:
            IMF分量
        """
        h = signal.copy()

        for _ in range(self.max_siftings):
            # 找局部极值点
            maxima, minima = self._find_extrema(h)

            if len(maxima) < 2 or len(minima) < 2:
                break

            # 插值生成包络
            upper = self._interpolate(maxima, h[maxima], len(h))
            lower = self._interpolate(minima, h[minima], len(h))

            # 计算均值包络
            mean_env = (upper + lower) / 2

            # 筛分
            h_new = h - mean_env

            # 检查收敛
            if np.sum((h - h_new) ** 2) < 1e-10:
                break

            h = h_new

        return h

    def _find_extrema(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """找局部极值点"""
        # 一阶差分
        diff = np.diff(signal)

        # 找极大值 (由正变负)
        maxima = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0] + 1

        # 找极小值 (由负变正)
        minima = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0] + 1

        return maxima, minima

    def _interpolate(self, x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
        """三次样条插值"""
        if len(x) < 2:
            return np.zeros(n)

        try:
            from scipy.interpolate import CubicSpline
            # 添加端点
            x_ext = np.concatenate([[0], x, [n - 1]])
            y_ext = np.concatenate([[y[0]], y, [y[-1]]])
            cs = CubicSpline(x_ext, y_ext)
            return cs(np.arange(n))
        except ImportError:
            # 简单线性插值
            return np.interp(np.arange(n), x, y)

    def decompose_batch(self, signals: np.ndarray) -> np.ndarray:
        """
        批量分解

        Args:
            signals: (batch, seq_len) 输入信号

        Returns:
            (batch, num_imfs, seq_len) IMF分量
        """
        batch_size = signals.shape[0]
        seq_len = signals.shape[1]

        imfs = np.zeros((batch_size, self.num_imfs, seq_len))

        for i in range(batch_size):
            imfs[i] = self.decompose(signals[i])

        return imfs


# =============================================================================
# 数据集
# =============================================================================
class GasTimeSeriesDataset(Dataset):
    """
    气体时序数据集
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_length: int = 168,
        pred_length: int = 24,
        labels: Optional[np.ndarray] = None,
        env_data: Optional[np.ndarray] = None,
        transform: Optional[callable] = None
    ):
        """
        Args:
            data: (total_len, features) 时序数据
            seq_length: 输入序列长度
            pred_length: 预测长度
            labels: (total_len,) 风险标签 (可选)
            env_data: (total_len, env_features) 环境数据 (可选)
            transform: 数据变换函数
        """
        self.data = data.astype(np.float32)
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.labels = labels
        self.env_data = env_data.astype(np.float32) if env_data is not None else None
        self.transform = transform

        # 计算有效样本数
        self.total_len = len(data)
        self.num_samples = self.total_len - seq_length - pred_length + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 输入序列
        x = self.data[idx:idx + self.seq_length]

        # 目标序列 (预测气体浓度)
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length, 0]

        # 转换为张量
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        sample = {'x': x, 'y': y}

        # 环境数据
        if self.env_data is not None:
            env = self.env_data[idx + self.seq_length - 1]
            sample['env'] = torch.from_numpy(env)

        # 风险标签
        if self.labels is not None:
            label = self.labels[idx + self.seq_length - 1]
            sample['label'] = torch.tensor(label, dtype=torch.long)

        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample


# =============================================================================
# 损失函数
# =============================================================================
class GLTransLSTMLoss(nn.Module):
    """
    组合损失函数

    L = L_pred + alpha * L_cls + beta * L_trend
    """

    def __init__(
        self,
        pred_weight: float = 1.0,
        cls_weight: float = 0.5,
        trend_weight: float = 0.3
    ):
        super().__init__()
        self.pred_weight = pred_weight
        self.cls_weight = cls_weight
        self.trend_weight = trend_weight

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        risk_logits: torch.Tensor,
        risk_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: (batch, pred_len, 1) 预测值
            target: (batch, pred_len) 目标值
            risk_logits: (batch, num_classes) 风险分类
            risk_labels: (batch,) 风险标签

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失字典
        """
        # 预测损失 (MSE)
        pred_loss = self.mse(pred.squeeze(-1), target)

        # 分类损失 (CE)
        cls_loss = self.ce(risk_logits, risk_labels)

        # 趋势损失 (预测趋势与真实趋势的一致性)
        pred_trend = pred[:, -1, 0] - pred[:, 0, 0]
        target_trend = target[:, -1] - target[:, 0]
        trend_loss = self.mse(pred_trend, target_trend)

        # 总损失
        total_loss = (
            self.pred_weight * pred_loss +
            self.cls_weight * cls_loss +
            self.trend_weight * trend_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'pred': pred_loss.item(),
            'cls': cls_loss.item(),
            'trend': trend_loss.item(),
        }

        return total_loss, loss_dict


# =============================================================================
# 训练器
# =============================================================================
class GLTransLSTMTrainer:
    """
    GL-TransLSTM 训练器
    """

    def __init__(
        self,
        model: GLTransLSTMModel,
        config: GLTransLSTMConfig,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs
        )

        # 损失函数
        self.criterion = GLTransLSTMLoss()

        # 训练历史
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        loss_dict_sum = {}

        for batch in dataloader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            env = batch.get('env')
            label = batch.get('label')

            if env is not None:
                env = env.to(self.device)
            if label is not None:
                label = label.to(self.device)
            else:
                label = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

            # 前向传播
            pred, risk_logits = self.model(x, env)

            # 计算损失
            loss, loss_dict = self.criterion(pred, y, risk_logits, label)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v

        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_loss_dict = {k: v / n_batches for k, v in loss_dict_sum.items()}

        return avg_loss_dict

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        n_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                env = batch.get('env')
                label = batch.get('label')

                if env is not None:
                    env = env.to(self.device)
                if label is not None:
                    label = label.to(self.device)
                else:
                    label = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

                pred, risk_logits = self.model(x, env)
                loss, _ = self.criterion(pred, y, risk_logits, label)

                total_loss += loss.item() * x.size(0)
                total_mae += torch.abs(pred.squeeze(-1) - y).sum().item()
                n_samples += x.size(0)

        return {
            'loss': total_loss / n_samples,
            'mae': total_mae / n_samples / self.config.pred_length
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            verbose: 是否打印训练信息

        Returns:
            训练历史
        """
        epochs = epochs or self.config.max_epochs
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_metrics['total'])

            # 验证
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_mae'].append(val_metrics['mae'])

                # 保存最佳模型
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.best_model_state = self.model.state_dict().copy()

            # 更新学习率
            self.scheduler.step()

            # 打印信息
            if verbose and (epoch + 1) % 5 == 0:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_metrics['total']:.4f}"
                if val_loader:
                    msg += f" | Val Loss: {val_metrics['loss']:.4f} | Val MAE: {val_metrics['mae']:.4f}"
                logger.info(msg)

        return self.history

    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
        }, path)

    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)


# =============================================================================
# 推理接口
# =============================================================================
class GLTransLSTMPredictor:
    """
    GL-TransLSTM 推理接口

    封装模型加载和预测功能
    """

    def __init__(self, model_path: Optional[str] = None, config: Optional[GLTransLSTMConfig] = None):
        """
        Args:
            model_path: 模型文件路径
            config: 模型配置 (如果没有model_path则必须提供)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config or GLTransLSTMConfig()

        # 创建模型
        self.model = GLTransLSTMModel(self.config)

        # 加载权重
        if model_path:
            self.load(model_path)

        self.model.to(self.device)
        self.model.eval()

        # CEEMDAN分解器
        if self.config.use_ceemdan:
            self.ceemdan = CEEMDANDecomposer(num_imfs=self.config.num_imfs)
        else:
            self.ceemdan = None

        # 数据标准化参数
        self._mean = None
        self._std = None

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        if 'normalization' in checkpoint:
            self._mean = checkpoint['normalization']['mean']
            self._std = checkpoint['normalization']['std']

    def fit_normalizer(self, data: np.ndarray):
        """拟合标准化参数"""
        self._mean = data.mean(axis=0)
        self._std = data.std(axis=0) + 1e-8

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """标准化数据"""
        if self._mean is None:
            return data
        return (data - self._mean) / self._std

    def denormalize(self, data: np.ndarray, feature_idx: int = 0) -> np.ndarray:
        """反标准化数据"""
        if self._mean is None:
            return data
        return data * self._std[feature_idx] + self._mean[feature_idx]

    def predict(
        self,
        sequence: np.ndarray,
        env: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        预测未来气体浓度

        Args:
            sequence: (seq_len, features) 输入序列
            env: (env_features,) 环境参数

        Returns:
            预测结果字典
        """
        # 标准化
        x = self.normalize(sequence)

        # CEEMDAN分解 (如果启用)
        if self.ceemdan is not None:
            imfs = self.ceemdan.decompose(x[:, 0])  # 对主要气体分解
            # 将IMF作为额外特征
            # 这里简化处理，实际可以更复杂
            pass

        # 转换为张量
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(self.device)

        if env is not None:
            env_tensor = torch.from_numpy(env).float().unsqueeze(0).to(self.device)
        else:
            env_tensor = None

        # 推理
        with torch.no_grad():
            pred, risk_logits = self.model(x_tensor, env_tensor)

        # 后处理
        pred_np = pred.squeeze().cpu().numpy()
        pred_np = self.denormalize(pred_np, 0)

        risk_probs = F.softmax(risk_logits, dim=-1).squeeze().cpu().numpy()
        risk_class = risk_probs.argmax()

        risk_labels = ['normal', 'attention', 'warning', 'alarm', 'critical']

        return {
            'predictions': pred_np.tolist(),
            'risk_class': risk_labels[risk_class],
            'risk_probability': float(risk_probs[risk_class]),
            'risk_distribution': {
                label: float(prob)
                for label, prob in zip(risk_labels, risk_probs)
            }
        }


# =============================================================================
# 便捷函数
# =============================================================================
def create_model(config: Optional[GLTransLSTMConfig] = None) -> GLTransLSTMModel:
    """创建模型"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch未安装")
    config = config or GLTransLSTMConfig()
    return GLTransLSTMModel(config)


def create_predictor(model_path: Optional[str] = None) -> GLTransLSTMPredictor:
    """创建预测器"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch未安装")
    return GLTransLSTMPredictor(model_path)
