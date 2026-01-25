#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GL-TransLSTM 气体浓度预测模型 - 完整PyTorch实现
================================================

基于下一步迭代方案实现:
- Transformer全局自注意力机制
- BiLSTM门控记忆单元
- CEEMDAN多尺度分解
- 物理感知门控注意机制
- 训练器和预测器完整接口

功能:
1. 多气体浓度时序预测
2. 泄漏检测与预警
3. 概率预测 (不确定性量化)
4. 在线学习支持

版本: 2.0.0
作者: 变电站AI巡检系统
"""

from __future__ import annotations
import logging
import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    logger.warning("PyTorch不可用，部分功能将受限")


# =============================================================================
# 枚举和数据结构
# =============================================================================

class GasType(Enum):
    """气体类型"""
    SF6 = "SF6"
    H2 = "H2"
    CO = "CO"
    C2H2 = "C2H2"
    CH4 = "CH4"
    C2H4 = "C2H4"
    C2H6 = "C2H6"
    O2 = "O2"
    CO2 = "CO2"


class PredictionHorizon(Enum):
    """预测时间范围"""
    HOUR_1 = "1h"
    HOUR_6 = "6h"
    HOUR_24 = "24h"
    DAY_7 = "7d"


@dataclass
class GLTransLSTMConfig:
    """GL-TransLSTM 模型配置"""
    # 数据配置
    input_dim: int = 8                    # 输入特征维度 (多气体+环境)
    output_dim: int = 4                   # 输出维度 (预测的气体数)
    sequence_length: int = 168            # 输入序列长度 (7天 * 24小时)
    prediction_horizon: int = 24          # 预测步数
    
    # Transformer配置
    d_model: int = 128                    # 模型维度
    nhead: int = 8                        # 注意力头数
    num_encoder_layers: int = 3           # 编码器层数
    num_decoder_layers: int = 2           # 解码器层数
    dim_feedforward: int = 512            # FFN维度
    transformer_dropout: float = 0.1
    
    # LSTM配置
    lstm_hidden_dim: int = 128            # LSTM隐藏维度
    lstm_num_layers: int = 2              # LSTM层数
    lstm_dropout: float = 0.1
    bidirectional: bool = True            # 双向LSTM
    
    # CEEMDAN配置
    use_ceemdan: bool = True
    num_imfs: int = 8                     # IMF分量数
    ceemdan_trials: int = 100
    ceemdan_noise_std: float = 0.2
    
    # 物理感知门控配置
    use_physical_gate: bool = True
    temp_weight: float = 0.3
    pressure_weight: float = 0.4
    humidity_weight: float = 0.3
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # 概率预测
    use_probabilistic: bool = True
    num_quantiles: int = 9                # 分位数数量
    
    # 告警阈值
    alarm_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "SF6": 1000.0,
        "H2": 150.0,
        "CO": 300.0,
        "C2H2": 5.0
    })


@dataclass
class GasPrediction:
    """气体预测结果"""
    gas_type: str
    current_value: float
    predictions: List[float]
    timestamps: List[float]
    confidence: float
    trend: str                            # "rising", "falling", "stable"
    leak_probability: float
    time_to_alarm: Optional[float] = None
    prediction_intervals: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakDetectionResult:
    """泄漏检测结果"""
    detected: bool
    gas_type: str
    leak_rate: float
    severity: str                         # "none", "low", "medium", "high", "critical"
    estimated_source: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0


# =============================================================================
# CEEMDAN 信号分解
# =============================================================================

class CEEMDANDecomposer:
    """
    CEEMDAN (Complete Ensemble EMD with Adaptive Noise) 信号分解器
    
    将时序信号分解为多个本征模态函数(IMF)，提取多尺度特征
    """
    
    def __init__(self, config: GLTransLSTMConfig):
        self.num_imfs = config.num_imfs
        self.trials = config.ceemdan_trials
        self.noise_std = config.ceemdan_noise_std
    
    def decompose(self, signal: np.ndarray) -> np.ndarray:
        """
        分解信号为IMF分量
        
        Args:
            signal: 输入信号 [seq_len]
            
        Returns:
            IMF分量 [num_imfs, seq_len]
        """
        try:
            from PyEMD import CEEMDAN
            
            ceemdan = CEEMDAN(
                trials=self.trials,
                epsilon=self.noise_std
            )
            
            imfs = ceemdan.ceemdan(signal)
            
            # 补齐或截断到固定数量的IMF
            if len(imfs) < self.num_imfs:
                padding = np.zeros((self.num_imfs - len(imfs), len(signal)))
                imfs = np.vstack([imfs, padding])
            else:
                imfs = imfs[:self.num_imfs]
            
            return imfs
            
        except ImportError:
            # 简化版本：使用傅里叶分解近似
            return self._fourier_decompose(signal)
    
    def _fourier_decompose(self, signal: np.ndarray) -> np.ndarray:
        """傅里叶分解近似"""
        n = len(signal)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n)
        
        imfs = np.zeros((self.num_imfs, n))
        
        # 将频谱分成多个频带
        freq_bands = np.linspace(0, 0.5, self.num_imfs + 1)
        
        for i in range(self.num_imfs):
            mask = (np.abs(freqs) >= freq_bands[i]) & (np.abs(freqs) < freq_bands[i + 1])
            filtered_fft = np.zeros_like(fft)
            filtered_fft[mask] = fft[mask]
            imfs[i] = np.real(np.fft.ifft(filtered_fft))
        
        return imfs
    
    def batch_decompose(self, signals: np.ndarray) -> np.ndarray:
        """
        批量分解信号
        
        Args:
            signals: 输入信号 [batch, seq_len] 或 [batch, features, seq_len]
            
        Returns:
            IMF分量
        """
        if signals.ndim == 2:
            batch_size, seq_len = signals.shape
            result = np.zeros((batch_size, self.num_imfs, seq_len))
            
            for i in range(batch_size):
                result[i] = self.decompose(signals[i])
            
            return result
        else:
            batch_size, features, seq_len = signals.shape
            result = np.zeros((batch_size, features, self.num_imfs, seq_len))
            
            for i in range(batch_size):
                for j in range(features):
                    result[i, j] = self.decompose(signals[i, j])
            
            return result


# =============================================================================
# 物理感知门控模块
# =============================================================================

if TORCH_AVAILABLE:
    
    class PhysicalAwareGate(nn.Module):
        """
        物理感知门控模块
        
        融合环境因素(温度、压力、湿度)调整预测权重
        """
        
        def __init__(self, config: GLTransLSTMConfig):
            super().__init__()
            
            self.temp_weight = config.temp_weight
            self.pressure_weight = config.pressure_weight
            self.humidity_weight = config.humidity_weight
            
            # 环境特征嵌入
            self.env_embed = nn.Sequential(
                nn.Linear(3, 32),  # 温度、压力、湿度
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.Tanh()
            )
            
            # 门控网络
            self.gate_net = nn.Sequential(
                nn.Linear(16 + config.d_model, config.d_model),
                nn.Sigmoid()
            )
        
        def forward(self, 
                    features: torch.Tensor, 
                    env_data: torch.Tensor) -> torch.Tensor:
            """
            Args:
                features: 模型特征 [batch, seq, d_model]
                env_data: 环境数据 [batch, 3] (温度, 压力, 湿度)
                
            Returns:
                门控后的特征
            """
            # 嵌入环境特征
            env_embed = self.env_embed(env_data)  # [batch, 16]
            
            # 扩展到序列长度
            env_embed = env_embed.unsqueeze(1).expand(-1, features.size(1), -1)
            
            # 计算门控
            gate_input = torch.cat([env_embed, features], dim=-1)
            gate = self.gate_net(gate_input)  # [batch, seq, d_model]
            
            return features * gate
    
    
    # =========================================================================
    # Transformer编码器
    # =========================================================================
    
    class PositionalEncoding(nn.Module):
        """位置编码"""
        
        def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
    
    
    class GLTransLSTMEncoder(nn.Module):
        """
        GL-TransLSTM 编码器
        
        融合Transformer和BiLSTM
        """
        
        def __init__(self, config: GLTransLSTMConfig):
            super().__init__()
            self.config = config
            
            # 输入投影
            self.input_projection = nn.Linear(config.input_dim, config.d_model)
            
            # CEEMDAN嵌入 (如果启用)
            if config.use_ceemdan:
                self.imf_projection = nn.Linear(config.num_imfs, config.d_model // 4)
            
            # 位置编码
            self.pos_encoder = PositionalEncoding(
                config.d_model, 
                config.sequence_length,
                config.transformer_dropout
            )
            
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.transformer_dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.num_encoder_layers
            )
            
            # BiLSTM
            self.lstm = nn.LSTM(
                input_size=config.d_model,
                hidden_size=config.lstm_hidden_dim,
                num_layers=config.lstm_num_layers,
                dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
                bidirectional=config.bidirectional,
                batch_first=True
            )
            
            # LSTM输出投影
            lstm_output_dim = config.lstm_hidden_dim * 2 if config.bidirectional else config.lstm_hidden_dim
            self.lstm_projection = nn.Linear(lstm_output_dim, config.d_model)
            
            # 融合门控
            self.fusion_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.Sigmoid()
            )
            
            # 物理感知门控
            if config.use_physical_gate:
                self.physical_gate = PhysicalAwareGate(config)
        
        def forward(self, 
                    x: torch.Tensor,
                    imf_features: Optional[torch.Tensor] = None,
                    env_data: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Args:
                x: 输入序列 [batch, seq_len, input_dim]
                imf_features: IMF特征 [batch, seq_len, num_imfs] (可选)
                env_data: 环境数据 [batch, 3] (可选)
                
            Returns:
                编码特征 [batch, seq_len, d_model]
            """
            batch_size, seq_len, _ = x.shape
            
            # 输入投影
            h = self.input_projection(x)  # [batch, seq, d_model]
            
            # 融合IMF特征
            if self.config.use_ceemdan and imf_features is not None:
                imf_embed = self.imf_projection(imf_features)  # [batch, seq, d_model//4]
                h = h + F.pad(imf_embed, (0, h.size(-1) - imf_embed.size(-1)))
            
            # 位置编码
            h = self.pos_encoder(h)
            
            # Transformer编码
            transformer_out = self.transformer_encoder(h)
            
            # BiLSTM编码
            lstm_out, _ = self.lstm(h)
            lstm_out = self.lstm_projection(lstm_out)
            
            # 融合Transformer和LSTM输出
            combined = torch.cat([transformer_out, lstm_out], dim=-1)
            gate = self.fusion_gate(combined)
            
            output = gate * transformer_out + (1 - gate) * lstm_out
            
            # 物理感知门控
            if self.config.use_physical_gate and env_data is not None:
                output = self.physical_gate(output, env_data)
            
            return output
    
    
    # =========================================================================
    # GL-TransLSTM 完整模型
    # =========================================================================
    
    class GLTransLSTMModel(nn.Module):
        """
        GL-TransLSTM 气体预测模型
        
        完整架构:
        1. CEEMDAN多尺度分解
        2. Transformer全局注意力
        3. BiLSTM时序建模
        4. 物理感知门控
        5. 概率预测头
        """
        
        def __init__(self, config: GLTransLSTMConfig):
            super().__init__()
            self.config = config
            
            # 编码器
            self.encoder = GLTransLSTMEncoder(config)
            
            # 解码器注意力
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.transformer_dropout,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=config.num_decoder_layers
            )
            
            # 预测头
            if config.use_probabilistic:
                # 分位数预测
                self.prediction_head = nn.Linear(
                    config.d_model, 
                    config.output_dim * config.num_quantiles
                )
            else:
                # 点预测
                self.prediction_head = nn.Linear(config.d_model, config.output_dim)
            
            # 目标嵌入
            self.target_embed = nn.Linear(config.output_dim, config.d_model)
            self.target_pos_encoder = PositionalEncoding(
                config.d_model,
                config.prediction_horizon
            )
        
        def forward(self,
                    src: torch.Tensor,
                    tgt: Optional[torch.Tensor] = None,
                    imf_features: Optional[torch.Tensor] = None,
                    env_data: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Args:
                src: 源序列 [batch, src_len, input_dim]
                tgt: 目标序列 [batch, tgt_len, output_dim] (训练时)
                imf_features: IMF特征
                env_data: 环境数据
                
            Returns:
                预测值 [batch, pred_len, output_dim] 或 
                       [batch, pred_len, output_dim * num_quantiles]
            """
            batch_size = src.size(0)
            
            # 编码
            memory = self.encoder(src, imf_features, env_data)
            
            if tgt is not None:
                # 训练模式：使用真实目标序列
                tgt_embed = self.target_embed(tgt)
                tgt_embed = self.target_pos_encoder(tgt_embed)
                
                # 生成因果掩码
                tgt_len = tgt.size(1)
                tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(src.device)
                
                # 解码
                output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            else:
                # 推理模式：自回归生成
                output = self._autoregressive_decode(memory, batch_size, src.device)
            
            # 预测
            predictions = self.prediction_head(output)
            
            if self.config.use_probabilistic:
                # 重塑为 [batch, pred_len, output_dim, num_quantiles]
                predictions = predictions.view(
                    batch_size, -1, self.config.output_dim, self.config.num_quantiles
                )
            
            return predictions
        
        def _autoregressive_decode(self, 
                                    memory: torch.Tensor,
                                    batch_size: int,
                                    device: torch.device) -> torch.Tensor:
            """自回归解码"""
            # 初始输入 (全零)
            tgt = torch.zeros(
                batch_size, 1, self.config.output_dim, 
                device=device
            )
            
            outputs = []
            
            for i in range(self.config.prediction_horizon):
                tgt_embed = self.target_embed(tgt)
                tgt_embed = self.target_pos_encoder(tgt_embed)
                
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(device)
                
                output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                
                # 取最后一步
                last_output = output[:, -1:, :]
                outputs.append(last_output)
                
                # 预测下一步
                if not self.config.use_probabilistic:
                    next_pred = self.prediction_head(last_output)
                else:
                    pred = self.prediction_head(last_output)
                    # 取中位数作为下一步输入
                    pred = pred.view(batch_size, 1, self.config.output_dim, -1)
                    next_pred = pred[:, :, :, self.config.num_quantiles // 2]
                
                # 更新输入
                tgt = torch.cat([tgt, next_pred], dim=1)
            
            return torch.cat(outputs, dim=1)
        
        def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
            """生成因果掩码"""
            mask = torch.triu(torch.ones(sz, sz), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            return mask
    
    
    # =========================================================================
    # 损失函数
    # =========================================================================
    
    class QuantileLoss(nn.Module):
        """分位数损失函数"""
        
        def __init__(self, quantiles: List[float]):
            super().__init__()
            self.quantiles = quantiles
        
        def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Args:
                predictions: [batch, seq, features, num_quantiles]
                targets: [batch, seq, features]
                
            Returns:
                损失值
            """
            losses = []
            
            for i, q in enumerate(self.quantiles):
                pred = predictions[:, :, :, i]
                error = targets - pred
                
                loss = torch.max(q * error, (q - 1) * error)
                losses.append(loss)
            
            return torch.mean(torch.stack(losses))


# =============================================================================
# 训练器
# =============================================================================

class GLTransLSTMTrainer:
    """GL-TransLSTM 训练器"""
    
    def __init__(self, config: GLTransLSTMConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch不可用，无法初始化训练器")
        
        # 创建模型
        self.model = GLTransLSTMModel(config).to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # 损失函数
        if config.use_probabilistic:
            quantiles = [i / (config.num_quantiles + 1) for i in range(1, config.num_quantiles + 1)]
            self.criterion = QuantileLoss(quantiles)
        else:
            self.criterion = nn.MSELoss()
        
        # CEEMDAN分解器
        if config.use_ceemdan:
            self.ceemdan = CEEMDANDecomposer(config)
        
        # 训练历史
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        logger.info(f"训练器初始化完成，设备: {self.device}")
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            
            # 可选的IMF特征和环境数据
            imf_features = batch.get("imf_features")
            env_data = batch.get("env_data")
            
            if imf_features is not None:
                imf_features = imf_features.to(self.device)
            if env_data is not None:
                env_data = env_data.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            predictions = self.model(
                src, 
                tgt[:, :-1, :],  # 教师强制
                imf_features, 
                env_data
            )
            
            # 计算损失
            if self.config.use_probabilistic:
                target = tgt[:, 1:, :]
                loss = self.criterion(predictions, target)
            else:
                loss = self.criterion(predictions, tgt[:, 1:, :])
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)
                
                imf_features = batch.get("imf_features")
                env_data = batch.get("env_data")
                
                if imf_features is not None:
                    imf_features = imf_features.to(self.device)
                if env_data is not None:
                    env_data = env_data.to(self.device)
                
                predictions = self.model(src, None, imf_features, env_data)
                
                if self.config.use_probabilistic:
                    # 使用中位数计算验证损失
                    median_pred = predictions[:, :, :, self.config.num_quantiles // 2]
                    loss = F.mse_loss(median_pred, tgt)
                else:
                    loss = self.criterion(predictions, tgt)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            
        Returns:
            训练历史
        """
        num_epochs = num_epochs or self.config.num_epochs
        best_val_loss = float('inf')
        
        logger.info(f"开始训练，共 {num_epochs} 轮")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            
            # 验证
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
            else:
                val_loss = None
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)
            
            # 日志
            elapsed = time.time() - start_time
            val_str = f", Val Loss: {val_loss:.6f}" if val_loss else ""
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f}{val_str} - "
                f"LR: {current_lr:.6f} - "
                f"Time: {elapsed:.1f}s"
            )
        
        logger.info("训练完成")
        return self.history
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "history": self.history
        }, path)
        logger.info(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info(f"检查点已加载: {path}")


# =============================================================================
# 预测器
# =============================================================================

class GLTransLSTMPredictor:
    """GL-TransLSTM 预测器"""
    
    def __init__(self, config: GLTransLSTMConfig, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch不可用，使用简化预测")
            self.model = None
        else:
            self.model = GLTransLSTMModel(config).to(self.device)
            self.model.eval()
            
            if model_path:
                self.load_model(model_path)
        
        # CEEMDAN分解器
        if config.use_ceemdan:
            self.ceemdan = CEEMDANDecomposer(config)
        
        # 历史数据缓冲
        self._history_buffer: Dict[str, deque] = {
            gas.value: deque(maxlen=config.sequence_length)
            for gas in GasType
        }
        self._env_buffer: deque = deque(maxlen=config.sequence_length)
        
        # 归一化参数
        self._normalization_params = {
            "SF6": {"mean": 900, "std": 100},
            "H2": {"mean": 80, "std": 30},
            "CO": {"mean": 150, "std": 50},
            "C2H2": {"mean": 3, "std": 2},
            "temperature": {"mean": 25, "std": 10},
            "humidity": {"mean": 50, "std": 15},
            "pressure": {"mean": 101.3, "std": 5}
        }
        
        logger.info("预测器初始化完成")
    
    def load_model(self, path: str):
        """加载模型"""
        if self.model is None:
            logger.warning("模型不可用")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info(f"模型已加载: {path}")
    
    def add_reading(self, 
                    gas_type: str, 
                    value: float,
                    timestamp: float = None,
                    environmental: Optional[Dict[str, float]] = None):
        """
        添加气体读数
        
        Args:
            gas_type: 气体类型
            value: 浓度值
            timestamp: 时间戳
            environmental: 环境数据 {"temperature", "humidity", "pressure"}
        """
        if gas_type in self._history_buffer:
            self._history_buffer[gas_type].append(value)
        
        if environmental:
            self._env_buffer.append(environmental)
    
    def predict(self, 
                gas_type: str,
                horizon: PredictionHorizon = PredictionHorizon.HOUR_24) -> GasPrediction:
        """
        预测气体浓度
        
        Args:
            gas_type: 气体类型
            horizon: 预测范围
            
        Returns:
            预测结果
        """
        # 获取历史数据
        history = list(self._history_buffer.get(gas_type, []))
        
        if len(history) < 24:
            return self._create_insufficient_data_result(gas_type)
        
        # 确定预测步数
        horizon_steps = {
            PredictionHorizon.HOUR_1: 1,
            PredictionHorizon.HOUR_6: 6,
            PredictionHorizon.HOUR_24: 24,
            PredictionHorizon.DAY_7: 168
        }
        num_steps = min(horizon_steps.get(horizon, 24), self.config.prediction_horizon)
        
        if self.model is not None and TORCH_AVAILABLE:
            return self._predict_with_model(gas_type, history, num_steps)
        else:
            return self._predict_with_fallback(gas_type, history, num_steps)
    
    def _predict_with_model(self, 
                             gas_type: str, 
                             history: List[float],
                             num_steps: int) -> GasPrediction:
        """使用深度学习模型预测"""
        try:
            # 准备输入
            src = self._prepare_input(history)
            src = torch.FloatTensor(src).unsqueeze(0).to(self.device)
            
            # 准备环境数据
            env_data = self._prepare_env_data()
            if env_data is not None:
                env_data = torch.FloatTensor(env_data).unsqueeze(0).to(self.device)
            
            # 准备IMF特征
            imf_features = None
            if self.config.use_ceemdan and len(history) >= self.config.sequence_length:
                imf = self.ceemdan.decompose(np.array(history[-self.config.sequence_length:]))
                imf_features = torch.FloatTensor(imf.T).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                predictions = self.model(src, None, imf_features, env_data)
            
            # 处理输出
            if self.config.use_probabilistic:
                # 提取分位数
                predictions = predictions.cpu().numpy()[0, :num_steps, :, :]
                gas_idx = list(GasType).index(GasType(gas_type)) % self.config.output_dim
                
                median_pred = predictions[:, gas_idx, self.config.num_quantiles // 2]
                lower_pred = predictions[:, gas_idx, 1]
                upper_pred = predictions[:, gas_idx, -2]
                
                pred_values = self._denormalize(median_pred, gas_type).tolist()
                intervals = {
                    "lower": self._denormalize(lower_pred, gas_type).tolist(),
                    "upper": self._denormalize(upper_pred, gas_type).tolist()
                }
            else:
                predictions = predictions.cpu().numpy()[0, :num_steps, :]
                gas_idx = list(GasType).index(GasType(gas_type)) % self.config.output_dim
                pred_values = self._denormalize(predictions[:, gas_idx], gas_type).tolist()
                intervals = None
            
            # 分析趋势
            current = history[-1]
            trend = self._analyze_trend(pred_values)
            
            # 计算泄漏概率
            leak_prob = self._calculate_leak_probability(history, pred_values, gas_type)
            
            # 计算超阈值时间
            threshold = self.config.alarm_thresholds.get(gas_type, float('inf'))
            time_to_alarm = self._calculate_time_to_alarm(pred_values, threshold)
            
            return GasPrediction(
                gas_type=gas_type,
                current_value=current,
                predictions=pred_values,
                timestamps=list(range(1, len(pred_values) + 1)),
                confidence=0.85,
                trend=trend,
                leak_probability=leak_prob,
                time_to_alarm=time_to_alarm,
                prediction_intervals=intervals,
                metadata={"method": "GL-TransLSTM", "model_version": "2.0.0"}
            )
            
        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return self._predict_with_fallback(gas_type, history, num_steps)
    
    def _predict_with_fallback(self, 
                                gas_type: str, 
                                history: List[float],
                                num_steps: int) -> GasPrediction:
        """回退预测方法 (统计方法)"""
        arr = np.array(history)
        
        # 移动平均 + 线性趋势
        window = min(24, len(arr))
        smoothed = np.convolve(arr, np.ones(window)/window, mode='valid')
        
        if len(smoothed) >= 2:
            slope = (smoothed[-1] - smoothed[0]) / len(smoothed)
        else:
            slope = 0
        
        base = smoothed[-1] if len(smoothed) > 0 else arr[-1]
        predictions = [base + slope * (i + 1) + np.random.normal(0, arr.std() * 0.1) 
                      for i in range(num_steps)]
        
        current = history[-1]
        trend = self._analyze_trend(predictions)
        leak_prob = self._calculate_leak_probability(history, predictions, gas_type)
        
        threshold = self.config.alarm_thresholds.get(gas_type, float('inf'))
        time_to_alarm = self._calculate_time_to_alarm(predictions, threshold)
        
        return GasPrediction(
            gas_type=gas_type,
            current_value=current,
            predictions=predictions,
            timestamps=list(range(1, len(predictions) + 1)),
            confidence=0.6,
            trend=trend,
            leak_probability=leak_prob,
            time_to_alarm=time_to_alarm,
            metadata={"method": "statistical_fallback"}
        )
    
    def detect_leak(self) -> LeakDetectionResult:
        """检测泄漏"""
        max_leak_rate = 0.0
        leak_gas = "SF6"
        
        for gas_type, buffer in self._history_buffer.items():
            if len(buffer) < 10:
                continue
            
            values = list(buffer)
            recent = values[-10:]
            
            # 计算变化率
            rate = (recent[-1] - recent[0]) / 10
            
            if rate > max_leak_rate:
                max_leak_rate = rate
                leak_gas = gas_type
        
        # 确定严重程度
        threshold = self.config.alarm_thresholds.get(leak_gas, 1000) * 0.01
        
        if max_leak_rate < threshold * 0.5:
            severity = "none"
        elif max_leak_rate < threshold:
            severity = "low"
        elif max_leak_rate < threshold * 2:
            severity = "medium"
        elif max_leak_rate < threshold * 5:
            severity = "high"
        else:
            severity = "critical"
        
        detected = severity != "none"
        
        recommendations = []
        if detected:
            if severity in ["high", "critical"]:
                recommendations = [
                    "立即停机检查",
                    "启动应急响应程序",
                    "通知相关人员"
                ]
            elif severity == "medium":
                recommendations = [
                    "增加监测频率",
                    "安排检查计划",
                    "准备维修材料"
                ]
            else:
                recommendations = [
                    "持续监测",
                    "记录异常数据"
                ]
        
        return LeakDetectionResult(
            detected=detected,
            gas_type=leak_gas,
            leak_rate=max_leak_rate,
            severity=severity,
            recommendations=recommendations,
            confidence=0.8 if len(list(self._history_buffer.values())[0]) > 50 else 0.5
        )
    
    def _prepare_input(self, history: List[float]) -> np.ndarray:
        """准备模型输入"""
        seq_len = min(len(history), self.config.sequence_length)
        values = np.array(history[-seq_len:])
        
        # 归一化
        normalized = (values - values.mean()) / (values.std() + 1e-8)
        
        # 补齐序列长度
        if len(normalized) < self.config.sequence_length:
            padding = np.zeros(self.config.sequence_length - len(normalized))
            normalized = np.concatenate([padding, normalized])
        
        # 扩展特征维度
        features = np.zeros((self.config.sequence_length, self.config.input_dim))
        features[:, 0] = normalized
        
        return features
    
    def _prepare_env_data(self) -> Optional[np.ndarray]:
        """准备环境数据"""
        if not self._env_buffer:
            return None
        
        latest = self._env_buffer[-1]
        
        return np.array([
            latest.get("temperature", 25),
            latest.get("pressure", 101.3),
            latest.get("humidity", 50)
        ])
    
    def _denormalize(self, values: np.ndarray, gas_type: str) -> np.ndarray:
        """反归一化"""
        params = self._normalization_params.get(gas_type, {"mean": 100, "std": 50})
        return values * params["std"] + params["mean"]
    
    def _analyze_trend(self, predictions: List[float]) -> str:
        """分析趋势"""
        if len(predictions) < 2:
            return "stable"
        
        slope = (predictions[-1] - predictions[0]) / len(predictions)
        
        if slope > 0.5:
            return "rising"
        elif slope < -0.5:
            return "falling"
        else:
            return "stable"
    
    def _calculate_leak_probability(self, 
                                     history: List[float], 
                                     predictions: List[float],
                                     gas_type: str) -> float:
        """计算泄漏概率"""
        if len(history) < 10:
            return 0.0
        
        # 基于变化率
        recent = history[-10:]
        hist_rate = (recent[-1] - recent[0]) / 10
        
        pred_rate = (predictions[-1] - history[-1]) / len(predictions) if predictions else 0
        
        combined_rate = (hist_rate + pred_rate) / 2
        threshold = self.config.alarm_thresholds.get(gas_type, 1000) * 0.01
        
        prob = min(1.0, max(0.0, combined_rate / threshold))
        return prob
    
    def _calculate_time_to_alarm(self, predictions: List[float], threshold: float) -> Optional[float]:
        """计算超阈值时间"""
        for i, val in enumerate(predictions):
            if val >= threshold:
                return float(i + 1)
        return None
    
    def _create_insufficient_data_result(self, gas_type: str) -> GasPrediction:
        """创建数据不足时的结果"""
        history = list(self._history_buffer.get(gas_type, []))
        current = history[-1] if history else 0.0
        
        return GasPrediction(
            gas_type=gas_type,
            current_value=current,
            predictions=[],
            timestamps=[],
            confidence=0.0,
            trend="unknown",
            leak_probability=0.0,
            metadata={"error": "历史数据不足"}
        )
    
    def clear_history(self):
        """清空历史数据"""
        for buffer in self._history_buffer.values():
            buffer.clear()
        self._env_buffer.clear()


# =============================================================================
# 数据集
# =============================================================================

if TORCH_AVAILABLE:
    
    class GasTimeSeriesDataset(Dataset):
        """气体时序数据集"""
        
        def __init__(self, 
                     data: np.ndarray,
                     sequence_length: int,
                     prediction_horizon: int,
                     feature_cols: List[int] = None,
                     target_cols: List[int] = None):
            """
            Args:
                data: 数据数组 [num_samples, num_features]
                sequence_length: 输入序列长度
                prediction_horizon: 预测长度
                feature_cols: 特征列索引
                target_cols: 目标列索引
            """
            self.data = data
            self.sequence_length = sequence_length
            self.prediction_horizon = prediction_horizon
            self.feature_cols = feature_cols or list(range(data.shape[1]))
            self.target_cols = target_cols or [0, 1, 2, 3]  # 默认前4个气体
            
            self.num_samples = len(data) - sequence_length - prediction_horizon + 1
        
        def __len__(self) -> int:
            return self.num_samples
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            # 输入序列
            src_start = idx
            src_end = idx + self.sequence_length
            src = self.data[src_start:src_end, self.feature_cols]
            
            # 目标序列
            tgt_start = src_end
            tgt_end = tgt_start + self.prediction_horizon
            tgt = self.data[tgt_start:tgt_end, self.target_cols]
            
            return {
                "src": torch.FloatTensor(src),
                "tgt": torch.FloatTensor(tgt)
            }


# =============================================================================
# 便捷函数
# =============================================================================

def create_gas_predictor(
    sequence_length: int = 168,
    prediction_horizon: int = 24,
    use_ceemdan: bool = True,
    use_physical_gate: bool = True,
    model_path: Optional[str] = None
) -> GLTransLSTMPredictor:
    """
    创建气体预测器
    
    Args:
        sequence_length: 输入序列长度
        prediction_horizon: 预测步数
        use_ceemdan: 是否使用CEEMDAN
        use_physical_gate: 是否使用物理门控
        model_path: 模型路径
        
    Returns:
        预测器实例
    """
    config = GLTransLSTMConfig(
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        use_ceemdan=use_ceemdan,
        use_physical_gate=use_physical_gate
    )
    
    return GLTransLSTMPredictor(config, model_path)


# =============================================================================
# 使用示例
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    config = GLTransLSTMConfig(
        sequence_length=168,
        prediction_horizon=24,
        use_ceemdan=True,
        use_physical_gate=True
    )
    
    # 创建预测器
    predictor = GLTransLSTMPredictor(config)
    
    # 添加模拟数据
    import numpy as np
    np.random.seed(42)
    
    for i in range(200):
        # 模拟SF6浓度 (带有趋势和噪声)
        sf6 = 900 + i * 0.1 + np.sin(i / 24 * 2 * np.pi) * 20 + np.random.normal(0, 10)
        predictor.add_reading(
            "SF6", 
            sf6,
            timestamp=time.time() - (200 - i) * 3600,
            environmental={
                "temperature": 25 + np.random.normal(0, 2),
                "humidity": 50 + np.random.normal(0, 5),
                "pressure": 101.3 + np.random.normal(0, 1)
            }
        )
    
    # 预测
    prediction = predictor.predict("SF6", PredictionHorizon.HOUR_24)
    
    print(f"\n当前值: {prediction.current_value:.2f} ppm")
    print(f"预测趋势: {prediction.trend}")
    print(f"置信度: {prediction.confidence:.2%}")
    print(f"泄漏概率: {prediction.leak_probability:.2%}")
    print(f"预测值 (前5步): {[f'{v:.2f}' for v in prediction.predictions[:5]]}")
    
    if prediction.time_to_alarm:
        print(f"预计超阈值时间: {prediction.time_to_alarm:.1f} 小时")
    
    # 泄漏检测
    leak_result = predictor.detect_leak()
    print(f"\n泄漏检测: {leak_result.detected}")
    print(f"严重程度: {leak_result.severity}")
    if leak_result.recommendations:
        print("建议措施:")
        for rec in leak_result.recommendations:
            print(f"  - {rec}")
