#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序预测模型训练模块
支持LSTM/Transformer/Informer架构

功能:
1. LSTM时序预测
2. Transformer时序预测
3. Informer长序列预测
4. 多变量预测
5. 概率预测 (不确定性估计)

作者: AI巡检系统
版本: 1.0.0
"""

import os
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


# =============================================================================
# 配置类
# =============================================================================
@dataclass
class TimeSeriesTrainingConfig:
    """时序模型训练配置"""
    # 数据配置
    data_root: str = "data/timeseries"
    input_length: int = 168  # 输入序列长度 (7天*24小时)
    prediction_length: int = 24  # 预测长度
    num_features: int = 8  # 输入特征数
    num_targets: int = 4  # 预测目标数
    
    # 特征列表
    feature_names: List[str] = field(default_factory=lambda: [
        "SF6", "H2", "CO", "C2H2",  # 气体浓度
        "temperature", "humidity", "pressure", "load"  # 环境参数
    ])
    target_names: List[str] = field(default_factory=lambda: [
        "SF6", "H2", "CO", "C2H2"
    ])
    
    # 模型配置
    model_type: str = "informer"  # lstm, transformer, informer
    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    
    # Informer特有配置
    factor: int = 5  # ProbSparse attention factor
    distil: bool = True  # 是否使用蒸馏
    
    # 训练配置
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    
    # 损失配置
    use_probabilistic: bool = True  # 是否使用概率预测
    
    # 保存配置
    save_dir: str = "checkpoints/timeseries"
    save_freq: int = 10


# =============================================================================
# 时序模型
# =============================================================================
if TORCH_AVAILABLE:
    
    # =========================================================================
    # LSTM模型
    # =========================================================================
    class LSTMPredictor(nn.Module):
        """LSTM时序预测模型"""
        
        def __init__(self, config: TimeSeriesTrainingConfig):
            super().__init__()
            
            self.input_projection = nn.Linear(config.num_features, config.hidden_dim)
            
            self.lstm = nn.LSTM(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
                bidirectional=False
            )
            
            # 输出投影
            if config.use_probabilistic:
                # 输出均值和方差
                self.output_mu = nn.Linear(config.hidden_dim, config.num_targets)
                self.output_sigma = nn.Linear(config.hidden_dim, config.num_targets)
            else:
                self.output_projection = nn.Linear(config.hidden_dim, config.num_targets)
            
            self.prediction_length = config.prediction_length
            self.config = config
        
        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Args:
                x: (B, seq_len, num_features)
            Returns:
                dict with predictions
            """
            # 输入投影
            x = self.input_projection(x)  # (B, seq_len, hidden_dim)
            
            # LSTM编码
            lstm_out, (h_n, c_n) = self.lstm(x)  # (B, seq_len, hidden_dim)
            
            # 使用最后一个时间步的隐藏状态
            last_hidden = lstm_out[:, -1, :]  # (B, hidden_dim)
            
            # 自回归预测
            predictions = []
            current_hidden = last_hidden.unsqueeze(1)  # (B, 1, hidden_dim)
            h, c = h_n, c_n
            
            for t in range(self.prediction_length):
                lstm_out_t, (h, c) = self.lstm(current_hidden, (h, c))
                current_hidden = lstm_out_t
                predictions.append(lstm_out_t.squeeze(1))
            
            predictions = torch.stack(predictions, dim=1)  # (B, pred_len, hidden_dim)
            
            if self.config.use_probabilistic:
                mu = self.output_mu(predictions)  # (B, pred_len, num_targets)
                sigma = F.softplus(self.output_sigma(predictions)) + 1e-6
                return {"mu": mu, "sigma": sigma}
            else:
                pred = self.output_projection(predictions)
                return {"prediction": pred}


    # =========================================================================
    # Transformer模型
    # =========================================================================
    class PositionalEncoding(nn.Module):
        """位置编码"""
        
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, :x.size(1), :]


    class TransformerPredictor(nn.Module):
        """Transformer时序预测模型"""
        
        def __init__(self, config: TimeSeriesTrainingConfig):
            super().__init__()
            
            self.input_projection = nn.Linear(config.num_features, config.hidden_dim)
            self.target_projection = nn.Linear(config.num_targets, config.hidden_dim)
            
            self.pos_encoder = PositionalEncoding(config.hidden_dim)
            
            self.transformer = nn.Transformer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                num_encoder_layers=config.num_layers,
                num_decoder_layers=config.num_layers,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True
            )
            
            if config.use_probabilistic:
                self.output_mu = nn.Linear(config.hidden_dim, config.num_targets)
                self.output_sigma = nn.Linear(config.hidden_dim, config.num_targets)
            else:
                self.output_projection = nn.Linear(config.hidden_dim, config.num_targets)
            
            self.prediction_length = config.prediction_length
            self.config = config
            
            # 解码器输入 (学习的查询)
            self.decoder_query = nn.Parameter(
                torch.randn(1, config.prediction_length, config.hidden_dim)
            )
        
        def forward(self, x: torch.Tensor, 
                    y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            """
            Args:
                x: (B, enc_len, num_features) 编码器输入
                y: (B, dec_len, num_targets) 解码器输入 (训练时)
            """
            B = x.shape[0]
            
            # 编码器
            enc_input = self.input_projection(x)
            enc_input = self.pos_encoder(enc_input)
            
            # 解码器输入
            if y is not None:
                dec_input = self.target_projection(y)
            else:
                dec_input = self.decoder_query.expand(B, -1, -1)
            dec_input = self.pos_encoder(dec_input)
            
            # 生成掩码 (因果掩码)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                dec_input.size(1), device=x.device
            )
            
            # Transformer
            output = self.transformer(
                enc_input, dec_input,
                tgt_mask=tgt_mask
            )  # (B, dec_len, hidden_dim)
            
            if self.config.use_probabilistic:
                mu = self.output_mu(output)
                sigma = F.softplus(self.output_sigma(output)) + 1e-6
                return {"mu": mu, "sigma": sigma}
            else:
                pred = self.output_projection(output)
                return {"prediction": pred}


    # =========================================================================
    # Informer模型
    # =========================================================================
    class ProbAttention(nn.Module):
        """ProbSparse Self-attention (Informer核心组件)"""
        
        def __init__(self, d_model: int, n_heads: int, factor: int = 5, 
                     dropout: float = 0.1):
            super().__init__()
            
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.factor = factor
            
            self.W_Q = nn.Linear(d_model, d_model)
            self.W_K = nn.Linear(d_model, d_model)
            self.W_V = nn.Linear(d_model, d_model)
            self.W_O = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
        
        def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, 
                     sample_k: int, n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """ProbSparse采样"""
            B, H, L_Q, D = Q.shape
            _, _, L_K, _ = K.shape
            
            # 随机采样K
            K_sample = K[:, :, torch.randint(L_K, (sample_k,)), :]
            
            # 计算Q与采样K的注意力
            Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))  # (B, H, L_Q, sample_k)
            
            # 计算稀疏性度量
            M = Q_K_sample.max(dim=-1)[0] - Q_K_sample.mean(dim=-1)  # (B, H, L_Q)
            
            # 选择top-u的queries
            M_top = M.topk(n_top, sorted=False)[1]  # (B, H, n_top)
            
            # 减少计算: 只对top queries计算完整注意力
            Q_reduce = torch.gather(
                Q, 2,
                M_top.unsqueeze(-1).expand(-1, -1, -1, D)
            )  # (B, H, n_top, D)
            
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # (B, H, n_top, L_K)
            
            return Q_K, M_top
        
        def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                    values: torch.Tensor) -> torch.Tensor:
            B, L_Q, _ = queries.shape
            _, L_K, _ = keys.shape
            H = self.n_heads
            
            Q = self.W_Q(queries).view(B, L_Q, H, self.d_k).transpose(1, 2)
            K = self.W_K(keys).view(B, L_K, H, self.d_k).transpose(1, 2)
            V = self.W_V(values).view(B, L_K, H, self.d_k).transpose(1, 2)
            
            # 确定采样参数
            U = self.factor * int(np.ceil(np.log(L_K + 1)))
            u = self.factor * int(np.ceil(np.log(L_Q + 1)))
            
            U = min(U, L_K)
            u = min(u, L_Q)
            
            # ProbSparse注意力
            scores_top, index = self._prob_QK(Q, K, sample_k=U, n_top=u)
            
            # Softmax
            scale = 1.0 / math.sqrt(self.d_k)
            attn = self.dropout(F.softmax(scale * scores_top, dim=-1))
            
            # 聚合
            V_reduce = torch.matmul(attn, V)  # (B, H, u, D)
            
            # 初始化输出为均值
            context = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1).clone()
            
            # 填充top queries的结果
            context = context.scatter(
                2,
                index.unsqueeze(-1).expand(-1, -1, -1, self.d_k),
                V_reduce
            )
            
            context = context.transpose(1, 2).contiguous().view(B, L_Q, -1)
            output = self.W_O(context)
            
            return output


    class InformerEncoderLayer(nn.Module):
        """Informer编码器层"""
        
        def __init__(self, d_model: int, n_heads: int, factor: int = 5,
                     dropout: float = 0.1):
            super().__init__()
            
            self.attention = ProbAttention(d_model, n_heads, factor, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            self.norm2 = nn.LayerNorm(d_model)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Self-attention
            attn_out = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            
            # FFN
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            
            return x


    class ConvDistillLayer(nn.Module):
        """蒸馏层 (用于减少序列长度)"""
        
        def __init__(self, d_model: int):
            super().__init__()
            self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
            self.norm = nn.BatchNorm1d(d_model)
            self.pool = nn.MaxPool1d(2, stride=2)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, L, D)
            x = x.transpose(1, 2)  # (B, D, L)
            x = self.pool(F.gelu(self.norm(self.conv(x))))
            x = x.transpose(1, 2)  # (B, L//2, D)
            return x


    class InformerEncoder(nn.Module):
        """Informer编码器"""
        
        def __init__(self, config: TimeSeriesTrainingConfig):
            super().__init__()
            
            self.layers = nn.ModuleList([
                InformerEncoderLayer(
                    config.hidden_dim, config.num_heads, 
                    config.factor, config.dropout
                )
                for _ in range(config.num_layers)
            ])
            
            if config.distil:
                self.distil_layers = nn.ModuleList([
                    ConvDistillLayer(config.hidden_dim)
                    for _ in range(config.num_layers - 1)
                ])
            else:
                self.distil_layers = None
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if self.distil_layers is not None and i < len(self.distil_layers):
                    x = self.distil_layers[i](x)
            return x


    class InformerDecoder(nn.Module):
        """Informer解码器"""
        
        def __init__(self, config: TimeSeriesTrainingConfig):
            super().__init__()
            
            self.layers = nn.ModuleList()
            
            for _ in range(config.num_layers):
                self.layers.append(nn.ModuleDict({
                    'self_attn': nn.MultiheadAttention(
                        config.hidden_dim, config.num_heads,
                        dropout=config.dropout, batch_first=True
                    ),
                    'cross_attn': nn.MultiheadAttention(
                        config.hidden_dim, config.num_heads,
                        dropout=config.dropout, batch_first=True
                    ),
                    'norm1': nn.LayerNorm(config.hidden_dim),
                    'norm2': nn.LayerNorm(config.hidden_dim),
                    'norm3': nn.LayerNorm(config.hidden_dim),
                    'ffn': nn.Sequential(
                        nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                        nn.Dropout(config.dropout)
                    )
                }))
        
        def forward(self, x: torch.Tensor, memory: torch.Tensor,
                    tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            for layer in self.layers:
                # Self-attention
                attn_out, _ = layer['self_attn'](x, x, x, attn_mask=tgt_mask)
                x = layer['norm1'](x + attn_out)
                
                # Cross-attention
                attn_out, _ = layer['cross_attn'](x, memory, memory)
                x = layer['norm2'](x + attn_out)
                
                # FFN
                x = layer['norm3'](x + layer['ffn'](x))
            
            return x


    class Informer(nn.Module):
        """Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"""
        
        def __init__(self, config: TimeSeriesTrainingConfig):
            super().__init__()
            
            # 输入嵌入
            self.enc_embedding = nn.Linear(config.num_features, config.hidden_dim)
            self.dec_embedding = nn.Linear(config.num_targets, config.hidden_dim)
            
            # 位置编码
            self.pos_encoder = PositionalEncoding(config.hidden_dim)
            
            # 编码器和解码器
            self.encoder = InformerEncoder(config)
            self.decoder = InformerDecoder(config)
            
            # 输出投影
            if config.use_probabilistic:
                self.output_mu = nn.Linear(config.hidden_dim, config.num_targets)
                self.output_sigma = nn.Linear(config.hidden_dim, config.num_targets)
            else:
                self.output_projection = nn.Linear(config.hidden_dim, config.num_targets)
            
            self.config = config
            
            # 解码器起始token
            self.start_token = nn.Parameter(
                torch.randn(1, config.prediction_length // 2, config.hidden_dim)
            )
        
        def forward(self, x_enc: torch.Tensor, 
                    x_dec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            """
            Args:
                x_enc: (B, enc_len, num_features) 编码器输入
                x_dec: (B, label_len + pred_len, num_targets) 解码器输入 (训练时)
            """
            B = x_enc.shape[0]
            
            # 编码器嵌入
            enc_embed = self.enc_embedding(x_enc)
            enc_embed = self.pos_encoder(enc_embed)
            
            # 编码
            enc_out = self.encoder(enc_embed)
            
            # 解码器嵌入
            if x_dec is not None:
                dec_embed = self.dec_embedding(x_dec)
            else:
                # 使用学习的起始token + 零填充
                start = self.start_token.expand(B, -1, -1)
                zeros = torch.zeros(B, self.config.prediction_length, 
                                   self.config.hidden_dim, device=x_enc.device)
                dec_embed = torch.cat([start, zeros], dim=1)
            
            dec_embed = self.pos_encoder(dec_embed)
            
            # 生成因果掩码
            tgt_len = dec_embed.size(1)
            tgt_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=x_enc.device), diagonal=1
            ).bool()
            
            # 解码
            dec_out = self.decoder(dec_embed, enc_out, tgt_mask)
            
            # 只取预测部分
            pred_out = dec_out[:, -self.config.prediction_length:, :]
            
            if self.config.use_probabilistic:
                mu = self.output_mu(pred_out)
                sigma = F.softplus(self.output_sigma(pred_out)) + 1e-6
                return {"mu": mu, "sigma": sigma}
            else:
                pred = self.output_projection(pred_out)
                return {"prediction": pred}


    # =========================================================================
    # 数据集
    # =========================================================================
    class TimeSeriesDataset(Dataset):
        """时序预测数据集"""
        
        def __init__(self,
                     data_root: str,
                     input_length: int = 168,
                     prediction_length: int = 24,
                     num_features: int = 8,
                     num_targets: int = 4,
                     split: str = "train",
                     normalize: bool = True):
            self.data_root = Path(data_root)
            self.input_length = input_length
            self.prediction_length = prediction_length
            self.num_features = num_features
            self.num_targets = num_targets
            self.split = split
            self.normalize = normalize
            
            # 加载数据
            self.data = self._load_data()
            
            # 计算归一化参数
            if self.normalize and len(self.data) > 0:
                self.mean = np.mean(self.data, axis=0)
                self.std = np.std(self.data, axis=0) + 1e-8
            else:
                self.mean = np.zeros(num_features)
                self.std = np.ones(num_features)
        
        def _load_data(self) -> np.ndarray:
            """加载数据"""
            data_file = self.data_root / f"{self.split}.csv"
            
            if data_file.exists():
                # 假设CSV格式: timestamp, SF6, H2, CO, C2H2, temp, humidity, pressure, load
                data = np.loadtxt(data_file, delimiter=',', skiprows=1)
                return data[:, 1:]  # 去掉时间戳列
            else:
                logger.warning(f"数据文件不存在: {data_file}, 使用虚拟数据")
                return np.array([])
        
        def __len__(self) -> int:
            if len(self.data) > 0:
                return len(self.data) - self.input_length - self.prediction_length + 1
            else:
                return 1000  # 虚拟数据集大小
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            if len(self.data) == 0:
                return self._generate_dummy_sample()
            
            # 提取序列
            x = self.data[idx:idx + self.input_length]
            y = self.data[idx + self.input_length:idx + self.input_length + self.prediction_length]
            
            # 归一化
            if self.normalize:
                x = (x - self.mean) / self.std
                y = (y - self.mean[:self.num_targets]) / self.std[:self.num_targets]
            
            return {
                "x": torch.from_numpy(x.astype(np.float32)),
                "y": torch.from_numpy(y[:, :self.num_targets].astype(np.float32))
            }
        
        def _generate_dummy_sample(self) -> Dict[str, torch.Tensor]:
            """生成虚拟样本"""
            # 生成带趋势和季节性的时序数据
            total_len = self.input_length + self.prediction_length
            t = np.arange(total_len)
            
            data = np.zeros((total_len, self.num_features))
            
            for i in range(self.num_features):
                # 基础信号
                trend = 0.001 * t
                seasonal = 0.1 * np.sin(2 * np.pi * t / 24)  # 日周期
                noise = 0.05 * np.random.randn(total_len)
                
                # 不同特征有不同的基线
                baseline = [1000, 50, 100, 1, 25, 50, 101, 80][i]
                scale = [100, 20, 30, 0.5, 5, 10, 2, 10][i]
                
                data[:, i] = baseline + scale * (trend + seasonal + noise)
            
            x = data[:self.input_length]
            y = data[self.input_length:, :self.num_targets]
            
            return {
                "x": torch.from_numpy(x.astype(np.float32)),
                "y": torch.from_numpy(y.astype(np.float32))
            }
        
        def inverse_transform(self, data: np.ndarray, 
                             targets_only: bool = True) -> np.ndarray:
            """反归一化"""
            if targets_only:
                return data * self.std[:self.num_targets] + self.mean[:self.num_targets]
            else:
                return data * self.std + self.mean


    # =========================================================================
    # 损失函数
    # =========================================================================
    class GaussianNLLLoss(nn.Module):
        """高斯负对数似然损失 (概率预测)"""
        
        def forward(self, mu: torch.Tensor, sigma: torch.Tensor, 
                    target: torch.Tensor) -> torch.Tensor:
            """
            Args:
                mu: (B, L, D) 预测均值
                sigma: (B, L, D) 预测标准差
                target: (B, L, D) 真实值
            """
            # NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
            var = sigma ** 2
            nll = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
            return nll.mean()


    class QuantileLoss(nn.Module):
        """分位数损失"""
        
        def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
            super().__init__()
            self.quantiles = quantiles
        
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """
            Args:
                pred: (B, L, D, Q) 分位数预测
                target: (B, L, D) 真实值
            """
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = target - pred[..., i]
                losses.append(torch.max((q - 1) * errors, q * errors))
            
            return torch.stack(losses).mean()


# =============================================================================
# 训练器
# =============================================================================
class TimeSeriesTrainer:
    """时序模型训练器"""
    
    def __init__(self, config: TimeSeriesTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 损失函数
        if config.use_probabilistic:
            self.criterion = GaussianNLLLoss()
        else:
            self.criterion = nn.MSELoss()
        
        # 数据加载器
        self.train_loader = self._create_dataloader("train", True)
        self.val_loader = self._create_dataloader("val", False)
        
        # 学习率调度器
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=len(self.train_loader)
        )
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        
        os.makedirs(config.save_dir, exist_ok=True)
    
    def _create_model(self) -> nn.Module:
        if self.config.model_type == "lstm":
            return LSTMPredictor(self.config)
        elif self.config.model_type == "transformer":
            return TransformerPredictor(self.config)
        elif self.config.model_type == "informer":
            return Informer(self.config)
        else:
            return Informer(self.config)
    
    def _create_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = TimeSeriesDataset(
            data_root=self.config.data_root,
            input_length=self.config.input_length,
            prediction_length=self.config.prediction_length,
            num_features=self.config.num_features,
            num_targets=self.config.num_targets,
            split=split
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(x)
            
            if self.config.use_probabilistic:
                loss = self.criterion(outputs["mu"], outputs["sigma"], y)
            else:
                loss = self.criterion(outputs["prediction"], y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {"loss": total_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            
            outputs = self.model(x)
            
            if self.config.use_probabilistic:
                pred = outputs["mu"]
                loss = self.criterion(outputs["mu"], outputs["sigma"], y)
            else:
                pred = outputs["prediction"]
                loss = self.criterion(pred, y)
            
            # MAE
            mae = torch.abs(pred - y).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "mae": total_mae / num_batches
        }
    
    def train(self):
        """完整训练流程"""
        logger.info(f"开始训练时序预测模型: {self.config.model_type}")
        logger.info(f"设备: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            self.train_history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics
            })
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}"
            )
            
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best.pth")
            
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pth")
        
        self.save_checkpoint("final.pth")
        logger.info("训练完成!")
    
    def save_checkpoint(self, filename: str):
        path = os.path.join(self.config.save_dir, filename)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "train_history": self.train_history
        }
        
        torch.save(checkpoint, path)
        logger.info(f"保存检查点: {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"加载检查点: {path}")


# =============================================================================
# 导出ONNX
# =============================================================================
def export_timeseries_to_onnx(model: nn.Module,
                              save_path: str,
                              input_length: int = 168,
                              num_features: int = 8) -> bool:
    """导出时序模型到ONNX"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return False
    
    model.eval()
    
    dummy_input = torch.randn(1, input_length, num_features)
    
    try:
        # 只导出推理路径
        class InferenceWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                out = self.model(x)
                if "mu" in out:
                    return out["mu"], out["sigma"]
                else:
                    return out["prediction"]
        
        wrapper = InferenceWrapper(model)
        
        torch.onnx.export(
            wrapper,
            dummy_input,
            save_path,
            input_names=["input"],
            output_names=["mu", "sigma"] if hasattr(model, 'output_mu') else ["prediction"],
            dynamic_axes={
                "input": {0: "batch", 1: "seq_len"},
            },
            opset_version=13
        )
        logger.info(f"ONNX模型导出成功: {save_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX导出失败: {e}")
        return False


# =============================================================================
# 入口函数
# =============================================================================
def train_timeseries_model(config: Optional[TimeSeriesTrainingConfig] = None):
    """训练时序模型的入口函数"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return None
    
    if config is None:
        config = TimeSeriesTrainingConfig()
    
    trainer = TimeSeriesTrainer(config)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = TimeSeriesTrainingConfig(
        model_type="informer",
        batch_size=16,
        num_epochs=10,
        save_dir="checkpoints/timeseries"
    )
    
    train_timeseries_model(config)
