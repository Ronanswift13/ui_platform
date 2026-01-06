#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
声学异常检测模型训练模块
支持Transformer/Autoencoder/Contrastive Learning

功能:
1. 音频Transformer异常检测
2. 变分自编码器 (VAE)
3. 对比学习 (SimCLR风格)
4. 多尺度特征融合
5. 局部放电专用检测器

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
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
class AcousticTrainingConfig:
    """声学模型训练配置"""
    # 数据配置
    data_root: str = "data/acoustic"
    sample_rate: int = 16000
    duration: float = 2.0  # 秒
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    
    # 模型配置
    model_type: str = "transformer"  # transformer, vae, contrastive
    feature_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # 异常类型
    anomaly_types: List[str] = field(default_factory=lambda: [
        "normal", "partial_discharge", "corona", 
        "mechanical_fault", "transformer_hum"
    ])
    
    # 训练配置
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    num_workers: int = 4
    
    # 对比学习配置
    temperature: float = 0.07
    
    # 保存配置
    save_dir: str = "checkpoints/acoustic"
    save_freq: int = 10


# =============================================================================
# 音频特征提取
# =============================================================================
if TORCH_AVAILABLE:
    class MelSpectrogramLayer(nn.Module):
        """可微分Mel频谱计算层"""
        
        def __init__(self, 
                     sample_rate: int = 16000,
                     n_fft: int = 2048,
                     hop_length: int = 512,
                     n_mels: int = 128):
            super().__init__()
            self.sample_rate = sample_rate
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.n_mels = n_mels
            
            # 创建Mel滤波器组
            mel_fb = self._create_mel_filterbank()
            self.register_buffer('mel_fb', mel_fb)
            
            # Hann窗
            window = torch.hann_window(n_fft)
            self.register_buffer('window', window)
        
        def _create_mel_filterbank(self) -> torch.Tensor:
            """创建Mel滤波器组"""
            n_freqs = self.n_fft // 2 + 1
            
            # Mel刻度转换
            low_freq_mel = 0
            high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
            mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)
            
            bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
            
            filterbank = np.zeros((self.n_mels, n_freqs))
            for m in range(1, self.n_mels + 1):
                f_m_minus = bin_points[m - 1]
                f_m = bin_points[m]
                f_m_plus = bin_points[m + 1]
                
                for k in range(f_m_minus, f_m):
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-8)
                for k in range(f_m, f_m_plus):
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-8)
            
            return torch.from_numpy(filterbank).float()
        
        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            """
            Args:
                waveform: (B, samples) 或 (B, 1, samples)
            Returns:
                mel_spec: (B, n_mels, time)
            """
            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)
            
            # STFT
            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            
            # 功率谱
            power_spec = torch.abs(spec) ** 2  # (B, n_freqs, time)
            
            # Mel滤波
            mel_spec = torch.matmul(self.mel_fb, power_spec)  # (B, n_mels, time)
            
            # 对数变换
            mel_spec = torch.log(mel_spec + 1e-8)
            
            return mel_spec


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
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer('pe', pe)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (B, seq_len, d_model)
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class AudioTransformerEncoder(nn.Module):
        """音频Transformer编码器"""
        
        def __init__(self, config: AcousticTrainingConfig):
            super().__init__()
            
            self.mel_layer = MelSpectrogramLayer(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels
            )
            
            # 输入投影
            self.input_projection = nn.Linear(config.n_mels, config.feature_dim)
            
            # 位置编码
            self.pos_encoder = PositionalEncoding(
                config.feature_dim, 
                max_len=1000,
                dropout=config.dropout
            )
            
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.feature_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feature_dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            
            # CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.feature_dim))
            
            # 输出投影
            self.output_projection = nn.Linear(config.feature_dim, config.feature_dim)
        
        def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                waveform: (B, samples) 或 (B, 1, samples)
            Returns:
                global_feat: (B, feature_dim) CLS token
                seq_feat: (B, seq_len, feature_dim) 序列特征
            """
            # Mel频谱
            mel_spec = self.mel_layer(waveform)  # (B, n_mels, time)
            mel_spec = mel_spec.transpose(1, 2)  # (B, time, n_mels)
            
            # 输入投影
            x = self.input_projection(mel_spec)  # (B, time, feature_dim)
            
            # 添加CLS token
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, time+1, feature_dim)
            
            # 位置编码
            x = self.pos_encoder(x)
            
            # Transformer
            seq_feat = self.transformer(x)  # (B, time+1, feature_dim)
            
            # 提取CLS token
            global_feat = seq_feat[:, 0, :]  # (B, feature_dim)
            global_feat = self.output_projection(global_feat)
            
            return global_feat, seq_feat[:, 1:, :]


    # =========================================================================
    # 异常检测模型
    # =========================================================================
    class AcousticAnomalyTransformer(nn.Module):
        """
        Transformer-based 声学异常检测
        结合重建误差和分类的混合方法
        """
        
        def __init__(self, config: AcousticTrainingConfig):
            super().__init__()
            
            self.encoder = AudioTransformerEncoder(config)
            
            # 解码器用于重建
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.feature_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feature_dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers // 2)
            
            # 重建投影
            self.reconstruction_head = nn.Linear(config.feature_dim, config.n_mels)
            
            # 异常分类头
            self.anomaly_head = nn.Sequential(
                nn.Linear(config.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, len(config.anomaly_types))
            )
            
            # 异常分数头 (回归)
            self.score_head = nn.Sequential(
                nn.Linear(config.feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
            self.config = config
        
        def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Args:
                waveform: (B, samples)
            Returns:
                dict with keys: global_feat, reconstruction, anomaly_logits, anomaly_score
            """
            # 编码
            global_feat, seq_feat = self.encoder(waveform)
            
            # 解码重建
            memory = seq_feat
            tgt = seq_feat  # 自回归目标
            reconstructed = self.decoder(tgt, memory)
            reconstructed = self.reconstruction_head(reconstructed)  # (B, time, n_mels)
            
            # 异常分类
            anomaly_logits = self.anomaly_head(global_feat)  # (B, num_classes)
            
            # 异常分数
            anomaly_score = self.score_head(global_feat).squeeze(-1)  # (B,)
            
            return {
                "global_feat": global_feat,
                "reconstruction": reconstructed,
                "anomaly_logits": anomaly_logits,
                "anomaly_score": anomaly_score
            }


    class AcousticVAE(nn.Module):
        """变分自编码器用于异常检测"""
        
        def __init__(self, config: AcousticTrainingConfig):
            super().__init__()
            
            self.mel_layer = MelSpectrogramLayer(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                n_mels=config.n_mels
            )
            
            # 编码器
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            
            # 计算编码后的空间尺寸
            self.latent_dim = config.feature_dim
            
            # VAE的mu和logvar
            self.fc_mu = nn.Linear(256 * 8 * 4, self.latent_dim)
            self.fc_logvar = nn.Linear(256 * 8 * 4, self.latent_dim)
            
            # 解码器
            self.fc_decode = nn.Linear(self.latent_dim, 256 * 8 * 4)
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            )
            
            self.config = config
        
        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder(x)
            h = h.view(h.size(0), -1)
            return self.fc_mu(h), self.fc_logvar(h)
        
        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def decode(self, z: torch.Tensor) -> torch.Tensor:
            h = self.fc_decode(z)
            h = h.view(-1, 256, 8, 4)
            return self.decoder(h)
        
        def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
            # Mel频谱
            mel_spec = self.mel_layer(waveform)  # (B, n_mels, time)
            mel_spec = mel_spec.unsqueeze(1)  # (B, 1, n_mels, time)
            
            # 编码
            mu, logvar = self.encode(mel_spec)
            z = self.reparameterize(mu, logvar)
            
            # 解码
            reconstruction = self.decode(z)
            
            # 异常分数 = 重建误差
            recon_error = F.mse_loss(reconstruction, mel_spec, reduction='none')
            anomaly_score = recon_error.mean(dim=[1, 2, 3])
            
            return {
                "reconstruction": reconstruction,
                "mu": mu,
                "logvar": logvar,
                "z": z,
                "anomaly_score": anomaly_score
            }


    class ContrastiveAcousticModel(nn.Module):
        """对比学习声学模型 (SimCLR风格)"""
        
        def __init__(self, config: AcousticTrainingConfig):
            super().__init__()
            
            self.encoder = AudioTransformerEncoder(config)
            
            # 投影头
            self.projector = nn.Sequential(
                nn.Linear(config.feature_dim, config.feature_dim),
                nn.ReLU(),
                nn.Linear(config.feature_dim, 128)
            )
            
            # 分类头 (用于下游任务)
            self.classifier = nn.Linear(config.feature_dim, len(config.anomaly_types))
            
            self.config = config
        
        def forward(self, waveform: torch.Tensor, return_projection: bool = False):
            global_feat, _ = self.encoder(waveform)
            
            if return_projection:
                projection = self.projector(global_feat)
                projection = F.normalize(projection, dim=-1)
                return global_feat, projection
            
            return global_feat
        
        def classify(self, waveform: torch.Tensor) -> torch.Tensor:
            global_feat, _ = self.encoder(waveform)
            return self.classifier(global_feat)


    # =========================================================================
    # 数据集
    # =========================================================================
    class AcousticDataset(Dataset):
        """声学异常检测数据集"""
        
        def __init__(self,
                     data_root: str,
                     sample_rate: int = 16000,
                     duration: float = 2.0,
                     split: str = "train",
                     augment: bool = True):
            self.data_root = Path(data_root)
            self.sample_rate = sample_rate
            self.duration = duration
            self.num_samples = int(sample_rate * duration)
            self.augment = augment
            self.split = split
            
            # 加载数据列表
            self.samples = self._load_samples()
            
            # 标签映射
            self.label_map = {
                "normal": 0,
                "partial_discharge": 1,
                "corona": 2,
                "mechanical_fault": 3,
                "transformer_hum": 4
            }
        
        def _load_samples(self) -> List[Dict]:
            """加载样本列表"""
            samples = []
            
            split_dir = self.data_root / self.split
            if not split_dir.exists():
                logger.warning(f"数据目录不存在: {split_dir}, 使用虚拟数据")
                return []
            
            for label_dir in split_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                
                label = label_dir.name
                
                for audio_file in label_dir.glob("*.wav"):
                    samples.append({
                        "path": str(audio_file),
                        "label": label
                    })
            
            return samples
        
        def __len__(self) -> int:
            return len(self.samples) if self.samples else 1000
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            if not self.samples:
                return self._generate_dummy_sample()
            
            sample = self.samples[idx]
            
            # 加载音频
            waveform = self._load_audio(sample["path"])
            
            # 数据增强
            if self.augment:
                waveform = self._augment(waveform)
            
            label = self.label_map.get(sample["label"], 0)
            is_anomaly = 0 if sample["label"] == "normal" else 1
            
            return {
                "waveform": torch.from_numpy(waveform).float(),
                "label": torch.tensor(label, dtype=torch.long),
                "is_anomaly": torch.tensor(is_anomaly, dtype=torch.float)
            }
        
        def _generate_dummy_sample(self) -> Dict[str, torch.Tensor]:
            """生成虚拟样本"""
            # 生成正常声音 (白噪声 + 100Hz正弦波)
            t = np.linspace(0, self.duration, self.num_samples)
            waveform = np.random.randn(self.num_samples) * 0.01
            waveform += 0.1 * np.sin(2 * np.pi * 100 * t)
            
            # 随机添加异常
            is_anomaly = np.random.random() > 0.7
            label = 0
            
            if is_anomaly:
                anomaly_type = np.random.randint(1, 5)
                label = anomaly_type
                
                if anomaly_type == 1:  # 局部放电 - 高频脉冲
                    pulse_times = np.random.choice(self.num_samples, 20, replace=False)
                    for pt in pulse_times:
                        if pt + 50 < self.num_samples:
                            waveform[pt:pt+50] += 0.5 * np.exp(-np.linspace(0, 5, 50))
                
                elif anomaly_type == 2:  # 电晕 - 嘶嘶声
                    waveform += 0.2 * np.random.randn(self.num_samples) * np.sin(2 * np.pi * 5000 * t)
                
                elif anomaly_type == 3:  # 机械故障 - 周期敲击
                    for i in range(10):
                        impact_time = int(i * self.num_samples / 10)
                        if impact_time + 200 < self.num_samples:
                            waveform[impact_time:impact_time+200] += 0.3 * np.exp(-np.linspace(0, 10, 200))
                
                elif anomaly_type == 4:  # 变压器异常嗡鸣
                    waveform += 0.3 * np.sin(2 * np.pi * 120 * t)
                    waveform += 0.2 * np.sin(2 * np.pi * 240 * t)
            
            # 归一化
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
            
            return {
                "waveform": torch.from_numpy(waveform.astype(np.float32)),
                "label": torch.tensor(label, dtype=torch.long),
                "is_anomaly": torch.tensor(float(is_anomaly), dtype=torch.float)
            }
        
        def _load_audio(self, path: str) -> np.ndarray:
            """加载音频文件"""
            try:
                import soundfile as sf
                waveform, sr = sf.read(path)
            except ImportError:
                # 回退到scipy
                from scipy.io import wavfile
                sr, waveform = wavfile.read(path)
                waveform = waveform.astype(np.float32) / 32768.0
            
            # 转单声道
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            
            # 重采样 (简化处理)
            if sr != self.sample_rate:
                # 线性插值重采样
                old_len = len(waveform)
                new_len = int(old_len * self.sample_rate / sr)
                waveform = np.interp(
                    np.linspace(0, old_len, new_len),
                    np.arange(old_len),
                    waveform
                )
            
            # 截断或填充
            if len(waveform) > self.num_samples:
                start = np.random.randint(0, len(waveform) - self.num_samples)
                waveform = waveform[start:start + self.num_samples]
            elif len(waveform) < self.num_samples:
                waveform = np.pad(waveform, (0, self.num_samples - len(waveform)))
            
            return waveform.astype(np.float32)
        
        def _augment(self, waveform: np.ndarray) -> np.ndarray:
            """数据增强"""
            # 时间偏移
            if np.random.random() > 0.5:
                shift = np.random.randint(-1000, 1000)
                waveform = np.roll(waveform, shift)
            
            # 添加噪声
            if np.random.random() > 0.5:
                noise_level = np.random.uniform(0.001, 0.01)
                waveform += noise_level * np.random.randn(len(waveform))
            
            # 音量变化
            if np.random.random() > 0.5:
                gain = np.random.uniform(0.8, 1.2)
                waveform *= gain
            
            # 时间拉伸 (简化版)
            if np.random.random() > 0.7:
                rate = np.random.uniform(0.9, 1.1)
                new_len = int(len(waveform) * rate)
                waveform = np.interp(
                    np.linspace(0, len(waveform), self.num_samples),
                    np.arange(new_len),
                    np.interp(np.linspace(0, len(waveform), new_len), np.arange(len(waveform)), waveform)
                )
            
            return waveform


    # =========================================================================
    # 损失函数
    # =========================================================================
    class AnomalyDetectionLoss(nn.Module):
        """混合异常检测损失"""
        
        def __init__(self, 
                     recon_weight: float = 1.0,
                     cls_weight: float = 1.0,
                     score_weight: float = 0.5):
            super().__init__()
            self.recon_weight = recon_weight
            self.cls_weight = cls_weight
            self.score_weight = score_weight
            self.ce_loss = nn.CrossEntropyLoss()
            self.bce_loss = nn.BCELoss()
        
        def forward(self, 
                    outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    mel_spec: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Args:
                outputs: 模型输出
                targets: 标签
                mel_spec: 原始Mel频谱
            """
            losses = {}
            
            # 重建损失
            if "reconstruction" in outputs:
                recon = outputs["reconstruction"]
                if mel_spec.dim() == 3:
                    mel_spec_target = mel_spec.transpose(1, 2)  # (B, time, n_mels)
                else:
                    mel_spec_target = mel_spec
                
                # 调整尺寸
                if recon.shape != mel_spec_target.shape:
                    min_len = min(recon.shape[1], mel_spec_target.shape[1])
                    recon = recon[:, :min_len, :]
                    mel_spec_target = mel_spec_target[:, :min_len, :]
                
                losses["recon"] = F.mse_loss(recon, mel_spec_target) * self.recon_weight
            
            # 分类损失
            if "anomaly_logits" in outputs and "label" in targets:
                losses["cls"] = self.ce_loss(
                    outputs["anomaly_logits"], 
                    targets["label"]
                ) * self.cls_weight
            
            # 异常分数损失
            if "anomaly_score" in outputs and "is_anomaly" in targets:
                losses["score"] = self.bce_loss(
                    outputs["anomaly_score"],
                    targets["is_anomaly"]
                ) * self.score_weight
            
            losses["total"] = sum(losses.values())
            
            return losses


    class NTXentLoss(nn.Module):
        """NT-Xent损失 (对比学习)"""
        
        def __init__(self, temperature: float = 0.07):
            super().__init__()
            self.temperature = temperature
        
        def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
            """
            Args:
                z_i, z_j: (B, D) 两个增强视图的特征
            """
            B = z_i.shape[0]
            
            # 拼接
            z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
            
            # 相似度矩阵
            sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)
            
            # 掩码对角线
            mask = torch.eye(2 * B, device=z.device).bool()
            sim.masked_fill_(mask, -float('inf'))
            
            # 正样本对的位置
            pos_mask = torch.zeros(2 * B, 2 * B, device=z.device).bool()
            pos_mask[:B, B:] = torch.eye(B, device=z.device).bool()
            pos_mask[B:, :B] = torch.eye(B, device=z.device).bool()
            
            # InfoNCE损失
            exp_sim = torch.exp(sim)
            log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
            
            loss = -log_prob[pos_mask].mean()
            
            return loss


# =============================================================================
# 训练器
# =============================================================================
class AcousticTrainer:
    """声学模型训练器"""
    
    def __init__(self, config: AcousticTrainingConfig):
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
        
        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # 损失函数
        self.criterion = self._create_criterion()
        
        # 数据加载器
        self.train_loader = self._create_dataloader("train", True)
        self.val_loader = self._create_dataloader("val", False)
        
        # Mel层用于计算目标
        self.mel_layer = MelSpectrogramLayer(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels
        ).to(self.device)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        
        os.makedirs(config.save_dir, exist_ok=True)
    
    def _create_model(self) -> nn.Module:
        if self.config.model_type == "transformer":
            return AcousticAnomalyTransformer(self.config)
        elif self.config.model_type == "vae":
            return AcousticVAE(self.config)
        elif self.config.model_type == "contrastive":
            return ContrastiveAcousticModel(self.config)
        else:
            return AcousticAnomalyTransformer(self.config)
    
    def _create_criterion(self):
        if self.config.model_type == "contrastive":
            return NTXentLoss(self.config.temperature)
        else:
            return AnomalyDetectionLoss()
    
    def _create_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = AcousticDataset(
            data_root=self.config.data_root,
            sample_rate=self.config.sample_rate,
            duration=self.config.duration,
            split=split,
            augment=shuffle
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
        
        total_losses = {}
        num_batches = 0
        
        for batch in self.train_loader:
            waveform = batch["waveform"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.model_type == "contrastive":
                # 对比学习需要两个增强视图
                _, proj1 = self.model(waveform, return_projection=True)
                # 简单增强: 添加噪声
                waveform_aug = waveform + 0.01 * torch.randn_like(waveform)
                _, proj2 = self.model(waveform_aug, return_projection=True)
                
                loss = self.criterion(proj1, proj2)
                losses = {"total": loss, "contrastive": loss}
            else:
                outputs = self.model(waveform)
                mel_spec = self.mel_layer(waveform)
                
                targets = {
                    "label": batch["label"].to(self.device),
                    "is_anomaly": batch["is_anomaly"].to(self.device)
                }
                
                losses = self.criterion(outputs, targets, mel_spec)
            
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        total_losses = {}
        num_batches = 0
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            waveform = batch["waveform"].to(self.device)
            
            if self.config.model_type == "contrastive":
                _, proj1 = self.model(waveform, return_projection=True)
                loss = torch.tensor(0.0)  # 对比学习验证时不计算损失
                losses = {"total": loss}
            else:
                outputs = self.model(waveform)
                mel_spec = self.mel_layer(waveform)
                
                targets = {
                    "label": batch["label"].to(self.device),
                    "is_anomaly": batch["is_anomaly"].to(self.device)
                }
                
                losses = self.criterion(outputs, targets, mel_spec)
                
                # 计算准确率
                if "anomaly_logits" in outputs:
                    pred = outputs["anomaly_logits"].argmax(dim=1)
                    correct += (pred == targets["label"]).sum().item()
                    total += targets["label"].size(0)
            
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1
        
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        if total > 0:
            avg_losses["accuracy"] = correct / total
        
        return avg_losses
    
    def train(self):
        """完整训练流程"""
        logger.info(f"开始训练声学异常检测模型: {self.config.model_type}")
        logger.info(f"设备: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            train_losses = self.train_epoch()
            val_losses = self.validate()
            
            self.scheduler.step()
            
            self.train_history.append({
                "epoch": epoch,
                "train": train_losses,
                "val": val_losses
            })
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_losses['total']:.4f}, "
                f"Val Loss: {val_losses['total']:.4f}"
            )
            
            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
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
            "scheduler_state_dict": self.scheduler.state_dict(),
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
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        logger.info(f"加载检查点: {path}")


# =============================================================================
# 导出ONNX
# =============================================================================
def export_acoustic_to_onnx(model: nn.Module,
                            save_path: str,
                            sample_rate: int = 16000,
                            duration: float = 2.0) -> bool:
    """导出声学模型到ONNX"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return False
    
    model.eval()
    
    num_samples = int(sample_rate * duration)
    dummy_waveform = torch.randn(1, num_samples)
    
    try:
        torch.onnx.export(
            model,
            dummy_waveform,
            save_path,
            input_names=["waveform"],
            output_names=["global_feat", "reconstruction", "anomaly_logits", "anomaly_score"],
            dynamic_axes={
                "waveform": {0: "batch", 1: "samples"},
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
def train_acoustic_model(config: Optional[AcousticTrainingConfig] = None):
    """训练声学模型的入口函数"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return None
    
    if config is None:
        config = AcousticTrainingConfig()
    
    trainer = AcousticTrainer(config)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = AcousticTrainingConfig(
        model_type="transformer",
        batch_size=8,
        num_epochs=10,
        save_dir="checkpoints/acoustic"
    )
    
    train_acoustic_model(config)
