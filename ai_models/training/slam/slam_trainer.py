#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SLAM模型训练模块
支持LOAM/LIO-SAM风格的深度学习SLAM训练

功能:
1. 点云特征提取网络 (PointNet++/KPConv)
2. 位姿估计网络 (DeepLIO)
3. 点云配准网络 (DCP/RPMNet)
4. 回环检测网络 (PointNetVLAD)
5. 语义分割网络 (RandLA-Net)

作者: AI巡检系统
版本: 1.0.0
"""

import os
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
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
class SLAMTrainingConfig:
    """SLAM训练配置"""
    # 数据配置
    data_root: str = "data/slam"
    train_sequences: List[str] = field(default_factory=lambda: ["00", "01", "02"])
    val_sequences: List[str] = field(default_factory=lambda: ["03"])
    num_points: int = 16384
    voxel_size: float = 0.3
    
    # 模型配置
    model_type: str = "deep_lio"  # deep_lio, pointnet_slam, dcp
    feature_dim: int = 256
    use_normals: bool = True
    use_intensity: bool = True
    
    # 训练配置
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    
    # 损失权重
    pose_loss_weight: float = 1.0
    rotation_loss_weight: float = 1.0
    translation_loss_weight: float = 1.0
    
    # 保存配置
    save_dir: str = "checkpoints/slam"
    save_freq: int = 10
    log_freq: int = 100


# =============================================================================
# 点云特征提取网络
# =============================================================================
if TORCH_AVAILABLE:
    class PointNetEncoder(nn.Module):
        """PointNet特征编码器"""
        
        def __init__(self, in_channels: int = 3, feature_dim: int = 256):
            super().__init__()
            
            self.conv1 = nn.Conv1d(in_channels, 64, 1)
            self.conv2 = nn.Conv1d(64, 128, 1)
            self.conv3 = nn.Conv1d(128, 256, 1)
            self.conv4 = nn.Conv1d(256, 512, 1)
            self.conv5 = nn.Conv1d(512, feature_dim, 1)
            
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(256)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(feature_dim)
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (B, N, C) 点云
            Returns:
                global_feat: (B, feature_dim) 全局特征
                point_feat: (B, N, feature_dim) 点级特征
            """
            x = x.transpose(2, 1)  # (B, C, N)
            
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            point_feat = F.relu(self.bn5(self.conv5(x)))  # (B, D, N)
            
            global_feat = torch.max(point_feat, dim=2)[0]  # (B, D)
            
            return global_feat, point_feat.transpose(2, 1)


    class EdgeConv(nn.Module):
        """EdgeConv层 (DGCNN)"""
        
        def __init__(self, in_channels: int, out_channels: int, k: int = 20):
            super().__init__()
            self.k = k
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (B, C, N)
            Returns:
                (B, out_channels, N)
            """
            B, C, N = x.shape
            
            # KNN
            inner = -2 * torch.matmul(x.transpose(2, 1), x)
            xx = torch.sum(x ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            
            idx = pairwise_distance.topk(k=self.k, dim=-1)[1]  # (B, N, k)
            
            # 获取邻居特征
            idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
            idx = idx + idx_base
            idx = idx.view(-1)
            
            x = x.transpose(2, 1).contiguous()  # (B, N, C)
            neighbors = x.view(B * N, -1)[idx, :].view(B, N, self.k, C)
            x = x.view(B, N, 1, C).expand(-1, -1, self.k, -1)
            
            # EdgeConv
            edge_feat = torch.cat([neighbors - x, x], dim=3)  # (B, N, k, 2C)
            edge_feat = edge_feat.permute(0, 3, 1, 2)  # (B, 2C, N, k)
            
            out = self.conv(edge_feat)  # (B, out_channels, N, k)
            out = out.max(dim=-1)[0]  # (B, out_channels, N)
            
            return out


    class DGCNNEncoder(nn.Module):
        """DGCNN特征编码器"""
        
        def __init__(self, in_channels: int = 3, feature_dim: int = 256, k: int = 20):
            super().__init__()
            
            self.k = k
            
            self.edge_conv1 = EdgeConv(in_channels, 64, k)
            self.edge_conv2 = EdgeConv(64, 64, k)
            self.edge_conv3 = EdgeConv(64, 128, k)
            self.edge_conv4 = EdgeConv(128, 256, k)
            
            self.conv = nn.Sequential(
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Conv1d(512, feature_dim, 1),
                nn.BatchNorm1d(feature_dim),
                nn.LeakyReLU(0.2)
            )
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (B, N, C) 点云
            Returns:
                global_feat: (B, feature_dim)
                point_feat: (B, N, feature_dim)
            """
            x = x.transpose(2, 1)  # (B, C, N)
            
            x1 = self.edge_conv1(x)
            x2 = self.edge_conv2(x1)
            x3 = self.edge_conv3(x2)
            x4 = self.edge_conv4(x3)
            
            x = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
            point_feat = self.conv(x)  # (B, feature_dim, N)
            
            global_feat = torch.max(point_feat, dim=2)[0]  # (B, feature_dim)
            
            return global_feat, point_feat.transpose(2, 1)


    # =========================================================================
    # 位姿估计网络
    # =========================================================================
    class DeepLIO(nn.Module):
        """
        Deep LiDAR-Inertial Odometry
        基于深度学习的LiDAR里程计
        """
        
        def __init__(self, config: SLAMTrainingConfig):
            super().__init__()
            
            in_channels = 3
            if config.use_normals:
                in_channels += 3
            if config.use_intensity:
                in_channels += 1
            
            self.encoder = DGCNNEncoder(
                in_channels=in_channels,
                feature_dim=config.feature_dim,
                k=20
            )
            
            # 位姿回归头
            self.pose_head = nn.Sequential(
                nn.Linear(config.feature_dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 6)  # [tx, ty, tz, rx, ry, rz]
            )
            
        def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """
            预测从source到target的相对位姿
            
            Args:
                source: (B, N, C) 源点云
                target: (B, N, C) 目标点云
            
            Returns:
                pose: (B, 6) [tx, ty, tz, rx, ry, rz]
            """
            feat_src, _ = self.encoder(source)
            feat_tgt, _ = self.encoder(target)
            
            combined = torch.cat([feat_src, feat_tgt], dim=1)
            pose = self.pose_head(combined)
            
            return pose


    class DeepPointCorrespondence(nn.Module):
        """
        Deep Closest Point (DCP)
        基于深度学习的点云配准
        """
        
        def __init__(self, feature_dim: int = 256, num_heads: int = 4):
            super().__init__()
            
            self.encoder = DGCNNEncoder(in_channels=3, feature_dim=feature_dim)
            
            # Transformer注意力
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                batch_first=True
            )
            
            # 指针网络
            self.pointer = nn.Sequential(
                nn.Linear(feature_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            
            # SVD层用于计算刚性变换
            self.svd_weights = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1)
            )
        
        def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            计算从source到target的刚性变换
            
            Args:
                source: (B, N, 3)
                target: (B, M, 3)
            
            Returns:
                dict with R (B, 3, 3) and t (B, 3)
            """
            B, N, _ = source.shape
            
            # 编码
            _, feat_src = self.encoder(source)  # (B, N, D)
            _, feat_tgt = self.encoder(target)  # (B, M, D)
            
            # 交叉注意力
            attended_feat, _ = self.attention(feat_src, feat_tgt, feat_tgt)
            
            # 计算软对应
            # 使用注意力权重作为软对应
            attn_weights = torch.bmm(feat_src, feat_tgt.transpose(2, 1))
            attn_weights = F.softmax(attn_weights / np.sqrt(feat_src.shape[-1]), dim=-1)
            
            # 软对应点
            matched_target = torch.bmm(attn_weights, target)  # (B, N, 3)
            
            # 使用SVD计算刚性变换
            weights = self.svd_weights(attended_feat).squeeze(-1)  # (B, N)
            
            R, t = self._weighted_svd(source, matched_target, weights)
            
            return {"R": R, "t": t, "correspondence": attn_weights}
        
        def _weighted_svd(self, src: torch.Tensor, tgt: torch.Tensor, 
                         weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """加权SVD计算刚性变换"""
            B = src.shape[0]
            
            # 加权质心
            weights = weights.unsqueeze(-1)  # (B, N, 1)
            w_sum = weights.sum(dim=1, keepdim=True)
            
            src_centroid = (src * weights).sum(dim=1, keepdim=True) / w_sum
            tgt_centroid = (tgt * weights).sum(dim=1, keepdim=True) / w_sum
            
            src_centered = src - src_centroid
            tgt_centered = tgt - tgt_centroid
            
            # 加权协方差
            H = torch.bmm(
                (src_centered * weights).transpose(2, 1),
                tgt_centered
            )  # (B, 3, 3)
            
            # SVD
            U, S, Vh = torch.linalg.svd(H)
            R = torch.bmm(Vh.transpose(2, 1), U.transpose(2, 1))
            
            # 处理反射
            det = torch.linalg.det(R)
            det_sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
            Vh_corrected = Vh.clone()
            Vh_corrected[:, 2, :] *= det_sign.squeeze(-1)
            R = torch.bmm(Vh_corrected.transpose(2, 1), U.transpose(2, 1))
            
            t = tgt_centroid.squeeze(1) - torch.bmm(R, src_centroid.transpose(2, 1)).squeeze(-1)
            
            return R, t


    # =========================================================================
    # 回环检测网络
    # =========================================================================
    class PointNetVLAD(nn.Module):
        """
        PointNetVLAD: 基于VLAD的点云全局描述子
        用于回环检测
        """
        
        def __init__(self, feature_dim: int = 256, num_clusters: int = 64):
            super().__init__()
            
            self.encoder = PointNetEncoder(in_channels=3, feature_dim=feature_dim)
            self.num_clusters = num_clusters
            
            # VLAD聚类中心 (可学习)
            self.cluster_centers = nn.Parameter(
                torch.randn(num_clusters, feature_dim)
            )
            
            # 软分配
            self.fc_soft_assign = nn.Sequential(
                nn.Linear(feature_dim, num_clusters),
                nn.Softmax(dim=-1)
            )
            
            # 最终描述子
            self.fc_final = nn.Sequential(
                nn.Linear(num_clusters * feature_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            生成点云的全局描述子
            
            Args:
                x: (B, N, 3) 点云
            
            Returns:
                descriptor: (B, 256) 全局描述子
            """
            B, N, _ = x.shape
            
            # 点级特征
            _, point_feat = self.encoder(x)  # (B, N, D)
            D = point_feat.shape[-1]
            
            # 软分配
            soft_assign = self.fc_soft_assign(point_feat)  # (B, N, K)
            
            # VLAD聚合
            residuals = point_feat.unsqueeze(2) - self.cluster_centers.unsqueeze(0).unsqueeze(0)
            # (B, N, K, D)
            
            vlad = (soft_assign.unsqueeze(-1) * residuals).sum(dim=1)  # (B, K, D)
            
            # L2归一化
            vlad = F.normalize(vlad, p=2, dim=-1)
            vlad = vlad.view(B, -1)  # (B, K*D)
            
            # 最终描述子
            descriptor = self.fc_final(vlad)
            descriptor = F.normalize(descriptor, p=2, dim=-1)
            
            return descriptor


    # =========================================================================
    # 语义分割网络
    # =========================================================================
    class RandLANet(nn.Module):
        """
        RandLA-Net: 高效点云语义分割
        用于变电站场景理解
        """
        
        def __init__(self, num_classes: int = 8, feature_dim: int = 32):
            super().__init__()
            
            self.fc_start = nn.Linear(3, 8)
            
            # 编码器
            self.encoder = nn.ModuleList([
                self._make_encoder_block(8, 16),
                self._make_encoder_block(32, 64),
                self._make_encoder_block(128, 128),
                self._make_encoder_block(256, 256),
            ])
            
            # 解码器
            self.decoder = nn.ModuleList([
                self._make_decoder_block(512, 256),
                self._make_decoder_block(256, 128),
                self._make_decoder_block(128, 32),
                self._make_decoder_block(32, 8),
            ])
            
            # 分类头
            self.fc_end = nn.Sequential(
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, num_classes)
            )
        
        def _make_encoder_block(self, in_dim: int, out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim * 2)
            )
        
        def _make_decoder_block(self, in_dim: int, out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU()
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (B, N, 3) 点云
            Returns:
                logits: (B, N, num_classes)
            """
            B, N, _ = x.shape
            
            # 初始特征
            feat = self.fc_start(x)  # (B, N, 8)
            
            # 编码 (简化版,不含随机采样)
            encoder_feats = [feat]
            for enc in self.encoder:
                feat = enc(feat)
                encoder_feats.append(feat)
            
            # 解码
            for i, dec in enumerate(self.decoder):
                skip = encoder_feats[-(i+2)]
                feat = torch.cat([feat[..., :skip.shape[-1]], skip], dim=-1)
                feat = dec(feat)
            
            # 分类
            logits = self.fc_end(feat)
            
            return logits


    # =========================================================================
    # 数据集
    # =========================================================================
    class SLAMDataset(Dataset):
        """SLAM训练数据集"""
        
        def __init__(self, 
                     data_root: str,
                     sequences: List[str],
                     num_points: int = 16384,
                     voxel_size: float = 0.3,
                     augment: bool = True):
            self.data_root = Path(data_root)
            self.sequences = sequences
            self.num_points = num_points
            self.voxel_size = voxel_size
            self.augment = augment
            
            # 加载数据对
            self.pairs = self._load_pairs()
        
        def _load_pairs(self) -> List[Dict]:
            """加载点云对和位姿"""
            pairs = []
            
            for seq in self.sequences:
                seq_dir = self.data_root / seq
                
                # 检查目录是否存在
                if not seq_dir.exists():
                    logger.warning(f"序列目录不存在: {seq_dir}")
                    continue
                
                # 加载位姿文件
                pose_file = seq_dir / "poses.txt"
                if not pose_file.exists():
                    continue
                    
                poses = np.loadtxt(pose_file).reshape(-1, 3, 4)
                
                # 创建帧对
                pc_dir = seq_dir / "velodyne"
                if not pc_dir.exists():
                    continue
                    
                pc_files = sorted(pc_dir.glob("*.bin"))
                
                for i in range(len(pc_files) - 1):
                    pairs.append({
                        "source": str(pc_files[i]),
                        "target": str(pc_files[i + 1]),
                        "pose_src": poses[i],
                        "pose_tgt": poses[i + 1]
                    })
            
            return pairs
        
        def __len__(self) -> int:
            return len(self.pairs) if self.pairs else 1000  # 虚拟数据集大小
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            if not self.pairs:
                # 生成虚拟数据用于测试
                return self._generate_dummy_data()
            
            pair = self.pairs[idx]
            
            # 加载点云
            source = self._load_point_cloud(pair["source"])
            target = self._load_point_cloud(pair["target"])
            
            # 计算相对位姿
            T_src = np.eye(4)
            T_src[:3, :] = pair["pose_src"]
            T_tgt = np.eye(4)
            T_tgt[:3, :] = pair["pose_tgt"]
            
            T_rel = np.linalg.inv(T_src) @ T_tgt
            
            # 提取平移和旋转
            translation = T_rel[:3, 3]
            rotation = self._rotation_matrix_to_euler(T_rel[:3, :3])
            pose = np.concatenate([translation, rotation])
            
            # 数据增强
            if self.augment:
                source, target = self._augment(source, target)
            
            return {
                "source": torch.from_numpy(source).float(),
                "target": torch.from_numpy(target).float(),
                "pose": torch.from_numpy(pose).float()
            }
        
        def _generate_dummy_data(self) -> Dict[str, torch.Tensor]:
            """生成虚拟训练数据"""
            source = np.random.randn(self.num_points, 3).astype(np.float32)
            
            # 随机变换
            angle = np.random.uniform(-0.1, 0.1, 3)
            translation = np.random.uniform(-0.5, 0.5, 3)
            
            R = self._euler_to_rotation_matrix(angle)
            target = (source @ R.T + translation).astype(np.float32)
            
            pose = np.concatenate([translation, angle]).astype(np.float32)
            
            return {
                "source": torch.from_numpy(source),
                "target": torch.from_numpy(target),
                "pose": torch.from_numpy(pose)
            }
        
        def _load_point_cloud(self, path: str) -> np.ndarray:
            """加载点云文件"""
            if path.endswith(".bin"):
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
                points = points[:, :3]  # 只取xyz
            elif path.endswith(".npy"):
                points = np.load(path)
            else:
                raise ValueError(f"不支持的点云格式: {path}")
            
            # 随机采样
            if len(points) > self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
            elif len(points) < self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=True)
                points = points[indices]
            
            return points.astype(np.float32)
        
        def _augment(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """数据增强"""
            # 随机旋转 (绕z轴)
            if np.random.random() > 0.5:
                angle = np.random.uniform(-np.pi, np.pi)
                c, s = np.cos(angle), np.sin(angle)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                source = source @ R.T
                target = target @ R.T
            
            # 随机缩放
            if np.random.random() > 0.5:
                scale = np.random.uniform(0.95, 1.05)
                source *= scale
                target *= scale
            
            # 随机抖动
            source += np.random.normal(0, 0.01, source.shape)
            target += np.random.normal(0, 0.01, target.shape)
            
            return source, target
        
        @staticmethod
        def _rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
            """旋转矩阵转欧拉角"""
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0
            
            return np.array([x, y, z])
        
        @staticmethod
        def _euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
            """欧拉角转旋转矩阵"""
            rx, ry, rz = euler
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ])
            
            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ])
            
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]
            ])
            
            return Rz @ Ry @ Rx


    # =========================================================================
    # 损失函数
    # =========================================================================
    class PoseLoss(nn.Module):
        """位姿损失函数"""
        
        def __init__(self, translation_weight: float = 1.0, rotation_weight: float = 1.0):
            super().__init__()
            self.translation_weight = translation_weight
            self.rotation_weight = rotation_weight
        
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Args:
                pred: (B, 6) [tx, ty, tz, rx, ry, rz]
                target: (B, 6)
            """
            trans_loss = F.mse_loss(pred[:, :3], target[:, :3])
            rot_loss = F.mse_loss(pred[:, 3:], target[:, 3:])
            
            total_loss = (self.translation_weight * trans_loss + 
                         self.rotation_weight * rot_loss)
            
            return {
                "total": total_loss,
                "translation": trans_loss,
                "rotation": rot_loss
            }


    class ChamferLoss(nn.Module):
        """Chamfer距离损失"""
        
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """
            Args:
                pred: (B, N, 3)
                target: (B, M, 3)
            """
            # pred到target的距离
            diff = pred.unsqueeze(2) - target.unsqueeze(1)  # (B, N, M, 3)
            dist = torch.sum(diff ** 2, dim=-1)  # (B, N, M)
            
            dist1 = torch.min(dist, dim=2)[0]  # (B, N)
            dist2 = torch.min(dist, dim=1)[0]  # (B, M)
            
            return torch.mean(dist1) + torch.mean(dist2)


# =============================================================================
# 训练器
# =============================================================================
class SLAMTrainer:
    """SLAM模型训练器"""
    
    def __init__(self, config: SLAMTrainingConfig):
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
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
        
        # 损失函数
        self.criterion = PoseLoss(
            translation_weight=config.translation_loss_weight,
            rotation_weight=config.rotation_loss_weight
        )
        
        # 数据加载器
        self.train_loader = self._create_dataloader(config.train_sequences, True)
        self.val_loader = self._create_dataloader(config.val_sequences, False)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
    
    def _create_model(self) -> nn.Module:
        """根据配置创建模型"""
        if self.config.model_type == "deep_lio":
            return DeepLIO(self.config)
        elif self.config.model_type == "dcp":
            return DeepPointCorrespondence(self.config.feature_dim)
        elif self.config.model_type == "vlad":
            return PointNetVLAD(self.config.feature_dim)
        else:
            return DeepLIO(self.config)
    
    def _create_dataloader(self, sequences: List[str], shuffle: bool) -> DataLoader:
        """创建数据加载器"""
        dataset = SLAMDataset(
            data_root=self.config.data_root,
            sequences=sequences,
            num_points=self.config.num_points,
            voxel_size=self.config.voxel_size,
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
        total_loss = 0.0
        trans_loss = 0.0
        rot_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)
            pose_gt = batch["pose"].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            pose_pred = self.model(source, target)
            
            # 计算损失
            losses = self.criterion(pose_pred, pose_gt)
            
            # 反向传播
            losses["total"].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += losses["total"].item()
            trans_loss += losses["translation"].item()
            rot_loss += losses["rotation"].item()
            num_batches += 1
            
            if batch_idx % self.config.log_freq == 0:
                logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {losses['total'].item():.4f}"
                )
        
        return {
            "total": total_loss / num_batches,
            "translation": trans_loss / num_batches,
            "rotation": rot_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        trans_loss = 0.0
        rot_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            source = batch["source"].to(self.device)
            target = batch["target"].to(self.device)
            pose_gt = batch["pose"].to(self.device)
            
            pose_pred = self.model(source, target)
            losses = self.criterion(pose_pred, pose_gt)
            
            total_loss += losses["total"].item()
            trans_loss += losses["translation"].item()
            rot_loss += losses["rotation"].item()
            num_batches += 1
        
        return {
            "total": total_loss / num_batches,
            "translation": trans_loss / num_batches,
            "rotation": rot_loss / num_batches
        }
    
    def train(self):
        """完整训练流程"""
        logger.info(f"开始训练 SLAM模型: {self.config.model_type}")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练序列: {self.config.train_sequences}")
        logger.info(f"验证序列: {self.config.val_sequences}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch()
            
            # 验证
            val_losses = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录
            self.train_history.append({
                "epoch": epoch,
                "train": train_losses,
                "val": val_losses,
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_losses['total']:.4f}, "
                f"Val Loss: {val_losses['total']:.4f}"
            )
            
            # 保存最佳模型
            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self.save_checkpoint("best.pth")
            
            # 定期保存
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pth")
        
        logger.info("训练完成!")
        self.save_checkpoint("final.pth")
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
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
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_history = checkpoint.get("train_history", [])
        
        logger.info(f"加载检查点: {path}, Epoch: {self.current_epoch}")


# =============================================================================
# 导出ONNX
# =============================================================================
def export_slam_to_onnx(model: nn.Module, 
                        save_path: str,
                        num_points: int = 16384) -> bool:
    """导出SLAM模型到ONNX"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return False
    
    model.eval()
    
    # 创建虚拟输入
    dummy_source = torch.randn(1, num_points, 3)
    dummy_target = torch.randn(1, num_points, 3)
    
    try:
        torch.onnx.export(
            model,
            (dummy_source, dummy_target),
            save_path,
            input_names=["source", "target"],
            output_names=["pose"],
            dynamic_axes={
                "source": {0: "batch", 1: "num_points"},
                "target": {0: "batch", 1: "num_points"},
                "pose": {0: "batch"}
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
def train_slam_model(config: Optional[SLAMTrainingConfig] = None):
    """训练SLAM模型的入口函数"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装,无法训练模型")
        return None
    
    if config is None:
        config = SLAMTrainingConfig()
    
    trainer = SLAMTrainer(config)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 示例训练
    config = SLAMTrainingConfig(
        data_root="data/slam",
        model_type="deep_lio",
        batch_size=4,
        num_epochs=10,
        save_dir="checkpoints/slam"
    )
    
    train_slam_model(config)
