#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合网络训练模块
支持跨模态注意力融合

功能:
1. 跨模态注意力机制
2. 多尺度特征融合
3. 动态权重学习
4. 对比学习预训练
5. 综合健康评估

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
    from torch.optim.lr_scheduler import CosineAnnealingLR
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
class FusionTrainingConfig:
    """融合网络训练配置"""
    # 模态配置
    modalities: Dict[str, Dict] = field(default_factory=lambda: {
        "visual": {"feature_dim": 512, "weight": 0.3},
        "pointcloud": {"feature_dim": 256, "weight": 0.25},
        "audio": {"feature_dim": 128, "weight": 0.2},
        "thermal": {"feature_dim": 128, "weight": 0.15},
        "timeseries": {"feature_dim": 64, "weight": 0.1}
    })
    
    # 模型配置
    fusion_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
    # 融合类型
    fusion_type: str = "cross_attention"  # early, late, cross_attention, hierarchical
    
    # 输出配置
    num_classes: int = 5  # 故障类型数
    output_health_score: bool = True
    
    # 训练配置
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    
    # 对比学习配置
    use_contrastive: bool = True
    temperature: float = 0.07
    
    # 保存配置
    save_dir: str = "checkpoints/fusion"
    save_freq: int = 10


# =============================================================================
# 融合网络模型
# =============================================================================
if TORCH_AVAILABLE:
    
    # =========================================================================
    # 模态编码器
    # =========================================================================
    class ModalityEncoder(nn.Module):
        """通用模态编码器"""
        
        def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)


    # =========================================================================
    # 跨模态注意力
    # =========================================================================
    class CrossModalAttention(nn.Module):
        """跨模态注意力机制"""
        
        def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, query: torch.Tensor, key: torch.Tensor, 
                    value: torch.Tensor) -> torch.Tensor:
            """
            跨模态注意力
            
            Args:
                query: (B, N_q, D) 查询模态
                key: (B, N_k, D) 键模态
                value: (B, N_k, D) 值模态
            
            Returns:
                (B, N_q, D) 注意力输出
            """
            B, N_q, D = query.shape
            N_k = key.shape[1]
            
            # 投影
            Q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
            
            # 注意力得分
            attn = (Q @ K.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # 聚合
            out = (attn @ V).transpose(1, 2).reshape(B, N_q, D)
            out = self.out_proj(out)
            
            return out


    class CrossModalTransformerLayer(nn.Module):
        """跨模态Transformer层"""
        
        def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            
            # 自注意力
            self.self_attn = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )
            self.norm1 = nn.LayerNorm(dim)
            
            # 跨模态注意力
            self.cross_attn = CrossModalAttention(dim, num_heads, dropout)
            self.norm2 = nn.LayerNorm(dim)
            
            # FFN
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            )
            self.norm3 = nn.LayerNorm(dim)
        
        def forward(self, x: torch.Tensor, 
                    context: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (B, N, D) 当前模态
                context: (B, M, D) 上下文模态
            """
            # 自注意力
            attn_out, _ = self.self_attn(x, x, x)
            x = self.norm1(x + attn_out)
            
            # 跨模态注意力
            cross_out = self.cross_attn(x, context, context)
            x = self.norm2(x + cross_out)
            
            # FFN
            x = self.norm3(x + self.ffn(x))
            
            return x


    # =========================================================================
    # 多模态融合网络
    # =========================================================================
    class MultimodalFusionNetwork(nn.Module):
        """
        多模态融合网络
        支持多种融合策略: early, late, cross_attention, hierarchical
        """
        
        def __init__(self, config: FusionTrainingConfig):
            super().__init__()
            self.config = config
            
            # 模态编码器
            self.modality_encoders = nn.ModuleDict()
            for name, modality_config in config.modalities.items():
                self.modality_encoders[name] = ModalityEncoder(
                    modality_config["feature_dim"],
                    config.fusion_dim,
                    config.dropout
                )
            
            # 模态token (用于Transformer)
            self.modality_tokens = nn.ParameterDict()
            for name in config.modalities.keys():
                self.modality_tokens[name] = nn.Parameter(
                    torch.randn(1, 1, config.fusion_dim)
                )
            
            # 融合层
            if config.fusion_type == "cross_attention":
                self.fusion_layers = nn.ModuleList([
                    CrossModalTransformerLayer(
                        config.fusion_dim, config.num_heads, config.dropout
                    )
                    for _ in range(config.num_layers)
                ])
            elif config.fusion_type == "hierarchical":
                self._build_hierarchical_fusion()
            
            # 全局聚合
            self.global_attention = nn.MultiheadAttention(
                config.fusion_dim, config.num_heads,
                dropout=config.dropout, batch_first=True
            )
            
            # CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.fusion_dim))
            
            # 输出头
            self.classifier = nn.Sequential(
                nn.Linear(config.fusion_dim, 256),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(256, config.num_classes)
            )
            
            if config.output_health_score:
                self.health_score_head = nn.Sequential(
                    nn.Linear(config.fusion_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            
            # 模态权重预测器 (动态权重)
            self.weight_predictor = nn.Sequential(
                nn.Linear(config.fusion_dim * len(config.modalities), 128),
                nn.ReLU(),
                nn.Linear(128, len(config.modalities)),
                nn.Softmax(dim=-1)
            )
            
            # 投影头 (对比学习)
            if config.use_contrastive:
                self.projector = nn.Sequential(
                    nn.Linear(config.fusion_dim, config.fusion_dim),
                    nn.ReLU(),
                    nn.Linear(config.fusion_dim, 128)
                )
        
        def _build_hierarchical_fusion(self):
            """构建层次化融合结构"""
            # 低级融合: visual + thermal
            self.low_level_fusion = CrossModalTransformerLayer(
                self.config.fusion_dim, self.config.num_heads, self.config.dropout
            )
            
            # 中级融合: audio + timeseries
            self.mid_level_fusion = CrossModalTransformerLayer(
                self.config.fusion_dim, self.config.num_heads, self.config.dropout
            )
            
            # 高级融合: all
            self.high_level_fusion = nn.ModuleList([
                CrossModalTransformerLayer(
                    self.config.fusion_dim, self.config.num_heads, self.config.dropout
                )
                for _ in range(2)
            ])
        
        def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """
            Args:
                modality_features: {modality_name: (B, D) or (B, N, D)}
            
            Returns:
                dict with: logits, health_score, fused_features, modality_weights
            """
            B = list(modality_features.values())[0].shape[0]
            
            # 编码各模态
            encoded_features = {}
            for name, feat in modality_features.items():
                if name not in self.modality_encoders:
                    continue
                
                # 确保是3D tensor (B, N, D)
                if feat.dim() == 2:
                    feat = feat.unsqueeze(1)
                
                encoded = self.modality_encoders[name](feat)
                
                # 添加模态token
                token = self.modality_tokens[name].expand(B, -1, -1)
                encoded = torch.cat([token, encoded], dim=1)
                
                encoded_features[name] = encoded
            
            # 融合
            if self.config.fusion_type == "cross_attention":
                fused = self._cross_attention_fusion(encoded_features)
            elif self.config.fusion_type == "hierarchical":
                fused = self._hierarchical_fusion(encoded_features)
            elif self.config.fusion_type == "early":
                fused = self._early_fusion(encoded_features)
            else:  # late
                fused = self._late_fusion(encoded_features)
            
            # 全局聚合
            cls_token = self.cls_token.expand(B, -1, -1)
            fused_with_cls = torch.cat([cls_token, fused], dim=1)
            
            global_feat, _ = self.global_attention(
                fused_with_cls, fused_with_cls, fused_with_cls
            )
            global_feat = global_feat[:, 0, :]  # CLS token
            
            # 计算动态权重
            modality_feats = torch.cat([
                encoded_features[name][:, 0, :] 
                for name in sorted(encoded_features.keys())
            ], dim=-1)
            modality_weights = self.weight_predictor(modality_feats)
            
            # 输出
            outputs = {
                "fused_features": global_feat,
                "logits": self.classifier(global_feat),
                "modality_weights": modality_weights
            }
            
            if self.config.output_health_score:
                outputs["health_score"] = self.health_score_head(global_feat).squeeze(-1) * 100
            
            if self.config.use_contrastive:
                outputs["projection"] = F.normalize(self.projector(global_feat), dim=-1)
            
            return outputs
        
        def _cross_attention_fusion(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
            """跨模态注意力融合"""
            # 拼接所有模态
            all_features = torch.cat(list(features.values()), dim=1)
            
            # 对每个模态应用跨模态注意力
            fused_list = []
            for name, feat in features.items():
                # 其他模态作为context
                context = torch.cat([
                    f for n, f in features.items() if n != name
                ], dim=1)
                
                for layer in self.fusion_layers:
                    feat = layer(feat, context)
                
                fused_list.append(feat[:, 0, :])  # 只取模态token
            
            return torch.stack(fused_list, dim=1)  # (B, num_modalities, D)
        
        def _hierarchical_fusion(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
            """层次化融合"""
            # 低级: visual + thermal
            if "visual" in features and "thermal" in features:
                low = self.low_level_fusion(features["visual"], features["thermal"])
            else:
                low = list(features.values())[0]
            
            # 中级: audio + timeseries
            if "audio" in features and "timeseries" in features:
                mid = self.mid_level_fusion(features["audio"], features["timeseries"])
            else:
                mid = list(features.values())[-1]
            
            # 高级: 所有
            all_features = torch.cat([low, mid], dim=1)
            
            for layer in self.high_level_fusion:
                all_features = layer(all_features, all_features)
            
            return all_features
        
        def _early_fusion(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
            """早期融合 (特征拼接)"""
            return torch.cat(list(features.values()), dim=1)
        
        def _late_fusion(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
            """晚期融合 (加权平均)"""
            weights = torch.tensor(
                [self.config.modalities[name]["weight"] for name in features.keys()],
                device=list(features.values())[0].device
            )
            weights = weights / weights.sum()
            
            fused = sum(
                w * feat[:, 0, :] 
                for w, feat in zip(weights, features.values())
            )
            
            return fused.unsqueeze(1)


    # =========================================================================
    # 对比学习损失
    # =========================================================================
    class MultimodalContrastiveLoss(nn.Module):
        """多模态对比学习损失"""
        
        def __init__(self, temperature: float = 0.07):
            super().__init__()
            self.temperature = temperature
        
        def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
            """
            Args:
                z1, z2: (B, D) 两个视图的特征
            """
            B = z1.shape[0]
            
            # 拼接
            z = torch.cat([z1, z2], dim=0)  # (2B, D)
            
            # 相似度
            sim = torch.mm(z, z.t()) / self.temperature
            
            # 掩码
            mask = torch.eye(2 * B, device=z.device).bool()
            sim.masked_fill_(mask, -float('inf'))
            
            # 正样本对
            pos_mask = torch.zeros(2 * B, 2 * B, device=z.device).bool()
            pos_mask[:B, B:] = torch.eye(B, device=z.device).bool()
            pos_mask[B:, :B] = torch.eye(B, device=z.device).bool()
            
            # NCE loss
            exp_sim = torch.exp(sim)
            log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
            loss = -log_prob[pos_mask].mean()
            
            return loss


    class FusionLoss(nn.Module):
        """融合网络综合损失"""
        
        def __init__(self, 
                     cls_weight: float = 1.0,
                     health_weight: float = 0.5,
                     contrastive_weight: float = 0.3,
                     temperature: float = 0.07):
            super().__init__()
            
            self.cls_weight = cls_weight
            self.health_weight = health_weight
            self.contrastive_weight = contrastive_weight
            
            self.ce_loss = nn.CrossEntropyLoss()
            self.mse_loss = nn.MSELoss()
            self.contrastive_loss = MultimodalContrastiveLoss(temperature)
        
        def forward(self, 
                    outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    aug_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
            """
            Args:
                outputs: 模型输出
                targets: 标签
                aug_outputs: 增强视图的输出 (对比学习)
            """
            losses = {}
            
            # 分类损失
            if "logits" in outputs and "label" in targets:
                losses["cls"] = self.ce_loss(outputs["logits"], targets["label"]) * self.cls_weight
            
            # 健康评分损失
            if "health_score" in outputs and "health_score" in targets:
                losses["health"] = self.mse_loss(
                    outputs["health_score"], 
                    targets["health_score"]
                ) * self.health_weight
            
            # 对比学习损失
            if aug_outputs is not None and "projection" in outputs:
                losses["contrastive"] = self.contrastive_loss(
                    outputs["projection"],
                    aug_outputs["projection"]
                ) * self.contrastive_weight
            
            losses["total"] = sum(losses.values())
            
            return losses


    # =========================================================================
    # 数据集
    # =========================================================================
    class MultimodalDataset(Dataset):
        """多模态数据集"""
        
        def __init__(self,
                     data_root: str,
                     modalities: Dict[str, Dict],
                     split: str = "train",
                     augment: bool = True):
            self.data_root = Path(data_root)
            self.modalities = modalities
            self.split = split
            self.augment = augment
            
            # 加载数据列表
            self.samples = self._load_samples()
        
        def _load_samples(self) -> List[Dict]:
            """加载样本列表"""
            samples = []
            
            manifest_file = self.data_root / f"{self.split}_manifest.json"
            if manifest_file.exists():
                import json
                with open(manifest_file) as f:
                    samples = json.load(f)
            else:
                logger.warning(f"Manifest不存在: {manifest_file}, 使用虚拟数据")
            
            return samples
        
        def __len__(self) -> int:
            return len(self.samples) if self.samples else 500
        
        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            if not self.samples:
                return self._generate_dummy_sample()
            
            sample = self.samples[idx]
            
            # 加载各模态特征
            features = {}
            for name, config in self.modalities.items():
                feat_path = self.data_root / sample.get(f"{name}_path", "")
                if feat_path.exists():
                    features[name] = torch.from_numpy(np.load(feat_path)).float()
                else:
                    features[name] = torch.randn(config["feature_dim"])
            
            # 标签
            label = sample.get("label", 0)
            health_score = sample.get("health_score", 100.0)
            
            return {
                "features": features,
                "label": torch.tensor(label, dtype=torch.long),
                "health_score": torch.tensor(health_score, dtype=torch.float)
            }
        
        def _generate_dummy_sample(self) -> Dict[str, torch.Tensor]:
            """生成虚拟样本"""
            features = {}
            
            for name, config in self.modalities.items():
                features[name] = torch.randn(config["feature_dim"])
            
            # 随机标签
            label = np.random.randint(0, 5)
            health_score = np.random.uniform(0, 100)
            
            return {
                "features": features,
                "label": torch.tensor(label, dtype=torch.long),
                "health_score": torch.tensor(health_score, dtype=torch.float)
            }


# =============================================================================
# 训练器
# =============================================================================
class FusionTrainer:
    """融合网络训练器"""
    
    def __init__(self, config: FusionTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        self.model = MultimodalFusionNetwork(config)
        self.model = self.model.to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        
        # 损失函数
        self.criterion = FusionLoss(
            temperature=config.temperature
        )
        
        # 数据加载器
        self.train_loader = self._create_dataloader("train", True)
        self.val_loader = self._create_dataloader("val", False)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        
        os.makedirs(config.save_dir, exist_ok=True)
    
    def _create_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = MultimodalDataset(
            data_root="data/multimodal",
            modalities=self.config.modalities,
            split=split,
            augment=shuffle
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """自定义collate函数"""
        features = {}
        for name in self.config.modalities.keys():
            features[name] = torch.stack([
                sample["features"][name] for sample in batch
            ])
        
        labels = torch.stack([sample["label"] for sample in batch])
        health_scores = torch.stack([sample["health_score"] for sample in batch])
        
        return {
            "features": features,
            "label": labels,
            "health_score": health_scores
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_losses = {}
        num_batches = 0
        
        for batch in self.train_loader:
            # 移动到设备
            features = {k: v.to(self.device) for k, v in batch["features"].items()}
            targets = {
                "label": batch["label"].to(self.device),
                "health_score": batch["health_score"].to(self.device)
            }
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(features)
            
            # 对比学习: 创建增强视图
            aug_outputs = None
            if self.config.use_contrastive:
                # 简单的特征扰动增强
                aug_features = {
                    k: v + 0.1 * torch.randn_like(v) 
                    for k, v in features.items()
                }
                aug_outputs = self.model(aug_features)
            
            # 计算损失
            losses = self.criterion(outputs, targets, aug_outputs)
            
            # 反向传播
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 累积损失
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
        correct = 0
        total = 0
        health_mae = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            features = {k: v.to(self.device) for k, v in batch["features"].items()}
            targets = {
                "label": batch["label"].to(self.device),
                "health_score": batch["health_score"].to(self.device)
            }
            
            outputs = self.model(features)
            losses = self.criterion(outputs, targets)
            
            # 准确率
            pred = outputs["logits"].argmax(dim=1)
            correct += (pred == targets["label"]).sum().item()
            total += targets["label"].size(0)
            
            # 健康评分MAE
            if "health_score" in outputs:
                health_mae += torch.abs(
                    outputs["health_score"] - targets["health_score"]
                ).mean().item()
            
            for k, v in losses.items():
                if k not in total_losses:
                    total_losses[k] = 0.0
                total_losses[k] += v.item()
            num_batches += 1
        
        metrics = {k: v / num_batches for k, v in total_losses.items()}
        metrics["accuracy"] = correct / total
        metrics["health_mae"] = health_mae / num_batches
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        logger.info(f"开始训练多模态融合网络")
        logger.info(f"融合类型: {self.config.fusion_type}")
        logger.info(f"设备: {self.device}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            self.train_history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics
            })
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['total']:.4f}, "
                f"Val Loss: {val_metrics['total']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
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
def export_fusion_to_onnx(model: nn.Module,
                          config: FusionTrainingConfig,
                          save_path: str) -> bool:
    """导出融合模型到ONNX"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return False
    
    model.eval()
    
    # 创建虚拟输入
    dummy_inputs = {}
    for name, modality_config in config.modalities.items():
        dummy_inputs[name] = torch.randn(1, modality_config["feature_dim"])
    
    try:
        # ONNX不支持dict输入,需要包装
        class ONNXWrapper(nn.Module):
            def __init__(self, model, modality_names):
                super().__init__()
                self.model = model
                self.modality_names = modality_names
            
            def forward(self, *args):
                features = {
                    name: feat for name, feat in zip(self.modality_names, args)
                }
                out = self.model(features)
                return out["logits"], out.get("health_score", torch.zeros(1))
        
        wrapper = ONNXWrapper(model, list(config.modalities.keys()))
        inputs = tuple(dummy_inputs[name] for name in config.modalities.keys())
        
        torch.onnx.export(
            wrapper,
            inputs,
            save_path,
            input_names=list(config.modalities.keys()),
            output_names=["logits", "health_score"],
            dynamic_axes={
                name: {0: "batch"} for name in config.modalities.keys()
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
def train_fusion_model(config: Optional[FusionTrainingConfig] = None):
    """训练融合模型的入口函数"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return None
    
    if config is None:
        config = FusionTrainingConfig()
    
    trainer = FusionTrainer(config)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = FusionTrainingConfig(
        fusion_type="cross_attention",
        batch_size=16,
        num_epochs=10,
        save_dir="checkpoints/fusion"
    )
    
    train_fusion_model(config)
