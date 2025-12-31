#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力融合增强模块
多模态注意力机制

功能:
1. 自注意力融合
2. 跨模态注意力
3. 门控融合
4. 动态模态权重
5. 层次化融合

作者: AI巡检系统
版本: 1.0.0
"""

import os
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AttentionFusionConfig:
    """注意力融合配置"""
    # 模态配置
    modality_dims: Dict[str, int] = None  # {"visual": 512, "audio": 256, ...}
    fusion_dim: int = 512
    
    # 注意力配置
    num_heads: int = 8
    dropout: float = 0.1
    
    # 融合策略
    fusion_type: str = "cross_attention"  # cross_attention, gated, hierarchical
    num_layers: int = 4
    
    def __post_init__(self):
        if self.modality_dims is None:
            self.modality_dims = {
                "visual": 512,
                "pointcloud": 256,
                "audio": 128,
                "thermal": 128,
                "timeseries": 64
            }


if TORCH_AVAILABLE:
    # =========================================================================
    # 注意力组件
    # =========================================================================
    class MultiHeadAttention(nn.Module):
        """多头注意力"""
        
        def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
            super().__init__()
            
            assert d_model % num_heads == 0
            
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = math.sqrt(self.d_k)
        
        def forward(self, query: torch.Tensor, key: torch.Tensor, 
                   value: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Args:
                query: (B, L_q, D)
                key: (B, L_k, D)
                value: (B, L_v, D)
                mask: (B, L_q, L_k) 可选掩码
            
            Returns:
                output: (B, L_q, D)
                attention: (B, H, L_q, L_k) 可选
            """
            B = query.size(0)
            
            # 线性投影
            Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
            
            # 注意力分数
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
            
            attention = F.softmax(scores, dim=-1)
            attention = self.dropout(attention)
            
            # 加权聚合
            context = torch.matmul(attention, V)
            context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)
            
            output = self.W_o(context)
            
            if return_attention:
                return output, attention
            return output, None


    class CrossModalAttention(nn.Module):
        """跨模态注意力"""
        
        def __init__(self, query_dim: int, key_dim: int, 
                     num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            
            self.query_proj = nn.Linear(query_dim, query_dim)
            self.key_proj = nn.Linear(key_dim, query_dim)
            self.value_proj = nn.Linear(key_dim, query_dim)
            
            self.attention = MultiHeadAttention(query_dim, num_heads, dropout)
            
            self.norm = nn.LayerNorm(query_dim)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, query_modality: torch.Tensor,
                   key_modality: torch.Tensor) -> torch.Tensor:
            """
            Args:
                query_modality: (B, L_q, D_q) 查询模态
                key_modality: (B, L_k, D_k) 键值模态
            
            Returns:
                fused: (B, L_q, D_q) 融合后的特征
            """
            # 投影
            Q = self.query_proj(query_modality)
            K = self.key_proj(key_modality)
            V = self.value_proj(key_modality)
            
            # 跨模态注意力
            attended, _ = self.attention(Q, K, V)
            
            # 残差连接
            output = self.norm(query_modality + self.dropout(attended))
            
            return output


    class GatedFusion(nn.Module):
        """门控融合"""
        
        def __init__(self, modality_dims: Dict[str, int], fusion_dim: int):
            super().__init__()
            
            self.modality_names = list(modality_dims.keys())
            
            # 投影层
            self.projections = nn.ModuleDict({
                name: nn.Linear(dim, fusion_dim)
                for name, dim in modality_dims.items()
            })
            
            # 门控网络
            total_dim = sum(modality_dims.values())
            self.gate_network = nn.Sequential(
                nn.Linear(total_dim, len(modality_dims) * 2),
                nn.ReLU(),
                nn.Linear(len(modality_dims) * 2, len(modality_dims)),
                nn.Sigmoid()
            )
            
            self.fusion_dim = fusion_dim
        
        def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
            """
            Args:
                modalities: {name: (B, D_i)} 各模态特征
            
            Returns:
                fused: (B, fusion_dim) 融合特征
            """
            # 计算门控权重
            concat_features = torch.cat(
                [modalities[name] for name in self.modality_names], dim=-1
            )
            gates = self.gate_network(concat_features)  # (B, M)
            
            # 投影并加权融合
            projected = []
            for i, name in enumerate(self.modality_names):
                proj = self.projections[name](modalities[name])  # (B, fusion_dim)
                gated = proj * gates[:, i:i+1]  # 门控
                projected.append(gated)
            
            # 求和融合
            fused = sum(projected)
            
            return fused


    class ModalityTokenFusion(nn.Module):
        """模态Token融合 (类似ViT)"""
        
        def __init__(self, modality_dims: Dict[str, int], fusion_dim: int,
                     num_heads: int = 8, num_layers: int = 4, dropout: float = 0.1):
            super().__init__()
            
            self.modality_names = list(modality_dims.keys())
            num_modalities = len(modality_dims)
            
            # 模态投影
            self.projections = nn.ModuleDict({
                name: nn.Linear(dim, fusion_dim)
                for name, dim in modality_dims.items()
            })
            
            # 可学习的模态token
            self.modality_tokens = nn.ParameterDict({
                name: nn.Parameter(torch.randn(1, 1, fusion_dim))
                for name in modality_dims.keys()
            })
            
            # CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, fusion_dim))
            
            # 位置编码
            self.position_embedding = nn.Parameter(
                torch.randn(1, num_modalities + 1, fusion_dim)
            )
            
            # Transformer层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            self.norm = nn.LayerNorm(fusion_dim)
        
        def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
            """
            Args:
                modalities: {name: (B, D_i)} 各模态特征
            
            Returns:
                fused: (B, fusion_dim) 融合特征
            """
            B = list(modalities.values())[0].size(0)
            
            # 投影各模态
            tokens = []
            for name in self.modality_names:
                feat = modalities[name]
                
                # 投影
                proj = self.projections[name](feat)  # (B, fusion_dim)
                
                # 添加模态token
                modality_token = self.modality_tokens[name].expand(B, -1, -1)
                proj = proj.unsqueeze(1) + modality_token  # (B, 1, fusion_dim)
                
                tokens.append(proj)
            
            # 添加CLS token
            cls_token = self.cls_token.expand(B, -1, -1)
            tokens.insert(0, cls_token)
            
            # 拼接
            sequence = torch.cat(tokens, dim=1)  # (B, M+1, fusion_dim)
            
            # 添加位置编码
            sequence = sequence + self.position_embedding[:, :sequence.size(1)]
            
            # Transformer
            output = self.transformer(sequence)
            output = self.norm(output)
            
            # 返回CLS token作为融合特征
            return output[:, 0]


    class HierarchicalFusion(nn.Module):
        """层次化融合"""
        
        def __init__(self, modality_dims: Dict[str, int], fusion_dim: int,
                     num_heads: int = 8, dropout: float = 0.1):
            super().__init__()
            
            # 定义融合层次
            # Level 1: 空间模态 (visual + pointcloud + thermal)
            # Level 2: 时序模态 (audio + timeseries)
            # Level 3: 全局融合
            
            self.spatial_modalities = ["visual", "pointcloud", "thermal"]
            self.temporal_modalities = ["audio", "timeseries"]
            
            # 投影层
            self.projections = nn.ModuleDict({
                name: nn.Linear(dim, fusion_dim)
                for name, dim in modality_dims.items()
            })
            
            # Level 1: 空间融合
            self.spatial_attention = CrossModalAttention(
                fusion_dim, fusion_dim, num_heads, dropout
            )
            self.spatial_fusion = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU()
            )
            
            # Level 2: 时序融合
            self.temporal_attention = CrossModalAttention(
                fusion_dim, fusion_dim, num_heads, dropout
            )
            self.temporal_fusion = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU()
            )
            
            # Level 3: 全局融合
            self.global_attention = MultiHeadAttention(fusion_dim, num_heads, dropout)
            self.global_fusion = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU()
            )
        
        def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
            """
            Args:
                modalities: {name: (B, D_i)} 各模态特征
            
            Returns:
                fused: (B, fusion_dim) 融合特征
            """
            # 投影所有模态
            projected = {
                name: self.projections[name](modalities[name]).unsqueeze(1)
                for name in modalities.keys()
            }
            
            # Level 1: 空间模态融合
            spatial_feats = []
            for name in self.spatial_modalities:
                if name in projected:
                    spatial_feats.append(projected[name])
            
            if spatial_feats:
                # 跨模态注意力
                spatial_concat = torch.cat(spatial_feats, dim=1)
                spatial_attended = self.spatial_attention(
                    spatial_concat, spatial_concat
                )
                spatial_fused = self.spatial_fusion(
                    spatial_attended.view(spatial_attended.size(0), -1)
                )
            else:
                spatial_fused = torch.zeros(
                    list(modalities.values())[0].size(0), 
                    self.spatial_fusion[0].out_features,
                    device=list(modalities.values())[0].device
                )
            
            # Level 2: 时序模态融合
            temporal_feats = []
            for name in self.temporal_modalities:
                if name in projected:
                    temporal_feats.append(projected[name])
            
            if temporal_feats:
                temporal_concat = torch.cat(temporal_feats, dim=1)
                temporal_attended = self.temporal_attention(
                    temporal_concat, temporal_concat
                )
                temporal_fused = self.temporal_fusion(
                    temporal_attended.view(temporal_attended.size(0), -1)
                )
            else:
                temporal_fused = torch.zeros_like(spatial_fused)
            
            # Level 3: 全局融合
            global_input = torch.stack([spatial_fused, temporal_fused], dim=1)
            global_attended, _ = self.global_attention(
                global_input, global_input, global_input
            )
            
            global_fused = self.global_fusion(
                global_attended.view(global_attended.size(0), -1)
            )
            
            return global_fused


    class DynamicModalityWeighting(nn.Module):
        """动态模态权重"""
        
        def __init__(self, modality_dims: Dict[str, int], fusion_dim: int):
            super().__init__()
            
            self.modality_names = list(modality_dims.keys())
            num_modalities = len(modality_dims)
            
            # 投影层
            self.projections = nn.ModuleDict({
                name: nn.Linear(dim, fusion_dim)
                for name, dim in modality_dims.items()
            })
            
            # 质量评估网络 (为每个模态评估其可靠性)
            self.quality_networks = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                for name, dim in modality_dims.items()
            })
            
            # 全局权重预测
            self.global_weight_predictor = nn.Sequential(
                nn.Linear(num_modalities, num_modalities * 2),
                nn.ReLU(),
                nn.Linear(num_modalities * 2, num_modalities),
                nn.Softmax(dim=-1)
            )
        
        def forward(self, modalities: Dict[str, torch.Tensor],
                   return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Args:
                modalities: {name: (B, D_i)} 各模态特征
            
            Returns:
                fused: (B, fusion_dim) 融合特征
                weights: (B, M) 模态权重 (可选)
            """
            B = list(modalities.values())[0].size(0)
            
            # 评估各模态质量
            qualities = []
            for name in self.modality_names:
                q = self.quality_networks[name](modalities[name])  # (B, 1)
                qualities.append(q)
            
            qualities = torch.cat(qualities, dim=-1)  # (B, M)
            
            # 计算动态权重
            weights = self.global_weight_predictor(qualities)  # (B, M)
            
            # 加权融合
            fused = torch.zeros(B, list(self.projections.values())[0].out_features,
                               device=qualities.device)
            
            for i, name in enumerate(self.modality_names):
                proj = self.projections[name](modalities[name])  # (B, fusion_dim)
                fused = fused + proj * weights[:, i:i+1]
            
            if return_weights:
                return fused, weights
            return fused, None


    # =========================================================================
    # 完整的注意力融合网络
    # =========================================================================
    class AttentionFusionNetwork(nn.Module):
        """完整的注意力融合网络"""
        
        def __init__(self, config: AttentionFusionConfig):
            super().__init__()
            
            self.config = config
            self.modality_names = list(config.modality_dims.keys())
            
            # 选择融合类型
            if config.fusion_type == "cross_attention":
                self.fusion = ModalityTokenFusion(
                    config.modality_dims, config.fusion_dim,
                    config.num_heads, config.num_layers, config.dropout
                )
            elif config.fusion_type == "gated":
                self.fusion = GatedFusion(config.modality_dims, config.fusion_dim)
            elif config.fusion_type == "hierarchical":
                self.fusion = HierarchicalFusion(
                    config.modality_dims, config.fusion_dim,
                    config.num_heads, config.dropout
                )
            elif config.fusion_type == "dynamic":
                self.fusion = DynamicModalityWeighting(
                    config.modality_dims, config.fusion_dim
                )
            else:
                self.fusion = ModalityTokenFusion(
                    config.modality_dims, config.fusion_dim,
                    config.num_heads, config.num_layers, config.dropout
                )
            
            # 输出头
            self.classifier = nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_dim // 2, 5)  # 5类故障
            )
            
            self.health_predictor = nn.Sequential(
                nn.Linear(config.fusion_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, modalities: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """
            Args:
                modalities: {name: (B, D_i)} 各模态特征
            
            Returns:
                outputs: {
                    "fused": (B, fusion_dim) 融合特征
                    "logits": (B, num_classes) 分类logits
                    "health_score": (B, 1) 健康评分
                }
            """
            # 融合
            if isinstance(self.fusion, DynamicModalityWeighting):
                fused, weights = self.fusion(modalities, return_weights=True)
            else:
                fused = self.fusion(modalities)
                weights = None
            
            # 分类
            logits = self.classifier(fused)
            
            # 健康评分
            health_score = self.health_predictor(fused) * 100
            
            outputs = {
                "fused": fused,
                "logits": logits,
                "health_score": health_score
            }
            
            if weights is not None:
                outputs["modality_weights"] = weights
            
            return outputs


# =============================================================================
# 入口函数
# =============================================================================
def create_attention_fusion(config: Optional[AttentionFusionConfig] = None) -> 'AttentionFusionNetwork':
    """创建注意力融合网络"""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch未安装")
        return None
    
    if config is None:
        config = AttentionFusionConfig()
    
    return AttentionFusionNetwork(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if TORCH_AVAILABLE:
        # 示例
        config = AttentionFusionConfig(
            modality_dims={
                "visual": 512,
                "pointcloud": 256,
                "audio": 128,
                "thermal": 128,
                "timeseries": 64
            },
            fusion_dim=512,
            num_heads=8,
            fusion_type="cross_attention"
        )
        
        model = create_attention_fusion(config)
        
        # 测试
        batch_size = 4
        modalities = {
            "visual": torch.randn(batch_size, 512),
            "pointcloud": torch.randn(batch_size, 256),
            "audio": torch.randn(batch_size, 128),
            "thermal": torch.randn(batch_size, 128),
            "timeseries": torch.randn(batch_size, 64)
        }
        
        outputs = model(modalities)
        
        print(f"融合特征形状: {outputs['fused'].shape}")
        print(f"分类logits形状: {outputs['logits'].shape}")
        print(f"健康评分形状: {outputs['health_score'].shape}")
