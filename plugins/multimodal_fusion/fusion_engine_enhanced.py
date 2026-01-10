# -*- coding: utf-8 -*-
"""
增强版多模态融合引擎
====================

实现方案要求的功能：
1. 基于Transformer的cross-attention模块
2. 贝叶斯网络决策级关联分析
3. 运行时策略切换接口
4. 动态模态权重优化

作者: AI巡检系统
版本: 2.0.0
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import time
import json

logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构定义
# =============================================================================

class FusionStrategy(str, Enum):
    """融合策略枚举"""
    EARLY = "early"           # 早期融合(特征级)
    LATE = "late"             # 晚期融合(决策级)
    ATTENTION = "attention"   # 注意力融合
    HYBRID = "hybrid"         # 混合融合
    BAYESIAN = "bayesian"     # 贝叶斯决策融合


@dataclass
class ModalityData:
    """模态数据结构"""
    modality_type: str
    features: np.ndarray
    confidence: float
    detections: List[Dict]
    metadata: Dict = field(default_factory=dict)
    timestamp: float = 0.0
    quality_score: float = 1.0


@dataclass
class FusionResult:
    """融合结果"""
    success: bool
    overall_status: str
    confidence: float
    fused_detections: List[Dict]
    modality_contributions: Dict[str, float]
    diagnostic_report: Dict
    recommendations: List[str]
    fault_chain: List[Dict] = field(default_factory=list)
    bayesian_inference: Optional[Dict] = None


# =============================================================================
# Transformer Cross-Attention 模块
# =============================================================================

class CrossAttentionLayer:
    """
    跨模态注意力层
    
    实现多模态特征的交叉注意力机制，学习模态间的相关性
    """
    
    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout比率
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # 初始化权重矩阵 (使用Xavier初始化)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        
        # LayerNorm参数
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """数值稳定的softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer Normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + eps) + self.beta
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        前向传播
        
        Args:
            query: 查询矩阵 [batch, seq_q, d_model]
            key: 键矩阵 [batch, seq_k, d_model]
            value: 值矩阵 [batch, seq_v, d_model]
            mask: 注意力掩码
            
        Returns:
            output: 输出 [batch, seq_q, d_model]
            attention_weights: 注意力权重 [batch, n_heads, seq_q, seq_k]
        """
        batch_size = query.shape[0] if len(query.shape) == 3 else 1
        if len(query.shape) == 2:
            query = query.reshape(1, *query.shape)
            key = key.reshape(1, *key.shape)
            value = value.reshape(1, *value.shape)
        
        # 线性变换
        Q = np.dot(query, self.W_q)  # [batch, seq_q, d_model]
        K = np.dot(key, self.W_k)    # [batch, seq_k, d_model]
        V = np.dot(value, self.W_v)  # [batch, seq_v, d_model]
        
        # 分头
        seq_q, seq_k = Q.shape[1], K.shape[1]
        Q = Q.reshape(batch_size, seq_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 计算注意力分数
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask * (-1e9)
        
        attention_weights = self._softmax(scores, axis=-1)
        
        # 应用Dropout (推理时跳过)
        # attention_weights = self._dropout(attention_weights)
        
        # 加权求和
        context = np.matmul(attention_weights, V)  # [batch, n_heads, seq_q, d_k]
        
        # 合并头
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_q, self.d_model)
        
        # 输出投影
        output = np.dot(context, self.W_o)
        
        # 残差连接 + LayerNorm
        output = self._layer_norm(output + query)
        
        return output.squeeze(0) if batch_size == 1 else output, attention_weights


class TransformerFusionBlock:
    """
    Transformer融合块
    
    包含自注意力和跨模态注意力
    """
    
    def __init__(self, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: FFN隐藏层维度
        """
        self.d_model = d_model
        self.self_attention = CrossAttentionLayer(d_model, n_heads)
        self.cross_attention = CrossAttentionLayer(d_model, n_heads)
        
        # FFN层
        scale = np.sqrt(2.0 / (d_model + d_ff))
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)
        
        # LayerNorm
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU激活函数"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer Normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + eps) + self.beta
    
    def forward(self, x: np.ndarray, context: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        前向传播
        
        Args:
            x: 输入特征 [seq, d_model]
            context: 上下文特征 [seq_ctx, d_model]
            
        Returns:
            output: 输出特征
            attention_maps: 注意力图
        """
        attention_maps = {}
        
        # 自注意力
        x, self_attn = self.self_attention.forward(x, x, x)
        attention_maps['self_attention'] = self_attn
        
        # 跨模态注意力
        x, cross_attn = self.cross_attention.forward(x, context, context)
        attention_maps['cross_attention'] = cross_attn
        
        # FFN
        ffn_out = np.dot(x, self.W1) + self.b1
        ffn_out = self._gelu(ffn_out)
        ffn_out = np.dot(ffn_out, self.W2) + self.b2
        
        # 残差连接 + LayerNorm
        output = self._layer_norm(x + ffn_out)
        
        return output, attention_maps


# =============================================================================
# 贝叶斯网络决策融合
# =============================================================================

class BayesianDecisionNetwork:
    """
    贝叶斯网络决策级关联分析
    
    利用贝叶斯推断对不同插件的告警结果进行逻辑推断，
    识别跨模态故障链
    """
    
    def __init__(self):
        # 先验概率 P(Fault)
        self.prior_fault_prob = {
            'transformer_fault': 0.01,
            'switch_fault': 0.008,
            'insulation_fault': 0.015,
            'connection_fault': 0.012,
            'cooling_fault': 0.005,
            'gas_leak': 0.008,
            'partial_discharge': 0.02,
            'overheating': 0.018,
            'mechanical_fault': 0.01,
            'normal': 0.90
        }
        
        # 条件概率 P(Evidence|Fault)
        # evidence: {modality: {detection_type: probability}}
        self.likelihood = self._init_likelihood_table()
        
        # 故障依赖关系图 (用于故障链推断)
        self.fault_dependencies = self._init_fault_dependencies()
        
    def _init_likelihood_table(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """初始化似然表"""
        return {
            'transformer_fault': {
                'visual': {'oil_leak': 0.7, 'surface_damage': 0.5, 'discoloration': 0.6},
                'thermal': {'hotspot': 0.8, 'temperature_rise': 0.7},
                'acoustic': {'abnormal_sound': 0.6, 'partial_discharge': 0.4},
                'gas': {'H2_high': 0.85, 'CO_high': 0.7, 'C2H2_high': 0.9},
            },
            'insulation_fault': {
                'visual': {'crack': 0.6, 'contamination': 0.5},
                'thermal': {'hotspot': 0.7},
                'acoustic': {'partial_discharge': 0.9, 'corona': 0.8},
                'hyperspectral': {'aging': 0.85, 'moisture': 0.7},
            },
            'gas_leak': {
                'gas': {'SF6_low': 0.95, 'pressure_drop': 0.9},
                'visual': {'frost': 0.6, 'oil_stain': 0.5},
                'acoustic': {'hissing': 0.8},
            },
            'partial_discharge': {
                'acoustic': {'ultrasonic': 0.9, 'pd_pattern': 0.85},
                'thermal': {'local_heating': 0.6},
                'visual': {'treeing': 0.7, 'tracking': 0.75},
            },
            'overheating': {
                'thermal': {'temperature_high': 0.95, 'hotspot': 0.9},
                'visual': {'discoloration': 0.7, 'deformation': 0.6},
                'gas': {'CO_high': 0.6, 'H2_high': 0.5},
            },
            'normal': {
                'visual': {'normal': 0.95},
                'thermal': {'normal': 0.95},
                'acoustic': {'normal': 0.95},
                'gas': {'normal': 0.95},
                'hyperspectral': {'normal': 0.95},
            }
        }
    
    def _init_fault_dependencies(self) -> Dict[str, List[str]]:
        """初始化故障依赖关系"""
        return {
            'transformer_fault': ['overheating', 'insulation_fault', 'gas_leak'],
            'insulation_fault': ['partial_discharge', 'overheating'],
            'partial_discharge': ['insulation_fault'],
            'overheating': ['transformer_fault', 'connection_fault'],
            'gas_leak': ['mechanical_fault'],
            'connection_fault': ['overheating'],
        }
    
    def compute_posterior(self, evidences: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        计算后验概率 P(Fault|Evidence) 使用贝叶斯定理
        
        Args:
            evidences: 各模态的检测证据 {modality: {detection_type: confidence}}
            
        Returns:
            posterior: 各故障类型的后验概率
        """
        posteriors = {}
        
        for fault_type, prior in self.prior_fault_prob.items():
            # 计算似然 P(Evidence|Fault)
            likelihood = 1.0
            for modality, detections in evidences.items():
                if fault_type in self.likelihood and modality in self.likelihood[fault_type]:
                    fault_likelihoods = self.likelihood[fault_type][modality]
                    for detection, confidence in detections.items():
                        if detection in fault_likelihoods:
                            # 使用置信度加权的似然
                            likelihood *= fault_likelihoods[detection] ** confidence
                        else:
                            # 未知检测类型，给予中性似然
                            likelihood *= 0.5
                else:
                    # 该故障类型不产生该模态的证据
                    likelihood *= 0.1
            
            # 贝叶斯公式: P(Fault|E) ∝ P(E|Fault) * P(Fault)
            posteriors[fault_type] = likelihood * prior
        
        # 归一化
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}
        
        return posteriors
    
    def infer_fault_chain(self, posteriors: Dict[str, float], 
                          threshold: float = 0.1) -> List[Dict]:
        """
        推断故障链
        
        Args:
            posteriors: 各故障类型的后验概率
            threshold: 概率阈值
            
        Returns:
            fault_chain: 故障链列表
        """
        fault_chain = []
        
        # 找出高概率故障
        significant_faults = {k: v for k, v in posteriors.items() 
                           if v >= threshold and k != 'normal'}
        
        if not significant_faults:
            return fault_chain
        
        # 按概率排序
        sorted_faults = sorted(significant_faults.items(), key=lambda x: x[1], reverse=True)
        
        # 构建故障链
        visited = set()
        for fault, prob in sorted_faults:
            if fault in visited:
                continue
            
            chain = {
                'root_fault': fault,
                'probability': prob,
                'related_faults': [],
                'propagation_path': []
            }
            
            # 查找依赖故障
            if fault in self.fault_dependencies:
                for related_fault in self.fault_dependencies[fault]:
                    if related_fault in significant_faults:
                        chain['related_faults'].append({
                            'fault': related_fault,
                            'probability': posteriors.get(related_fault, 0)
                        })
                        chain['propagation_path'].append(f"{fault} -> {related_fault}")
                        visited.add(related_fault)
            
            visited.add(fault)
            fault_chain.append(chain)
        
        return fault_chain


# =============================================================================
# 增强版多模态注意力融合
# =============================================================================

class EnhancedAttentionFusion:
    """
    增强版注意力融合机制
    
    结合Transformer cross-attention和动态权重学习
    """
    
    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int = 256, n_heads: int = 8):
        """
        Args:
            modality_dims: 各模态的特征维度
            hidden_dim: 隐藏层维度
            n_heads: 注意力头数
        """
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # 特征投影层 (将各模态特征投影到统一维度)
        self.projections = {}
        for modality, dim in modality_dims.items():
            scale = np.sqrt(2.0 / (dim + hidden_dim))
            self.projections[modality] = {
                'W': np.random.randn(dim, hidden_dim) * scale,
                'b': np.zeros(hidden_dim)
            }
        
        # Transformer融合块
        self.fusion_block = TransformerFusionBlock(hidden_dim, n_heads)
        
        # 动态权重网络
        self.weight_network = {
            'W1': np.random.randn(hidden_dim * len(modality_dims), hidden_dim) * 0.01,
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, len(modality_dims)) * 0.01,
            'b2': np.zeros(len(modality_dims))
        }
        
        # 模态名称映射
        self.modality_names = list(modality_dims.keys())
        
        # 历史权重记录
        self.weight_history = defaultdict(list)
        
    def _project_features(self, modality: str, features: np.ndarray) -> np.ndarray:
        """将模态特征投影到统一维度"""
        if modality not in self.projections:
            # 动态创建投影
            dim = features.shape[-1]
            scale = np.sqrt(2.0 / (dim + self.hidden_dim))
            self.projections[modality] = {
                'W': np.random.randn(dim, self.hidden_dim) * scale,
                'b': np.zeros(self.hidden_dim)
            }
        
        proj = self.projections[modality]
        return np.dot(features, proj['W']) + proj['b']
    
    def _compute_dynamic_weights(self, projected_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算动态模态权重"""
        # 将所有特征拼接
        concat_features = []
        for modality in self.modality_names:
            if modality in projected_features:
                feat = projected_features[modality]
                if len(feat.shape) > 1:
                    feat = np.mean(feat, axis=0)
                concat_features.append(feat)
            else:
                concat_features.append(np.zeros(self.hidden_dim))
        
        concat = np.concatenate(concat_features)
        
        # 通过权重网络
        h = np.dot(concat, self.weight_network['W1']) + self.weight_network['b1']
        h = np.maximum(0, h)  # ReLU
        logits = np.dot(h, self.weight_network['W2']) + self.weight_network['b2']
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        weights = exp_logits / np.sum(exp_logits)
        
        return {self.modality_names[i]: float(weights[i]) for i in range(len(self.modality_names))}
    
    def fuse(self, modality_data: Dict[str, ModalityData]) -> Tuple[np.ndarray, Dict[str, float], Dict]:
        """
        执行注意力融合
        
        Args:
            modality_data: 各模态数据
            
        Returns:
            fused_features: 融合后的特征
            weights: 模态权重
            attention_info: 注意力信息
        """
        # 特征投影
        projected = {}
        for modality, data in modality_data.items():
            if data.features is not None and len(data.features) > 0:
                projected[modality] = self._project_features(modality, data.features)
        
        if not projected:
            return np.zeros(self.hidden_dim), {}, {}
        
        # 计算动态权重
        weights = self._compute_dynamic_weights(projected)
        
        # 更新历史
        for modality, weight in weights.items():
            self.weight_history[modality].append(weight)
            if len(self.weight_history[modality]) > 100:
                self.weight_history[modality].pop(0)
        
        # 构建特征序列 (将各模态特征视为序列tokens)
        feature_list = []
        modality_order = []
        for modality, feat in projected.items():
            if len(feat.shape) == 1:
                feat = feat.reshape(1, -1)
            feature_list.append(feat)
            modality_order.append(modality)
        
        if len(feature_list) > 1:
            # 多模态融合
            all_features = np.concatenate(feature_list, axis=0)
            
            # Transformer融合
            attention_info = {}
            fused = all_features.copy()
            
            # 对每个模态应用跨模态注意力
            for i, (modality, feat) in enumerate(zip(modality_order, feature_list)):
                # 其他模态作为context
                context_idx = [j for j in range(len(feature_list)) if j != i]
                if context_idx:
                    context = np.concatenate([feature_list[j] for j in context_idx], axis=0)
                    fused_feat, attn_maps = self.fusion_block.forward(feat, context)
                    attention_info[modality] = attn_maps
                    
                    # 加权融合
                    start_idx = sum(f.shape[0] for f in feature_list[:i])
                    end_idx = start_idx + feat.shape[0]
                    fused[start_idx:end_idx] = fused_feat * weights.get(modality, 1.0)
            
            # 全局聚合
            fused_features = np.mean(fused, axis=0)
        else:
            # 单模态
            fused_features = np.mean(feature_list[0], axis=0)
            attention_info = {}
        
        return fused_features, weights, attention_info


# =============================================================================
# 融合策略切换接口
# =============================================================================

class FusionStrategyManager:
    """
    融合策略管理器
    
    支持运行时策略切换和自动策略选择
    """
    
    def __init__(self):
        self.current_strategy = FusionStrategy.HYBRID
        self.strategy_history = []
        
        # 策略性能统计
        self.strategy_performance = {
            strategy: {
                'accuracy': 0.0,
                'latency_ms': 0.0,
                'count': 0
            } for strategy in FusionStrategy
        }
        
        # 设备类型到策略的映射
        self.device_strategy_map = {
            'transformer': FusionStrategy.HYBRID,
            'switch': FusionStrategy.LATE,
            'busbar': FusionStrategy.ATTENTION,
            'capacitor': FusionStrategy.EARLY,
            'cable': FusionStrategy.BAYESIAN,
        }
        
        # 传感器可用性到策略的映射
        self.sensor_strategy_map = {
            frozenset(['visual']): FusionStrategy.EARLY,
            frozenset(['visual', 'thermal']): FusionStrategy.LATE,
            frozenset(['visual', 'acoustic']): FusionStrategy.ATTENTION,
            frozenset(['visual', 'thermal', 'acoustic', 'gas']): FusionStrategy.HYBRID,
        }
    
    def switch_strategy(self, strategy: FusionStrategy) -> bool:
        """
        切换融合策略
        
        Args:
            strategy: 目标策略
            
        Returns:
            success: 是否成功
        """
        old_strategy = self.current_strategy
        self.current_strategy = strategy
        
        self.strategy_history.append({
            'from': old_strategy.value,
            'to': strategy.value,
            'timestamp': time.time()
        })
        
        logger.info(f"融合策略切换: {old_strategy.value} -> {strategy.value}")
        return True
    
    def auto_select_strategy(self, 
                            device_type: Optional[str] = None,
                            available_sensors: Optional[List[str]] = None) -> FusionStrategy:
        """
        自动选择最佳策略
        
        Args:
            device_type: 设备类型
            available_sensors: 可用传感器列表
            
        Returns:
            strategy: 选择的策略
        """
        # 优先根据传感器可用性选择
        if available_sensors:
            sensor_set = frozenset(available_sensors)
            for sensors, strategy in self.sensor_strategy_map.items():
                if sensors <= sensor_set:
                    return strategy
        
        # 其次根据设备类型选择
        if device_type and device_type in self.device_strategy_map:
            return self.device_strategy_map[device_type]
        
        # 默认使用混合策略
        return FusionStrategy.HYBRID
    
    def update_performance(self, strategy: FusionStrategy, 
                          accuracy: float, latency_ms: float):
        """更新策略性能统计"""
        perf = self.strategy_performance[strategy]
        n = perf['count']
        
        # 增量更新平均值
        perf['accuracy'] = (perf['accuracy'] * n + accuracy) / (n + 1)
        perf['latency_ms'] = (perf['latency_ms'] * n + latency_ms) / (n + 1)
        perf['count'] = n + 1
    
    def get_best_strategy(self) -> FusionStrategy:
        """获取历史最佳策略"""
        best = None
        best_score = -1
        
        for strategy, perf in self.strategy_performance.items():
            if perf['count'] < 10:
                continue
            # 综合考虑准确率和延迟
            score = perf['accuracy'] - 0.01 * perf['latency_ms']
            if score > best_score:
                best_score = score
                best = strategy
        
        return best or FusionStrategy.HYBRID


# =============================================================================
# 增强版融合引擎主类
# =============================================================================

class EnhancedFusionEngine:
    """
    增强版多模态融合引擎
    
    整合所有融合功能：
    - Transformer cross-attention
    - 贝叶斯决策融合
    - 策略切换
    - 动态权重
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 融合配置
        """
        self.config = config or {}
        
        # 初始化各组件
        modality_dims = self.config.get('modality_dims', {
            'visual': 512,
            'thermal': 256,
            'acoustic': 128,
            'gas': 64,
            'hyperspectral': 224
        })
        
        self.attention_fusion = EnhancedAttentionFusion(
            modality_dims=modality_dims,
            hidden_dim=self.config.get('hidden_dim', 256),
            n_heads=self.config.get('n_heads', 8)
        )
        
        self.bayesian_network = BayesianDecisionNetwork()
        self.strategy_manager = FusionStrategyManager()
        
        # 模态权重 (用于传统融合)
        self.modality_weights = self.config.get('modality_weights', {
            'visual': 0.25,
            'thermal': 0.15,
            'acoustic': 0.20,
            'gas': 0.20,
            'hyperspectral': 0.20
        })
        
        # 诊断规则
        self.diagnostic_rules = self._init_diagnostic_rules()
        
        logger.info(f"EnhancedFusionEngine 初始化完成, 策略: {self.strategy_manager.current_strategy.value}")
    
    def _init_diagnostic_rules(self) -> List[Dict]:
        """初始化诊断规则"""
        return [
            {
                'name': '变压器综合过热诊断',
                'conditions': [
                    {'modality': 'thermal', 'detection': 'hotspot', 'min_confidence': 0.7},
                    {'modality': 'gas', 'detection': 'H2_high', 'min_confidence': 0.6}
                ],
                'min_match': 2,
                'diagnosis': '变压器内部过热，建议立即停机检查',
                'priority': 'critical',
                'fault_type': 'transformer_fault'
            },
            {
                'name': '绝缘劣化诊断',
                'conditions': [
                    {'modality': 'acoustic', 'detection': 'partial_discharge', 'min_confidence': 0.7},
                    {'modality': 'hyperspectral', 'detection': 'aging', 'min_confidence': 0.6}
                ],
                'min_match': 2,
                'diagnosis': '绝缘系统劣化，存在局放风险',
                'priority': 'alarm',
                'fault_type': 'insulation_fault'
            },
            {
                'name': 'SF6泄漏诊断',
                'conditions': [
                    {'modality': 'gas', 'detection': 'SF6_low', 'min_confidence': 0.8},
                    {'modality': 'acoustic', 'detection': 'hissing', 'min_confidence': 0.5}
                ],
                'min_match': 1,
                'diagnosis': 'SF6气体泄漏，需要检漏补气',
                'priority': 'warning',
                'fault_type': 'gas_leak'
            },
            {
                'name': '局部放电检测',
                'conditions': [
                    {'modality': 'acoustic', 'detection': 'ultrasonic', 'min_confidence': 0.8},
                    {'modality': 'thermal', 'detection': 'local_heating', 'min_confidence': 0.6}
                ],
                'min_match': 1,
                'diagnosis': '检测到局部放电，需进一步定位',
                'priority': 'warning',
                'fault_type': 'partial_discharge'
            }
        ]
    
    def set_strategy(self, strategy: str) -> bool:
        """设置融合策略"""
        try:
            fusion_strategy = FusionStrategy(strategy)
            return self.strategy_manager.switch_strategy(fusion_strategy)
        except ValueError:
            logger.error(f"无效的融合策略: {strategy}")
            return False
    
    def fuse(self, modality_data: Dict[str, ModalityData]) -> FusionResult:
        """
        执行多模态融合
        
        Args:
            modality_data: 各模态数据
            
        Returns:
            result: 融合结果
        """
        start_time = time.time()
        
        strategy = self.strategy_manager.current_strategy
        
        try:
            if strategy == FusionStrategy.EARLY:
                result = self._early_fusion(modality_data)
            elif strategy == FusionStrategy.LATE:
                result = self._late_fusion(modality_data)
            elif strategy == FusionStrategy.ATTENTION:
                result = self._attention_fusion(modality_data)
            elif strategy == FusionStrategy.BAYESIAN:
                result = self._bayesian_fusion(modality_data)
            else:  # HYBRID
                result = self._hybrid_fusion(modality_data)
            
            # 更新性能统计
            latency = (time.time() - start_time) * 1000
            self.strategy_manager.update_performance(
                strategy, 
                result.confidence, 
                latency
            )
            
            return result
            
        except Exception as e:
            logger.error(f"融合失败: {e}")
            return FusionResult(
                success=False,
                overall_status='error',
                confidence=0.0,
                fused_detections=[],
                modality_contributions={},
                diagnostic_report={'error': str(e)},
                recommendations=['融合处理失败，请检查输入数据']
            )
    
    def _early_fusion(self, modality_data: Dict[str, ModalityData]) -> FusionResult:
        """早期融合 (特征级)"""
        # 收集所有特征
        all_features = []
        contributions = {}
        
        for modality, data in modality_data.items():
            if data.features is not None and len(data.features) > 0:
                weight = self.modality_weights.get(modality, 0.1)
                weighted_feat = data.features * weight * data.quality_score
                all_features.append(weighted_feat)
                contributions[modality] = weight * data.confidence
        
        if not all_features:
            return self._empty_result()
        
        # 特征拼接
        fused_features = np.concatenate([f.flatten() for f in all_features])
        
        # 基于特征进行简单分类
        overall_status, confidence = self._classify_from_features(fused_features)
        
        return FusionResult(
            success=True,
            overall_status=overall_status,
            confidence=confidence,
            fused_detections=self._collect_detections(modality_data),
            modality_contributions=contributions,
            diagnostic_report={'method': 'early_fusion'},
            recommendations=self._generate_recommendations(overall_status)
        )
    
    def _late_fusion(self, modality_data: Dict[str, ModalityData]) -> FusionResult:
        """晚期融合 (决策级)"""
        # 收集各模态的决策
        decisions = {}
        contributions = {}
        
        for modality, data in modality_data.items():
            if data.detections:
                # 找出最高置信度的检测
                max_detection = max(data.detections, key=lambda x: x.get('confidence', 0))
                decisions[modality] = {
                    'status': max_detection.get('status', 'normal'),
                    'confidence': max_detection.get('confidence', data.confidence)
                }
                contributions[modality] = data.confidence * self.modality_weights.get(modality, 0.1)
        
        # 加权投票
        status_scores = defaultdict(float)
        for modality, decision in decisions.items():
            weight = self.modality_weights.get(modality, 0.1)
            status_scores[decision['status']] += weight * decision['confidence']
        
        if status_scores:
            overall_status = max(status_scores.items(), key=lambda x: x[1])[0]
            confidence = status_scores[overall_status] / sum(status_scores.values())
        else:
            overall_status = 'normal'
            confidence = 0.5
        
        return FusionResult(
            success=True,
            overall_status=overall_status,
            confidence=confidence,
            fused_detections=self._collect_detections(modality_data),
            modality_contributions=contributions,
            diagnostic_report={'method': 'late_fusion', 'decisions': decisions},
            recommendations=self._generate_recommendations(overall_status)
        )
    
    def _attention_fusion(self, modality_data: Dict[str, ModalityData]) -> FusionResult:
        """注意力融合"""
        # 使用增强版注意力融合
        fused_features, weights, attention_info = self.attention_fusion.fuse(modality_data)
        
        # 基于融合特征分类
        overall_status, confidence = self._classify_from_features(fused_features)
        
        # 应用诊断规则
        rule_matches = self._apply_diagnostic_rules(modality_data)
        if rule_matches:
            # 使用规则匹配的最高优先级
            priority_order = {'critical': 4, 'alarm': 3, 'warning': 2, 'attention': 1, 'normal': 0}
            best_match = max(rule_matches, key=lambda x: priority_order.get(x['priority'], 0))
            if priority_order.get(best_match['priority'], 0) > priority_order.get(overall_status, 0):
                overall_status = best_match['priority']
        
        return FusionResult(
            success=True,
            overall_status=overall_status,
            confidence=confidence,
            fused_detections=self._collect_detections(modality_data),
            modality_contributions=weights,
            diagnostic_report={
                'method': 'attention_fusion',
                'attention_info': {k: 'computed' for k in attention_info.keys()},
                'rule_matches': rule_matches
            },
            recommendations=self._generate_recommendations(overall_status, rule_matches)
        )
    
    def _bayesian_fusion(self, modality_data: Dict[str, ModalityData]) -> FusionResult:
        """贝叶斯决策融合"""
        # 收集证据
        evidences = {}
        for modality, data in modality_data.items():
            evidences[modality] = {}
            for detection in data.detections:
                det_type = detection.get('type', 'unknown')
                det_conf = detection.get('confidence', data.confidence)
                evidences[modality][det_type] = det_conf
        
        # 贝叶斯推断
        posteriors = self.bayesian_network.compute_posterior(evidences)
        fault_chain = self.bayesian_network.infer_fault_chain(posteriors)
        
        # 确定整体状态
        max_fault = max(posteriors.items(), key=lambda x: x[1] if x[0] != 'normal' else 0)
        if max_fault[0] == 'normal' or max_fault[1] < 0.3:
            overall_status = 'normal'
            confidence = posteriors.get('normal', 0.5)
        else:
            # 根据后验概率确定状态级别
            if max_fault[1] >= 0.7:
                overall_status = 'critical'
            elif max_fault[1] >= 0.5:
                overall_status = 'alarm'
            elif max_fault[1] >= 0.3:
                overall_status = 'warning'
            else:
                overall_status = 'attention'
            confidence = max_fault[1]
        
        # 计算贡献度
        contributions = {}
        for modality, data in modality_data.items():
            contributions[modality] = data.confidence * self.modality_weights.get(modality, 0.1)
        
        return FusionResult(
            success=True,
            overall_status=overall_status,
            confidence=confidence,
            fused_detections=self._collect_detections(modality_data),
            modality_contributions=contributions,
            diagnostic_report={
                'method': 'bayesian_fusion',
                'posteriors': posteriors,
                'most_likely_fault': max_fault[0]
            },
            recommendations=self._generate_recommendations(overall_status, fault_chain=fault_chain),
            fault_chain=fault_chain,
            bayesian_inference={'posteriors': posteriors}
        )
    
    def _hybrid_fusion(self, modality_data: Dict[str, ModalityData]) -> FusionResult:
        """混合融合"""
        # 同时执行注意力融合和贝叶斯融合
        attention_result = self._attention_fusion(modality_data)
        bayesian_result = self._bayesian_fusion(modality_data)
        
        # 综合两种方法的结果
        priority_order = {'critical': 4, 'alarm': 3, 'warning': 2, 'attention': 1, 'normal': 0}
        
        attn_priority = priority_order.get(attention_result.overall_status, 0)
        bayes_priority = priority_order.get(bayesian_result.overall_status, 0)
        
        if bayes_priority > attn_priority:
            overall_status = bayesian_result.overall_status
            confidence = (attention_result.confidence + bayesian_result.confidence * 1.2) / 2.2
        else:
            overall_status = attention_result.overall_status
            confidence = (attention_result.confidence * 1.2 + bayesian_result.confidence) / 2.2
        
        # 合并贡献度
        contributions = {}
        for modality in set(attention_result.modality_contributions.keys()) | set(bayesian_result.modality_contributions.keys()):
            attn_contrib = attention_result.modality_contributions.get(modality, 0)
            bayes_contrib = bayesian_result.modality_contributions.get(modality, 0)
            contributions[modality] = (attn_contrib + bayes_contrib) / 2
        
        return FusionResult(
            success=True,
            overall_status=overall_status,
            confidence=min(1.0, confidence),
            fused_detections=attention_result.fused_detections,
            modality_contributions=contributions,
            diagnostic_report={
                'method': 'hybrid_fusion',
                'attention_status': attention_result.overall_status,
                'bayesian_status': bayesian_result.overall_status,
                'bayesian_posteriors': bayesian_result.bayesian_inference
            },
            recommendations=self._merge_recommendations(
                attention_result.recommendations,
                bayesian_result.recommendations
            ),
            fault_chain=bayesian_result.fault_chain,
            bayesian_inference=bayesian_result.bayesian_inference
        )
    
    def _classify_from_features(self, features: np.ndarray) -> Tuple[str, float]:
        """从特征进行简单分类"""
        # 简化的分类逻辑 (实际应使用训练好的模型)
        feature_norm = np.linalg.norm(features)
        feature_mean = np.mean(np.abs(features))
        
        if feature_mean > 0.8:
            return 'critical', 0.9
        elif feature_mean > 0.6:
            return 'alarm', 0.8
        elif feature_mean > 0.4:
            return 'warning', 0.7
        elif feature_mean > 0.2:
            return 'attention', 0.6
        else:
            return 'normal', 0.85
    
    def _apply_diagnostic_rules(self, modality_data: Dict[str, ModalityData]) -> List[Dict]:
        """应用诊断规则"""
        matches = []
        
        for rule in self.diagnostic_rules:
            match_count = 0
            matched_conditions = []
            
            for condition in rule['conditions']:
                modality = condition['modality']
                if modality not in modality_data:
                    continue
                
                data = modality_data[modality]
                for detection in data.detections:
                    if (detection.get('type') == condition['detection'] and
                        detection.get('confidence', 0) >= condition['min_confidence']):
                        match_count += 1
                        matched_conditions.append(condition)
                        break
            
            if match_count >= rule['min_match']:
                matches.append({
                    'rule': rule['name'],
                    'diagnosis': rule['diagnosis'],
                    'priority': rule['priority'],
                    'fault_type': rule['fault_type'],
                    'matched_conditions': matched_conditions
                })
        
        return matches
    
    def _collect_detections(self, modality_data: Dict[str, ModalityData]) -> List[Dict]:
        """收集所有检测结果"""
        detections = []
        for modality, data in modality_data.items():
            for detection in data.detections:
                detection_copy = detection.copy()
                detection_copy['source_modality'] = modality
                detections.append(detection_copy)
        return detections
    
    def _generate_recommendations(self, status: str, 
                                  rule_matches: List[Dict] = None,
                                  fault_chain: List[Dict] = None) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于状态的通用建议
        status_recommendations = {
            'critical': ['立即停机检查', '通知值班人员', '启动应急预案'],
            'alarm': ['安排紧急检修', '加强监测频率', '准备备品备件'],
            'warning': ['计划性检修', '持续关注趋势', '记录异常数据'],
            'attention': ['增加巡检频次', '观察变化趋势'],
            'normal': ['继续常规监测']
        }
        
        recommendations.extend(status_recommendations.get(status, []))
        
        # 基于规则匹配的建议
        if rule_matches:
            for match in rule_matches:
                recommendations.append(f"[{match['rule']}] {match['diagnosis']}")
        
        # 基于故障链的建议
        if fault_chain:
            for chain in fault_chain:
                if chain['related_faults']:
                    related = ', '.join([f['fault'] for f in chain['related_faults']])
                    recommendations.append(f"注意关联故障: {chain['root_fault']} 可能导致 {related}")
        
        return list(set(recommendations))  # 去重
    
    def _merge_recommendations(self, rec1: List[str], rec2: List[str]) -> List[str]:
        """合并建议列表"""
        return list(set(rec1 + rec2))
    
    def _empty_result(self) -> FusionResult:
        """返回空结果"""
        return FusionResult(
            success=True,
            overall_status='normal',
            confidence=0.5,
            fused_detections=[],
            modality_contributions={},
            diagnostic_report={'warning': '无有效模态数据'},
            recommendations=['无法进行融合分析，请检查传感器状态']
        )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'FusionStrategy',
    'ModalityData',
    'FusionResult',
    'CrossAttentionLayer',
    'TransformerFusionBlock',
    'BayesianDecisionNetwork',
    'EnhancedAttentionFusion',
    'FusionStrategyManager',
    'EnhancedFusionEngine'
]
