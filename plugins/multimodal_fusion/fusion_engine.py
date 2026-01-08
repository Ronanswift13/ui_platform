# -*- coding: utf-8 -*-
"""
多模态融合引擎
实现多种融合策略：早期融合、晚期融合、注意力融合
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """融合策略枚举"""
    EARLY = "early"           # 早期融合(特征级)
    LATE = "late"             # 晚期融合(决策级)
    ATTENTION = "attention"   # 注意力融合
    HYBRID = "hybrid"         # 混合融合


@dataclass
class ModalityData:
    """模态数据结构"""
    modality_type: str          # 模态类型: visual, acoustic, gas, hyperspectral, thermal
    features: np.ndarray        # 特征向量
    confidence: float           # 置信度
    detections: List[Dict]      # 检测结果
    metadata: Dict = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class FusionResult:
    """融合结果"""
    success: bool
    overall_status: str         # normal, attention, warning, alarm, critical
    confidence: float
    fused_detections: List[Dict]
    modality_contributions: Dict[str, float]  # 各模态贡献度
    diagnostic_report: Dict
    recommendations: List[str]


class AttentionFusion:
    """
    注意力融合机制
    动态学习各模态的重要性权重
    """
    
    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int = 128):
        """
        Args:
            modality_dims: 各模态的特征维度 {'visual': 512, 'acoustic': 256, ...}
            hidden_dim: 隐藏层维度
        """
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        self.model_registry = None
        
        # 模态权重(可学习,初始化为均匀)
        n_modalities = len(modality_dims)
        self.modality_weights = {m: 1.0/n_modalities for m in modality_dims}
        
        # 历史注意力分数
        self.attention_history = defaultdict(list)
        
    def set_model_registry(self, registry):
        """设置模型注册器"""
        self.model_registry = registry
        
    def compute_attention_scores(self, modality_data: Dict[str, ModalityData]) -> Dict[str, float]:
        """
        计算各模态的注意力分数
        
        基于:
        1. 特征质量(非零比例、方差)
        2. 检测置信度
        3. 历史可靠性
        """
        scores = {}
        
        for modality_type, data in modality_data.items():
            # 基础分数
            base_score = data.confidence
            
            # 特征质量分数
            if data.features is not None and len(data.features) > 0:
                # 非零特征比例
                non_zero_ratio = np.count_nonzero(data.features) / len(data.features)
                # 特征方差(归一化)
                feature_var = np.var(data.features)
                feature_var_norm = min(1.0, feature_var / 10.0)
                
                quality_score = 0.5 * non_zero_ratio + 0.5 * feature_var_norm
            else:
                quality_score = 0.0
            
            # 检测结果可信度
            if data.detections:
                avg_detection_conf = np.mean([d.get('confidence', 0.5) for d in data.detections])
            else:
                avg_detection_conf = 0.5
            
            # 历史可靠性(最近10次)
            if self.attention_history[modality_type]:
                history = self.attention_history[modality_type][-10:]
                historical_reliability = np.mean(history)
            else:
                historical_reliability = 0.5
            
            # 综合分数
            scores[modality_type] = (
                0.3 * base_score +
                0.2 * quality_score +
                0.3 * avg_detection_conf +
                0.2 * historical_reliability
            )
        
        # 归一化
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            n = len(scores)
            scores = {k: 1.0/n for k in scores}
            
        return scores
    
    def fuse_features(self, modality_data: Dict[str, ModalityData]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        注意力加权特征融合
        
        Returns:
            融合特征向量, 注意力权重
        """
        attention_scores = self.compute_attention_scores(modality_data)
        
        # 收集所有特征
        all_features = []
        feature_weights = []
        
        for modality_type, data in modality_data.items():
            if data.features is not None and len(data.features) > 0:
                all_features.append(data.features)
                feature_weights.append(attention_scores.get(modality_type, 0.0))
        
        if not all_features:
            return np.array([]), attention_scores
        
        # 标准化各模态特征到相同维度后加权拼接
        # 这里采用简单的加权平均(假设特征已对齐)
        max_len = max(len(f) for f in all_features)
        
        weighted_features = np.zeros(max_len)
        total_weight = 0
        
        for features, weight in zip(all_features, feature_weights):
            # 零填充到相同长度
            padded = np.zeros(max_len)
            padded[:len(features)] = features
            weighted_features += weight * padded
            total_weight += weight
        
        if total_weight > 0:
            weighted_features /= total_weight
            
        # 更新历史
        for modality_type, score in attention_scores.items():
            self.attention_history[modality_type].append(score)
            # 保留最近100次
            if len(self.attention_history[modality_type]) > 100:
                self.attention_history[modality_type] = self.attention_history[modality_type][-100:]
        
        return weighted_features, attention_scores


class DecisionFusion:
    """
    决策级融合
    整合各模态的检测/诊断结果
    """
    
    # 故障类型映射
    FAULT_TYPE_MAPPING = {
        # 视觉检测
        'overheating': '过热',
        'oil_leakage': '油位异常/漏油',
        'corrosion': '腐蚀',
        'damage': '损坏',
        'deformation': '变形',
        
        # 声学检测
        'partial_discharge': '局部放电',
        'corona_discharge': '电晕放电',
        'poor_contact': '接触不良',
        'bearing_fault': '轴承故障',
        'transformer_hum': '变压器异响',
        
        # 气体检测
        'sf6_leak': 'SF6泄漏',
        'h2_abnormal': '氢气异常',
        'dga_fault': 'DGA故障',
        
        # 高光谱检测
        'insulation_aging': '绝缘老化',
        'surface_contamination': '表面污染',
        'moisture_ingress': '受潮'
    }
    
    # 严重程度权重
    SEVERITY_WEIGHTS = {
        'critical': 1.0,
        'alarm': 0.8,
        'warning': 0.5,
        'attention': 0.3,
        'normal': 0.0
    }
    
    def __init__(self):
        self.fault_correlations = self._init_fault_correlations()
        
    def _init_fault_correlations(self) -> Dict[Tuple[str, str], float]:
        """初始化故障关联矩阵"""
        correlations = {
            # 视觉过热 + 声学放电 = 高度相关
            ('overheating', 'partial_discharge'): 0.85,
            ('overheating', 'corona_discharge'): 0.80,
            
            # 漏油 + 气体异常 = 相关
            ('oil_leakage', 'h2_abnormal'): 0.75,
            ('oil_leakage', 'dga_fault'): 0.90,
            
            # 腐蚀 + 高光谱老化 = 相关
            ('corrosion', 'insulation_aging'): 0.70,
            ('corrosion', 'surface_contamination'): 0.60,
            
            # 过热 + DGA故障 = 高度相关
            ('overheating', 'dga_fault'): 0.85,
            ('overheating', 'h2_abnormal'): 0.70,
            
            # 受潮 + 局放 = 相关
            ('moisture_ingress', 'partial_discharge'): 0.75,
            
            # 接触不良 + 过热 = 高度相关
            ('poor_contact', 'overheating'): 0.90,
        }
        
        # 双向关联
        reverse = {(b, a): v for (a, b), v in correlations.items()}
        correlations.update(reverse)
        
        return correlations
    
    def correlate_detections(self, all_detections: List[Dict]) -> List[Dict]:
        """
        关联不同模态的检测结果
        识别相关联的故障模式
        """
        correlated_groups = []
        used_indices = set()
        
        for i, det1 in enumerate(all_detections):
            if i in used_indices:
                continue
                
            group = [det1]
            used_indices.add(i)
            
            fault1 = det1.get('fault_type', det1.get('type', ''))
            
            for j, det2 in enumerate(all_detections):
                if j in used_indices:
                    continue
                    
                fault2 = det2.get('fault_type', det2.get('type', ''))
                
                # 检查关联性
                correlation = self.fault_correlations.get((fault1, fault2), 0.0)
                
                # 同设备检测结果关联性更强
                if det1.get('device_id') == det2.get('device_id'):
                    correlation *= 1.2
                    
                if correlation > 0.5:
                    group.append(det2)
                    used_indices.add(j)
            
            correlated_groups.append(group)
        
        # 转换为关联检测结果
        correlated_detections = []
        for group in correlated_groups:
            if len(group) == 1:
                correlated_detections.append(group[0])
            else:
                # 合并关联的检测
                merged = self._merge_correlated_detections(group)
                correlated_detections.append(merged)
        
        return correlated_detections
    
    def _merge_correlated_detections(self, group: List[Dict]) -> Dict:
        """合并关联的检测结果"""
        # 取最高置信度
        max_conf = max(d.get('confidence', 0) for d in group)
        
        # 取最严重级别
        severities = [d.get('severity', 'normal') for d in group]
        severity_order = ['critical', 'alarm', 'warning', 'attention', 'normal']
        worst_severity = min(severities, key=lambda s: severity_order.index(s) if s in severity_order else 99)
        
        # 合并故障类型
        fault_types = list(set(d.get('fault_type', d.get('type', '')) for d in group))
        
        # 合并模态来源
        modalities = list(set(d.get('modality', '') for d in group))
        
        return {
            'fault_types': fault_types,
            'primary_fault': fault_types[0] if fault_types else 'unknown',
            'severity': worst_severity,
            'confidence': max_conf,
            'modalities': modalities,
            'correlated': True,
            'supporting_evidence': [
                {
                    'type': d.get('fault_type', d.get('type')),
                    'modality': d.get('modality'),
                    'confidence': d.get('confidence')
                }
                for d in group
            ],
            'device_id': group[0].get('device_id', 'unknown')
        }
    
    def compute_overall_status(self, detections: List[Dict]) -> Tuple[str, float]:
        """
        计算整体状态
        
        Returns:
            (状态级别, 置信度)
        """
        if not detections:
            return 'normal', 1.0
        
        # 收集所有严重程度
        severity_scores = []
        for det in detections:
            severity = det.get('severity', 'normal')
            confidence = det.get('confidence', 0.5)
            
            base_score = self.SEVERITY_WEIGHTS.get(severity, 0.0)
            
            # 多模态确认的检测权重更高
            if det.get('correlated', False):
                base_score *= 1.2
            
            severity_scores.append(base_score * confidence)
        
        # 取最高分数
        max_score = max(severity_scores)
        
        # 映射到状态级别
        if max_score >= 0.9:
            status = 'critical'
        elif max_score >= 0.7:
            status = 'alarm'
        elif max_score >= 0.4:
            status = 'warning'
        elif max_score >= 0.2:
            status = 'attention'
        else:
            status = 'normal'
        
        # 置信度基于检测数量和一致性
        confidence = min(1.0, 0.5 + 0.1 * len(detections))
        
        return status, confidence


class FusionEngine:
    """
    多模态融合引擎主类
    协调特征融合和决策融合
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 融合配置
        """
        self.config = config or {}
        self.strategy = FusionStrategy(self.config.get('strategy', 'hybrid'))
        
        # 初始化融合模块
        modality_dims = self.config.get('modality_dims', {
            'visual': 512,
            'acoustic': 256,
            'gas': 64,
            'hyperspectral': 224,
            'thermal': 128
        })
        
        self.attention_fusion = AttentionFusion(modality_dims)
        self.decision_fusion = DecisionFusion()
        
        self.model_registry = None
        self.fusion_model_available = False
        
        # 诊断规则
        self.diagnostic_rules = self._init_diagnostic_rules()
        
        logger.info(f"FusionEngine initialized with strategy: {self.strategy.value}")
    
    def _init_diagnostic_rules(self) -> List[Dict]:
        """初始化诊断规则"""
        return [
            {
                'name': '变压器综合过热诊断',
                'conditions': [
                    {'modality': 'visual', 'fault_type': 'overheating'},
                    {'modality': 'gas', 'fault_type': 'dga_fault'}
                ],
                'min_match': 2,
                'diagnosis': '变压器内部过热，建议立即停机检查',
                'priority': 'critical'
            },
            {
                'name': '绝缘劣化诊断',
                'conditions': [
                    {'modality': 'acoustic', 'fault_type': 'partial_discharge'},
                    {'modality': 'hyperspectral', 'fault_type': 'insulation_aging'}
                ],
                'min_match': 2,
                'diagnosis': '绝缘系统劣化，存在局放风险',
                'priority': 'alarm'
            },
            {
                'name': '接触电阻升高',
                'conditions': [
                    {'modality': 'acoustic', 'fault_type': 'poor_contact'},
                    {'modality': 'visual', 'fault_type': 'overheating'}
                ],
                'min_match': 2,
                'diagnosis': '连接点接触电阻升高，建议紧固检查',
                'priority': 'warning'
            },
            {
                'name': '密封失效',
                'conditions': [
                    {'modality': 'gas', 'fault_type': 'sf6_leak'},
                    {'modality': 'visual', 'fault_type': 'oil_leakage'}
                ],
                'min_match': 1,
                'diagnosis': '设备密封可能失效',
                'priority': 'warning'
            },
            {
                'name': '受潮故障',
                'conditions': [
                    {'modality': 'hyperspectral', 'fault_type': 'moisture_ingress'},
                    {'modality': 'acoustic', 'fault_type': 'partial_discharge'}
                ],
                'min_match': 2,
                'diagnosis': '设备受潮导致绝缘性能下降',
                'priority': 'alarm'
            }
        ]
    
    def set_model_registry(self, registry):
        """设置模型注册器"""
        self.model_registry = registry
        self.attention_fusion.set_model_registry(registry)
        
        # 检查融合模型是否可用
        if registry:
            try:
                model_info = registry.get_model_info('multimodal_fusion')
                self.fusion_model_available = model_info is not None
            except:
                self.fusion_model_available = False
                
        logger.info(f"Fusion model available: {self.fusion_model_available}")
    
    def fuse(self, modality_data: Dict[str, ModalityData]) -> FusionResult:
        """
        执行多模态融合
        
        Args:
            modality_data: 各模态数据 {modality_type: ModalityData}
            
        Returns:
            FusionResult
        """
        if not modality_data:
            return FusionResult(
                success=False,
                overall_status='unknown',
                confidence=0.0,
                fused_detections=[],
                modality_contributions={},
                diagnostic_report={},
                recommendations=['无可用模态数据']
            )
        
        try:
            # 1. 特征级融合
            fused_features, attention_weights = self.attention_fusion.fuse_features(modality_data)
            
            # 2. 收集所有检测结果
            all_detections = []
            for modality_type, data in modality_data.items():
                for det in data.detections:
                    det_copy = det.copy()
                    det_copy['modality'] = modality_type
                    all_detections.append(det_copy)
            
            # 3. 决策级融合 - 关联检测
            correlated_detections = self.decision_fusion.correlate_detections(all_detections)
            
            # 4. 深度学习融合(如果模型可用)
            if self.fusion_model_available and len(fused_features) > 0:
                dl_result = self._deep_learning_fusion(fused_features, correlated_detections)
                if dl_result:
                    correlated_detections = self._merge_dl_result(correlated_detections, dl_result)
            
            # 5. 计算整体状态
            overall_status, confidence = self.decision_fusion.compute_overall_status(correlated_detections)
            
            # 6. 生成诊断报告
            diagnostic_report = self._generate_diagnostic_report(
                modality_data, correlated_detections, attention_weights
            )
            
            # 7. 生成建议
            recommendations = self._generate_recommendations(
                correlated_detections, diagnostic_report
            )
            
            return FusionResult(
                success=True,
                overall_status=overall_status,
                confidence=confidence,
                fused_detections=correlated_detections,
                modality_contributions=attention_weights,
                diagnostic_report=diagnostic_report,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Fusion error: {str(e)}")
            return FusionResult(
                success=False,
                overall_status='error',
                confidence=0.0,
                fused_detections=[],
                modality_contributions={},
                diagnostic_report={'error': str(e)},
                recommendations=['融合处理异常，请检查输入数据']
            )
    
    def _deep_learning_fusion(self, features: np.ndarray, detections: List[Dict]) -> Optional[Dict]:
        """使用深度学习模型进行融合"""
        if not self.model_registry:
            return None
            
        try:
            # 准备输入
            # 特征向量 + 检测结果编码
            detection_encoding = self._encode_detections(detections)
            
            # 拼接输入
            if len(detection_encoding) > 0:
                combined_input = np.concatenate([features, detection_encoding])
            else:
                combined_input = features
            
            # 调整形状
            model_input = combined_input.reshape(1, -1).astype(np.float32)
            
            # 推理
            result = self.model_registry.infer('multimodal_fusion', {'input': model_input})
            
            if result and 'output' in result:
                output = result['output']
                return {
                    'fusion_score': float(output[0]) if len(output) > 0 else 0.5,
                    'fault_probs': output[1:].tolist() if len(output) > 1 else []
                }
                
        except Exception as e:
            logger.warning(f"Deep learning fusion failed: {str(e)}")
            
        return None
    
    def _encode_detections(self, detections: List[Dict], max_detections: int = 10) -> np.ndarray:
        """将检测结果编码为向量"""
        # 每个检测编码为[severity_score, confidence, fault_type_hash]
        encoding = np.zeros(max_detections * 3)
        
        severity_map = {'critical': 1.0, 'alarm': 0.8, 'warning': 0.5, 'attention': 0.3, 'normal': 0.0}
        
        for i, det in enumerate(detections[:max_detections]):
            idx = i * 3
            encoding[idx] = severity_map.get(det.get('severity', 'normal'), 0.0)
            encoding[idx + 1] = det.get('confidence', 0.5)
            # 简单的类型编码
            fault_type = det.get('fault_type', det.get('primary_fault', ''))
            encoding[idx + 2] = hash(fault_type) % 100 / 100.0
        
        return encoding
    
    def _merge_dl_result(self, detections: List[Dict], dl_result: Dict) -> List[Dict]:
        """合并深度学习结果到检测列表"""
        # 根据深度学习融合分数调整置信度
        fusion_score = dl_result.get('fusion_score', 0.5)
        
        for det in detections:
            original_conf = det.get('confidence', 0.5)
            # 加权平均
            det['confidence'] = 0.6 * original_conf + 0.4 * fusion_score
            det['dl_enhanced'] = True
        
        return detections
    
    def _generate_diagnostic_report(
        self,
        modality_data: Dict[str, ModalityData],
        detections: List[Dict],
        attention_weights: Dict[str, float]
    ) -> Dict:
        """生成诊断报告"""
        report = {
            'summary': {},
            'modality_analysis': {},
            'rule_matches': [],
            'fault_statistics': {}
        }
        
        # 模态分析
        for modality_type, data in modality_data.items():
            report['modality_analysis'][modality_type] = {
                'confidence': data.confidence,
                'detection_count': len(data.detections),
                'contribution': attention_weights.get(modality_type, 0.0)
            }
        
        # 规则匹配
        for rule in self.diagnostic_rules:
            matches = self._check_rule(rule, detections)
            if matches >= rule['min_match']:
                report['rule_matches'].append({
                    'rule_name': rule['name'],
                    'diagnosis': rule['diagnosis'],
                    'priority': rule['priority'],
                    'match_count': matches
                })
        
        # 故障统计
        fault_counts = defaultdict(int)
        for det in detections:
            fault_type = det.get('fault_type', det.get('primary_fault', 'unknown'))
            fault_counts[fault_type] += 1
        report['fault_statistics'] = dict(fault_counts)
        
        # 摘要
        report['summary'] = {
            'total_detections': len(detections),
            'modalities_used': len(modality_data),
            'rules_triggered': len(report['rule_matches']),
            'primary_concern': report['rule_matches'][0]['diagnosis'] if report['rule_matches'] else '无重大异常'
        }
        
        return report
    
    def _check_rule(self, rule: Dict, detections: List[Dict]) -> int:
        """检查诊断规则匹配数"""
        matches = 0
        for condition in rule['conditions']:
            for det in detections:
                modality_match = det.get('modality') == condition.get('modality')
                fault_match = (
                    det.get('fault_type') == condition.get('fault_type') or
                    condition.get('fault_type') in det.get('fault_types', [])
                )
                
                if modality_match and fault_match:
                    matches += 1
                    break
        return matches
    
    def _generate_recommendations(self, detections: List[Dict], report: Dict) -> List[str]:
        """生成处置建议"""
        recommendations = []
        
        # 基于规则匹配生成建议
        for match in report.get('rule_matches', []):
            priority = match['priority']
            if priority == 'critical':
                recommendations.append(f"【紧急】{match['diagnosis']}，建议立即处置")
            elif priority == 'alarm':
                recommendations.append(f"【警告】{match['diagnosis']}，建议24小时内处置")
            else:
                recommendations.append(f"【注意】{match['diagnosis']}，建议安排检修")
        
        # 基于故障统计生成建议
        fault_stats = report.get('fault_statistics', {})
        if fault_stats.get('overheating', 0) > 0:
            recommendations.append("检测到过热现象，建议进行红外热成像复核")
        if fault_stats.get('partial_discharge', 0) > 0:
            recommendations.append("检测到局部放电，建议进行超声波定位检测")
        if fault_stats.get('sf6_leak', 0) > 0:
            recommendations.append("SF6气体异常，建议进行气密性检测")
        
        # 如果无异常
        if not recommendations:
            recommendations.append("设备运行状态良好，建议继续定期监测")
        
        return recommendations


class MultimodalDataCollector:
    """
    多模态数据收集器
    协调各插件收集数据
    """
    
    def __init__(self):
        self.plugin_mapping = {
            'visual': 'transformer_monitoring',  # 视觉检测可来自多个插件
            'acoustic': 'acoustic_monitoring',
            'gas': 'gas_detection',
            'hyperspectral': 'hyperspectral_detection',
            'thermal': 'transformer_monitoring'  # 热成像可整合到变压器监测
        }
        self.plugins = {}
        
    def register_plugin(self, modality: str, plugin):
        """注册模态对应的插件"""
        self.plugins[modality] = plugin
        
    def collect(self, device_id: str, input_data: Dict) -> Dict[str, ModalityData]:
        """
        从各插件收集数据
        
        Args:
            device_id: 设备ID
            input_data: 输入数据，按模态组织
            
        Returns:
            Dict[modality_type, ModalityData]
        """
        collected_data = {}
        
        for modality, data in input_data.items():
            if modality not in self.plugins:
                logger.warning(f"No plugin registered for modality: {modality}")
                continue
            
            plugin = self.plugins[modality]
            
            try:
                # 调用插件处理
                result = plugin.process(data)
                
                if result.get('success', False):
                    collected_data[modality] = ModalityData(
                        modality_type=modality,
                        features=np.array(result.get('features', [])),
                        confidence=result.get('confidence', 0.5),
                        detections=result.get('detections', []),
                        metadata=result.get('metadata', {}),
                        timestamp=data.get('timestamp', 0.0)
                    )
            except Exception as e:
                logger.error(f"Error collecting {modality} data: {str(e)}")
        
        return collected_data
