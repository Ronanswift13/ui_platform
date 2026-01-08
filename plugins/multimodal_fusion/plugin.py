# -*- coding: utf-8 -*-
"""
多模态融合插件
整合声学、视觉、气体、高光谱等多种传感器数据进行综合故障诊断
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import time
from collections import defaultdict
import json
from pathlib import Path

# 处理相对导入（支持作为包导入和直接加载两种方式）
try:
    from .fusion_engine import (
        FusionEngine,
        ModalityData,
        FusionResult,
        MultimodalDataCollector,
        FusionStrategy
    )
except ImportError:
    import sys
    _plugin_dir = Path(__file__).parent
    if str(_plugin_dir) not in sys.path:
        sys.path.insert(0, str(_plugin_dir))
    from fusion_engine import (
        FusionEngine,
        ModalityData,
        FusionResult,
        MultimodalDataCollector,
        FusionStrategy
    )

logger = logging.getLogger(__name__)


class MultimodalFusionPlugin:
    """
    多模态融合插件
    
    功能:
    1. 整合多种传感器数据
    2. 特征级融合(注意力机制)
    3. 决策级融合(关联分析)
    4. 综合故障诊断
    5. 生成处置建议
    """
    
    PLUGIN_ID = 'multimodal_fusion'
    PLUGIN_NAME = '多模态融合诊断'
    VERSION = '1.0.0'
    
    # 支持的模态类型
    SUPPORTED_MODALITIES = [
        'visual',           # 可见光图像
        'thermal',          # 红外热成像
        'acoustic',         # 声学信号
        'ultrasonic',       # 超声信号
        'gas',              # 气体传感
        'hyperspectral',    # 高光谱成像
        'vibration'         # 振动信号
    ]
    
    def __init__(self, manifest=None, plugin_dir=None):
        """
        初始化多模态融合插件

        Args:
            manifest: 插件清单 (PluginManifest)
            plugin_dir: 插件目录 (Path)
        """
        self.manifest = manifest
        self.plugin_dir = plugin_dir

        # 从 manifest 获取配置或使用默认值
        config = {}
        if manifest and hasattr(manifest, 'config_schema'):
            config = manifest.config_schema or {}
        self.config = config
        
        # 融合配置
        fusion_config = {
            'strategy': self.config.get('fusion_strategy', 'hybrid'),
            'modality_dims': self.config.get('modality_dims', {
                'visual': 512,
                'thermal': 256,
                'acoustic': 256,
                'ultrasonic': 128,
                'gas': 64,
                'hyperspectral': 224,
                'vibration': 128
            })
        }
        
        # 初始化融合引擎
        self.fusion_engine = FusionEngine(fusion_config)
        
        # 数据收集器
        self.data_collector = MultimodalDataCollector()
        
        # 模型注册器
        self.model_registry = None
        
        # 历史融合结果(用于趋势分析)
        self.fusion_history: Dict[str, List[Dict]] = defaultdict(list)
        self.max_history_length = self.config.get('max_history_length', 100)
        
        # 报警累计计数
        self.alarm_accumulator: Dict[str, int] = defaultdict(int)
        self.alarm_threshold = self.config.get('alarm_threshold', 3)
        
        # 子插件引用
        self.sub_plugins: Dict[str, Any] = {}
        
        # 状态
        self.initialized = False
        
        logger.info(f"{self.PLUGIN_NAME} v{self.VERSION} 初始化完成")
    
    def init(self, context: Optional[Dict] = None) -> bool:
        """
        初始化插件
        
        Args:
            context: 上下文信息(包含其他插件引用等)
        """
        try:
            context = context or {}
            
            # 注册子插件
            if 'plugins' in context:
                for modality, plugin_id in self.data_collector.plugin_mapping.items():
                    if plugin_id in context['plugins']:
                        self.sub_plugins[modality] = context['plugins'][plugin_id]
                        self.data_collector.register_plugin(modality, context['plugins'][plugin_id])
                        logger.info(f"注册子插件: {modality} -> {plugin_id}")
            
            self.initialized = True
            logger.info(f"{self.PLUGIN_NAME} 初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"插件初始化失败: {str(e)}")
            return False
    
    def set_model_registry(self, registry):
        """设置模型注册器"""
        self.model_registry = registry
        self.fusion_engine.set_model_registry(registry)
        
        # 同时设置给子插件
        for modality, plugin in self.sub_plugins.items():
            if hasattr(plugin, 'set_model_registry'):
                plugin.set_model_registry(registry)
        
        logger.info("模型注册器已设置")
    
    def process(self, input_data: Dict) -> Dict:
        """
        处理多模态数据并进行融合诊断
        
        Args:
            input_data: 输入数据
                {
                    'device_id': str,           # 设备ID
                    'timestamp': float,         # 时间戳
                    'modalities': {             # 各模态数据
                        'visual': {...},
                        'acoustic': {...},
                        'gas': {...},
                        ...
                    },
                    'pre_processed': bool,      # 是否已预处理
                    'modality_results': {...}   # 预处理的模态结果(可选)
                }
                
        Returns:
            融合诊断结果
        """
        device_id = input_data.get('device_id', 'unknown')
        timestamp = input_data.get('timestamp', time.time())
        
        try:
            # 获取各模态数据
            if input_data.get('pre_processed', False) and 'modality_results' in input_data:
                # 使用预处理的结果
                modality_data = self._convert_to_modality_data(input_data['modality_results'])
            else:
                # 调用子插件处理
                modality_data = self._process_modalities(
                    device_id,
                    input_data.get('modalities', {})
                )
            
            if not modality_data:
                return {
                    'success': False,
                    'error': '无可用的模态数据',
                    'device_id': device_id,
                    'timestamp': timestamp
                }
            
            # 执行融合
            fusion_result = self.fusion_engine.fuse(modality_data)
            
            # 构建输出
            result = self._build_result(device_id, timestamp, fusion_result, modality_data)
            
            # 更新历史
            self._update_history(device_id, result)
            
            # 生成报警
            alarms = self._generate_alarms(device_id, fusion_result)
            result['alarms'] = alarms
            
            return result
            
        except Exception as e:
            logger.error(f"融合处理失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'device_id': device_id,
                'timestamp': timestamp
            }
    
    def _process_modalities(self, device_id: str, modalities: Dict) -> Dict[str, ModalityData]:
        """调用各子插件处理模态数据"""
        modality_data = {}
        
        for modality_type, data in modalities.items():
            if modality_type not in self.SUPPORTED_MODALITIES:
                logger.warning(f"不支持的模态类型: {modality_type}")
                continue
            
            # 检查是否有对应的子插件
            if modality_type in self.sub_plugins:
                plugin = self.sub_plugins[modality_type]
                try:
                    # 添加设备ID
                    data['device_id'] = device_id
                    
                    # 调用子插件
                    result = plugin.process(data)
                    
                    if result.get('success', False):
                        modality_data[modality_type] = ModalityData(
                            modality_type=modality_type,
                            features=np.array(result.get('features', [])),
                            confidence=result.get('confidence', 0.5),
                            detections=result.get('detections', []),
                            metadata=result.get('metadata', {}),
                            timestamp=data.get('timestamp', time.time())
                        )
                        
                except Exception as e:
                    logger.error(f"处理{modality_type}模态失败: {str(e)}")
            else:
                # 直接使用传入的数据(假设已处理)
                if 'detections' in data:
                    modality_data[modality_type] = ModalityData(
                        modality_type=modality_type,
                        features=np.array(data.get('features', [])),
                        confidence=data.get('confidence', 0.5),
                        detections=data.get('detections', []),
                        metadata=data.get('metadata', {}),
                        timestamp=data.get('timestamp', time.time())
                    )
        
        return modality_data
    
    def _convert_to_modality_data(self, modality_results: Dict) -> Dict[str, ModalityData]:
        """将预处理结果转换为ModalityData"""
        modality_data = {}
        
        for modality_type, result in modality_results.items():
            modality_data[modality_type] = ModalityData(
                modality_type=modality_type,
                features=np.array(result.get('features', [])),
                confidence=result.get('confidence', 0.5),
                detections=result.get('detections', []),
                metadata=result.get('metadata', {}),
                timestamp=result.get('timestamp', time.time())
            )
        
        return modality_data
    
    def _build_result(
        self,
        device_id: str,
        timestamp: float,
        fusion_result: FusionResult,
        modality_data: Dict[str, ModalityData]
    ) -> Dict:
        """构建输出结果"""
        return {
            'success': fusion_result.success,
            'device_id': device_id,
            'timestamp': timestamp,
            
            # 整体状态
            'overall_status': fusion_result.overall_status,
            'confidence': fusion_result.confidence,
            
            # 融合检测结果
            'detections': fusion_result.fused_detections,
            
            # 模态贡献度
            'modality_contributions': fusion_result.modality_contributions,
            
            # 诊断报告
            'diagnostic_report': fusion_result.diagnostic_report,
            
            # 处置建议
            'recommendations': fusion_result.recommendations,
            
            # 模态统计
            'modality_summary': {
                modality: {
                    'confidence': data.confidence,
                    'detection_count': len(data.detections)
                }
                for modality, data in modality_data.items()
            }
        }
    
    def _update_history(self, device_id: str, result: Dict):
        """更新历史记录"""
        history_entry = {
            'timestamp': result['timestamp'],
            'overall_status': result['overall_status'],
            'confidence': result['confidence'],
            'detection_count': len(result['detections'])
        }
        
        self.fusion_history[device_id].append(history_entry)
        
        # 限制历史长度
        if len(self.fusion_history[device_id]) > self.max_history_length:
            self.fusion_history[device_id] = self.fusion_history[device_id][-self.max_history_length:]
    
    def _generate_alarms(self, device_id: str, fusion_result: FusionResult) -> List[Dict]:
        """生成报警"""
        alarms = []
        
        # 状态级别报警
        status = fusion_result.overall_status
        if status in ['critical', 'alarm', 'warning']:
            alarm_key = f"{device_id}_{status}"
            self.alarm_accumulator[alarm_key] += 1
            
            if self.alarm_accumulator[alarm_key] >= self.alarm_threshold:
                alarms.append({
                    'type': 'status_alarm',
                    'level': status,
                    'message': f"设备{device_id}状态异常: {status}",
                    'confidence': fusion_result.confidence,
                    'recommendations': fusion_result.recommendations[:3],  # 前3条建议
                    'accumulated_count': self.alarm_accumulator[alarm_key]
                })
        else:
            # 重置计数
            for key in list(self.alarm_accumulator.keys()):
                if key.startswith(device_id):
                    self.alarm_accumulator[key] = 0
        
        # 关联故障报警
        for det in fusion_result.fused_detections:
            if det.get('correlated', False) and det.get('severity') in ['critical', 'alarm']:
                alarms.append({
                    'type': 'correlated_fault',
                    'level': det['severity'],
                    'fault_types': det.get('fault_types', []),
                    'modalities': det.get('modalities', []),
                    'confidence': det.get('confidence', 0.5),
                    'evidence': det.get('supporting_evidence', [])
                })
        
        # 规则触发报警
        for match in fusion_result.diagnostic_report.get('rule_matches', []):
            if match['priority'] in ['critical', 'alarm']:
                alarms.append({
                    'type': 'diagnostic_rule',
                    'level': match['priority'],
                    'rule_name': match['rule_name'],
                    'diagnosis': match['diagnosis']
                })
        
        return alarms
    
    def get_device_trend(self, device_id: str, hours: int = 24) -> Dict:
        """
        获取设备状态趋势
        
        Args:
            device_id: 设备ID
            hours: 回溯小时数
            
        Returns:
            趋势分析结果
        """
        history = self.fusion_history.get(device_id, [])
        
        if not history:
            return {
                'success': False,
                'error': '无历史数据'
            }
        
        # 过滤时间范围
        cutoff_time = time.time() - hours * 3600
        recent_history = [h for h in history if h['timestamp'] >= cutoff_time]
        
        if not recent_history:
            return {
                'success': False,
                'error': f'最近{hours}小时无数据'
            }
        
        # 状态统计
        status_counts = defaultdict(int)
        for h in recent_history:
            status_counts[h['overall_status']] += 1
        
        # 置信度趋势
        confidences = [h['confidence'] for h in recent_history]
        
        # 检测数量趋势
        detection_counts = [h['detection_count'] for h in recent_history]
        
        return {
            'success': True,
            'device_id': device_id,
            'time_range_hours': hours,
            'data_points': len(recent_history),
            'status_distribution': dict(status_counts),
            'confidence_stats': {
                'mean': np.mean(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'trend': 'increasing' if len(confidences) > 1 and confidences[-1] > confidences[0] else 'decreasing'
            },
            'detection_stats': {
                'mean': np.mean(detection_counts),
                'max': np.max(detection_counts),
                'total': sum(detection_counts)
            }
        }
    
    def batch_process(self, batch_data: List[Dict]) -> List[Dict]:
        """
        批量处理多个设备的数据
        
        Args:
            batch_data: 多个设备的输入数据列表
            
        Returns:
            处理结果列表
        """
        results = []
        
        for data in batch_data:
            result = self.process(data)
            results.append(result)
        
        return results
    
    def get_plugin_status(self) -> Dict:
        """获取插件状态"""
        return {
            'plugin_id': self.PLUGIN_ID,
            'plugin_name': self.PLUGIN_NAME,
            'version': self.VERSION,
            'initialized': self.initialized,
            'model_registry_set': self.model_registry is not None,
            'fusion_model_available': self.fusion_engine.fusion_model_available,
            'registered_sub_plugins': list(self.sub_plugins.keys()),
            'supported_modalities': self.SUPPORTED_MODALITIES,
            'devices_monitored': len(self.fusion_history),
            'total_history_entries': sum(len(h) for h in self.fusion_history.values())
        }
    
    def infer(self, frame, rois, context):
        """实现BasePlugin抽象方法 - 执行推理"""
        return []

    def postprocess(self, results, rules):
        """实现BasePlugin抽象方法 - 后处理"""
        return []

    def healthcheck(self):
        """实现BasePlugin抽象方法 - 健康检查"""
        try:
            from platform_core.plugin_manager.base import HealthStatus
            return HealthStatus(healthy=self.initialized, message="OK" if self.initialized else "未初始化")
        except ImportError:
            return {"healthy": self.initialized, "message": "OK" if self.initialized else "未初始化"}

    def shutdown(self):
        """关闭插件"""
        logger.info(f"{self.PLUGIN_NAME} 正在关闭...")
        
        # 保存历史数据(可选)
        try:
            history_file = f"/tmp/fusion_history_{int(time.time())}.json"
            serializable_history = {
                k: v for k, v in self.fusion_history.items()
            }
            with open(history_file, 'w') as f:
                json.dump(serializable_history, f)
            logger.info(f"历史数据已保存到: {history_file}")
        except Exception as e:
            logger.warning(f"保存历史数据失败: {str(e)}")
        
        self.initialized = False
        logger.info(f"{self.PLUGIN_NAME} 已关闭")


# 便捷函数
def create_plugin(config: Optional[Dict] = None) -> MultimodalFusionPlugin:
    """创建插件实例"""
    return MultimodalFusionPlugin(config)
