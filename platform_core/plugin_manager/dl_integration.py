"""
深度学习模型集成层
==================

基于UPDATE.md优化方案实现:
- 统一的深度学习模型访问接口
- 自动模型加载与管理
- 支持多模态检测
- 性能监控与缓存

用于将ai_models中的深度学习模型集成到现有插件中

版本: 3.0.0
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入深度学习模型
try:
    from ai_models.deep_learning import (
        YOLOv8ViTDetector, YOLOv8ViTConfig,
        GLTransLSTM, GLTransLSTMConfig,
        AcousticCNNLSTM, AcousticModelConfig,
        DeepSORTTracker, TrackerConfig,
        KeypointDetector, KeypointConfig,
    )
    DL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"深度学习模型导入失败: {e}")
    DL_AVAILABLE = False


@dataclass
class ModelStats:
    """模型统计信息"""
    name: str
    loaded: bool = False
    load_time: Optional[datetime] = None
    inference_count: int = 0
    total_inference_time: float = 0.0
    last_inference_time: Optional[datetime] = None
    errors: int = 0

    @property
    def avg_inference_time_ms(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return (self.total_inference_time / self.inference_count) * 1000


class DeepLearningIntegration:
    """
    深度学习集成层

    提供统一接口访问各种深度学习模型:
    - YOLOv8-ViT 目标检测
    - GL-TransLSTM 气体预测
    - CNN-LSTM 声学分析
    - DeepSORT 多目标跟踪
    - Keypoint 表计读数

    使用示例:
    ```python
    dl = DeepLearningIntegration()

    # 初始化指定模型
    dl.init_model('yolov8_vit', model_path='models/yolo.pt')

    # 检测
    detections = dl.detect(image)

    # 预测气体浓度
    prediction = dl.predict_gas(gas_history)

    # 声学分析
    anomalies = dl.analyze_acoustic(audio_features)
    ```
    """

    # 模型类型配置
    MODEL_CONFIGS = {
        'yolov8_vit': {
            'class': 'YOLOv8ViTDetector',
            'config_class': 'YOLOv8ViTConfig',
            'default_config': {
                'model_size': 'n',
                'confidence_threshold': 0.5,
                'use_thermal_fusion': False,
            }
        },
        'gl_translstm': {
            'class': 'GLTransLSTM',
            'config_class': 'GLTransLSTMConfig',
            'default_config': {
                'input_dim': 8,
                'hidden_dim': 128,
                'num_layers': 2,
            }
        },
        'acoustic_cnn_lstm': {
            'class': 'AcousticCNNLSTM',
            'config_class': 'AcousticModelConfig',
            'default_config': {
                'input_channels': 1,
                'num_classes': 5,
            }
        },
        'deep_sort': {
            'class': 'DeepSORTTracker',
            'config_class': 'TrackerConfig',
            'default_config': {
                'max_age': 30,
                'min_hits': 3,
            }
        },
        'keypoint': {
            'class': 'KeypointDetector',
            'config_class': 'KeypointConfig',
            'default_config': {
                'num_keypoints': 8,
                'enable_perspective_correction': True,
            }
        }
    }

    def __init__(self):
        """初始化集成层"""
        self._models: Dict[str, Any] = {}
        self._stats: Dict[str, ModelStats] = {}
        self._initialized = False

        if not DL_AVAILABLE:
            logger.warning("[DLIntegration] 深度学习模块不可用")

    @property
    def is_available(self) -> bool:
        """是否可用"""
        return DL_AVAILABLE

    def init_model(
        self,
        model_type: str,
        **kwargs
    ) -> bool:
        """
        初始化指定类型的模型

        Args:
            model_type: 模型类型 (yolov8_vit, gl_translstm, acoustic_cnn_lstm, etc.)
            **kwargs: 模型配置参数

        Returns:
            是否成功
        """
        if not DL_AVAILABLE:
            logger.error(f"[DLIntegration] 无法初始化 {model_type}: 深度学习模块不可用")
            return False

        if model_type not in self.MODEL_CONFIGS:
            logger.error(f"[DLIntegration] 未知模型类型: {model_type}")
            return False

        try:
            config_info = self.MODEL_CONFIGS[model_type]

            # 合并默认配置和用户配置
            config_params = {**config_info['default_config'], **kwargs}

            # 创建配置对象
            config_class_name = config_info['config_class']
            config_class = globals().get(config_class_name)
            if config_class is None:
                # 从ai_models导入
                from ai_models import deep_learning
                config_class = getattr(deep_learning, config_class_name)

            config = config_class(**config_params)

            # 创建模型实例
            model_class_name = config_info['class']
            model_class = globals().get(model_class_name)
            if model_class is None:
                from ai_models import deep_learning
                model_class = getattr(deep_learning, model_class_name)

            model = model_class(config)

            # 加载模型
            if hasattr(model, 'load'):
                model.load()
            elif hasattr(model, 'initialize'):
                model.initialize()

            self._models[model_type] = model
            self._stats[model_type] = ModelStats(
                name=model_type,
                loaded=True,
                load_time=datetime.now()
            )

            logger.info(f"[DLIntegration] 模型 {model_type} 初始化成功")
            return True

        except Exception as e:
            logger.error(f"[DLIntegration] 模型 {model_type} 初始化失败: {e}")
            self._stats[model_type] = ModelStats(name=model_type, loaded=False, errors=1)
            return False

    def get_model(self, model_type: str) -> Optional[Any]:
        """获取已加载的模型"""
        return self._models.get(model_type)

    def is_model_loaded(self, model_type: str) -> bool:
        """检查模型是否已加载"""
        return model_type in self._models

    # =========================================================================
    # 目标检测接口
    # =========================================================================

    def detect(
        self,
        image: np.ndarray,
        model_type: str = 'yolov8_vit',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        通用目标检测

        Args:
            image: 输入图像 (BGR格式)
            model_type: 模型类型
            **kwargs: 额外参数

        Returns:
            检测结果列表 [{bbox, label, confidence, ...}, ...]
        """
        model = self._models.get(model_type)
        if model is None:
            logger.warning(f"[DLIntegration] 模型 {model_type} 未加载")
            return []

        start_time = time.time()

        try:
            if hasattr(model, 'detect'):
                result = model.detect(image, **kwargs)
                detections = [
                    {
                        'bbox': d.bbox,
                        'label': d.class_name,
                        'confidence': d.confidence,
                        'class_id': d.class_id,
                    }
                    for d in result.detections
                ]
            else:
                detections = []

            # 更新统计
            elapsed = time.time() - start_time
            stats = self._stats.get(model_type)
            if stats:
                stats.inference_count += 1
                stats.total_inference_time += elapsed
                stats.last_inference_time = datetime.now()

            return detections

        except Exception as e:
            logger.error(f"[DLIntegration] 检测失败: {e}")
            stats = self._stats.get(model_type)
            if stats:
                stats.errors += 1
            return []

    def detect_with_thermal(
        self,
        visible_image: np.ndarray,
        thermal_image: np.ndarray,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        可见光+热成像融合检测

        Args:
            visible_image: 可见光图像
            thermal_image: 热成像图像
            **kwargs: 额外参数

        Returns:
            检测结果列表
        """
        model = self._models.get('yolov8_vit')
        if model is None:
            logger.warning("[DLIntegration] YOLOv8-ViT模型未加载")
            return []

        try:
            if hasattr(model, 'detect_fused'):
                result = model.detect_fused(visible_image, thermal_image, **kwargs)
                return [
                    {
                        'bbox': d.bbox,
                        'label': d.class_name,
                        'confidence': d.confidence,
                        'class_id': d.class_id,
                        'thermal_features': d.metadata.get('thermal_features'),
                    }
                    for d in result.detections
                ]
        except Exception as e:
            logger.error(f"[DLIntegration] 融合检测失败: {e}")

        return []

    # =========================================================================
    # 气体预测接口
    # =========================================================================

    def predict_gas(
        self,
        history: np.ndarray,
        steps: int = 6,
        **kwargs
    ) -> Dict[str, Any]:
        """
        气体浓度预测

        Args:
            history: 历史数据 (T, features)
            steps: 预测步数
            **kwargs: 额外参数

        Returns:
            预测结果 {predictions, confidence, trend, anomalies, ...}
        """
        model = self._models.get('gl_translstm')
        if model is None:
            logger.warning("[DLIntegration] GL-TransLSTM模型未加载")
            return {}

        start_time = time.time()

        try:
            if hasattr(model, 'predict'):
                result = model.predict(history, steps, **kwargs)
                prediction = {
                    'predictions': result.predictions,
                    'confidence': result.confidence,
                    'trend': result.trend,
                    'anomaly_probability': result.anomaly_probability,
                    'decomposition': result.decomposition,
                }
            else:
                prediction = {}

            # 更新统计
            elapsed = time.time() - start_time
            stats = self._stats.get('gl_translstm')
            if stats:
                stats.inference_count += 1
                stats.total_inference_time += elapsed
                stats.last_inference_time = datetime.now()

            return prediction

        except Exception as e:
            logger.error(f"[DLIntegration] 气体预测失败: {e}")
            stats = self._stats.get('gl_translstm')
            if stats:
                stats.errors += 1
            return {}

    def detect_gas_leak(
        self,
        current_values: Dict[str, float],
        history: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        气体泄漏检测

        Args:
            current_values: 当前气体浓度值
            history: 历史数据
            **kwargs: 额外参数

        Returns:
            泄漏检测结果
        """
        model = self._models.get('gl_translstm')
        if model is None:
            return {'has_leak': False, 'leak_probability': 0.0}

        try:
            if hasattr(model, 'detect_leak'):
                result = model.detect_leak(current_values, history, **kwargs)
                return {
                    'has_leak': result.has_leak,
                    'leak_probability': result.leak_probability,
                    'leak_type': result.leak_type,
                    'severity': result.severity,
                    'affected_gases': result.affected_gases,
                }
        except Exception as e:
            logger.error(f"[DLIntegration] 泄漏检测失败: {e}")

        return {'has_leak': False, 'leak_probability': 0.0}

    # =========================================================================
    # 声学分析接口
    # =========================================================================

    def analyze_acoustic(
        self,
        features: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        声学异常分析

        Args:
            features: 音频特征
            **kwargs: 额外参数

        Returns:
            分析结果 {anomalies, partial_discharge, ...}
        """
        model = self._models.get('acoustic_cnn_lstm')
        if model is None:
            logger.warning("[DLIntegration] CNN-LSTM声学模型未加载")
            return {}

        start_time = time.time()

        try:
            if hasattr(model, 'analyze'):
                result = model.analyze(features, **kwargs)
                analysis = {
                    'has_anomaly': result.has_anomaly,
                    'anomaly_type': result.anomaly_type,
                    'confidence': result.confidence,
                    'anomalies': [
                        {
                            'type': a.anomaly_type,
                            'confidence': a.confidence,
                            'start_time': a.start_time,
                            'duration': a.duration,
                        }
                        for a in result.anomalies
                    ],
                    'health_score': result.health_score,
                }
            else:
                analysis = {}

            # 更新统计
            elapsed = time.time() - start_time
            stats = self._stats.get('acoustic_cnn_lstm')
            if stats:
                stats.inference_count += 1
                stats.total_inference_time += elapsed
                stats.last_inference_time = datetime.now()

            return analysis

        except Exception as e:
            logger.error(f"[DLIntegration] 声学分析失败: {e}")
            stats = self._stats.get('acoustic_cnn_lstm')
            if stats:
                stats.errors += 1
            return {}

    def detect_partial_discharge(
        self,
        audio_signal: np.ndarray,
        sample_rate: int = 44100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        局部放电检测

        Args:
            audio_signal: 音频信号
            sample_rate: 采样率
            **kwargs: 额外参数

        Returns:
            局放检测结果
        """
        model = self._models.get('acoustic_cnn_lstm')
        if model is None:
            return {'has_pd': False, 'pd_probability': 0.0}

        try:
            if hasattr(model, 'detect_partial_discharge'):
                result = model.detect_partial_discharge(audio_signal, sample_rate, **kwargs)
                return {
                    'has_pd': result.has_pd,
                    'pd_probability': result.probability,
                    'pd_type': result.pd_type,
                    'severity': result.severity,
                    'frequency_range': result.frequency_range,
                }
        except Exception as e:
            logger.error(f"[DLIntegration] 局放检测失败: {e}")

        return {'has_pd': False, 'pd_probability': 0.0}

    # =========================================================================
    # 表计读数接口
    # =========================================================================

    def read_meter(
        self,
        image: np.ndarray,
        meter_type: str = 'pointer',
        scale_range: tuple = (0, 100),
        unit: str = '',
        **kwargs
    ) -> Dict[str, Any]:
        """
        表计读数

        Args:
            image: 表计图像
            meter_type: 表计类型 (pointer, digital, lcd)
            scale_range: 刻度范围
            unit: 单位
            **kwargs: 额外参数

        Returns:
            读数结果 {value, confidence, keypoints, ...}
        """
        model = self._models.get('keypoint')
        if model is None:
            logger.warning("[DLIntegration] Keypoint模型未加载")
            return {}

        start_time = time.time()

        try:
            if hasattr(model, 'detect_and_read'):
                from ai_models.deep_learning.keypoint_detector import MeterType
                mt = MeterType(meter_type) if meter_type in [e.value for e in MeterType] else MeterType.POINTER

                result = model.detect_and_read(image, mt, scale_range, unit, **kwargs)
                reading = {
                    'value': result.value,
                    'unit': result.unit,
                    'confidence': result.confidence,
                    'is_valid': result.is_valid,
                    'pointer_angle': result.pointer_angle,
                    'keypoints': [
                        {'x': kp.x, 'y': kp.y, 'name': kp.name, 'confidence': kp.confidence}
                        for kp in result.keypoints
                    ],
                    'dial_bbox': result.dial_bbox,
                }
            else:
                reading = {}

            # 更新统计
            elapsed = time.time() - start_time
            stats = self._stats.get('keypoint')
            if stats:
                stats.inference_count += 1
                stats.total_inference_time += elapsed
                stats.last_inference_time = datetime.now()

            return reading

        except Exception as e:
            logger.error(f"[DLIntegration] 表计读数失败: {e}")
            stats = self._stats.get('keypoint')
            if stats:
                stats.errors += 1
            return {}

    # =========================================================================
    # 多目标跟踪接口
    # =========================================================================

    def update_tracks(
        self,
        detections: List[Dict[str, Any]],
        features: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        更新多目标跟踪

        Args:
            detections: 检测结果列表
            features: 深度特征 (可选)
            **kwargs: 额外参数

        Returns:
            跟踪结果列表
        """
        model = self._models.get('deep_sort')
        if model is None:
            logger.warning("[DLIntegration] DeepSORT模型未加载")
            return []

        try:
            if hasattr(model, 'update'):
                tracks = model.update(detections, features)
                return [
                    {
                        'track_id': t.track_id,
                        'bbox': t.bbox,
                        'class_name': t.class_name,
                        'velocity': t.velocity,
                        'age': t.age,
                        'hits': t.hits,
                    }
                    for t in tracks
                ]
        except Exception as e:
            logger.error(f"[DLIntegration] 跟踪更新失败: {e}")

        return []

    # =========================================================================
    # 管理接口
    # =========================================================================

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型统计"""
        return {
            name: {
                'loaded': stats.loaded,
                'load_time': stats.load_time.isoformat() if stats.load_time else None,
                'inference_count': stats.inference_count,
                'avg_inference_time_ms': stats.avg_inference_time_ms,
                'last_inference': stats.last_inference_time.isoformat() if stats.last_inference_time else None,
                'errors': stats.errors,
            }
            for name, stats in self._stats.items()
        }

    def unload_model(self, model_type: str) -> bool:
        """卸载模型"""
        if model_type not in self._models:
            return False

        try:
            model = self._models[model_type]
            if hasattr(model, 'cleanup'):
                model.cleanup()

            del self._models[model_type]
            self._stats[model_type].loaded = False
            logger.info(f"[DLIntegration] 模型 {model_type} 已卸载")
            return True
        except Exception as e:
            logger.error(f"[DLIntegration] 卸载模型失败: {e}")
            return False

    def unload_all(self):
        """卸载所有模型"""
        for model_type in list(self._models.keys()):
            self.unload_model(model_type)


# 全局单例
_dl_integration: Optional[DeepLearningIntegration] = None


def get_dl_integration() -> DeepLearningIntegration:
    """获取全局深度学习集成实例"""
    global _dl_integration
    if _dl_integration is None:
        _dl_integration = DeepLearningIntegration()
    return _dl_integration


__all__ = [
    'DeepLearningIntegration',
    'get_dl_integration',
    'DL_AVAILABLE',
]
