"""
插件初始化启动器
输变电站全自动AI巡检方案 - 插件与模型注册中心集成

功能:
1. 应用启动时初始化模型注册中心
2. 加载并初始化所有插件
3. 将model_registry注入到各增强检测器
4. 提供统一的插件访问接口

使用方式:
    from platform_core.plugin_initializer import PluginInitializer
    
    # 初始化
    initializer = PluginInitializer()
    initializer.initialize_all()
    
    # 获取插件执行推理
    result = initializer.run_inspection("transformer_inspection", frame, rois)
"""

from __future__ import annotations
import os
import sys
import time
import logging
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Type
import yaml
import numpy as np

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from platform_core.model_registry_manager import (
    ModelRegistryManager,
    ModelRegistry,
    get_model_registry_manager,
    initialize_models
)

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """插件信息"""
    plugin_id: str
    plugin_name: str
    plugin_class: Any
    detector_class: Any
    plugin_instance: Any = None
    detector_instance: Any = None
    is_initialized: bool = False
    use_enhanced: bool = False
    error_message: Optional[str] = None


class PluginInitializer:
    """
    插件初始化器
    
    负责初始化所有插件并注入模型注册中心
    """
    
    # 插件映射配置
    PLUGIN_CONFIG = {
        "transformer_inspection": {
            "name": "主变巡视插件(A组)",
            "module": "plugins.transformer_inspection.plugin",
            "class": "TransformerInspectionPlugin",
            "detector_module": "plugins.transformer_inspection.detector_enhanced",
            "detector_class": "TransformerDetectorEnhanced",
            "fallback_detector_module": "plugins.transformer_inspection.detector",
            "fallback_detector_class": "TransformerDetector",
            "models": [
                "transformer_defect_yolov8n",
                "transformer_oil_unet",
                "transformer_silica_cnn",
                "transformer_thermal"
            ]
        },
        "switch_inspection": {
            "name": "开关间隔插件(B组)",
            "module": "plugins.switch_inspection.plugin",
            "class": "SwitchInspectionPlugin",
            "detector_module": "plugins.switch_inspection.detector_enhanced",
            "detector_class": "SwitchDetectorEnhanced",
            "fallback_detector_module": "plugins.switch_inspection.detector",
            "fallback_detector_class": "SwitchDetector",
            "models": [
                "switch_state_yolov8s",
                "switch_ocr_crnn"
            ]
        },
        "busbar_inspection": {
            "name": "母线巡视插件(C组)",
            "module": "plugins.busbar_inspection.plugin",
            "class": "BusbarInspectionPlugin",
            "detector_module": "plugins.busbar_inspection.detector_enhanced",
            "detector_class": "BusbarDetectorEnhanced",
            "fallback_detector_module": "plugins.busbar_inspection.detector",
            "fallback_detector_class": "BusbarDetector",
            "models": [
                "busbar_defect_yolov8m",
                "busbar_noise_classifier"
            ]
        },
        "capacitor_inspection": {
            "name": "电容器插件(D组)",
            "module": "plugins.capacitor_inspection.plugin",
            "class": "CapacitorInspectionPlugin",
            "detector_module": "plugins.capacitor_inspection.detector_enhanced",
            "detector_class": "CapacitorDetectorEnhanced",
            "fallback_detector_module": "plugins.capacitor_inspection.detector",
            "fallback_detector_class": "CapacitorDetector",
            "models": [
                "capacitor_unit_yolov8",
                "capacitor_intrusion_rtdetr"
            ]
        },
        "meter_reading": {
            "name": "表计读数插件(E组)",
            "module": "plugins.meter_reading.plugin",
            "class": "MeterReadingPlugin",
            "detector_module": "plugins.meter_reading.detector_enhanced",
            "detector_class": "MeterReadingDetectorEnhanced",
            "fallback_detector_module": "plugins.meter_reading.detector",
            "fallback_detector_class": "MeterReadingDetector",
            "models": [
                "meter_keypoint_hrnet",
                "meter_digit_crnn",
                "meter_type_classifier"
            ]
        }
    }
    
    def __init__(self, config_path: str = "configs/models_config.yaml"):
        self._config_path = config_path
        self._model_registry: Optional[ModelRegistry] = None
        self._plugins: Dict[str, PluginInfo] = {}
        self._initialized = False
    
    def initialize_all(self, 
                       preload_models: bool = True,
                       enable_enhanced: bool = True) -> Dict[str, bool]:
        """
        初始化所有组件
        
        Args:
            preload_models: 是否预加载模型
            enable_enhanced: 是否启用增强检测器
            
        Returns:
            Dict[str, bool]: 各插件初始化状态
        """
        results = {}
        
        # 1. 初始化模型注册中心
        logger.info("=" * 60)
        logger.info("开始初始化AI巡检系统")
        logger.info("=" * 60)
        
        if not self._initialize_model_registry():
            logger.error("模型注册中心初始化失败")
            return {"model_registry": False}
        
        results["model_registry"] = True
        
        # 2. 检查模型文件状态
        model_status = self._check_model_files()
        logger.info(f"\n模型文件状态: {model_status['available']}/{model_status['total']} 可用")
        
        # 3. 预加载可用模型
        if preload_models:
            self._preload_available_models()
        
        # 4. 初始化各插件
        for plugin_id in self.PLUGIN_CONFIG.keys():
            success = self._initialize_plugin(plugin_id, enable_enhanced)
            results[plugin_id] = success
        
        self._initialized = True
        
        # 5. 打印初始化摘要
        self._print_summary(results)
        
        return results
    
    def _initialize_model_registry(self) -> bool:
        """初始化模型注册中心"""
        logger.info("\n[步骤1] 初始化模型注册中心...")
        
        try:
            manager = get_model_registry_manager(self._config_path)
            if not manager.initialize():
                return False
            
            self._model_registry = manager.get_registry()
            return self._model_registry is not None
            
        except Exception as e:
            logger.error(f"模型注册中心初始化失败: {e}")
            return False
    
    def _check_model_files(self) -> Dict[str, Any]:
        """检查模型文件状态"""
        logger.info("\n[步骤2] 检查模型文件...")
        
        manager = get_model_registry_manager()
        status = manager.check_models()
        
        available = 0
        missing = []
        
        for model_id, info in status.items():
            if isinstance(info, dict):
                if info.get("exists", False):
                    available += 1
                    logger.info(f"  ✓ {model_id}: {info.get('size_mb', 0):.1f} MB")
                else:
                    missing.append(model_id)
                    logger.warning(f"  ✗ {model_id}: 文件不存在 ({info.get('path', '')})")
        
        return {
            "total": len(status),
            "available": available,
            "missing": missing
        }
    
    def _preload_available_models(self):
        """预加载可用模型"""
        logger.info("\n[步骤3] 预加载模型...")
        
        if self._model_registry is None:
            return
        
        loaded_count = 0
        for model_id in self._model_registry.list_models():
            try:
                if self._model_registry.load(model_id):
                    loaded_count += 1
            except Exception as e:
                logger.warning(f"  模型加载跳过 {model_id}: {e}")
        
        logger.info(f"  预加载完成: {loaded_count} 个模型")
    
    def _initialize_plugin(self, plugin_id: str, enable_enhanced: bool) -> bool:
        """初始化单个插件"""
        config = self.PLUGIN_CONFIG.get(plugin_id)
        if not config:
            logger.error(f"未知插件: {plugin_id}")
            return False
        
        logger.info(f"\n[插件初始化] {config['name']}...")
        
        plugin_info = PluginInfo(
            plugin_id=plugin_id,
            plugin_name=config["name"],
            plugin_class=None,
            detector_class=None
        )
        
        try:
            # 1. 尝试加载增强检测器
            detector_class = None
            use_enhanced = False
            
            if enable_enhanced:
                try:
                    detector_module = importlib.import_module(config["detector_module"])
                    detector_class = getattr(detector_module, config["detector_class"])
                    use_enhanced = True
                    logger.info(f"  ✓ 加载增强检测器: {config['detector_class']}")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"  增强检测器不可用: {e}")
            
            # 2. 回退到基础检测器
            if detector_class is None:
                try:
                    fallback_module = importlib.import_module(config["fallback_detector_module"])
                    detector_class = getattr(fallback_module, config["fallback_detector_class"])
                    logger.info(f"  ✓ 使用基础检测器: {config['fallback_detector_class']}")
                except (ImportError, AttributeError) as e:
                    logger.error(f"  ✗ 检测器加载失败: {e}")
                    plugin_info.error_message = str(e)
                    self._plugins[plugin_id] = plugin_info
                    return False
            
            plugin_info.detector_class = detector_class
            plugin_info.use_enhanced = use_enhanced
            
            # 3. 创建检测器实例并注入model_registry
            detector_config = self._get_detector_config(plugin_id)
            
            if use_enhanced and self._model_registry is not None:
                # 增强检测器需要model_registry
                plugin_info.detector_instance = detector_class(
                    config=detector_config,
                    model_registry=self._model_registry
                )
                logger.info(f"  ✓ 注入model_registry到增强检测器")
            else:
                # 基础检测器不需要model_registry
                plugin_info.detector_instance = detector_class(config=detector_config)
            
            # 4. 初始化检测器
            if hasattr(plugin_info.detector_instance, 'initialize'):
                plugin_info.detector_instance.initialize()
            
            plugin_info.is_initialized = True
            self._plugins[plugin_id] = plugin_info
            
            logger.info(f"  ✓ 插件初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ 插件初始化失败: {e}")
            plugin_info.error_message = str(e)
            self._plugins[plugin_id] = plugin_info
            return False
    
    def _get_detector_config(self, plugin_id: str) -> Dict[str, Any]:
        """获取检测器配置"""
        # 基础配置
        config = {
            "confidence_threshold": 0.5,
            "use_deep_learning": True,
        }
        
        # 插件特定配置
        plugin_configs = {
            "transformer_inspection": {
                "defect_confidence_threshold": 0.5,
                "oil_level_threshold": 0.5,
                "thermal_anomaly_threshold": 0.7,
            },
            "busbar_inspection": {
                "use_slicing": True,
                "tile_size": 1280,
                "overlap": 128,
                "small_target_threshold": 0.4,
            },
            "capacitor_inspection": {
                "tilt_angle_threshold": 5.0,
                "intrusion_confirm_frames": 3,
            },
            "meter_reading": {
                "max_retry": 3,
                "perspective_correction": True,
            },
            "switch_inspection": {
                "multi_evidence_fusion": True,
                "clarity_threshold": 0.7,
            }
        }
        
        if plugin_id in plugin_configs:
            config.update(plugin_configs[plugin_id])
        
        return config
    
    def _print_summary(self, results: Dict[str, bool]):
        """打印初始化摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("初始化摘要")
        logger.info("=" * 60)
        
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        for name, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            
            # 对于插件，显示额外信息
            if name in self._plugins:
                plugin_info = self._plugins[name]
                if plugin_info.use_enhanced:
                    status += " (增强版)"
                else:
                    status += " (基础版)"
                if plugin_info.error_message:
                    status += f" - {plugin_info.error_message}"
            
            logger.info(f"  {name}: {status}")
        
        logger.info("-" * 60)
        logger.info(f"总计: {success_count}/{total_count} 成功")
        logger.info("=" * 60)
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInfo]:
        """获取插件信息"""
        return self._plugins.get(plugin_id)
    
    def get_detector(self, plugin_id: str) -> Optional[Any]:
        """获取检测器实例"""
        plugin_info = self._plugins.get(plugin_id)
        if plugin_info and plugin_info.is_initialized:
            return plugin_info.detector_instance
        return None
    
    def run_inspection(self, plugin_id: str, 
                       frame: np.ndarray,
                       rois: Optional[List[Dict]] = None,
                       context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行巡检
        
        Args:
            plugin_id: 插件ID
            frame: 输入图像 (BGR格式)
            rois: ROI列表
            context: 上下文信息
            
        Returns:
            Dict: 巡检结果
        """
        start_time = time.time()
        
        detector = self.get_detector(plugin_id)
        if detector is None:
            return {
                "success": False,
                "error": f"插件未初始化: {plugin_id}",
                "processing_time_ms": 0
            }
        
        try:
            # 调用检测器
            if hasattr(detector, 'inspect'):
                result = detector.inspect(frame, rois or [], context or {})
            elif hasattr(detector, 'detect'):
                result = detector.detect(frame)
            else:
                return {
                    "success": False,
                    "error": "检测器没有可用的检测方法",
                    "processing_time_ms": 0
                }
            
            processing_time = (time.time() - start_time) * 1000
            
            # 标准化返回
            if isinstance(result, dict):
                result["processing_time_ms"] = processing_time
                result["plugin_id"] = plugin_id
                result["use_enhanced"] = self._plugins[plugin_id].use_enhanced
                return result
            else:
                return {
                    "success": True,
                    "results": result,
                    "processing_time_ms": processing_time,
                    "plugin_id": plugin_id,
                    "use_enhanced": self._plugins[plugin_id].use_enhanced
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "plugin_id": plugin_id
            }
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "initialized": self._initialized,
            "model_registry": {
                "available": self._model_registry is not None,
                "stats": get_model_registry_manager().get_stats() if self._model_registry else None
            },
            "plugins": {}
        }
        
        for plugin_id, info in self._plugins.items():
            status["plugins"][plugin_id] = {
                "name": info.plugin_name,
                "initialized": info.is_initialized,
                "use_enhanced": info.use_enhanced,
                "error": info.error_message
            }
        
        return status


# 全局初始化器实例
_initializer: Optional[PluginInitializer] = None


def get_plugin_initializer(config_path: str = "configs/models_config.yaml") -> PluginInitializer:
    """获取全局插件初始化器"""
    global _initializer
    if _initializer is None:
        _initializer = PluginInitializer(config_path)
    return _initializer


def initialize_inspection_system(config_path: str = "configs/models_config.yaml",
                                 preload_models: bool = True,
                                 enable_enhanced: bool = True) -> Dict[str, bool]:
    """
    初始化完整的巡检系统
    
    这是启动应用时的主入口点
    
    Args:
        config_path: 模型配置文件路径
        preload_models: 是否预加载模型
        enable_enhanced: 是否启用增强检测器
        
    Returns:
        Dict[str, bool]: 各组件初始化状态
    """
    initializer = get_plugin_initializer(config_path)
    return initializer.initialize_all(
        preload_models=preload_models,
        enable_enhanced=enable_enhanced
    )


# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 初始化系统
    results = initialize_inspection_system(
        config_path="configs/models_config.yaml",
        preload_models=True,
        enable_enhanced=True
    )
    
    # 获取初始化器
    initializer = get_plugin_initializer()
    
    # 测试巡检 (使用虚拟图像)
    if results.get("transformer_inspection"):
        dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        result = initializer.run_inspection("transformer_inspection", dummy_frame)
        print(f"\n主变巡检结果: {result}")
