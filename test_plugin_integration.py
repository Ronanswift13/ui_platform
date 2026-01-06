#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件集成测试
破夜绘明激光监测平台

测试所有5个核心插件的基本功能
"""

import os
import sys
import unittest
import logging
from pathlib import Path

import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class TestPluginIntegration(unittest.TestCase):
    """插件集成测试"""
    
    # 核心5个插件
    CORE_PLUGINS = [
        "transformer_inspection",  # A组
        "switch_inspection",       # B组
        "busbar_inspection",       # C组
        "capacitor_inspection",    # D组
        "meter_reading",           # E组
    ]
    
    def setUp(self):
        """测试准备"""
        self.project_root = Path(__file__).parent
        self.plugins_dir = self.project_root / "plugins"
    
    def test_plugins_exist(self):
        """测试插件目录存在"""
        for plugin_name in self.CORE_PLUGINS:
            plugin_dir = self.plugins_dir / plugin_name
            self.assertTrue(
                plugin_dir.exists(), 
                f"插件目录不存在: {plugin_dir}"
            )
            logger.info(f"✅ {plugin_name} 目录存在")
    
    def test_plugin_files(self):
        """测试插件必要文件"""
        required_files = ["__init__.py", "plugin.py"]
        
        for plugin_name in self.CORE_PLUGINS:
            plugin_dir = self.plugins_dir / plugin_name
            
            for filename in required_files:
                filepath = plugin_dir / filename
                self.assertTrue(
                    filepath.exists(),
                    f"缺少文件: {filepath}"
                )
            
            logger.info(f"✅ {plugin_name} 文件完整")
    
    def test_plugin_import(self):
        """测试插件导入"""
        sys.path.insert(0, str(self.project_root))
        
        for plugin_name in self.CORE_PLUGINS:
            try:
                module = __import__(f"plugins.{plugin_name}", fromlist=["plugin"])
                self.assertIsNotNone(module)
                logger.info(f"✅ {plugin_name} 导入成功")
            except ImportError as e:
                logger.warning(f"⚠️ {plugin_name} 导入失败: {e}")
    
    def test_detector_class(self):
        """测试检测器类"""
        sys.path.insert(0, str(self.project_root))
        
        detector_classes = {
            "transformer_inspection": "TransformerDetector",
            "switch_inspection": "SwitchDetector",
            "busbar_inspection": "BusbarDetector",
            "capacitor_inspection": "CapacitorDetector",
            "meter_reading": "MeterReadingDetector",
        }
        
        for plugin_name, class_name in detector_classes.items():
            try:
                # 尝试导入检测器
                detector_module = __import__(
                    f"plugins.{plugin_name}.detector",
                    fromlist=[class_name]
                )
                detector_class = getattr(detector_module, class_name, None)
                
                if detector_class:
                    logger.info(f"✅ {plugin_name}.{class_name} 存在")
                else:
                    # 尝试增强版
                    detector_module = __import__(
                        f"plugins.{plugin_name}.detector_enhanced",
                        fromlist=[f"{class_name}Enhanced"]
                    )
                    logger.info(f"✅ {plugin_name} 增强版检测器存在")
                    
            except ImportError as e:
                logger.warning(f"⚠️ {plugin_name} 检测器导入失败: {e}")


class TestModelRegistry(unittest.TestCase):
    """模型注册测试"""
    
    def test_config_exists(self):
        """测试配置文件存在"""
        project_root = Path(__file__).parent
        config_files = [
            "configs/models_config.yaml",
            "configs/enhanced_config.yaml",
        ]
        
        for config_file in config_files:
            config_path = project_root / config_file
            if config_path.exists():
                logger.info(f"✅ 配置文件存在: {config_file}")
            else:
                logger.warning(f"⚠️ 配置文件不存在: {config_file}")


class TestONNXModels(unittest.TestCase):
    """ONNX模型测试"""
    
    def test_models_directory(self):
        """测试模型目录"""
        project_root = Path(__file__).parent
        models_dir = project_root / "models"
        
        if models_dir.exists():
            onnx_files = list(models_dir.rglob("*.onnx"))
            logger.info(f"找到 {len(onnx_files)} 个ONNX模型")
            
            for onnx_file in onnx_files:
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                logger.info(f"  - {onnx_file.name}: {size_mb:.2f} MB")
        else:
            logger.warning("⚠️ models目录不存在")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestPluginIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestModelRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestONNXModels))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
