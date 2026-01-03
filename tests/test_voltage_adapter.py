#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变电站电压等级管理系统 - 单元测试
==========================================

运行测试:
    python -m pytest tests/test_voltage_adapter.py -v
    
或者直接运行:
    python tests/test_voltage_adapter.py

作者: 破夜绘明团队
日期: 2025
"""

import sys
import os
import unittest
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from platform_core.voltage_adapter_extended import (
    VoltageAdapterManager,
    VoltageCategory,
    VoltageLevel,
    get_all_voltage_categories,
    create_voltage_adapter,
    VOLTAGE_CONFIGS,
    MODEL_LIBRARIES,
    PLUGIN_CAPABILITIES,
)


class TestVoltageCategory(unittest.TestCase):
    """测试电压分类枚举"""
    
    def test_category_values(self):
        """测试分类值"""
        self.assertEqual(VoltageCategory.UHV.value, "特高压")
        self.assertEqual(VoltageCategory.EHV.value, "超高压")
        self.assertEqual(VoltageCategory.HV.value, "高压")
        self.assertEqual(VoltageCategory.MV.value, "中压")
        self.assertEqual(VoltageCategory.LV.value, "低压")
    
    def test_all_categories_exist(self):
        """测试所有分类存在"""
        categories = [c for c in VoltageCategory]
        self.assertEqual(len(categories), 5)


class TestVoltageConfigs(unittest.TestCase):
    """测试电压配置"""
    
    def test_config_exists_for_all_levels(self):
        """测试所有电压等级都有配置"""
        expected_levels = [
            "1000kV_AC", "±800kV_DC", "500kV_AC", "330kV_AC",
            "220kV", "110kV", "35kV", "10kV"
        ]
        for level in expected_levels:
            self.assertIn(level, VOLTAGE_CONFIGS, f"缺少配置: {level}")
    
    def test_uhv_config_has_special_features(self):
        """测试特高压配置有特殊功能"""
        config = VOLTAGE_CONFIGS.get("1000kV_AC")
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.gis)
        self.assertTrue(len(config.transformer.special_features) > 0)
    
    def test_thermal_thresholds_increase_with_voltage(self):
        """测试热成像阈值随电压等级增加"""
        uhv_thresh = VOLTAGE_CONFIGS["1000kV_AC"].transformer.thermal_thresholds
        ehv_thresh = VOLTAGE_CONFIGS["500kV_AC"].transformer.thermal_thresholds
        hv_thresh = VOLTAGE_CONFIGS["220kV"].transformer.thermal_thresholds
        lv_thresh = VOLTAGE_CONFIGS["10kV"].transformer.thermal_thresholds
        
        # 特高压阈值最高
        self.assertGreater(uhv_thresh["alarm"], ehv_thresh["alarm"])
        self.assertGreater(ehv_thresh["alarm"], hv_thresh["alarm"])
        self.assertGreater(hv_thresh["alarm"], lv_thresh["alarm"])


class TestVoltageAdapterManager(unittest.TestCase):
    """测试电压适配管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = VoltageAdapterManager(config_path="test_config.yaml")
    
    def tearDown(self):
        """清理测试环境"""
        # 删除测试配置文件
        test_config = Path("test_config.yaml")
        if test_config.exists():
            test_config.unlink()
    
    def test_initial_state(self):
        """测试初始状态"""
        self.assertIsNone(self.manager.current_level)
        self.assertIsNone(self.manager.current_category)
    
    def test_set_voltage_level_220kv(self):
        """测试设置220kV"""
        result = self.manager.set_voltage_level("220kV")
        self.assertTrue(result)
        self.assertEqual(self.manager.get_current_level(), "220kV")
        self.assertEqual(self.manager.get_current_category(), "高压")
    
    def test_set_voltage_level_500kv(self):
        """测试设置500kV"""
        result = self.manager.set_voltage_level("500kV_AC")
        self.assertTrue(result)
        self.assertEqual(self.manager.get_current_level(), "500kV_AC")
        self.assertEqual(self.manager.get_current_category(), "超高压")
    
    def test_set_voltage_level_1000kv(self):
        """测试设置1000kV特高压"""
        result = self.manager.set_voltage_level("1000kV_AC")
        self.assertTrue(result)
        self.assertEqual(self.manager.get_current_level(), "1000kV_AC")
        self.assertEqual(self.manager.get_current_category(), "特高压")
    
    def test_set_voltage_level_35kv(self):
        """测试设置35kV中压"""
        result = self.manager.set_voltage_level("35kV")
        self.assertTrue(result)
        self.assertEqual(self.manager.get_current_level(), "35kV")
        self.assertEqual(self.manager.get_current_category(), "中压")
    
    def test_set_voltage_level_10kv(self):
        """测试设置10kV低压"""
        result = self.manager.set_voltage_level("10kV")
        self.assertTrue(result)
        self.assertEqual(self.manager.get_current_level(), "10kV")
        self.assertEqual(self.manager.get_current_category(), "低压")
    
    def test_set_invalid_voltage_level(self):
        """测试设置无效电压等级"""
        result = self.manager.set_voltage_level("invalid_level")
        self.assertFalse(result)
    
    def test_normalize_voltage_level(self):
        """测试电压等级标准化"""
        # 测试大小写不敏感
        result = self.manager.set_voltage_level("220kv")
        self.assertTrue(result)
        self.assertEqual(self.manager.get_current_level(), "220kV")
    
    def test_get_equipment_config(self):
        """测试获取设备配置"""
        self.manager.set_voltage_level("220kV")
        
        config = self.manager.get_equipment_config("transformer")
        self.assertIsNotNone(config)
        self.assertIn("typical_models", config)
        self.assertIn("thermal_thresholds", config)
    
    def test_get_all_equipment_config(self):
        """测试获取所有设备配置"""
        self.manager.set_voltage_level("500kV_AC")
        
        config = self.manager.get_all_equipment_config()
        self.assertIn("transformer", config)
        self.assertIn("switch", config)
        self.assertIn("busbar", config)
        self.assertIn("capacitor", config)
        self.assertIn("meter", config)
    
    def test_get_model_path(self):
        """测试获取模型路径"""
        self.manager.set_voltage_level("220kV")
        
        path = self.manager.get_model_path("transformer", "defect_detection")
        self.assertIsNotNone(path)
        self.assertIn("220kV", path)
        self.assertIn("transformer", path)
    
    def test_get_detection_classes(self):
        """测试获取检测类别"""
        self.manager.set_voltage_level("500kV_AC")
        
        classes = self.manager.get_detection_classes("switch")
        self.assertIsInstance(classes, list)
        self.assertIn("breaker_open", classes)
        self.assertIn("breaker_closed", classes)
    
    def test_get_thermal_thresholds(self):
        """测试获取热成像阈值"""
        self.manager.set_voltage_level("220kV")
        
        thresholds = self.manager.get_thermal_thresholds()
        self.assertIn("normal", thresholds)
        self.assertIn("warning", thresholds)
        self.assertIn("alarm", thresholds)
        
        # 220kV的阈值
        self.assertEqual(thresholds["normal"], 60)
        self.assertEqual(thresholds["warning"], 75)
        self.assertEqual(thresholds["alarm"], 85)
    
    def test_get_angle_reference(self):
        """测试获取开关角度参考值"""
        self.manager.set_voltage_level("220kV")
        
        breaker_ref = self.manager.get_angle_reference("breaker")
        self.assertIn("open_deg", breaker_ref)
        self.assertIn("closed_deg", breaker_ref)
    
    def test_get_supported_plugins(self):
        """测试获取支持的插件"""
        # 特高压应该有更多特殊插件
        self.manager.set_voltage_level("1000kV_AC")
        uhv_plugins = self.manager.get_supported_plugins()
        
        self.manager.set_voltage_level("10kV")
        lv_plugins = self.manager.get_supported_plugins()
        
        # 验证插件列表不为空
        self.assertTrue(len(uhv_plugins) > 0)
        self.assertTrue(len(lv_plugins) > 0)
        
        # 验证特高压有特殊插件
        uhv_plugin_ids = [p["id"] for p in uhv_plugins]
        self.assertIn("uhv_bushing_monitor", uhv_plugin_ids)
    
    def test_get_special_features(self):
        """测试获取特殊功能"""
        self.manager.set_voltage_level("1000kV_AC")
        features = self.manager.get_special_features()
        
        self.assertIsInstance(features, list)
        self.assertTrue(len(features) > 0)
    
    def test_get_voltage_level_info(self):
        """测试获取完整电压等级信息"""
        self.manager.set_voltage_level("500kV_AC")
        
        info = self.manager.get_voltage_level_info()
        
        self.assertEqual(info["voltage_level"], "500kV_AC")
        self.assertEqual(info["category"], "超高压")
        self.assertIn("equipment_config", info)
        self.assertIn("model_library", info)
        self.assertIn("supported_plugins", info)
        self.assertIn("special_features", info)
    
    def test_get_available_voltage_levels(self):
        """测试获取所有可用电压等级"""
        levels = self.manager.get_available_voltage_levels()
        
        self.assertIn("特高压", levels)
        self.assertIn("超高压", levels)
        self.assertIn("高压", levels)
        self.assertIn("中压", levels)
        self.assertIn("低压", levels)


class TestHelperFunctions(unittest.TestCase):
    """测试辅助函数"""
    
    def test_get_all_voltage_categories(self):
        """测试获取所有电压分类"""
        categories = get_all_voltage_categories()
        
        self.assertEqual(len(categories), 5)
        
        category_names = [c["category"] for c in categories]
        self.assertIn("特高压", category_names)
        self.assertIn("超高压", category_names)
        self.assertIn("高压", category_names)
        self.assertIn("中压", category_names)
        self.assertIn("低压", category_names)
    
    def test_create_voltage_adapter(self):
        """测试创建电压适配器便捷函数"""
        adapter = create_voltage_adapter("220kV")
        
        self.assertEqual(adapter.get_current_level(), "220kV")
        self.assertEqual(adapter.get_current_category(), "高压")


class TestPluginCapabilities(unittest.TestCase):
    """测试插件功能定义"""
    
    def test_all_plugins_have_required_fields(self):
        """测试所有插件都有必需字段"""
        for plugin_id, capability in PLUGIN_CAPABILITIES.items():
            self.assertTrue(hasattr(capability, "name"), f"{plugin_id} 缺少 name")
            self.assertTrue(hasattr(capability, "description"), f"{plugin_id} 缺少 description")
            self.assertTrue(hasattr(capability, "supported_voltage_levels"), f"{plugin_id} 缺少 supported_voltage_levels")
            self.assertTrue(hasattr(capability, "detection_types"), f"{plugin_id} 缺少 detection_types")
            self.assertTrue(hasattr(capability, "requires_models"), f"{plugin_id} 缺少 requires_models")
    
    def test_uhv_specific_plugins(self):
        """测试特高压专有插件"""
        uhv_plugins = ["uhv_bushing_monitor", "uhv_corona_detection", "converter_valve_monitor"]
        
        for plugin_id in uhv_plugins:
            self.assertIn(plugin_id, PLUGIN_CAPABILITIES)
            capability = PLUGIN_CAPABILITIES[plugin_id]
            # 应该不支持所有电压等级
            self.assertNotEqual(capability.supported_voltage_levels, ["all"])
    
    def test_universal_plugins(self):
        """测试通用插件"""
        universal_plugins = ["transformer_monitor", "switch_state_detection", "thermal_imaging"]
        
        for plugin_id in universal_plugins:
            self.assertIn(plugin_id, PLUGIN_CAPABILITIES)
            capability = PLUGIN_CAPABILITIES[plugin_id]
            # 应该支持所有电压等级
            self.assertEqual(capability.supported_voltage_levels, ["all"])


class TestModelLibraries(unittest.TestCase):
    """测试模型库配置"""
    
    def test_model_libraries_exist(self):
        """测试模型库存在"""
        expected_libraries = ["1000kV_AC", "500kV_AC", "220kV", "110kV", "35kV", "10kV"]
        
        for lib in expected_libraries:
            self.assertIn(lib, MODEL_LIBRARIES, f"缺少模型库: {lib}")
    
    def test_model_library_structure(self):
        """测试模型库结构"""
        for level, library in MODEL_LIBRARIES.items():
            self.assertTrue(hasattr(library, "voltage_level"))
            self.assertTrue(hasattr(library, "base_path"))
            self.assertTrue(hasattr(library, "models"))
            self.assertIsInstance(library.models, dict)
    
    def test_transformer_models_exist(self):
        """测试变压器模型存在"""
        for level, library in MODEL_LIBRARIES.items():
            if "transformer" in library.models:
                self.assertIn("defect_detection", library.models["transformer"])


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_complete_workflow(self):
        """测试完整工作流程"""
        manager = VoltageAdapterManager()
        
        # 1. 设置电压等级
        self.assertTrue(manager.set_voltage_level("500kV_AC"))
        
        # 2. 获取设备配置
        transformer_config = manager.get_equipment_config("transformer")
        self.assertIsNotNone(transformer_config)
        
        # 3. 获取模型路径
        model_path = manager.get_model_path("transformer", "defect_detection")
        self.assertIsNotNone(model_path)
        
        # 4. 获取检测类别
        classes = manager.get_detection_classes("switch")
        self.assertTrue(len(classes) > 0)
        
        # 5. 获取支持的插件
        plugins = manager.get_supported_plugins()
        self.assertTrue(len(plugins) > 0)
        
        # 6. 切换电压等级
        self.assertTrue(manager.set_voltage_level("220kV"))
        self.assertEqual(manager.get_current_level(), "220kV")
    
    def test_voltage_level_switching(self):
        """测试电压等级切换"""
        manager = VoltageAdapterManager()
        
        levels = ["1000kV_AC", "500kV_AC", "220kV", "110kV", "35kV", "10kV"]
        
        for level in levels:
            self.assertTrue(manager.set_voltage_level(level), f"无法切换到 {level}")
            self.assertEqual(manager.get_current_level(), level)
            
            # 验证配置已更新
            config = manager.get_all_equipment_config()
            self.assertIsNotNone(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
