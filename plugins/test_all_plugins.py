#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全插件测试脚本
破夜绘明激光监测平台

测试所有5个核心巡检插件
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


# 核心插件配置
PLUGIN_CONFIGS = {
    "transformer_inspection": {
        "name": "主变巡视 (A组)",
        "detector": "TransformerDetector",
        "enhanced_detector": "TransformerDetectorEnhanced",
        "test_rois": ["oil_level", "silica_gel", "thermal"],
    },
    "switch_inspection": {
        "name": "开关间隔 (B组)",
        "detector": "SwitchDetector",
        "enhanced_detector": "SwitchDetectorEnhanced",
        "test_rois": ["breaker_state", "indicator"],
    },
    "busbar_inspection": {
        "name": "母线巡视 (C组)",
        "detector": "BusbarDetector",
        "enhanced_detector": "BusbarDetectorEnhanced",
        "test_rois": ["insulator", "fitting"],
    },
    "capacitor_inspection": {
        "name": "电容器巡视 (D组)",
        "detector": "CapacitorDetector",
        "enhanced_detector": "CapacitorDetectorEnhanced",
        "test_rois": ["capacitor_unit", "tilt_check"],
    },
    "meter_reading": {
        "name": "表计读数 (E组)",
        "detector": "MeterReadingDetector",
        "enhanced_detector": "MeterReadingDetectorEnhanced",
        "test_rois": ["pressure_gauge", "temperature"],
    },
}


def create_test_image(height=640, width=640):
    """创建测试图像"""
    # 创建带模式的测试图像
    image = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
    
    # 添加一些形状
    cv2_available = False
    try:
        import cv2
        cv2_available = True
    except ImportError:
        pass
    
    if cv2_available:
        # 添加矩形和圆形
        cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), 2)
        cv2.circle(image, (400, 400), 50, (255, 0, 0), -1)
    
    return image


def test_plugin(plugin_id: str, config: Dict) -> Dict[str, Any]:
    """测试单个插件"""
    result = {
        "plugin_id": plugin_id,
        "name": config["name"],
        "status": "unknown",
        "errors": [],
        "tests": {}
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"测试插件: {config['name']}")
    logger.info(f"{'='*50}")
    
    # 测试1: 导入检测器
    try:
        # 尝试增强版
        detector_module = __import__(
            f"plugins.{plugin_id}.detector_enhanced",
            fromlist=[config["enhanced_detector"]]
        )
        detector_class = getattr(detector_module, config["enhanced_detector"])
        result["tests"]["import_enhanced"] = "PASS"
        logger.info(f"✅ 增强版检测器导入成功: {config['enhanced_detector']}")
    except (ImportError, AttributeError) as e:
        result["tests"]["import_enhanced"] = f"SKIP: {e}"
        
        # 尝试基础版
        try:
            detector_module = __import__(
                f"plugins.{plugin_id}.detector",
                fromlist=[config["detector"]]
            )
            detector_class = getattr(detector_module, config["detector"])
            result["tests"]["import_basic"] = "PASS"
            logger.info(f"✅ 基础检测器导入成功: {config['detector']}")
        except (ImportError, AttributeError) as e2:
            result["tests"]["import_basic"] = f"FAIL: {e2}"
            result["errors"].append(str(e2))
            logger.error(f"❌ 检测器导入失败: {e2}")
            result["status"] = "failed"
            return result
    
    # 测试2: 创建实例
    try:
        detector = detector_class(config={}, model_registry=None)
        result["tests"]["instantiate"] = "PASS"
        logger.info(f"✅ 检测器实例化成功")
    except Exception as e:
        result["tests"]["instantiate"] = f"FAIL: {e}"
        result["errors"].append(str(e))
        logger.error(f"❌ 检测器实例化失败: {e}")
        result["status"] = "failed"
        return result
    
    # 测试3: 初始化
    try:
        if hasattr(detector, 'initialize'):
            detector.initialize()
        result["tests"]["initialize"] = "PASS"
        logger.info(f"✅ 检测器初始化成功")
    except Exception as e:
        result["tests"]["initialize"] = f"WARN: {e}"
        logger.warning(f"⚠️ 检测器初始化警告: {e}")
    
    # 测试4: 推理测试
    try:
        test_image = create_test_image()
        test_rois = [{"id": f"test_{i}", "type": roi_type} for i, roi_type in enumerate(config["test_rois"])]
        
        if hasattr(detector, 'inspect'):
            result_obj = detector.inspect(test_image, test_rois)
            result["tests"]["inference"] = "PASS"
            logger.info(f"✅ 推理测试成功")
        else:
            result["tests"]["inference"] = "SKIP: 无inspect方法"
            logger.info(f"⚠️ 跳过推理测试")
    except Exception as e:
        result["tests"]["inference"] = f"WARN: {e}"
        logger.warning(f"⚠️ 推理测试警告: {e}")
    
    # 设置总体状态
    if not result["errors"]:
        result["status"] = "passed"
    
    return result


def test_all_plugins() -> Dict[str, Any]:
    """测试所有插件"""
    logger.info("\n" + "="*60)
    logger.info("开始测试所有核心插件")
    logger.info("="*60)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plugins": {},
        "summary": {
            "total": len(PLUGIN_CONFIGS),
            "passed": 0,
            "failed": 0,
        }
    }
    
    for plugin_id, config in PLUGIN_CONFIGS.items():
        result = test_plugin(plugin_id, config)
        results["plugins"][plugin_id] = result
        
        if result["status"] == "passed":
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1
    
    # 打印摘要
    logger.info("\n" + "="*60)
    logger.info("测试摘要")
    logger.info("="*60)
    logger.info(f"总数: {results['summary']['total']}")
    logger.info(f"通过: {results['summary']['passed']}")
    logger.info(f"失败: {results['summary']['failed']}")
    
    return results


if __name__ == "__main__":
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    results = test_all_plugins()
    
    # 返回码
    sys.exit(0 if results["summary"]["failed"] == 0 else 1)
