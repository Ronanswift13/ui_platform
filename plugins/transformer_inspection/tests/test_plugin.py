#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主变自主巡视插件测试脚本
"""

import sys
from pathlib import Path

# 添加路径
PLUGIN_DIR = Path(__file__).parent.parent
PROJECT_ROOT = PLUGIN_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PLUGIN_DIR))

import numpy as np
import yaml


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result(success, message):
    status = "✓" if success else "✗"
    print(f"{status} {message}")


def test_detector():
    """测试检测器"""
    print("\n[测试 1/4] 测试检测器...")

    try:
        from plugins.transformer_inspection.detector import TransformerDetector

        # 加载配置
        config_path = PLUGIN_DIR / "configs" / "default.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 创建检测器
        detector = TransformerDetector(config)

        # 创建测试图像 (模拟有缺陷的图像)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # 添加深色区域模拟油漏
        test_image[100:200, 100:200] = [30, 30, 30]
        # 添加橙色区域模拟锈蚀
        test_image[200:280, 300:400] = [0, 100, 180]

        # 测试缺陷检测
        defects = detector.detect_defects(test_image, "radiator")
        print_result(True, f"检测到 {len(defects)} 个缺陷")

        for d in defects:
            print(f"    - {d['label']}: 置信度 {d['confidence']:.2f}")

        return True

    except Exception as e:
        print_result(False, f"检测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plugin_init():
    """测试插件初始化"""
    print("\n[测试 2/4] 测试插件初始化...")

    try:
        from platform_core.plugin_manager import PluginManager

        pm = PluginManager()
        plugin = pm.load_plugin("transformer_inspection")

        print_result(True, f"初始化结果: 成功, 状态: {plugin.status.value}")

        return True, plugin

    except Exception as e:
        print_result(False, f"插件初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_healthcheck(plugin):
    """测试健康检查"""
    print("\n[测试 3/4] 测试健康检查...")

    try:
        health = plugin.healthcheck()
        print_result(health.healthy, f"健康状态: {'正常' if health.healthy else '异常'}")
        print(f"    消息: {health.message}")
        if health.details:
            for k, v in health.details.items():
                print(f"    {k}: {v}")
        return health.healthy

    except Exception as e:
        print_result(False, f"健康检查失败: {e}")
        return False


def test_inference(plugin):
    """测试推理"""
    print("\n[测试 4/4] 测试推理...")

    try:
        from platform_core.plugin_manager.base import PluginContext
        from platform_core.schema.models import ROI, ROIType, BoundingBox

        # 创建测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:200, 100:200] = [30, 30, 30]  # 深色区域
        test_image[200:280, 300:400] = [0, 100, 180]  # 橙色区域

        # 创建ROI
        rois = [
            ROI(
                name="散热器ROI",
                component_id="comp_001",
                roi_type=ROIType.DEFECT,
                bbox=BoundingBox(x=0.1, y=0.1, width=0.4, height=0.4),
            ),
            ROI(
                name="呼吸器ROI",
                component_id="comp_002",
                roi_type=ROIType.STATE,
                bbox=BoundingBox(x=0.3, y=0.3, width=0.3, height=0.3),
            ),
        ]

        # 创建上下文
        context = PluginContext(
            task_id="test_task_001",
            site_id="test_site",
            device_id="test_device",
            component_id="transformer_01",
        )

        # 执行推理
        results = plugin.infer(test_image, rois, context)
        print_result(True, f"检测到 {len(results)} 个结果")

        for r in results:
            print(f"    - ROI: {r.roi_id}, 标签: {r.label}, 置信度: {r.confidence:.2f}")

        # 测试后处理
        alarms = plugin.postprocess(results, [])
        print_result(True, f"生成 {len(alarms)} 个告警")

        for a in alarms:
            print(f"    - [{a.level.value}] {a.title}")

        return True

    except Exception as e:
        print_result(False, f"推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print_header("主变自主巡视插件测试")

    all_passed = True

    # 测试检测器
    if not test_detector():
        all_passed = False

    # 测试插件初始化
    success, plugin = test_plugin_init()
    if not success or plugin is None:
        all_passed = False
    else:
        # 测试健康检查
        if not test_healthcheck(plugin):
            all_passed = False

        # 测试推理
        if not test_inference(plugin):
            all_passed = False

    print_header("测试完成!")

    if all_passed:
        print("\n所有测试通过! ✓\n")
        return 0
    else:
        print("\n部分测试失败! ✗\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
