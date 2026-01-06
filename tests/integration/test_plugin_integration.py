"""
完整插件集成测试
测试所有插件是否正确集成到平台中
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from platform_core.plugin_manager import PluginManager
from platform_core.logging import setup_logging
from platform_core.plugin_manager.base import PluginContext
from platform_core.schema.models import ROI, BoundingBox, ROIType


def test_plugin_discovery():
    """测试插件发现"""
    print("=" * 70)
    print("测试1: 插件发现")
    print("=" * 70)

    pm = PluginManager()
    plugins = pm.discover_plugins()

    expected_plugins = {
        "transformer_inspection": "主变自主巡视插件",
        "switch_inspection": "开关间隔自主巡视插件",
        "busbar_inspection": "母线自主巡视插件",
        "capacitor_inspection": "电容器自主巡视插件",
        "meter_reading": "表计无建模增强读数插件",
    }

    print(f"\n发现 {len(plugins)} 个插件:")
    for p in plugins:
        status = "✓" if p.id in expected_plugins else "✗"
        print(f"  {status} {p.id} v{p.version}: {p.name}")

    found_ids = {p.id for p in plugins}
    missing = set(expected_plugins.keys()) - found_ids

    if missing:
        print(f"\n✗ 缺少插件: {missing}")
        return False

    print("\n✓ 所有预期插件都被发现")
    return True


def test_plugin_loading():
    """测试插件加载和初始化"""
    print("\n" + "=" * 70)
    print("测试2: 插件加载和初始化")
    print("=" * 70)

    pm = PluginManager()
    plugins = pm.discover_plugins()

    loaded_count = 0
    healthy_count = 0

    for manifest in plugins:
        print(f"\n[{manifest.id}]")
        try:
            # 加载插件
            plugin = pm.load_plugin(manifest.id)
            print(f"  ✓ 加载成功")
            loaded_count += 1

            # 健康检查
            health = plugin.healthcheck()
            if health.healthy:
                print(f"  ✓ 健康检查通过: {health.message}")
                healthy_count += 1
            else:
                print(f"  ✗ 健康检查失败: {health.message}")

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")

    print(f"\n加载成功: {loaded_count}/{len(plugins)}")
    print(f"健康检查通过: {healthy_count}/{len(plugins)}")

    success = loaded_count == len(plugins) and healthy_count == len(plugins)
    if success:
        print("\n✓ 所有插件都成功加载并通过健康检查")
    else:
        print("\n✗ 部分插件加载或健康检查失败")

    return success


def test_plugin_inference():
    """测试插件推理功能"""
    print("\n" + "=" * 70)
    print("测试3: 插件推理功能")
    print("=" * 70)

    pm = PluginManager()

    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[100:380, 100:540] = [100, 100, 100]  # 添加一些内容

    # 创建测试ROI
    test_roi = ROI(
        id="test_roi_001",
        component_id="component_001",
        bbox=BoundingBox(x=0.1, y=0.1, width=0.8, height=0.8),
        roi_type=ROIType.DEFECT,
        name="测试ROI",
    )

    # 创建测试上下文
    context = PluginContext(
        task_id="test_task_001",
        site_id="site_001",
        device_id="device_001",
        component_id="component_001",
    )

    plugins_to_test = [
        "transformer_inspection",
        "switch_inspection",
        "busbar_inspection",
        "capacitor_inspection",
        "meter_reading",
    ]

    success_count = 0

    for plugin_id in plugins_to_test:
        print(f"\n[{plugin_id}]")
        try:
            plugin = pm.load_plugin(plugin_id)

            # 执行推理
            results = plugin.infer(test_image, [test_roi], context)

            print(f"  ✓ 推理执行成功")
            print(f"  - 返回结果数: {len(results)}")

            # 验证结果格式
            for i, result in enumerate(results[:3]):  # 只显示前3个
                print(f"  - 结果{i+1}: label={result.label}, confidence={result.confidence:.3f}")

            success_count += 1

        except Exception as e:
            print(f"  ✗ 推理执行失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n推理测试通过: {success_count}/{len(plugins_to_test)}")

    success = success_count == len(plugins_to_test)
    if success:
        print("\n✓ 所有插件推理功能正常")
    else:
        print("\n✗ 部分插件推理功能异常")

    return success


def test_plugin_postprocess():
    """测试插件后处理功能"""
    print("\n" + "=" * 70)
    print("测试4: 插件后处理功能")
    print("=" * 70)

    pm = PluginManager()

    # 创建测试图像和ROI
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_roi = ROI(
        id="test_roi_001",
        component_id="component_001",
        bbox=BoundingBox(x=0.1, y=0.1, width=0.8, height=0.8),
        roi_type=ROIType.DEFECT,
        name="测试ROI",
    )

    context = PluginContext(
        task_id="test_task_001",
        site_id="site_001",
        device_id="device_001",
        component_id="component_001",
    )

    plugins_to_test = ["transformer_inspection", "switch_inspection", "busbar_inspection"]
    success_count = 0

    for plugin_id in plugins_to_test:
        print(f"\n[{plugin_id}]")
        try:
            plugin = pm.load_plugin(plugin_id)

            # 执行推理
            results = plugin.infer(test_image, [test_roi], context)

            # 执行后处理
            alarms = plugin.postprocess(results, [])

            print(f"  ✓ 后处理执行成功")
            print(f"  - 生成告警数: {len(alarms)}")

            for i, alarm in enumerate(alarms[:3]):
                print(f"  - 告警{i+1}: {alarm.title} [{alarm.level.value}]")

            success_count += 1

        except Exception as e:
            print(f"  ✗ 后处理执行失败: {e}")

    print(f"\n后处理测试通过: {success_count}/{len(plugins_to_test)}")

    success = success_count == len(plugins_to_test)
    if success:
        print("\n✓ 所有插件后处理功能正常")
    else:
        print("\n✗ 部分插件后处理功能异常")

    return success


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("插件集成完整性测试")
    print("输变电激光星芒破夜绘明监测平台")
    print("=" * 70)

    # 初始化日志系统
    setup_logging(log_level='WARNING', console=False)

    # 运行所有测试
    tests = [
        ("插件发现", test_plugin_discovery),
        ("插件加载", test_plugin_loading),
        ("插件推理", test_plugin_inference),
        ("插件后处理", test_plugin_postprocess),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{test_name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 汇总结果
    print("\n\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, r in results if r)

    print(f"\n总计: {total} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {total - passed} 个")

    if passed == total:
        print("\n" + "=" * 70)
        print("✓✓✓ 所有测试通过！所有插件功能正常集成到平台中 ✓✓✓")
        print("=" * 70)
        return 0
    else:
        print("\n✗ 部分测试失败，请检查上述错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
