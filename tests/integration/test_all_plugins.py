"""
完整的插件测试脚本
测试所有5个插件的加载、初始化和基本功能
"""

import sys
from pathlib import Path
import json
import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from platform_core.plugin_manager.base import PluginManifest


def test_plugin(plugin_name: str, plugin_class_name: str) -> dict:
    """测试单个插件"""
    result = {
        "plugin": plugin_name,
        "import": False,
        "manifest": False,
        "config": False,
        "instantiate": False,
        "init": False,
        "healthcheck": False,
        "errors": []
    }

    try:
        # 1. 测试导入
        plugin_module = __import__(f"{plugin_name}.plugin", fromlist=[plugin_class_name])
        plugin_class = getattr(plugin_module, plugin_class_name)
        result["import"] = True

        # 2. 测试manifest加载
        plugins_dir = Path(__file__).parent
        manifest_path = plugins_dir / plugin_name / "manifest.json"
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        manifest = PluginManifest(**manifest_data)
        result["manifest"] = True

        # 3. 测试配置加载
        config_path = plugins_dir / plugin_name / "configs" / "default.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        result["config"] = True

        # 4. 测试实例化
        plugin_dir = plugins_dir / plugin_name
        plugin_instance = plugin_class(manifest, plugin_dir)
        result["instantiate"] = True

        # 5. 测试初始化
        init_success = plugin_instance.init(config)
        result["init"] = init_success

        if not init_success:
            result["errors"].append(f"初始化返回False: {getattr(plugin_instance, '_last_error', 'Unknown')}")

        # 6. 测试健康检查
        health = plugin_instance.healthcheck()
        result["healthcheck"] = health.healthy

        if not health.healthy:
            result["errors"].append(f"健康检查失败: {health.message}")

    except Exception as e:
        result["errors"].append(f"异常: {str(e)}")
        import traceback
        result["errors"].append(traceback.format_exc())

    return result


def main():
    """主测试函数"""
    print("=" * 70)
    print("插件完整性测试")
    print("=" * 70)
    print()

    plugins = [
        ("transformer_inspection", "TransformerInspectionPlugin", "主变自主巡视"),
        ("switch_inspection", "SwitchInspectionPlugin", "开关间隔巡视"),
        ("busbar_inspection", "BusbarInspectionPlugin", "母线巡视"),
        ("capacitor_inspection", "CapacitorInspectionPlugin", "电容器巡视"),
        ("meter_reading", "MeterReadingPlugin", "表计读数"),
    ]

    all_results = []

    for plugin_name, class_name, display_name in plugins:
        print(f"\n{'=' * 70}")
        print(f"测试插件: {display_name} ({plugin_name})")
        print(f"{'=' * 70}")

        result = test_plugin(plugin_name, class_name)
        all_results.append(result)

        # 打印结果
        print(f"  ✓ 导入模块: {'成功' if result['import'] else '失败'}")
        print(f"  ✓ 加载清单: {'成功' if result['manifest'] else '失败'}")
        print(f"  ✓ 加载配置: {'成功' if result['config'] else '失败'}")
        print(f"  ✓ 创建实例: {'成功' if result['instantiate'] else '失败'}")
        print(f"  ✓ 初始化: {'成功' if result['init'] else '失败'}")
        print(f"  ✓ 健康检查: {'成功' if result['healthcheck'] else '失败'}")

        if result['errors']:
            print(f"\n  错误信息:")
            for error in result['errors']:
                for line in error.split('\n'):
                    if line.strip():
                        print(f"    {line}")

    # 汇总统计
    print(f"\n\n{'=' * 70}")
    print("测试汇总")
    print(f"{'=' * 70}")

    total = len(all_results)
    passed = sum(1 for r in all_results if r['init'] and r['healthcheck'])

    print(f"\n总计: {total} 个插件")
    print(f"通过: {passed} 个插件")
    print(f"失败: {total - passed} 个插件")

    if passed == total:
        print("\n✓ 所有插件测试通过！")
        return 0
    else:
        print("\n✗ 部分插件测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
