#!/usr/bin/env python3
"""
插件诊断工具

帮助诊断为什么插件未被加载
"""

import os
import sys
from pathlib import Path
import json

print("=" * 60)
print("插件诊断工具")
print("=" * 60)

# 1. 检查当前目录结构
print("\n[步骤 1] 检查项目结构...")
current_dir = Path.cwd()
print(f"当前目录: {current_dir}")

# 查找 plugins 目录
possible_plugin_dirs = [
    current_dir / "plugins",
    current_dir.parent / "plugins",
    Path("/mnt/user-data/uploads").parent.parent / "plugins",
]

plugins_dir = None
for dir_path in possible_plugin_dirs:
    if dir_path.exists():
        plugins_dir = dir_path
        print(f"✓ 找到 plugins 目录: {plugins_dir}")
        break

if not plugins_dir:
    print("✗ 未找到 plugins 目录")
    print("\n可能的原因:")
    print("  1. 您不在项目根目录")
    print("  2. plugins 目录不存在")
    print("\n解决方案:")
    print("  cd /path/to/your/project")
    print("  mkdir -p plugins")
    sys.exit(1)

# 2. 检查 transformer_inspection 插件
print("\n[步骤 2] 检查主变巡视插件...")
transformer_dir = plugins_dir / "transformer_inspection"

if not transformer_dir.exists():
    print(f"✗ 插件目录不存在: {transformer_dir}")
    print("\n解决方案:")
    print(f"  cp -r /mnt/user-data/outputs/transformer_inspection {plugins_dir}/")
    sys.exit(1)
else:
    print(f"✓ 插件目录存在: {transformer_dir}")

# 3. 检查必需文件
print("\n[步骤 3] 检查插件文件...")
required_files = {
    "manifest.json": "插件清单",
    "plugin.py": "主插件文件",
    "detector.py": "检测器",
    "__init__.py": "模块初始化",
}

missing_files = []
for filename, description in required_files.items():
    file_path = transformer_dir / filename
    if file_path.exists():
        print(f"  ✓ {description}: {filename}")
    else:
        print(f"  ✗ {description}: {filename} (缺失)")
        missing_files.append(filename)

if missing_files:
    print(f"\n✗ 缺少 {len(missing_files)} 个文件，插件无法加载")
    sys.exit(1)

# 4. 验证 manifest.json
print("\n[步骤 4] 验证 manifest.json...")
manifest_path = transformer_dir / "manifest.json"
try:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    required_keys = ["id", "name", "version", "entrypoint", "plugin_class"]
    for key in required_keys:
        if key in manifest:
            print(f"  ✓ {key}: {manifest[key]}")
        else:
            print(f"  ✗ 缺少字段: {key}")
    
    if manifest.get("id") != "transformer_inspection":
        print(f"  ⚠ 警告: id 不是 'transformer_inspection'")
    
except json.JSONDecodeError as e:
    print(f"  ✗ JSON 格式错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ 读取失败: {e}")
    sys.exit(1)

# 5. 检查 platform_core
print("\n[步骤 5] 检查平台核心...")
platform_core = current_dir / "platform_core"
if not platform_core.exists():
    print(f"✗ platform_core 目录不存在")
    print("  这可能导致插件无法加载")
else:
    print(f"✓ platform_core 存在")
    
    # 检查插件管理器
    plugin_manager = platform_core / "plugin_manager"
    if plugin_manager.exists():
        print(f"  ✓ plugin_manager 模块存在")
    else:
        print(f"  ✗ plugin_manager 模块不存在")

# 6. 测试插件加载
print("\n[步骤 6] 测试插件加载...")
sys.path.insert(0, str(current_dir))

try:
    # 尝试导入平台模块
    from platform_core.plugin_manager import PluginManager
    
    pm = PluginManager()
    plugins = pm.discover_plugins()
    
    print(f"✓ 发现 {len(plugins)} 个插件:")
    for p in plugins:
        status = "✓" if p.id == "transformer_inspection" else " "
        print(f"  {status} {p.id} v{p.version}")
    
    # 检查是否找到我们的插件
    transformer_found = any(p.id == "transformer_inspection" for p in plugins)
    if transformer_found:
        print("\n✓ 主变巡视插件已被发现!")
        
        # 尝试加载
        try:
            plugin = pm.load_plugin("transformer_inspection")
            print(f"✓ 插件加载成功，状态: {plugin.status}")
        except Exception as e:
            print(f"✗ 插件加载失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n✗ 主变巡视插件未被发现")
        print("  可能的原因:")
        print("  1. manifest.json 格式错误")
        print("  2. 插件管理器配置问题")
        
except ImportError as e:
    print(f"✗ 无法导入平台模块: {e}")
    print("\n可能的原因:")
    print("  1. 不在项目根目录")
    print("  2. platform_core 未正确安装")
    print("\n解决方案:")
    print("  cd /path/to/your/project")
    print("  python -m pip install -e .")
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 7. 检查 UI 配置
print("\n[步骤 7] 检查 UI 配置...")
ui_server = current_dir / "apps" / "ui_server.py"
if ui_server.exists():
    print("✓ UI 服务器文件存在")
    
    # 检查模块配置
    with open(ui_server, 'r', encoding='utf-8') as f:
        content = f.read()
        if "transformer_inspection" in content or "MODULES" in content:
            print("  ✓ UI 配置包含模块定义")
        else:
            print("  ⚠ UI 配置可能需要更新")
else:
    print("✗ UI 服务器文件不存在")

print("\n" + "=" * 60)
print("诊断完成!")
print("=" * 60)

print("\n📋 诊断总结:")
print("1. 如果所有检查都通过，尝试重启平台: python run.py")
print("2. 如果插件未被发现，检查 manifest.json 格式")
print("3. 如果仍有问题，查看日志: logs/platform.log")
print("\n💡 快速修复命令:")
print(f"   cp -r /mnt/user-data/outputs/transformer_inspection {plugins_dir}/")
print("   python run.py")
