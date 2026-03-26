#!/usr/bin/env python3
"""
温度监测 - 独立运行入口
输变电激光星芒破夜绘明监测平台

可直接运行此文件启动插件独立服务:
    python run_standalone.py

或在VS Code中右键 → Run Python File

Web界面: http://localhost:8085
API文档: http://localhost:8085/docs
"""
import os
import sys
from pathlib import Path

# ============================================================
# 路径设置 - 支持独立运行和项目内运行两种模式
# ============================================================
PLUGIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLUGIN_DIR.parent.parent

# ============================================================
# 虚拟环境自动切换 - 防止系统 Python 缺少项目依赖
# 优先使用项目根目录 venv，其次兼容插件目录 .venv
# ============================================================
PROJECT_VENV_PYTHON = PROJECT_ROOT / 'venv' / 'bin' / 'python'
PLUGIN_VENV_PYTHON = PLUGIN_DIR / '.venv' / 'bin' / 'python'

if sys.prefix == sys.base_prefix:
    target_python = None
    if PROJECT_VENV_PYTHON.exists():
        target_python = PROJECT_VENV_PYTHON
    elif PLUGIN_VENV_PYTHON.exists():
        target_python = PLUGIN_VENV_PYTHON

    if target_python is not None and Path(sys.executable).resolve() != target_python.resolve():
        os.execv(str(target_python), [str(target_python)] + sys.argv)

    if target_python is None:
        print(
            f"\n[ERROR] 未检测到可用虚拟环境，当前解释器: {sys.executable}\n"
            f"请先在项目根目录创建并安装依赖:\n"
            f"  cd {PROJECT_ROOT}\n"
            f"  python3 -m venv venv\n"
            f"  ./venv/bin/pip install -r requirements.txt\n"
            f"然后使用以下命令启动:\n"
            f"  ./venv/bin/python {__file__}\n"
        )
        sys.exit(1)

# 添加项目根目录以支持 darkbreaker_sdk 和其他插件导入
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 添加 plugins 父目录以支持 plugins.xxx 导入
plugins_parent = PLUGIN_DIR.parent.parent
if str(plugins_parent) not in sys.path:
    sys.path.insert(0, str(plugins_parent))


def main():
    from darkbreaker_sdk.standalone import StandalonePluginRunner
    from plugins.temperature_monitoring.plugin import Plugin

    plugin = Plugin.create_standalone()
    runner = StandalonePluginRunner(
        plugin,
        plugin_templates_dir=PLUGIN_DIR / "standalone" / "templates",
        plugin_static_dir=PLUGIN_DIR / "standalone" / "static",
    )
    runner.run(host="0.0.0.0", port=8085)


if __name__ == "__main__":
    main()
