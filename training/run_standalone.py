#!/usr/bin/env python3
"""
训练调试平台 - 独立运行入口
输变电激光星芒破夜绘明监测平台

可直接运行此文件启动训练调试UI:
    python run_standalone.py

Web界面: http://localhost:8095
API文档: http://localhost:8095/docs
"""
import os
import sys
from pathlib import Path

# ============================================================
# 路径设置 - 支持独立运行和项目内运行两种模式
# ============================================================
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent

# ============================================================
# 虚拟环境自动切换 - 防止系统 Python 缺少项目依赖
# 优先使用项目根目录 venv，其次兼容训练目录 .venv
# ============================================================
PROJECT_VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
LOCAL_VENV_PYTHON = TRAINING_DIR / ".venv" / "bin" / "python"

if sys.prefix == sys.base_prefix:
    target_python = None
    if PROJECT_VENV_PYTHON.exists():
        target_python = PROJECT_VENV_PYTHON
    elif LOCAL_VENV_PYTHON.exists():
        target_python = LOCAL_VENV_PYTHON

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

# 添加项目根目录以支持所有模块导入
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from training.standalone.app import main as standalone_main
    standalone_main()


if __name__ == "__main__":
    main()
