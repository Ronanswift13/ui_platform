#!/usr/bin/env python3
"""多模态融合诊断插件 - 独立启动入口

Usage:
    # 从项目根目录运行:
    python -m plugins.multimodal_fusion

    # 从插件目录运行:
    cd plugins/multimodal_fusion && python -m standalone.app

    # 直接运行:
    python plugins/multimodal_fusion/run_standalone.py
"""
import sys
from pathlib import Path


def _setup_paths():
    """Setup import paths for dual-mode operation."""
    plugin_dir = Path(__file__).resolve().parent
    project_root = plugin_dir.parent.parent

    # Ensure project root is on path for absolute imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Ensure darkbreaker_sdk is importable
    sdk_dir = project_root / "darkbreaker_sdk"
    if sdk_dir.exists() and str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_setup_paths()

from plugins.multimodal_fusion.standalone.app import main

if __name__ == "__main__":
    main()
