#!/usr/bin/env python3
"""高光谱缺陷检测插件 - 独立启动入口

Usage:
    python -m plugins.hyperspectral_detection
"""
from plugins.hyperspectral_detection.standalone.app import main

if __name__ == "__main__":
    main()
