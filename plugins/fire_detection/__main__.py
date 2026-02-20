#!/usr/bin/env python3
"""消防监测插件 - 独立启动入口

Usage:
    python -m plugins.fire_detection
"""
from plugins.fire_detection.standalone.app import main

if __name__ == "__main__":
    main()
