#!/usr/bin/env python3
"""电容器自主巡视插件 - 独立启动入口

Usage:
    python -m plugins.capacitor_inspection
"""
from plugins.capacitor_inspection.standalone.app import main

if __name__ == "__main__":
    main()
