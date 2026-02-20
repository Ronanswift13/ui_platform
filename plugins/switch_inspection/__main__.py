#!/usr/bin/env python3
"""开关间隔自主巡视插件 - 独立启动入口

Usage:
    python -m plugins.switch_inspection
"""
from plugins.switch_inspection.standalone.app import main

if __name__ == "__main__":
    main()
