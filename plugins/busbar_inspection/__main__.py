#!/usr/bin/env python3
"""母线自主巡视插件 - 独立启动入口

Usage:
    python -m plugins.busbar_inspection
"""
from plugins.busbar_inspection.standalone.app import main

if __name__ == "__main__":
    main()
