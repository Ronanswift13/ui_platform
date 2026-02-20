#!/usr/bin/env python3
"""温度监测插件 - 独立启动入口

Usage:
    python -m plugins.temperature_monitoring
"""
from plugins.temperature_monitoring.standalone.app import main

if __name__ == "__main__":
    main()
