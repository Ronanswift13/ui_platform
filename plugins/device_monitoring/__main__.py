#!/usr/bin/env python3
"""设备状态监测插件 - 独立启动入口

Usage:
    python -m plugins.device_monitoring
"""
from plugins.device_monitoring.standalone.app import main

if __name__ == "__main__":
    main()
