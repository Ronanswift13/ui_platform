#!/usr/bin/env python3
"""鸟类监控插件 - 独立启动入口

Usage:
    python -m plugins.bird_monitoring
"""
from plugins.bird_monitoring.standalone.app import main

if __name__ == "__main__":
    main()
