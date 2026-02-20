#!/usr/bin/env python3
"""气体泄漏检测插件 - 独立启动入口

Usage:
    python -m plugins.gas_detection
"""
from plugins.gas_detection.standalone.app import main

if __name__ == "__main__":
    main()
