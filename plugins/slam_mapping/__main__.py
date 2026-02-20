#!/usr/bin/env python3
"""SLAM三维建图插件 - 独立启动入口

Usage:
    python -m plugins.slam_mapping
"""
from plugins.slam_mapping.standalone.app import main

if __name__ == "__main__":
    main()
