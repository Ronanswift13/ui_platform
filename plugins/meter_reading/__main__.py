#!/usr/bin/env python3
"""表计读数插件 - 独立启动入口

Usage:
    python -m plugins.meter_reading
"""
from plugins.meter_reading.standalone.app import main

if __name__ == "__main__":
    main()
