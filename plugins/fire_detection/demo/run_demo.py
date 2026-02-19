#!/usr/bin/env python3
"""Fire Detection - Demo"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


def main():
    from plugins.fire_detection.plugin import Plugin

    print("=== Fire Detection Demo ===")
    plugin = Plugin.create_standalone()
    print(f"Plugin: {plugin.id} v{plugin.version}")

    health = plugin.healthcheck()
    print(f"Health: {'OK' if health.healthy else 'FAIL'} - {health.message}")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = plugin.infer(frame, [], None)
    print(f"Results: {len(results)}")
    print("Done!")


if __name__ == "__main__":
    main()
