#!/usr/bin/env python3
"""Animal Detection - Demo"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


def main():
    from plugins.animal_detection.plugin import Plugin

    print("=== Animal Detection Demo ===")
    plugin = Plugin.create_standalone()
    print(f"Plugin: {plugin.id} v{plugin.version}")

    health = plugin.healthcheck()
    healthy = health.get("status", "error") == "healthy" if isinstance(health, dict) else getattr(health, "healthy", False)
    message = health.get("status", "") if isinstance(health, dict) else getattr(health, "message", "")
    print(f"Health: {'OK' if healthy else 'FAIL'} - {message}")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = plugin.infer(frame, [], None)
    print(f"Results: {results}")
    print("Done!")


if __name__ == "__main__":
    main()
