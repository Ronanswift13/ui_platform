#!/usr/bin/env python3
"""Demo script for Indoor Fence Plugin."""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


def main():
    from plugins.indoor_fence.plugin import Plugin

    print("=== Indoor Fence Plugin Demo ===")
    plugin = Plugin.create_standalone()
    print(f"Plugin: {plugin.id} v{plugin.version}")
    print(f"Status: {plugin.status}")

    health = plugin.healthcheck()
    print(f"Health: {'OK' if health.healthy else 'FAIL'} - {health.message}")

    # Test with dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    results = plugin.infer(frame, [], None)
    print(f"Detection results: {len(results)}")

    print("Demo complete!")


if __name__ == "__main__":
    main()
