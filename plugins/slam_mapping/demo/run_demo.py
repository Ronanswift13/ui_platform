#!/usr/bin/env python3
"""SLAM Mapping - Demo"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


def main():
    from plugins.slam_mapping.plugin import Plugin

    print("=== SLAM Mapping Demo ===")
    plugin = Plugin.create_standalone()
    print(f"Plugin: {plugin.PLUGIN_ID} v{plugin.PLUGIN_VERSION}")

    health = plugin.healthcheck()
    print(f"Health: {'OK' if health.healthy else 'FAIL'} - {health.message}")

    # Generate test point cloud
    np.random.seed(42)
    ground = np.random.randn(500, 3) * [20, 20, 0.1]
    ground[:, 2] = 0
    objects = np.random.randn(100, 3) * [1, 1, 2] + [5, 5, 1]
    points = np.vstack([ground, objects])

    result = plugin.process_point_cloud(points)
    print(f"Point cloud result: {result.get('obstacles_detected', 0)} obstacles")

    status = plugin.get_status()
    print(f"Status: frame_count={status['frame_count']}, devices={status['registered_devices']}")
    print("Done!")


if __name__ == "__main__":
    main()
