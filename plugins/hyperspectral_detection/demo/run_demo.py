#!/usr/bin/env python3
"""
Hyperspectral Detection - Demo Script
Demonstrates plugin functionality with simulated spectral data.

Usage:
    python -m plugins.hyperspectral_detection.demo.run_demo
"""
import sys
import json
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


def main():
    print("=" * 60)
    print("Hyperspectral Detection Plugin - Demo")
    print("=" * 60)

    from plugins.hyperspectral_detection.plugin import HyperspectralDetectionPlugin

    # Create standalone instance
    print("\n[1] Creating plugin instance...")
    plugin = HyperspectralDetectionPlugin.create_standalone()
    print(f"    Plugin: {plugin.name} v{plugin.version}")
    print(f"    Status: {plugin.status.value}")

    # Health check
    print("\n[2] Running health check...")
    health = plugin.healthcheck()
    print(f"    Healthy: {health.healthy}")
    print(f"    Message: {health.message}")

    # Process simulated hyperspectral image
    print("\n[3] Processing simulated hyperspectral image...")
    result = plugin.process({
        "device_id": "demo_device_001",
        "image": np.random.rand(224, 256, 256).astype(np.float32),
        "analysis_type": "full",
    })
    print(f"    Success: {result['success']}")
    print(f"    Overall status: {result.get('overall_status', 'N/A')}")
    if result.get("material_analysis"):
        print(f"    Primary material: {result['material_analysis'].get('primary_material', 'N/A')}")
    if result.get("recommendations"):
        print(f"    Recommendations:")
        for rec in result["recommendations"]:
            print(f"      - {rec}")

    # Plugin status
    print("\n[4] Plugin status:")
    status = plugin.get_plugin_status()
    print(f"    {json.dumps(status, indent=4, ensure_ascii=False)}")

    # Shutdown
    print("\n[5] Shutting down plugin...")
    plugin.shutdown()
    print(f"    Status: {plugin.status.value}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
