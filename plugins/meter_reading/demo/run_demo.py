#!/usr/bin/env python3
"""
Meter Reading - Demo Script
Demonstrates plugin functionality with simulated image data.

Usage:
    python -m plugins.meter_reading.demo.run_demo
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
    print("Meter Reading Plugin - Demo")
    print("=" * 60)

    from plugins.meter_reading.plugin import MeterReadingPlugin

    # Create standalone instance
    print("\n[1] Creating plugin instance...")
    plugin = MeterReadingPlugin.create_standalone()
    print(f"    Plugin: {plugin.id} v{plugin.version}")
    print(f"    Status: {plugin.status.value}")

    # Health check
    print("\n[2] Running health check...")
    health = plugin.healthcheck()
    print(f"    Healthy: {health.healthy}")
    print(f"    Message: {health.message}")

    # Simulated inference
    print("\n[3] Running simulated inference...")
    from darkbreaker_sdk.interfaces import PluginContext
    from darkbreaker_sdk.schemas import ROI, BoundingBox

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    context = PluginContext(
        task_id="demo-task",
        site_id="demo-site",
        device_id="demo-camera",
        component_id="meter-01",
    )
    rois = [ROI(
        id="roi-demo",
        name="pressure_gauge",
        component_id="meter-01",
        bbox=BoundingBox(x=0.2, y=0.2, width=0.6, height=0.6),
    )]

    results = plugin.infer(frame, rois, context)
    print(f"    Results: {len(results)}")

    alarms = plugin.postprocess(results, [])
    print(f"    Alarms: {len(alarms)}")

    # Final health check
    print("\n[4] Final health check...")
    health = plugin.healthcheck()
    print(f"    Healthy: {health.healthy}")
    print(f"    Details: {json.dumps(health.details, indent=4, ensure_ascii=False)}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
