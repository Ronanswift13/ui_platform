#!/usr/bin/env python3
"""
Capacitor Inspection - Demo Script
Demonstrates plugin functionality with simulated image data.

Usage:
    python -m plugins.capacitor_inspection.demo.run_demo
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
    print("Capacitor Inspection Plugin - Demo")
    print("=" * 60)

    from plugins.capacitor_inspection.plugin import CapacitorInspectionPlugin

    # Create standalone instance
    print("\n[1] Creating plugin instance...")
    plugin = CapacitorInspectionPlugin.create_standalone()
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
    from darkbreaker_sdk.schemas import BoundingBox, ROI, ROIType

    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    frame[120:420, 100:150] = [20, 20, 20]
    frame[110:420, 250:300] = [20, 20, 20]
    frame[130:420, 400:450] = [20, 20, 20]
    context = PluginContext(
        task_id="demo-task",
        site_id="demo-site",
        device_id="demo-camera",
        component_id="capacitor-bank-01",
    )
    rois = [ROI(
        id="roi-demo",
        name="capacitor_bank",
        component_id="capacitor-bank-01",
        roi_type=ROIType.DEFECT,
        bbox=BoundingBox(x=0.1, y=0.1, width=0.8, height=0.8),
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
