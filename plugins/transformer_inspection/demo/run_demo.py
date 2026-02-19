#!/usr/bin/env python3
"""
Demo script for the Transformer Inspection plugin.

Demonstrates standalone creation, healthcheck, and inference with a dummy frame.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from plugins.transformer_inspection.plugin import TransformerInspectionPlugin
from darkbreaker_sdk.interfaces import PluginContext
from darkbreaker_sdk.schemas import ROI, BoundingBox


def main():
    print("=" * 60)
    print("Transformer Inspection Plugin - Demo")
    print("=" * 60)

    # 1. Create standalone instance
    print("\n[1] Creating standalone plugin instance...")
    plugin = TransformerInspectionPlugin.create_standalone()
    print(f"    Plugin ID: {plugin.id}")
    print(f"    Version:   {plugin.version}")

    # 2. Healthcheck
    print("\n[2] Running healthcheck...")
    health = plugin.healthcheck()
    print(f"    Healthy: {health.healthy}")
    print(f"    Message: {health.message}")

    # 3. Test inference with dummy frame
    print("\n[3] Running inference with dummy frame...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    context = PluginContext(
        task_id="demo-task-001",
        site_id="demo-site",
        device_id="demo-camera",
        component_id="transformer-01",
    )

    rois = [
        ROI(
            id="roi-1",
            name="transformer_body",
            component_id="transformer-01",
            bbox=BoundingBox(x=0.1, y=0.1, width=0.8, height=0.8),
        )
    ]

    results = plugin.infer(frame=dummy_frame, rois=rois, context=context)
    print(f"    Results: {len(results)} detections")

    for r in results:
        print(f"      - {r.label}: confidence={r.confidence:.3f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
