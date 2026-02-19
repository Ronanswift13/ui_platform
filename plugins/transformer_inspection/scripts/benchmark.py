#!/usr/bin/env python3
"""
Performance benchmark for the Transformer Inspection plugin.

Runs 100 inference iterations and reports timing statistics.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from plugins.transformer_inspection.plugin import TransformerInspectionPlugin
from darkbreaker_sdk.interfaces import PluginContext
from darkbreaker_sdk.schemas import ROI, BoundingBox

NUM_ITERATIONS = 100


def main():
    print("Transformer Inspection - Performance Benchmark")
    print("=" * 60)

    plugin = TransformerInspectionPlugin.create_standalone()

    context = PluginContext(
        task_id="benchmark-task",
        site_id="benchmark-site",
        device_id="benchmark-camera",
        component_id="transformer-01",
    )

    rois = [
        ROI(
            id="roi-bench",
            name="transformer_body",
            component_id="transformer-01",
            bbox=BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0),
        )
    ]

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warm-up
    print("Warming up...")
    for _ in range(5):
        plugin.infer(frame=dummy_frame, rois=rois, context=context)

    # Benchmark
    print(f"Running {NUM_ITERATIONS} iterations...")
    times = []
    for i in range(NUM_ITERATIONS):
        start = time.perf_counter()
        plugin.infer(frame=dummy_frame, rois=rois, context=context)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times_arr = np.array(times)
    print(f"\nResults ({NUM_ITERATIONS} iterations):")
    print(f"  Mean:   {times_arr.mean():.2f} ms")
    print(f"  Median: {np.median(times_arr):.2f} ms")
    print(f"  Std:    {times_arr.std():.2f} ms")
    print(f"  Min:    {times_arr.min():.2f} ms")
    print(f"  Max:    {times_arr.max():.2f} ms")
    print(f"  P95:    {np.percentile(times_arr, 95):.2f} ms")
    print(f"  P99:    {np.percentile(times_arr, 99):.2f} ms")


if __name__ == "__main__":
    main()
