#!/usr/bin/env python3
"""Performance benchmarking script for Indoor Fence Plugin."""
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


def main():
    from plugins.indoor_fence.plugin import Plugin

    print("=== Indoor Fence Plugin - Performance Benchmark ===\n")

    # Create standalone plugin
    print("Creating plugin instance...")
    plugin = Plugin.create_standalone()
    print(f"Plugin: {plugin.id} v{plugin.version}")
    print(f"Status: {plugin.status}\n")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Warmup: 5 frames
    print("Warming up (5 frames)...")
    for i in range(5):
        plugin.infer(frame, [], None)
    print("Warmup complete.\n")

    # Benchmark: 100 inferences
    num_iterations = 100
    print(f"Running benchmark ({num_iterations} inferences)...")
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        plugin.infer(frame, [], None)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    # Statistics
    times_arr = np.array(times)
    print(f"\n--- Benchmark Results ---")
    print(f"Total iterations: {num_iterations}")
    print(f"Mean:   {times_arr.mean():.2f} ms")
    print(f"Median: {np.median(times_arr):.2f} ms")
    print(f"Std:    {times_arr.std():.2f} ms")
    print(f"Min:    {times_arr.min():.2f} ms")
    print(f"Max:    {times_arr.max():.2f} ms")
    print(f"P95:    {np.percentile(times_arr, 95):.2f} ms")
    print(f"P99:    {np.percentile(times_arr, 99):.2f} ms")
    print(f"Throughput: {1000 / times_arr.mean():.1f} FPS")


if __name__ == "__main__":
    main()
