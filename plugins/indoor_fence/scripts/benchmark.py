#!/usr/bin/env python3
"""Performance benchmarking script for Indoor Fence Plugin."""

import argparse
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

from darkbreaker_sdk.interfaces import PluginContext


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indoor Fence benchmark")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--width", type=int, default=640, help="Input frame width")
    parser.add_argument("--height", type=int, default=480, help="Input frame height")
    parser.add_argument("--target-ms", type=float, default=100.0, help="Realtime target in milliseconds")
    return parser.parse_args()


def main() -> int:
    args = _build_args()
    from plugins.indoor_fence.plugin import Plugin

    print("=== Indoor Fence Plugin - Performance Benchmark ===\n")

    plugin = Plugin.create_standalone()
    print(f"Plugin: {plugin.id} v{plugin.version}")
    print(f"Status: {plugin.status}\n")

    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    context = PluginContext(
        task_id="benchmark",
        site_id="standalone",
        device_id="benchmark",
        component_id="perf",
    )

    print(f"Warming up ({args.warmup} frames)...")
    for _ in range(max(args.warmup, 0)):
        plugin.infer(frame, [], context)
    print("Warmup complete.\n")

    num_iterations = max(args.iterations, 1)
    print(f"Running benchmark ({num_iterations} inferences)...")
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        plugin.infer(frame, [], context)
        elapsed = (time.perf_counter() - start) * 1000.0
        times.append(elapsed)

    times_arr = np.array(times, dtype=np.float64)
    mean_ms = float(times_arr.mean())
    p95_ms = float(np.percentile(times_arr, 95))
    p99_ms = float(np.percentile(times_arr, 99))

    print("\n--- Benchmark Results ---")
    print(f"Total iterations: {num_iterations}")
    print(f"Mean:   {mean_ms:.2f} ms")
    print(f"Median: {np.median(times_arr):.2f} ms")
    print(f"Std:    {times_arr.std():.2f} ms")
    print(f"Min:    {times_arr.min():.2f} ms")
    print(f"Max:    {times_arr.max():.2f} ms")
    print(f"P95:    {p95_ms:.2f} ms")
    print(f"P99:    {p99_ms:.2f} ms")
    print(f"Throughput: {1000.0 / mean_ms:.1f} FPS")

    target_pass = mean_ms <= args.target_ms
    print(f"Realtime target ({args.target_ms:.1f}ms): {'PASS' if target_pass else 'FAIL'}")

    plugin.cleanup()
    return 0 if target_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
