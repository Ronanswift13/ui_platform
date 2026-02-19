#!/usr/bin/env python3
"""Device Monitoring - Benchmark Script"""
import sys
import time
import statistics
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


def main():
    print("=" * 60)
    print("Device Monitoring Plugin - Benchmark")
    print("=" * 60)

    from plugins.device_monitoring.plugin import Plugin

    plugin = Plugin.create_standalone()
    print(f"Plugin: {plugin.name} v{plugin.version}")

    num_warmup = 5
    num_iterations = 100

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print(f"\nWarmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        plugin.infer(frame, [], None)

    print(f"Benchmarking ({num_iterations} iterations)...")
    latencies = []
    for i in range(num_iterations):
        start = time.perf_counter()
        plugin.infer(frame, [], None)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    print("\n--- Results ---")
    print(f"  Iterations:  {num_iterations}")
    print(f"  Mean:        {statistics.mean(latencies):.2f} ms")
    print(f"  Median:      {statistics.median(latencies):.2f} ms")
    print(f"  Std Dev:     {statistics.stdev(latencies):.2f} ms")
    print(f"  Min:         {min(latencies):.2f} ms")
    print(f"  Max:         {max(latencies):.2f} ms")
    print(f"  P95:         {sorted(latencies)[int(0.95 * len(latencies))]:.2f} ms")
    print(f"  P99:         {sorted(latencies)[int(0.99 * len(latencies))]:.2f} ms")
    print(f"  Throughput:  {1000 / statistics.mean(latencies):.1f} inferences/sec")

    plugin.shutdown()
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
