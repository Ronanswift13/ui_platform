#!/usr/bin/env python3
"""
Acoustic Monitoring - Benchmark Script
Measures plugin performance on simulated audio data.

Usage:
    python -m plugins.acoustic_monitoring.scripts.benchmark
"""
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
    print("Acoustic Monitoring Plugin - Benchmark")
    print("=" * 60)

    from plugins.acoustic_monitoring.plugin import Plugin

    plugin = Plugin.create_standalone()
    print(f"Plugin: {plugin.name} v{plugin.version}")

    # Benchmark parameters
    num_warmup = 5
    num_iterations = 50
    sample_rate = 16000
    duration = 2.0

    # Generate test audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (
        np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 100 * t)
        + 0.1 * np.random.randn(len(t))
    ).astype(np.float32)

    test_input = {
        "audio": audio,
        "sample_rate": sample_rate,
        "device_id": "bench_device_001",
    }

    # Warmup
    print(f"\nWarmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        plugin.process(test_input)

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    latencies = []
    for i in range(num_iterations):
        start = time.perf_counter()
        result = plugin.process(test_input)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        if not result.get("success"):
            print(f"  WARNING: iteration {i} failed: {result.get('error')}")

    # Results
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
