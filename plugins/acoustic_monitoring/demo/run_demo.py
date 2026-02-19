#!/usr/bin/env python3
"""
Acoustic Monitoring - Demo Script
Demonstrates plugin functionality with simulated audio data.

Usage:
    python -m plugins.acoustic_monitoring.demo.run_demo
"""
import sys
import json
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np


def main():
    print("=" * 60)
    print("Acoustic Monitoring Plugin - Demo")
    print("=" * 60)

    from plugins.acoustic_monitoring.plugin import Plugin

    # Create standalone instance
    print("\n[1] Creating plugin instance...")
    plugin = Plugin.create_standalone()
    print(f"    Plugin: {plugin.name} v{plugin.version}")
    print(f"    Status: {plugin.status.value}")

    # Health check
    print("\n[2] Running health check...")
    health = plugin.healthcheck()
    print(f"    Healthy: {health.healthy}")
    print(f"    Message: {health.message}")

    # Generate simulated audio
    print("\n[3] Processing simulated normal audio...")
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    normal_audio = (
        np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 100 * t)
        + 0.1 * np.random.randn(len(t))
    ).astype(np.float32)

    result = plugin.process({
        "audio": normal_audio,
        "sample_rate": sample_rate,
        "device_id": "demo_transformer_001",
    })

    print(f"    Success: {result['success']}")
    print(f"    Anomaly detected: {result['anomaly_detected']}")
    print(f"    Anomaly type: {result['anomaly_type']}")
    print(f"    Anomaly score: {result['anomaly_score']:.4f}")
    print(f"    Confidence: {result['confidence']:.4f}")

    freq = result.get("frequency_analysis", {})
    print(f"    Dominant frequency: {freq.get('dominant_frequency', 'N/A')} Hz")
    print(f"    Noise level: {freq.get('noise_level_db', 'N/A'):.1f} dB")

    # Simulate anomaly
    print("\n[4] Processing simulated anomaly audio...")
    anomaly_audio = normal_audio.copy()
    # Add high-frequency burst (simulated partial discharge)
    burst_start = int(0.5 * sample_rate)
    burst_len = int(0.01 * sample_rate)
    anomaly_audio[burst_start:burst_start + burst_len] += 3.0 * np.random.randn(burst_len).astype(np.float32)

    result2 = plugin.process({
        "audio": anomaly_audio,
        "sample_rate": sample_rate,
        "device_id": "demo_transformer_002",
    })

    print(f"    Success: {result2['success']}")
    print(f"    Anomaly detected: {result2['anomaly_detected']}")
    print(f"    Anomaly type: {result2['anomaly_type']}")
    print(f"    Anomaly score: {result2['anomaly_score']:.4f}")

    if result2.get("alarms"):
        print(f"    Alarms: {len(result2['alarms'])}")
        for alarm in result2["alarms"]:
            print(f"      - {alarm['message']} (level: {alarm['level']})")

    if result2.get("recommendations"):
        print(f"    Recommendations:")
        for rec in result2["recommendations"]:
            print(f"      - {rec}")

    # Plugin status
    print("\n[5] Plugin status:")
    status_info = plugin.get_plugin_status()
    print(f"    {json.dumps(status_info, indent=4, ensure_ascii=False)}")

    # Shutdown
    print("\n[6] Shutting down plugin...")
    plugin.shutdown()
    print(f"    Status: {plugin.status.value}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
