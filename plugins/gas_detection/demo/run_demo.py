#!/usr/bin/env python3
"""
Gas Detection - Demo Script
Demonstrates plugin functionality with simulated gas data.

Usage:
    python -m plugins.gas_detection.demo.run_demo
"""
import sys
import json
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    print("=" * 60)
    print("Gas Detection Plugin - Demo")
    print("=" * 60)

    from plugins.gas_detection.plugin import GasDetectionPlugin

    # Create standalone instance
    print("\n[1] Creating plugin instance...")
    plugin = GasDetectionPlugin.create_standalone()
    print(f"    Plugin: {plugin.name} v{plugin.version}")
    print(f"    Status: {plugin.status.value}")

    # Health check
    print("\n[2] Running health check...")
    health = plugin.healthcheck()
    print(f"    Healthy: {health.healthy}")
    print(f"    Message: {health.message}")

    # Process normal gas readings
    print("\n[3] Processing normal gas readings...")
    result = plugin.process({
        "device_id": "demo_device_001",
        "gas_readings": {
            "SF6": 200,
            "H2": 30,
            "CO": 50,
        },
        "environmental": {
            "temperature": 25.0,
            "humidity": 60.0,
        },
    })
    print(f"    Success: {result['success']}")
    print(f"    Overall status: {result.get('overall_status', 'N/A')}")
    print(f"    Health index: {result.get('health_index', 'N/A')}")

    # Process elevated gas readings
    print("\n[4] Processing elevated gas readings...")
    result2 = plugin.process({
        "device_id": "demo_device_002",
        "gas_readings": {
            "SF6": 1200,
            "H2": 200,
            "CO": 400,
        },
        "environmental": {
            "temperature": 35.0,
            "humidity": 80.0,
        },
    })
    print(f"    Success: {result2['success']}")
    print(f"    Overall status: {result2.get('overall_status', 'N/A')}")
    print(f"    Alarms: {len(result2.get('alarms', []))}")
    if result2.get("recommendations"):
        print(f"    Recommendations:")
        for rec in result2["recommendations"]:
            print(f"      - {rec}")

    # Plugin status
    print("\n[5] Plugin status:")
    status = plugin.get_plugin_status()
    print(f"    {json.dumps(status, indent=4, ensure_ascii=False)}")

    # Shutdown
    print("\n[6] Shutting down plugin...")
    plugin.shutdown()
    print(f"    Status: {plugin.status.value}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
