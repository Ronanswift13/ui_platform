#!/usr/bin/env python3
"""
Multimodal Fusion - Demo Script
Demonstrates plugin functionality with simulated multi-modal data.

Usage:
    python -m plugins.multimodal_fusion.demo.run_demo
"""
import sys
import json
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    print("=" * 60)
    print("Multimodal Fusion Plugin - Demo")
    print("=" * 60)

    from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin

    # Create standalone instance
    print("\n[1] Creating plugin instance...")
    plugin = MultimodalFusionPlugin.create_standalone()
    print(f"    Plugin: {plugin.name} v{plugin.version}")
    print(f"    Status: {plugin.status.value}")

    # Health check
    print("\n[2] Running health check...")
    health = plugin.healthcheck()
    print(f"    Healthy: {health.healthy}")
    print(f"    Message: {health.message}")

    # Process simulated multi-modal data
    print("\n[3] Processing simulated multi-modal data...")
    result = plugin.process({
        "device_id": "demo_device_001",
        "modalities": {
            "visual": {"confidence": 0.9, "status": "normal"},
            "acoustic": {"confidence": 0.85, "status": "normal"},
            "gas": {"confidence": 0.8, "status": "warning"},
            "thermal": {"confidence": 0.92, "status": "normal"},
        },
        "fusion_strategy": "hybrid",
    })
    print(f"    Success: {result['success']}")
    print(f"    Overall status: {result.get('overall_status', 'N/A')}")
    print(f"    Confidence: {result.get('confidence', 'N/A')}")
    if result.get("recommendations"):
        print(f"    Recommendations:")
        for rec in result["recommendations"]:
            print(f"      - {rec}")

    # Plugin status
    print("\n[4] Plugin status:")
    status = plugin.get_plugin_status()
    print(f"    {json.dumps(status, indent=4, ensure_ascii=False)}")

    # Shutdown
    print("\n[5] Shutting down plugin...")
    plugin.shutdown()
    print(f"    Status: {plugin.status.value}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
