#!/usr/bin/env python3
"""Minimal local replay demo for action_event_monitoring."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


PLUGIN_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PLUGIN_DIR.parent.parent

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from plugins.action_event_monitoring.plugin import Plugin


def main() -> None:
    config_path = PLUGIN_DIR / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    plugin = Plugin.create_standalone(config)
    sample = {
        "device_id": "relay_demo_01",
        "timestamp": datetime.now().isoformat(),
        "signal_changes": [
            {
                "signal_id": "SOE-DEMO-TRIP",
                "signal_name": "Demo Protection Trip",
                "signal_group": "protection",
                "action_type": "protection_trip",
                "action_desc": "demo protection trip event",
                "value_after": "1",
                "severity_hint": "warning",
            }
        ],
        "context": {"task_id": "action-event-demo", "site_id": "demo"},
    }

    result = plugin.process(sample)
    print("=== Action Event Monitoring Demo ===")
    print(f"Plugin: {plugin.id} v{plugin.version}")
    print(f"Healthy: {plugin.healthcheck().healthy}")
    print(f"Success: {result['success']}")
    print(f"Status: {result['status']}")
    print(f"Stored events: {len(result.get('stored_event_ids', []))}")
    print(f"Analysis triggered: {result.get('analysis_triggered')}")
    print("Summary:")
    print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
