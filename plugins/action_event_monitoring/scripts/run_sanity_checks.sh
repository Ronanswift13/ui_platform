#!/usr/bin/env bash
# run_sanity_checks.sh
# action_event_monitoring 最小治理级校验入口

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$SCRIPT_DIR")"
PLUGINS_DIR="$(dirname "$PLUGIN_DIR")"
REPO_DIR="$(dirname "$PLUGINS_DIR")"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "[ERROR] python interpreter not found (expected python3 or python)"
  exit 127
fi

cd "$PLUGIN_DIR"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================"
echo "Sanity Checks - action_event_monitoring"
echo "========================================"

"$PYTHON_CMD" - <<'PY'
import json
from pathlib import Path

import yaml

from plugins.action_event_monitoring.plugin import ActionEventMonitoringPlugin, Plugin

assert Plugin is ActionEventMonitoringPlugin

plugin_dir = Path.cwd()

manifest = json.loads((plugin_dir / "manifest.json").read_text(encoding="utf-8"))
assert manifest["id"] == "action_event_monitoring", manifest
assert manifest["config_file"] == "configs/default.yaml", manifest
assert manifest["entrypoint"] == "plugin.py", manifest
assert manifest["plugin_class"] == "ActionEventMonitoringPlugin", manifest
assert manifest.get("runtime_requirements") == "requirements.txt", manifest
assert manifest["standalone"]["port"] == 8097, manifest
assert (plugin_dir / "requirements.txt").exists()
assert (plugin_dir / "demo" / "run_demo.py").exists()
assert (plugin_dir / "__main__.py").exists()
assert (plugin_dir / "run_standalone.py").exists()

config = yaml.safe_load((plugin_dir / "configs" / "default.yaml").read_text(encoding="utf-8"))
assert isinstance(config.get("subscriptions"), list), config
assert "analysis" in config, config

plugin = ActionEventMonitoringPlugin()
assert plugin.init(config) is True

result = plugin.process({
    "signal_id": "SIG-SANITY-1",
    "signal_name": "Sanity Protection Trip",
    "action_type": "protection_trip",
    "action_desc": "保护出口",
    "station_id": "ST-DALI-220",
    "bay_id": "BAY-DALI-JIA",
    "secondary_device_id": "PR-JIA-MAIN",
    "source_system": "simulated",
    "timestamp": "2026-04-08T12:00:00",
})

assert result["success"] is True, result
assert isinstance(result["stored_event_ids"], list) and result["stored_event_ids"], result

status = plugin.get_status()
assert status["initialized"] is True, status
assert status["stats"]["events_received"] >= 1, status
assert status["stats"]["events_stored"] >= 1, status

print("[OK] manifest.json parsed")
print("[OK] configs/default.yaml parsed")
print("[OK] plugin import/init succeeded")
print("[OK] sample process() succeeded")
print("[OK] get_status() statistics updated")
print("[OK] local entrypoints declared")
print("[INFO] analysis_triggered=", result.get("analysis_triggered"))
print("[INFO] root_cause_category=", (result.get("analysis_result") or {}).get("root_cause_category"))
PY

"$PYTHON_CMD" -m pytest tests -q

echo "[PASS] Sanity checks completed"
