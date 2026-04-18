#!/usr/bin/env bash
# run_sanity_checks.sh — {{PLUGIN_NAME}}
# 最小治理级校验入口

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
echo "Sanity Checks - {{PLUGIN_NAME}}"
echo "========================================"

"$PYTHON_CMD" - <<'PY'
import json
from pathlib import Path

plugin_dir = Path.cwd()

# 1. manifest.json 可解析
manifest = json.loads((plugin_dir / "manifest.json").read_text(encoding="utf-8"))
assert "id" in manifest, "manifest.json missing 'id'"
print(f"[OK] manifest.json parsed — id={manifest['id']}")

# 2. plugin.py 可导入
# <!-- BUSINESS: 替换为实际的 import 路径和插件类名 -->
# from plugins.{{PLUGIN_NAME}}.plugin import MyPlugin
# plugin = MyPlugin()
# assert plugin is not None
# print("[OK] plugin import succeeded")

# 3. configs 可加载（如存在）
config_file = plugin_dir / "configs" / "default.yaml"
if config_file.exists():
    import yaml
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert isinstance(config, dict), "config is not a dict"
    print("[OK] configs/default.yaml parsed")
else:
    print("[SKIP] configs/default.yaml not found")

print("[PASS] Basic sanity checks completed")
PY

echo "[PASS] Sanity checks completed"
