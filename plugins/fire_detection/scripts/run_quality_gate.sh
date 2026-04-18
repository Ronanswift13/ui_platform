#!/usr/bin/env bash
# Quality gate for fire_detection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "[ERROR] python interpreter not found"; exit 127
fi

echo "========================================"
echo "Quality Gate - fire_detection"
echo "========================================"

# Stage 1: Governance assets
echo "[Stage 1] Checking governance assets..."
for f in manifest.json configs/default.yaml plugin.py standalone/app.py; do
  if [[ ! -f "$f" ]]; then
    echo "[FAIL] Missing: $f"; exit 1
  fi
done
echo "[PASS] Stage 1"

# Stage 2: Tests
echo "[Stage 2] Running tests..."
"$PYTHON_CMD" -m pytest tests/ -q --tb=short
echo "[PASS] Stage 2"

echo "========================================"
echo "[PASS] Quality gate complete"
echo "========================================"
