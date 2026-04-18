#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "========================================"
echo "Quality Gate - thermal"
echo "========================================"

# Stage 1: Manifest and config
echo "[Stage 1] Checking governance assets..."
for f in manifest.json configs/default.yaml plugin.py standalone/app.py; do
  if [[ ! -f "$f" ]]; then
    echo "[FAIL] Missing: $f"
    exit 1
  fi
done
echo "[PASS] Stage 1"

# Stage 2: Tests
echo "[Stage 2] Running tests..."
python3 -m pytest tests/ -q --tb=short
echo "[PASS] Stage 2"

echo "========================================"
echo "[PASS] Quality gate complete"
echo "========================================"
