#!/usr/bin/env bash
# Governance + architecture quality gate for hyperspectral_detection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "[ERROR] python interpreter not found"
  exit 127
fi

echo "========================================"
echo "Quality Gate - hyperspectral_detection"
echo "========================================"

echo "[1/4] Required governance assets"
REQUIRED_FILES=(
  "manifest.json"
  "plugin.py"
  "configs/default.yaml"
  "tests/conftest.py"
  "tests/test_manifest_contract.py"
  "tests/test_config_contract.py"
  "tests/test_plugin_contract.py"
  "tests/test_standalone.py"
  "scripts/run_targeted_tests.sh"
  "scripts/run_quality_gate.sh"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -e "$file" ]]; then
    echo "[MISSING] $file"
    MISSING=1
  fi
done
if [[ "$MISSING" -ne 0 ]]; then
  echo "[BLOCKED] Missing governance assets"
  exit 1
fi
echo "[OK] All governance assets present"

echo "[2/4] Architecture and anti-pattern checks"
if grep -qE "except\s*:|except\s+Exception\s*:\s*pass" plugin.py 2>/dev/null; then
  echo "[FAIL] Silent exception anti-pattern detected in plugin.py"
  grep -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py
  exit 1
fi
echo "[OK] No anti-patterns detected"

echo "[3/4] Running test suite"
"$PYTHON_CMD" -m pytest tests/ -q

echo "[4/4] Security scan"
if grep -rqE "rm -rf|sudo |os\.remove\(|shutil\.rmtree\(" scripts/ 2>/dev/null; then
  echo "[FAIL] Destructive command detected in scripts/"
  grep -rn "rm -rf|sudo |os\.remove\(|shutil\.rmtree\(" scripts/
  exit 1
fi
echo "[OK] No security issues detected"

echo "[PASS] Quality gate completed"
