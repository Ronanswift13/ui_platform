#!/usr/bin/env bash
# Regression gate for transformer_inspection.

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

run_optional_pytest() {
  set +e
  "$PYTHON_CMD" -m pytest "$@"
  local code=$?
  set -e
  if [[ $code -eq 5 ]]; then
    echo "[SKIP] No tests collected for: $*"
    return 0
  fi
  return $code
}

echo "========================================"
echo "Regression Gate - transformer_inspection"
echo "========================================"
echo "[NOTE] No real image replay dataset is configured yet."
echo "[NOTE] Regression currently means full pytest + demo smoke + optional static checks."
echo "[NOTE] benchmark.py is opt-in and not part of the default HF gate."

echo "[1/5] Targeted gate"
"$SCRIPT_DIR/run_targeted_tests.sh" all

echo "[2/5] Full pytest"
if "$PYTHON_CMD" -m pytest --help 2>/dev/null | grep -q -- "--cov"; then
  "$PYTHON_CMD" -m pytest tests/ -q \
    --cov=plugins.transformer_inspection \
    --cov-report=term-missing
else
  "$PYTHON_CMD" -m pytest tests/ -q
fi

echo "[3/5] Demo smoke"
"$PYTHON_CMD" demo/run_demo.py

echo "[4/5] Optional regression marker tests"
run_optional_pytest tests/ -m regression -q

echo "[5/5] Optional static/security tools"
if [[ "${RUN_BENCHMARK:-0}" == "1" ]]; then
  "$PYTHON_CMD" scripts/benchmark.py
else
  echo "[SKIP] benchmark.py not enabled (set RUN_BENCHMARK=1 to run)"
fi

if command -v mypy >/dev/null 2>&1; then
  mypy . --ignore-missing-imports
else
  echo "[SKIP] mypy not installed"
fi

if command -v flake8 >/dev/null 2>&1; then
  flake8 . --max-line-length=100 --exclude=__pycache__,*.pyc
else
  echo "[SKIP] flake8 not installed"
fi

if command -v bandit >/dev/null 2>&1; then
  bandit -r . -ll -q --exclude __pycache__,tests
else
  echo "[SKIP] bandit not installed"
fi

echo "[PASS] Regression gate completed"
