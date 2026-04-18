#!/usr/bin/env bash
# run_regression_tests.sh — {{PLUGIN_NAME}}
# 发布前全量回归门禁: targeted → 全量 pytest → regression marker → 静态检查

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

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

run_pytest_allow_empty() {
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
echo "Regression Gate - {{PLUGIN_NAME}}"
echo "========================================"

# Stage 1: 快速门禁
echo "[1/4] Targeted gate"
"$SCRIPT_DIR/run_targeted_tests.sh" all

# Stage 2: 全量 pytest（含覆盖率，如可用）
echo "[2/4] Full pytest"
if "$PYTHON_CMD" -m pytest --help 2>/dev/null | grep -q -- "--cov"; then
  "$PYTHON_CMD" -m pytest tests/ -v --tb=short \
    --cov={{COV_TARGET}} \
    --cov-report=term-missing
else
  "$PYTHON_CMD" -m pytest tests/ -v --tb=short
fi

# Stage 3: regression marker（如存在）
echo "[3/4] Regression marker tests"
run_pytest_allow_empty tests/ -m regression -v --tb=short

# Stage 4: 静态与安全（工具存在则执行）
echo "[4/4] Static and security checks"
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
