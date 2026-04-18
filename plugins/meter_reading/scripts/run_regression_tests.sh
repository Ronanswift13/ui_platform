#!/usr/bin/env bash
# run_regression_tests.sh
# 职责:
# 1) 执行发布前全量回归门禁
# 2) 串联 targeted -> 全量 pytest -> regression 目录 -> 静态/安全检查
# 3) 对空 regression / fixtures 输出可审计的 skip，而不是假通过

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

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

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
echo "Regression Gate - meter_reading"
echo "========================================"

echo "[1/4] Targeted gate"
"$SCRIPT_DIR/run_targeted_tests.sh" all

echo "[2/4] Full pytest"
if "$PYTHON_CMD" -m pytest --help 2>/dev/null | grep -q -- "--cov"; then
  "$PYTHON_CMD" -m pytest tests/ -v --tb=short \
    --cov=. \
    --cov-config=.coveragerc \
    --cov-report=term-missing
else
  "$PYTHON_CMD" -m pytest tests/ -v --tb=short
fi

echo "[3/4] Regression dataset checks"
if find tests/regression -name "test_*.py" -print -quit 2>/dev/null | grep -q .; then
  run_pytest_allow_empty tests/regression -v --tb=short
else
  echo "[SKIP] tests/regression/ has no test_*.py"
fi

if find tests/fixtures -type f -print -quit 2>/dev/null | grep -q .; then
  echo "[INFO] tests/fixtures/ contains regression assets"
else
  echo "[SKIP] tests/fixtures/ is empty"
fi

echo "[4/4] Static and security checks"
if command -v mypy >/dev/null 2>&1; then
  mypy . --ignore-missing-imports --exclude '__pycache__|htmlcov|.pytest_cache'
else
  echo "[SKIP] mypy not installed"
fi

if command -v flake8 >/dev/null 2>&1; then
  flake8 . --max-line-length=100 --exclude=__pycache__,*.pyc,.agent_skills,.claude,htmlcov,.pytest_cache
else
  echo "[SKIP] flake8 not installed"
fi

if command -v bandit >/dev/null 2>&1; then
  bandit -r . -ll -q --exclude __pycache__,tests,.agent_skills,.claude,htmlcov
else
  echo "[SKIP] bandit not installed"
fi

echo "[PASS] Regression gate completed"
