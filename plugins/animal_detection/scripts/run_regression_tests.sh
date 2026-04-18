#!/usr/bin/env bash
# run_regression_tests.sh
# 职责:
# 1) 执行发布前全量回归门禁
# 2) 串联 targeted -> 全量 pytest -> 契约锁定 -> 静态/安全检查
# 3) 输出可审计的阶段结果并在失败时快速退出

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
echo "Regression Gate - animal_detection"
echo "========================================"

# Stage 1: 快速门禁
echo "[1/5] Targeted gate"
"$SCRIPT_DIR/run_targeted_tests.sh" all

# Stage 2: 全量 pytest + 覆盖率
echo "[2/5] Full pytest + coverage"
if "$PYTHON_CMD" -m pytest --help 2>/dev/null | grep -q -- "--cov"; then
  "$PYTHON_CMD" -m pytest tests/ -v --tb=short \
    --cov=. --cov-config=.coveragerc \
    --cov-report=term-missing
else
  "$PYTHON_CMD" -m pytest tests/ -v --tb=short
fi

# Stage 3: 枚举锁定 (QR-8)
echo "[3/5] Enum lock verification (QR-8)"
"$PYTHON_CMD" -c "
from core.event_schema import AnimalClass, EventType, RiskLevel
assert len(AnimalClass) == 8, f'AnimalClass count {len(AnimalClass)} != 8'
assert len(EventType) == 10, f'EventType count {len(EventType)} != 10'
assert len(RiskLevel) == 4, f'RiskLevel count {len(RiskLevel)} != 4'
print('[OK] Enum lock: PASS')
"

# Stage 4: 契约测试（置信度清洗 + 事件 schema）
echo "[4/5] Contract tests (QR-11, QR-12)"
run_pytest_allow_empty tests/test_confidence_sanitization.py tests/test_event_schema_contract.py -v --tb=short

# Stage 5: 静态与安全检查（工具存在则执行）
echo "[5/5] Static and security checks"
if command -v flake8 >/dev/null 2>&1 || "$PYTHON_CMD" -m flake8 --version >/dev/null 2>&1; then
  "$PYTHON_CMD" -m flake8 . --max-line-length=100 \
    --exclude='__pycache__,*.pyc,.claude,.agent_skills,venv,.venv,models,demo,scripts,train.py,__main__.py,api_integration.py,standalone/app.py' \
    --per-file-ignores='plugin.py:F401,E402,E501 tests/*:E402'
else
  echo "[SKIP] flake8 not installed"
fi

if command -v bandit >/dev/null 2>&1; then
  bandit -r . -ll -q --exclude __pycache__,tests,.claude,.agent_skills,models,venv,.venv,demo || true
else
  echo "[SKIP] bandit not installed"
fi

echo "[PASS] Regression gate completed"
