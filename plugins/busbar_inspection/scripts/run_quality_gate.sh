#!/usr/bin/env bash
# run_quality_gate.sh
# 质量闸门: 架构检查 + 反模式扫描 + 测试门禁

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
  echo "[ERROR] python interpreter not found"
  exit 127
fi

PLUGIN_NAME="${PLUGIN_NAME:-busbar_inspection}"

echo "========================================"
echo "Quality Gate - $PLUGIN_NAME"
echo "========================================"

# Stage 1: 架构检查（快速失败，不依赖 pytest）
echo "[1/3] Architecture checks"
ARCH_FAIL=0

if rg -n "darkbreaker_sdk|standalone" detector_enhanced.py >/dev/null 2>&1; then
  echo "[FAIL] detector_enhanced.py has forbidden dependency"
  rg -n "darkbreaker_sdk|standalone" detector_enhanced.py
  ARCH_FAIL=1
fi

if rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py detector_enhanced.py 2>/dev/null; then
  echo "[FAIL] silent exception anti-pattern detected"
  ARCH_FAIL=1
fi

if rg -n "\bprint\(" plugin.py detector_enhanced.py 2>/dev/null; then
  echo "[FAIL] print() found in production modules"
  ARCH_FAIL=1
fi

if [[ $ARCH_FAIL -ne 0 ]]; then
  echo "[BLOCKED] Architecture checks failed — skipping remaining stages"
  exit 1
fi
echo "[OK] Architecture checks passed"

# Stage 2: 全量回归门禁（内部已串联 targeted + 全量 pytest + 静态检查）
echo "[2/3] Regression gate (includes targeted + full pytest + static checks)"
"$SCRIPT_DIR/run_regression_tests.sh"

# Stage 3: 安全扫描（仅检查，不阻断——输出供人工审查）
echo "[3/3] Security scan (informational)"
if command -v bandit >/dev/null 2>&1; then
  bandit -r . -ll -q --exclude __pycache__,tests || true
else
  echo "[SKIP] bandit not installed"
fi

echo "========================================"
echo "[PASS] Quality gate completed"
echo "========================================"
