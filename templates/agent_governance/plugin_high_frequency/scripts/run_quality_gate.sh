#!/usr/bin/env bash
# run_quality_gate.sh — {{PLUGIN_NAME}}
# 质量闸门: 架构检查 + 反模式扫描 + 回归门禁

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

PLUGIN_NAME="${PLUGIN_NAME:-{{PLUGIN_NAME}}}"

echo "========================================"
echo "Quality Gate - $PLUGIN_NAME"
echo "========================================"

# Stage 1: 架构检查（快速失败）
echo "[1/3] Architecture checks"
ARCH_FAIL=0

# 检查检测器是否引入禁止依赖
# <!-- BUSINESS: 按实际检测器文件名替换 -->
for f in plugin.py {{DETECTOR_FILE}}; do
  [[ -f "$f" ]] || continue
  if rg -n "darkbreaker_sdk|standalone" "$f" >/dev/null 2>&1; then
    echo "[FAIL] $f has forbidden dependency"
    rg -n "darkbreaker_sdk|standalone" "$f"
    ARCH_FAIL=1
  fi
done

# 检查静默异常
if rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py {{DETECTOR_FILE}} 2>/dev/null; then
  echo "[FAIL] silent exception anti-pattern detected"
  ARCH_FAIL=1
fi

# 检查 print
if rg -n "\bprint\(" plugin.py {{DETECTOR_FILE}} 2>/dev/null; then
  echo "[FAIL] print() found in production modules"
  ARCH_FAIL=1
fi

if [[ $ARCH_FAIL -ne 0 ]]; then
  echo "[BLOCKED] Architecture checks failed — skipping remaining stages"
  exit 1
fi
echo "[OK] Architecture checks passed"

# Stage 2: 全量回归门禁
echo "[2/3] Regression gate"
"$SCRIPT_DIR/run_regression_tests.sh"

# Stage 3: 安全扫描（仅检查，不阻断）
echo "[3/3] Security scan (informational)"
if command -v bandit >/dev/null 2>&1; then
  bandit -r . -ll -q --exclude __pycache__,tests || true
else
  echo "[SKIP] bandit not installed"
fi

echo "========================================"
echo "[PASS] Quality gate completed"
echo "========================================"
