#!/usr/bin/env bash
# run_quality_gate.sh
# 质量闸门: 架构检查 + regression gate + 安全扫描

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PLUGIN_NAME="${PLUGIN_NAME:-meter_reading}"

echo "========================================"
echo "Quality Gate - $PLUGIN_NAME"
echo "========================================"

echo "[1/3] Architecture and contract checks"
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

if rg -n "LOW_CONFIDENCE" plugin.py detector_enhanced.py 2>/dev/null; then
  echo "[FAIL] deprecated LOW_CONFIDENCE state found"
  ARCH_FAIL=1
fi

if [[ $ARCH_FAIL -ne 0 ]]; then
  echo "[BLOCKED] Architecture checks failed - skipping remaining stages"
  exit 1
fi
echo "[OK] Architecture checks passed"

echo "[2/3] Regression gate (includes targeted + full pytest + static checks)"
"$SCRIPT_DIR/run_regression_tests.sh"

echo "[3/3] Security scan (informational)"
if rg -n "requests\.|http://|https://|urllib|aiohttp" plugin.py detector_enhanced.py standalone 2>/dev/null; then
  echo "[WARN] Potential network usage found above"
else
  echo "[OK] No obvious network calls"
fi

if rg -n "cv2\.imwrite|imwrite\(|Image\.save|open\(.*['\"]wb" plugin.py detector_enhanced.py standalone scripts 2>/dev/null; then
  echo "[WARN] Potential binary file write found above"
else
  echo "[OK] No obvious binary image persistence"
fi

echo "========================================"
echo "[PASS] Quality gate completed"
echo "========================================"
