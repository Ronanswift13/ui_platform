#!/usr/bin/env bash
# run_targeted_tests.sh — {{PLUGIN_NAME}}
# 快速执行模块级测试（L0/L1），对缺失测试文件快速失败

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

MODULE="${1:-all}"
shift || true

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_targeted_tests.sh [module] [pytest_args...]

Modules:
{{MODULE_LIST}}
  all         - 运行 tests/ 下非回归测试
USAGE
}

if [[ "$MODULE" == "-h" || "$MODULE" == "--help" ]]; then
  usage
  exit 0
fi

# <!-- BUSINESS: 按插件实际模块替换 case 分支 -->
case "$MODULE" in
  plugin)
    TEST_FILES=("tests/test_plugin.py")
    ;;
  all)
    TEST_FILES=("tests")
    ;;
  *)
    echo "[ERROR] Unknown module: $MODULE"
    usage
    exit 2
    ;;
esac

if [[ "$MODULE" != "all" ]]; then
  MISSING=0
  for f in "${TEST_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "[MISSING] $f"
      MISSING=1
    fi
  done
  if [[ "$MISSING" -ne 0 ]]; then
    echo "[FAIL] Missing targeted tests for module '$MODULE'"
    exit 2
  fi
fi

echo "========================================"
echo "Targeted Tests - module: $MODULE"
echo "========================================"

if [[ "$MODULE" == "all" ]]; then
  "$PYTHON_CMD" -m pytest tests/ -v --tb=short -m "not regression" "$@"
else
  "$PYTHON_CMD" -m pytest "${TEST_FILES[@]}" -v --tb=short "$@"
fi

echo "[PASS] Targeted tests completed"
