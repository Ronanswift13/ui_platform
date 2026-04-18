#!/usr/bin/env bash
# run_targeted_tests.sh
# 职责:
# 1) 快速执行模块级测试（L0/L1）
# 2) 对缺失测试文件快速失败，避免假绿色
# 3) 为回归脚本提供前置门禁

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
  detector    - 检测模块 (detector + onnx_inference)
  tracker     - 跟踪模块
  thermal     - 热验证模块
  event       - 事件契约
  plugin      - 插件接口
  standalone  - 独立运行
  all         - 运行 tests/ 下非回归测试
USAGE
}

if [[ "$MODULE" == "-h" || "$MODULE" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODULE" in
  detector)
    TEST_FILES=("tests/test_onnx_inference.py" "tests/test_detector.py")
    ;;
  tracker)
    TEST_FILES=("tests/test_plugin.py")
    PYTEST_EXTRA=(-k "tracker")
    ;;
  thermal)
    TEST_FILES=("tests/test_plugin.py")
    PYTEST_EXTRA=(-k "thermal")
    ;;
  event)
    TEST_FILES=("tests/test_event_schema_contract.py")
    ;;
  plugin)
    TEST_FILES=("tests/test_plugin.py")
    ;;
  standalone)
    TEST_FILES=("tests/test_standalone.py")
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
    if [[ ! -f "$f" && ! -d "$f" ]]; then
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
  "$PYTHON_CMD" -m pytest "${TEST_FILES[@]}" -v --tb=short ${PYTEST_EXTRA[@]+"${PYTEST_EXTRA[@]}"} "$@"
fi

echo "[PASS] Targeted tests completed"
