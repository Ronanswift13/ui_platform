#!/usr/bin/env bash
# run_targeted_tests.sh
# 职责:
# 1) 快速执行 meter_reading 模块级测试（L0/L1）
# 2) 对缺失测试文件快速失败，阻断假绿色
# 3) 为 regression / quality gate 提供前置门禁

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

MODULE="${1:-all}"
shift || true

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_targeted_tests.sh [module] [pytest_args...]

Modules:
  analog      - 模拟表链路与降级链
  digital     - 数字表 / 七段码 OCR 清洗
  led         - LED HSV 分类与可分离性
  validation  - 输入校验、置信度、状态集
  plugin      - plugin 接口集成
  contract    - 输出结构与 metadata 合同
  all         - 运行 tests/ 下所有非 regression 测试
USAGE
}

if [[ "$MODULE" == "-h" || "$MODULE" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODULE" in
  analog)
    TEST_FILES=("tests/test_analog_meter.py")
    ;;
  digital)
    TEST_FILES=("tests/test_digital_ocr.py")
    ;;
  led)
    TEST_FILES=("tests/test_led_indicator.py")
    ;;
  validation)
    TEST_FILES=("tests/test_input_validation.py" "tests/test_confidence.py")
    ;;
  plugin)
    TEST_FILES=("tests/test_plugin_integration.py")
    ;;
  contract)
    TEST_FILES=("tests/test_output_structure.py")
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
else
  if ! find tests -name "test_*.py" -print -quit 2>/dev/null | grep -q .; then
    echo "[FAIL] No test_*.py files found under tests/"
    exit 2
  fi
fi

echo "========================================"
echo "Targeted Tests - module: $MODULE"
echo "========================================"

if [[ "$MODULE" == "all" ]]; then
  "$PYTHON_CMD" -m pytest tests/ -v --tb=short -m "not regression and not smoke" "$@"
else
  "$PYTHON_CMD" -m pytest "${TEST_FILES[@]}" -v --tb=short "$@"
fi

echo "[PASS] Targeted tests completed"
