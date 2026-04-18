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
  standalone  - standalone 启动与健康检查
  plugin      - plugin 接口契约
  detector    - detector 算法契约
  quality     - 质量门禁与原因码
  config      - 配置映射契约
  all         - 运行 tests/ 下非回归测试
USAGE
}

if [[ "$MODULE" == "-h" || "$MODULE" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODULE" in
  standalone)
    TEST_FILES=("tests/test_standalone.py")
    ;;
  plugin)
    TEST_FILES=("tests/test_plugin_contract.py" "tests/test_plugin_postprocess.py")
    ;;
  detector)
    TEST_FILES=("tests/test_detector_contract.py" "tests/test_fallback_chain.py")
    ;;
  quality)
    TEST_FILES=("tests/test_quality_gate_contract.py" "tests/test_reason_code_contract.py")
    ;;
  config)
    TEST_FILES=("tests/test_config_contract.py")
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
