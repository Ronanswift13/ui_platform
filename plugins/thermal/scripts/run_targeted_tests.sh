#!/usr/bin/env bash
# Fast module-scoped tests for thermal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

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

MODULE="${1:-all}"
shift || true

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_targeted_tests.sh [module] [pytest_args...]

Modules:
  standalone   standalone startup and runner contract
  plugin       plugin infer/postprocess contract
  config       config and manifest/runtime contract
  all          run the full local test suite
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
    TEST_FILES=("tests/test_plugin_contract.py")
    ;;
  config)
    TEST_FILES=("tests/test_config_contract.py" "tests/test_manifest_contract.py")
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
  for test_file in "${TEST_FILES[@]}"; do
    if [[ ! -f "$test_file" ]]; then
      echo "[MISSING] $test_file"
      MISSING=1
    fi
  done
  if [[ "$MISSING" -ne 0 ]]; then
    echo "[FAIL] Missing targeted tests for module '$MODULE'"
    exit 2
  fi
fi

echo "========================================"
echo "Targeted Tests - thermal"
echo "module=$MODULE"
echo "python=$PYTHON_CMD"
echo "========================================"

if [[ "$MODULE" == "all" ]]; then
  "$PYTHON_CMD" -m pytest tests/ -q "$@"
else
  "$PYTHON_CMD" -m pytest "${TEST_FILES[@]}" -q "$@"
fi

echo "[PASS] Targeted tests completed"
