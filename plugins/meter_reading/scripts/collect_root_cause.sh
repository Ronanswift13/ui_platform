#!/usr/bin/env bash
# collect_root_cause.sh
# 收集一次失败所需的最小根因材料

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

OUT_DIR="${1:-./data/root_cause}"
STAMP="$(date +%Y%m%d-%H%M%S)"
CASE_DIR="$OUT_DIR/$STAMP"
mkdir -p "$CASE_DIR"

echo "Collecting root cause materials into: $CASE_DIR"

git rev-parse --is-inside-work-tree >/dev/null 2>&1 && {
  git status --short > "$CASE_DIR/git_status.txt" || true
  git diff > "$CASE_DIR/git_diff.patch" || true
}

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pwd=$(pwd)"
  echo "python=$($PYTHON_CMD --version 2>&1)"
  echo "opencv=$($PYTHON_CMD -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo not_installed)"
  echo "numpy=$($PYTHON_CMD -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo not_installed)"
} > "$CASE_DIR/env.txt"

cp -f configs/default.yaml "$CASE_DIR/default.yaml" 2>/dev/null || true
cp -f PROJECT_CARD.md "$CASE_DIR/PROJECT_CARD.md" 2>/dev/null || true
cp -f CLAUDE.md "$CASE_DIR/CLAUDE.md" 2>/dev/null || true
cp -f .agent_skills/02_algorithm_contract.md "$CASE_DIR/02_algorithm_contract.md" 2>/dev/null || true
cp -f .agent_skills/03_test_strategy.md "$CASE_DIR/03_test_strategy.md" 2>/dev/null || true

"$PYTHON_CMD" - <<'PY' > "$CASE_DIR/config_summary.txt" 2>/dev/null || true
import yaml

with open("configs/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(f"confidence_threshold={cfg.get('inference', {}).get('confidence_threshold')}")
print(f"manual_review_threshold={cfg.get('fallback', {}).get('manual_review_threshold')}")
print(f"retry_count={cfg.get('fallback', {}).get('retry_count')}")
print(f"max_rotation={cfg.get('perspective_correction', {}).get('max_rotation')}")
print(f"angle_range={cfg.get('pointer_detection', {}).get('angle_range')}")
print(f"led_brightness_threshold={cfg.get('led_detection', {}).get('brightness_threshold')}")
print(f"max_reading_time_ms={cfg.get('performance', {}).get('max_reading_time_ms')}")
PY

set +e
./scripts/run_targeted_tests.sh all > "$CASE_DIR/targeted_tests.log" 2>&1
TARGETED_CODE=$?
./scripts/run_regression_tests.sh > "$CASE_DIR/regression_tests.log" 2>&1
REGRESSION_CODE=$?
set -e

{
  echo "targeted_exit_code=$TARGETED_CODE"
  echo "regression_exit_code=$REGRESSION_CODE"
} >> "$CASE_DIR/env.txt"

echo "done"
echo "- env: $CASE_DIR/env.txt"
echo "- config summary: $CASE_DIR/config_summary.txt"
echo "- status: $CASE_DIR/git_status.txt"
echo "- targeted log: $CASE_DIR/targeted_tests.log"
echo "- regression log: $CASE_DIR/regression_tests.log"
