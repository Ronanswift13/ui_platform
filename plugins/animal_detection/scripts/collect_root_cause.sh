#!/usr/bin/env bash
# collect_root_cause.sh
# 收集一次失败所需的最小根因材料

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

OUT_DIR="${1:-./data/root_cause}"
STAMP="$(date +%Y%m%d-%H%M%S)"
CASE_DIR="$OUT_DIR/$STAMP"
mkdir -p "$CASE_DIR"

echo "Collecting root cause materials into: $CASE_DIR"

# Git 状态
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && {
  git status --short > "$CASE_DIR/git_status.txt" || true
  git diff > "$CASE_DIR/git_diff.patch" || true
  git log --oneline -10 > "$CASE_DIR/git_log.txt" || true
}

# 环境快照
{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python3 || command -v python || echo 'not_found')"
} > "$CASE_DIR/env.txt"

# 关键配置与规则文件快照
cp -f configs/default.yaml "$CASE_DIR/default.yaml" 2>/dev/null || true
cp -f PROJECT_CARD.md "$CASE_DIR/PROJECT_CARD.md" 2>/dev/null || true
cp -f CLAUDE.md "$CASE_DIR/CLAUDE.md" 2>/dev/null || true
cp -f .agent_skills/02_algorithm_contract.md "$CASE_DIR/02_algorithm_contract.md" 2>/dev/null || true

# 错误日志快照
if [[ -d "logs" ]]; then
  for logfile in logs/*.log logs/*.jsonl; do
    if [[ -f "$logfile" ]]; then
      tail -50 "$logfile" > "$CASE_DIR/$(basename "$logfile")" 2>/dev/null || true
    fi
  done
fi

# 运行最小复现并保存输出（失败也继续）
set +e
./scripts/run_targeted_tests.sh all > "$CASE_DIR/targeted_tests.log" 2>&1
TARGETED_CODE=$?
./scripts/run_regression_tests.sh > "$CASE_DIR/regression_tests.log" 2>&1
REGRESSION_CODE=$?
set -e

echo "targeted_exit_code=$TARGETED_CODE" >> "$CASE_DIR/env.txt"
echo "regression_exit_code=$REGRESSION_CODE" >> "$CASE_DIR/env.txt"

# 模型状态
if [[ -f "models/animal_yolov8n.onnx" ]]; then
  echo "model_status=present" >> "$CASE_DIR/env.txt"
else
  echo "model_status=missing" >> "$CASE_DIR/env.txt"
fi

echo "done"
echo "- env: $CASE_DIR/env.txt"
echo "- status: $CASE_DIR/git_status.txt"
echo "- targeted log: $CASE_DIR/targeted_tests.log"
echo "- regression log: $CASE_DIR/regression_tests.log"
