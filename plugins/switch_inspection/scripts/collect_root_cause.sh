#!/usr/bin/env bash
# Collect minimum reproducibility materials for switch_inspection failures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

OUT_DIR="${1:-./data/root_cause}"
STAMP="$(date +%Y%m%d-%H%M%S)"
CASE_DIR="$OUT_DIR/$STAMP"
mkdir -p "$CASE_DIR"

echo "Collecting root cause materials into: $CASE_DIR"

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_dir=$PROJECT_DIR"
  echo "repo_root=$REPO_ROOT"
  echo "python=$(command -v python3 || command -v python || echo 'not_found')"
} > "$CASE_DIR/env.txt"

cp -f configs/default.yaml "$CASE_DIR/default.yaml" 2>/dev/null || true
cp -f README.md "$CASE_DIR/README.md" 2>/dev/null || true
cp -f PROJECT_CARD.md "$CASE_DIR/PROJECT_CARD.md" 2>/dev/null || true
cp -f .agent_skills/02_algorithm_contract.md "$CASE_DIR/02_algorithm_contract.md" 2>/dev/null || true
cp -f .agent_skills/04_quality_audit.md "$CASE_DIR/04_quality_audit.md" 2>/dev/null || true
cp -f .agent_skills/07_learning_log.md "$CASE_DIR/07_learning_log.md" 2>/dev/null || true
cp -f .agent_skills/08_task_routing.md "$CASE_DIR/08_task_routing.md" 2>/dev/null || true

if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$REPO_ROOT" status --short -- plugins/switch_inspection > "$CASE_DIR/git_status.txt" || true
  git -C "$REPO_ROOT" diff -- plugins/switch_inspection > "$CASE_DIR/git_diff.patch" || true
fi

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
echo "- targeted log: $CASE_DIR/targeted_tests.log"
echo "- regression log: $CASE_DIR/regression_tests.log"
