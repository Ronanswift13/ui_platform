#!/usr/bin/env bash
# Governance + architecture quality gate for capacitor_inspection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PROJECT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "========================================"
echo "Quality Gate - capacitor_inspection"
echo "========================================"

echo "[1/4] Required governance assets"
REQUIRED_FILES=(
  ".agent_skills/00_project_context.md"
  ".agent_skills/01_architecture_rules.md"
  ".agent_skills/02_algorithm_contract.md"
  ".agent_skills/03_test_strategy.md"
  ".agent_skills/04_quality_audit.md"
  ".agent_skills/05_security_boundary.md"
  ".agent_skills/06_refactor_policy.md"
  ".agent_skills/07_learning_log.md"
  ".agent_skills/08_task_routing.md"
  ".claude/commands/implement.md"
  ".claude/commands/repair.md"
  ".claude/commands/audit.md"
  "scripts/run_targeted_tests.sh"
  "scripts/run_regression_tests.sh"
  "scripts/run_quality_gate.sh"
  "scripts/collect_root_cause.sh"
  "README.md"
  "PROJECT_CARD.md"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -e "$file" ]]; then
    echo "[MISSING] $file"
    MISSING=1
  fi
done
if [[ "$MISSING" -ne 0 ]]; then
  echo "[BLOCKED] Missing governance assets"
  exit 1
fi

echo "[2/4] Architecture and anti-pattern checks"
if rg -n "darkbreaker_sdk|standalone" detector_enhanced.py >/dev/null 2>&1; then
  echo "[FAIL] detector_enhanced.py has forbidden dependency"
  rg -n "darkbreaker_sdk|standalone" detector_enhanced.py
  exit 1
fi

if rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py detector_enhanced.py >/dev/null 2>&1; then
  echo "[FAIL] Silent exception anti-pattern detected"
  rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py detector_enhanced.py
  exit 1
fi

if rg -n "\bprint\(" plugin.py detector_enhanced.py >/dev/null 2>&1; then
  echo "[FAIL] print() found in production modules"
  rg -n "\bprint\(" plugin.py detector_enhanced.py
  exit 1
fi

for script_name in run_targeted_tests.sh run_regression_tests.sh run_quality_gate.sh collect_root_cause.sh; do
  if ! rg -n "$script_name" .agent_skills/08_task_routing.md >/dev/null 2>&1; then
    echo "[FAIL] 08_task_routing.md missing $script_name"
    exit 1
  fi
done

if rg -n "CLAUDE\.md|脚本不存在" .claude/commands >/dev/null 2>&1; then
  echo "[FAIL] stale command template content detected"
  rg -n "CLAUDE\.md|脚本不存在" .claude/commands
  exit 1
fi

if ! rg -n "\.agent_skills/08_task_routing\.md" .claude/commands >/dev/null 2>&1; then
  echo "[FAIL] commands are not routed through 08_task_routing.md"
  exit 1
fi

echo "[3/4] Regression gate"
"$SCRIPT_DIR/run_regression_tests.sh"

echo "[4/4] Security scan"
if rg -n "requests\.|http://|https://|urllib|aiohttp" \
  plugin.py detector_enhanced.py standalone demo \
  scripts/run_targeted_tests.sh scripts/run_regression_tests.sh scripts/collect_root_cause.sh >/dev/null 2>&1; then
  echo "[FAIL] Network call detected"
  rg -n "requests\.|http://|https://|urllib|aiohttp" \
    plugin.py detector_enhanced.py standalone demo \
    scripts/run_targeted_tests.sh scripts/run_regression_tests.sh scripts/collect_root_cause.sh
  exit 1
fi

if rg -n "rm -rf|sudo |os\.remove\(|shutil\.rmtree\(" \
  scripts/run_targeted_tests.sh scripts/run_regression_tests.sh scripts/collect_root_cause.sh >/dev/null 2>&1; then
  echo "[FAIL] Destructive command detected"
  rg -n "rm -rf|sudo |os\.remove\(|shutil\.rmtree\(" \
    scripts/run_targeted_tests.sh scripts/run_regression_tests.sh scripts/collect_root_cause.sh
  exit 1
fi

echo "[PASS] Quality gate completed"
