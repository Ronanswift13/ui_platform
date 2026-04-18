#!/usr/bin/env python3
"""Governance contract tests for HF operations."""

from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def test_required_governance_assets_exist():
    required = [
        ".agent_skills/00_project_context.md",
        ".agent_skills/01_architecture_rules.md",
        ".agent_skills/02_algorithm_contract.md",
        ".agent_skills/03_test_strategy.md",
        ".agent_skills/04_quality_audit.md",
        ".agent_skills/05_security_boundary.md",
        ".agent_skills/06_refactor_policy.md",
        ".agent_skills/07_learning_log.md",
        ".agent_skills/08_task_routing.md",
        ".claude/commands/implement.md",
        ".claude/commands/repair.md",
        ".claude/commands/audit.md",
        "scripts/run_targeted_tests.sh",
        "scripts/run_regression_tests.sh",
        "scripts/run_quality_gate.sh",
        "scripts/collect_root_cause.sh",
        "README.md",
        "PROJECT_CARD.md",
    ]
    for relpath in required:
        assert (PLUGIN_ROOT / relpath).exists(), relpath


def test_task_routing_mentions_governance_scripts():
    routing = (PLUGIN_ROOT / ".agent_skills" / "08_task_routing.md").read_text(encoding="utf-8")
    for name in [
        "run_targeted_tests.sh",
        "run_regression_tests.sh",
        "run_quality_gate.sh",
        "collect_root_cause.sh",
    ]:
        assert name in routing


def test_production_modules_do_not_use_print():
    for relpath in ["plugin.py", "detector_enhanced.py", "switch_consistency.py"]:
        text = (PLUGIN_ROOT / relpath).read_text(encoding="utf-8")
        assert "print(" not in text
