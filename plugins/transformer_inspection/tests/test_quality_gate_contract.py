"""Governance asset contract tests for transformer_inspection."""

from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_required_governance_assets_exist():
    """HF local governance files must exist inside the plugin directory."""
    required_files = [
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

    for relative_path in required_files:
        assert (PROJECT_DIR / relative_path).exists(), relative_path


def test_task_routing_mentions_all_governance_scripts():
    """08_task_routing.md must route implement/repair/audit into the real scripts."""
    content = (PROJECT_DIR / ".agent_skills/08_task_routing.md").read_text(encoding="utf-8")
    for script_name in (
        "run_targeted_tests.sh",
        "run_regression_tests.sh",
        "run_quality_gate.sh",
        "collect_root_cause.sh",
    ):
        assert script_name in content


def test_commands_reference_task_routing_and_no_claude_placeholder():
    """Commands should be grounded in local routing instead of stale template files."""
    commands_dir = PROJECT_DIR / ".claude" / "commands"

    for command_name in ("implement.md", "repair.md", "audit.md"):
        content = (commands_dir / command_name).read_text(encoding="utf-8")
        assert ".agent_skills/08_task_routing.md" in content
        assert "CLAUDE.md" not in content
        assert "脚本不存在" not in content
