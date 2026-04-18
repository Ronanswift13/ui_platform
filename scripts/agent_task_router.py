#!/usr/bin/env python3
"""
DarkBreaker Agent Task Router
===============================
将 .agent_skills/08_task_routing.md 中的路由规则转化为可执行的任务入口。

复用 audit_agent_governance.py 的治理等级判断逻辑，根据目标对象的治理等级
和任务类型，输出 [TARGET] / [READ_ORDER] / [EXECUTION_PLAN] / [WRITEBACK] 四段式计划。

用法:
    python3 scripts/agent_task_router.py plugins/busbar_inspection repair
    python3 scripts/agent_task_router.py ui audit --dry-run
    python3 scripts/agent_task_router.py plugins/animal_detection implement --run
    python3 scripts/agent_task_router.py plugins/action_event_monitoring upgrade --module detector
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 复用 audit_agent_governance 的治理判断逻辑
# ---------------------------------------------------------------------------

# 把 audit_agent_governance 加入 import 路径
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from audit_agent_governance import (
    GovernanceResult,
    scan_target,
    ROOT_SKILLS,
    PLUGIN_SKILLS,
    QUALITY_SCRIPTS,
)

# ---------------------------------------------------------------------------
# 常量: 任务类型与脚本映射
# ---------------------------------------------------------------------------

VALID_TASKS = ("implement", "repair", "audit", "upgrade")

# HF 插件脚本执行序列 (按任务类型)
HF_PLUGIN_SCRIPTS = {
    "implement": [
        "scripts/run_targeted_tests.sh",
        "scripts/run_regression_tests.sh",
        "scripts/run_quality_gate.sh",
    ],
    "repair": [
        "scripts/run_targeted_tests.sh",
        "scripts/run_regression_tests.sh",
        "scripts/collect_root_cause.sh",
    ],
    "audit": [
        "scripts/run_quality_gate.sh",
        "scripts/run_regression_tests.sh",
    ],
    "upgrade": [
        "scripts/run_targeted_tests.sh",
        "scripts/run_regression_tests.sh",
        "scripts/run_quality_gate.sh",
    ],
}

# STD 插件: 有 agent_skills 但脚本体系不完整
STD_PLUGIN_SCRIPTS = {
    "implement": ["scripts/run_targeted_tests.sh", "scripts/run_quality_gate.sh"],
    "repair": ["scripts/run_targeted_tests.sh", "scripts/collect_root_cause.sh"],
    "audit": ["scripts/run_quality_gate.sh"],
    "upgrade": ["scripts/run_targeted_tests.sh", "scripts/run_quality_gate.sh"],
}

# MIN 插件: 仅 sanity check
MIN_PLUGIN_SCRIPTS = {
    "implement": ["scripts/run_sanity_checks.sh"],
    "repair": ["scripts/run_sanity_checks.sh"],
    "audit": ["scripts/run_sanity_checks.sh"],
    "upgrade": ["scripts/run_sanity_checks.sh"],
}

# UI 专用脚本
UI_SCRIPTS = {
    "implement": [
        "scripts/check_ui_contract.sh",
        "scripts/check_three_state_coverage.sh",
        "scripts/run_quality_gate.sh",
    ],
    "repair": [
        "scripts/check_ui_contract.sh",
        "scripts/collect_root_cause.sh",
    ],
    "audit": [
        "scripts/check_ui_contract.sh",
        "scripts/check_three_state_coverage.sh",
        "scripts/run_quality_gate.sh",
    ],
    "upgrade": [
        "scripts/check_ui_contract.sh",
        "scripts/run_quality_gate.sh",
    ],
}

# ---------------------------------------------------------------------------
# 读取顺序规则
# ---------------------------------------------------------------------------

# 通用 Prerequisites (插件 / UI)
COMMON_PREREQS = [
    "CLAUDE.md",
    "PROJECT_CARD.md",
    ".agent_skills/00_project_context.md",
    ".agent_skills/01_architecture_rules.md",
]

# Root 专用 Prerequisites (root .agent_skills 使用不同命名)
ROOT_PREREQS = [
    "CLAUDE.md",
    "PROJECT_CARD.md",
    ".agent_skills/00_project_identity.md",
    ".agent_skills/01_architecture_map.md",
]

# 按任务类型追加的读取文件 (插件)
TASK_READ_EXTRA = {
    "implement": [
        ".agent_skills/02_algorithm_contract.md",
        ".agent_skills/03_test_strategy.md",
        ".agent_skills/05_security_boundary.md",
    ],
    "repair": [
        ".agent_skills/04_quality_audit.md",
        ".agent_skills/03_test_strategy.md",
        ".agent_skills/06_refactor_policy.md",
        ".agent_skills/07_learning_log.md",
    ],
    "audit": [
        ".agent_skills/04_quality_audit.md",
        ".agent_skills/03_test_strategy.md",
        ".agent_skills/02_algorithm_contract.md",
        ".agent_skills/05_security_boundary.md",
    ],
    "upgrade": [
        "manifest.json",
        ".agent_skills/02_algorithm_contract.md",
        ".agent_skills/03_test_strategy.md",
    ],
}

# Root 专用追加文件
ROOT_READ_EXTRA = {
    "implement": [
        ".agent_skills/02_plugin_registry.md",
        ".agent_skills/05_task_routing.md",
        ".agent_skills/06_testing_strategy.md",
    ],
    "repair": [
        ".agent_skills/04_naming_conventions.md",
        ".agent_skills/06_testing_strategy.md",
        ".agent_skills/07_config_hierarchy.md",
    ],
    "audit": [
        ".agent_skills/02_plugin_registry.md",
        ".agent_skills/05_task_routing.md",
        ".agent_skills/06_testing_strategy.md",
        ".agent_skills/08_governance_rules.md",
    ],
    "upgrade": [
        ".agent_skills/02_plugin_registry.md",
        ".agent_skills/03_tech_stack.md",
        ".agent_skills/08_governance_rules.md",
    ],
}

# UI 专用读取顺序 (覆盖通用规则)
UI_READ_EXTRA = {
    "implement": [
        ".agent_skills/02_ui_contract.md",
        ".agent_skills/05_security_boundary.md",
        ".agent_skills/04_quality_audit.md",
    ],
    "repair": [
        ".agent_skills/04_quality_audit.md",
        ".agent_skills/03_test_strategy.md",
        ".agent_skills/07_learning_log.md",
    ],
    "audit": [
        ".agent_skills/04_quality_audit.md",
        ".agent_skills/03_test_strategy.md",
        ".agent_skills/02_ui_contract.md",
    ],
    "upgrade": [
        ".agent_skills/02_ui_contract.md",
        ".agent_skills/04_quality_audit.md",
    ],
}

# 回写建议
WRITEBACK_RULES = {
    "implement": [
        "更新 .agent_skills/07_learning_log.md (记录实现决策)",
        "如有新模块，更新 .agent_skills/02_algorithm_contract.md",
        "如有新测试，更新 .agent_skills/03_test_strategy.md",
    ],
    "repair": [
        "更新 .agent_skills/07_learning_log.md (记录根因与修复)",
        "如有架构调整，更新 .agent_skills/06_refactor_policy.md",
    ],
    "audit": [
        "更新 .agent_skills/04_quality_audit.md (记录审计发现)",
        "如有安全问题，更新 .agent_skills/05_security_boundary.md",
    ],
    "upgrade": [
        "更新 manifest.json (版本与依赖)",
        "更新 .agent_skills/07_learning_log.md (记录升级决策)",
        "如有 API 变更，更新 .agent_skills/02_algorithm_contract.md",
    ],
}


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------

def find_repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "plugins").is_dir() and (cwd / "platform_core").is_dir():
        return cwd
    script_dir = Path(__file__).resolve().parent.parent
    if (script_dir / "plugins").is_dir():
        return script_dir
    return cwd


def resolve_target(repo_root: Path, target_str: str) -> tuple[Path, str]:
    """解析目标路径，返回 (绝对路径, scope)。"""
    target_str = target_str.rstrip("/")

    if target_str in (".", "root", ""):
        return repo_root, "root"
    if target_str == "ui":
        return repo_root / "ui", "ui"

    # plugins/xxx 或 xxx (尝试在 plugins/ 下查找)
    if target_str.startswith("plugins/"):
        target_path = repo_root / target_str
    else:
        target_path = repo_root / "plugins" / target_str

    if target_path.is_dir():
        return target_path, "plugin"

    return target_path, "invalid"


def get_governance(repo_root: Path, target_path: Path, scope: str) -> GovernanceResult:
    """获取目标的治理状态。"""
    name = target_path.name if scope != "root" else "root"
    return scan_target(name, target_path, scope)


def build_read_order(
    target_path: Path, scope: str, task: str, gov: GovernanceResult
) -> list[dict]:
    """构建文件读取顺序，返回 [{path, exists, priority}]。"""
    files = []

    # 前置文件 (root 使用专用命名)
    prereqs = ROOT_PREREQS if scope == "root" else COMMON_PREREQS
    for f in prereqs:
        p = target_path / f
        files.append({"path": str(p), "relative": f, "exists": p.is_file(), "priority": "必读"})

    # 任务路由文件本身 (root 的路由文件是 05_task_routing.md)
    if scope == "root":
        routing_rel = ".agent_skills/05_task_routing.md"
    else:
        routing_rel = ".agent_skills/08_task_routing.md"
    routing_file = target_path / routing_rel
    files.append({
        "path": str(routing_file),
        "relative": routing_rel,
        "exists": routing_file.is_file(),
        "priority": "路由参考",
    })

    # 按 scope + task 追加
    if scope == "root":
        extras = ROOT_READ_EXTRA.get(task, [])
    elif scope == "ui":
        extras = UI_READ_EXTRA.get(task, [])
    else:
        extras = TASK_READ_EXTRA.get(task, [])
    for f in extras:
        p = target_path / f
        # 去重
        if any(item["relative"] == f for item in files):
            continue
        files.append({
            "path": str(p),
            "relative": f,
            "exists": p.is_file(),
            "priority": "任务相关",
        })

    return files


def build_execution_plan(
    target_path: Path, scope: str, task: str, gov: GovernanceResult, module: Optional[str]
) -> list[dict]:
    """构建脚本执行计划，返回 [{script, exists, command, note}]。"""
    level = gov.governance_level

    # L0: 不具备自动路由条件
    if level == "L0":
        return []

    # 选择脚本映射
    if scope == "ui":
        script_map = UI_SCRIPTS
    elif level == "HF":
        script_map = HF_PLUGIN_SCRIPTS
    elif level == "STD":
        script_map = STD_PLUGIN_SCRIPTS
    else:  # MIN
        script_map = MIN_PLUGIN_SCRIPTS

    scripts = script_map.get(task, [])
    plan = []
    for s in scripts:
        script_path = target_path / s
        exists = script_path.is_file()

        # 构建命令
        cmd_parts = [str(script_path)]
        if module and "targeted" in s:
            cmd_parts.append(module)

        plan.append({
            "script": s,
            "exists": exists,
            "command": " ".join(cmd_parts),
            "note": "" if exists else "MISSING — 将被 SKIP",
        })

    return plan


def build_writeback(task: str) -> list[str]:
    """返回回写建议。"""
    return WRITEBACK_RULES.get(task, [])


# ---------------------------------------------------------------------------
# 输出格式化
# ---------------------------------------------------------------------------

def format_section(title: str, content: str) -> str:
    sep = "=" * 72
    return f"\n{sep}\n[{title}]\n{sep}\n{content}\n"


def format_target(
    target_path: Path, scope: str, gov: GovernanceResult, task: str, module: Optional[str]
) -> str:
    lines = [
        f"  对象名称:     {gov.name}",
        f"  对象范围:     {scope}",
        f"  对象类型:     {gov.object_type}",
        f"  治理等级:     {gov.governance_level}",
        f"  目标路径:     {target_path}",
        f"  任务类型:     {task}",
    ]
    if module:
        lines.append(f"  目标模块:     {module}")

    # 路由来源
    if scope == "root":
        routing_file = target_path / ".agent_skills" / "05_task_routing.md"
    else:
        routing_file = target_path / ".agent_skills" / "08_task_routing.md"
    if routing_file.is_file():
        lines.append(f"  路由来源:     {routing_file} (局部路由)")
    else:
        lines.append(f"  路由来源:     root 级规则 (回退 — 当前对象缺局部路由)")

    if gov.governance_level == "L0":
        lines.append(f"  ⚠ 警告:       当前对象不具备自动路由条件 (L0 占位态)")

    return "\n".join(lines)


def format_read_order(read_order: list[dict]) -> str:
    lines = []
    for i, item in enumerate(read_order, 1):
        status = "✓" if item["exists"] else "✗"
        lines.append(f"  {i:2d}. [{status}] {item['relative']:<50} ({item['priority']})")
    missing = [item for item in read_order if not item["exists"]]
    if missing:
        lines.append("")
        lines.append(f"  ⚠ {len(missing)} 个文件不存在，将跳过")
    return "\n".join(lines)


def format_execution_plan(plan: list[dict], gov: GovernanceResult) -> str:
    if gov.governance_level == "L0":
        return "  ⚠ L0 占位态 — 不具备自动路由条件，无可执行脚本\n  建议: 先完成最小治理升级 (plugin.py + manifest.json + tests/)"

    if not plan:
        return "  (无可执行脚本)"

    lines = []
    for i, step in enumerate(plan, 1):
        status = "READY" if step["exists"] else "SKIP"
        lines.append(f"  {i}. [{status}] {step['script']}")
        lines.append(f"     命令: {step['command']}")
        if step["note"]:
            lines.append(f"     备注: {step['note']}")
    return "\n".join(lines)


def format_writeback(wb: list[str]) -> str:
    lines = []
    for i, item in enumerate(wb, 1):
        lines.append(f"  {i}. {item}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 执行模式
# ---------------------------------------------------------------------------

def execute_plan(plan: list[dict], target_path: Path) -> list[dict]:
    """实际执行脚本，返回结果列表。"""
    results = []
    for step in plan:
        if not step["exists"]:
            results.append({**step, "status": "SKIP", "output": "脚本不存在"})
            continue

        script_abs = target_path / step["script"]
        cmd = step["command"]
        print(f"  ▶ 执行: {cmd}")

        try:
            proc = subprocess.run(
                ["bash", str(script_abs)] + (
                    step["command"].split()[1:] if len(step["command"].split()) > 1 else []
                ),
                cwd=str(target_path),
                capture_output=True,
                text=True,
                timeout=300,
            )
            status = "PASS" if proc.returncode == 0 else "FAIL"
            output = proc.stdout[-500:] if proc.stdout else ""
            if proc.stderr:
                output += f"\n[stderr] {proc.stderr[-300:]}"
            results.append({**step, "status": status, "output": output.strip()})
            print(f"    → {status}")
        except subprocess.TimeoutExpired:
            results.append({**step, "status": "FAIL", "output": "执行超时 (300s)"})
            print(f"    → FAIL (超时)")
        except Exception as e:
            results.append({**step, "status": "FAIL", "output": str(e)})
            print(f"    → FAIL ({e})")

    return results


def format_execution_results(results: list[dict]) -> str:
    lines = ["  执行结果:"]
    for i, r in enumerate(results, 1):
        lines.append(f"  {i}. [{r['status']}] {r['script']}")
        if r["output"]:
            for line in r["output"].split("\n")[:5]:
                lines.append(f"       {line}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    lines.append(f"\n  汇总: {passed} PASS / {failed} FAIL / {skipped} SKIP")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON 输出
# ---------------------------------------------------------------------------

def build_json_output(
    gov: GovernanceResult, task: str, module: Optional[str],
    read_order: list[dict], plan: list[dict], writeback: list[str],
    execution_results: Optional[list[dict]] = None,
) -> dict:
    return {
        "target": {
            "name": gov.name,
            "scope": gov.scope,
            "object_type": gov.object_type,
            "governance_level": gov.governance_level,
        },
        "task": task,
        "module": module,
        "read_order": read_order,
        "execution_plan": [
            {"script": s["script"], "exists": s["exists"], "command": s["command"]}
            for s in plan
        ],
        "writeback": writeback,
        "execution_results": execution_results,
    }


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DarkBreaker Agent Task Router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s plugins/busbar_inspection repair
  %(prog)s ui audit --dry-run
  %(prog)s plugins/animal_detection implement --run --module detector
  %(prog)s plugins/action_event_monitoring upgrade
  %(prog)s root audit
""",
    )
    parser.add_argument("target", help="目标路径 (root / ui / plugins/xxx)")
    parser.add_argument("task", choices=VALID_TASKS, help="任务类型")
    parser.add_argument("--module", "-m", default=None, help="目标模块名 (传给 targeted tests)")
    parser.add_argument("--run", action="store_true", help="实际执行脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅输出计划 (默认行为)")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    parser.add_argument("--root", type=str, default=None, help="指定仓库根目录")
    args = parser.parse_args()

    repo_root = Path(args.root) if args.root else find_repo_root()
    if not (repo_root / "plugins").is_dir():
        print(f"错误: {repo_root} 不是有效的 DarkBreaker 仓库根目录", file=sys.stderr)
        sys.exit(1)

    # 解析目标
    target_path, scope = resolve_target(repo_root, args.target)
    if scope == "invalid" or not target_path.is_dir():
        print(f"错误: 目标路径不存在或无效 — {target_path}", file=sys.stderr)
        sys.exit(1)

    # 获取治理状态
    gov = get_governance(repo_root, target_path, scope)

    # 构建四段式输出
    read_order = build_read_order(target_path, scope, args.task, gov)
    plan = build_execution_plan(target_path, scope, args.task, gov, args.module)
    writeback = build_writeback(args.task)

    execution_results = None

    # JSON 输出
    if args.json:
        if args.run and plan:
            execution_results = execute_plan(plan, target_path)
        data = build_json_output(gov, args.task, args.module, read_order, plan, writeback, execution_results)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    # 文本输出
    output = []
    output.append(format_section("TARGET", format_target(target_path, scope, gov, args.task, args.module)))
    output.append(format_section("READ_ORDER", format_read_order(read_order)))

    if args.run and plan:
        output.append(format_section("EXECUTION_PLAN", format_execution_plan(plan, gov)))
        print("\n".join(output))
        print(format_section("EXECUTION", ""))
        execution_results = execute_plan(plan, target_path)
        print(format_execution_results(execution_results))
    else:
        output.append(format_section("EXECUTION_PLAN", format_execution_plan(plan, gov)))
        if not args.run:
            output.append("  模式: dry-run (添加 --run 以实际执行)")

    output.append(format_section("WRITEBACK", format_writeback(writeback)))

    print("\n".join(output))


if __name__ == "__main__":
    main()
