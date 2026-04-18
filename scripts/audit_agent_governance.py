#!/usr/bin/env python3
"""
DarkBreaker Agent Governance Scanner
=====================================
扫描仓库 root / ui / plugins/* 的治理状态，输出治理等级、对象类型、缺失项。

用法:
    python scripts/audit_agent_governance.py              # 终端表格 + MD + JSON
    python scripts/audit_agent_governance.py --stdout-only  # 仅终端输出
    python scripts/audit_agent_governance.py --markdown-only # 仅生成 MD 报告
    python scripts/audit_agent_governance.py --json          # 仅生成 JSON 报告
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# root .agent_skills 编号体系 (00~08)
ROOT_SKILLS = [f"{i:02d}" for i in range(9)]

# plugin .agent_skills 编号体系 (00~08，部分有 09+)
PLUGIN_SKILLS = [f"{i:02d}" for i in range(9)]

QUALITY_SCRIPTS = [
    "run_targeted_tests.sh",
    "run_regression_tests.sh",
    "run_quality_gate.sh",
    "collect_root_cause.sh",
    "run_sanity_checks.sh",
]

DOC_FILES = [
    "plugin.py",
    "detector.py",
    "detector_enhanced.py",
    "manifest.json",
    "README.md",
    "PROJECT_CARD.md",
    "CLAUDE.md",
]

DOC_DIRS = [
    "tests",
    "configs",
    "standalone",
    "models",
]

HF_ENHANCED_REQUIRED_FILES = [
    "plugin.py",
    "detector_enhanced.py",
    "manifest.json",
]

HF_ENHANCED_REQUIRED_DIRS = [
    "tests",
    "configs",
]


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class GovernanceResult:
    name: str
    scope: str  # root / ui / plugin
    object_type: str = "unknown"
    governance_level: str = "L0"

    has_agent_skills: bool = False
    agent_skills_files: list = field(default_factory=list)
    agent_skills_missing: list = field(default_factory=list)

    has_claude_commands: bool = False

    scripts_present: list = field(default_factory=list)
    scripts_missing: list = field(default_factory=list)

    files_present: list = field(default_factory=list)
    files_missing: list = field(default_factory=list)

    dirs_present: list = field(default_factory=list)
    dirs_missing: list = field(default_factory=list)

    hf_upgrade_candidate: str = "no"
    hf_upgrade_reason: str = ""
    estimated_upgrade_effort: str = "high"
    recommended_next_action: str = ""


# ---------------------------------------------------------------------------
# 扫描逻辑
# ---------------------------------------------------------------------------

def scan_agent_skills(base: Path, expected_prefixes: list[str]) -> tuple[bool, list, list]:
    """检查 .agent_skills/ 目录及其 00~08 文件是否齐全。"""
    skills_dir = base / ".agent_skills"
    if not skills_dir.is_dir():
        return False, [], list(expected_prefixes)

    existing = sorted(f.name for f in skills_dir.iterdir() if f.is_file() and f.suffix == ".md")
    existing_prefixes = {f.split("_")[0] for f in existing}

    present = [p for p in expected_prefixes if p in existing_prefixes]
    missing = [p for p in expected_prefixes if p not in existing_prefixes]
    return True, present, missing


def scan_scripts(base: Path) -> tuple[list, list]:
    """检查 scripts/ 下的质量门禁脚本。"""
    scripts_dir = base / "scripts"
    present, missing = [], []
    for s in QUALITY_SCRIPTS:
        if (scripts_dir / s).is_file():
            present.append(s)
        else:
            missing.append(s)
    return present, missing


def scan_files_and_dirs(base: Path) -> tuple[list, list, list, list]:
    """检查关键文件和目录。"""
    fp, fm, dp, dm = [], [], [], []
    for f in DOC_FILES:
        if (base / f).is_file():
            fp.append(f)
        else:
            fm.append(f)
    for d in DOC_DIRS:
        if (base / d).is_dir():
            dp.append(d)
        else:
            dm.append(d)
    return fp, fm, dp, dm


def classify_object_type(r: GovernanceResult) -> str:
    """根据检测到的文件推断对象类型。"""
    if r.scope == "root":
        return "root"
    if r.scope == "ui":
        return "ui"

    # plugin 分类
    has_plugin = "plugin.py" in r.files_present
    has_manifest = "manifest.json" in r.files_present
    has_detector_enhanced = "detector_enhanced.py" in r.files_present
    has_detector = "detector.py" in r.files_present
    has_tests = "tests" in r.dirs_present
    has_standalone = "standalone" in r.dirs_present

    # L0 占位态
    if not has_plugin and not has_manifest:
        return "placeholder"

    if has_detector_enhanced:
        return "plugin-enhanced-detector"
    if has_detector:
        return "plugin-light-detector"

    # 有 plugin.py 但无 detector — 可能是服务集成类或最小插件
    if has_standalone or has_tests:
        return "plugin-service-integration"

    return "plugin-minimal"


def determine_governance_level(r: GovernanceResult) -> str:
    """
    治理等级判定:
      L0  — 占位态（无 plugin.py / manifest.json）
      MIN — 最小治理级（有 plugin.py + manifest.json + 基础入口）
      STD — 标准治理级（有 .agent_skills/00~08，但 commands / 四脚本不完整）
      HF  — 高频治理级（agent_skills 齐全 + .claude/commands + 质量门禁脚本 ≥4）
    """
    has_plugin = "plugin.py" in r.files_present
    has_manifest = "manifest.json" in r.files_present

    # root / ui 特殊处理
    if r.scope in ("root", "ui"):
        if r.has_agent_skills and not r.agent_skills_missing:
            if r.has_claude_commands and len(r.scripts_present) >= 4:
                return "HF"
            return "STD"
        if r.has_agent_skills:
            return "MIN"
        return "L0"

    # plugin
    if not has_plugin and not has_manifest:
        return "L0"

    skills_complete = r.has_agent_skills and len(r.agent_skills_missing) == 0
    has_commands = r.has_claude_commands
    scripts_count = len(r.scripts_present)

    if skills_complete and has_commands and scripts_count >= 4:
        return "HF"
    if skills_complete:
        return "STD"
    if has_plugin and has_manifest:
        return "MIN"
    return "L0"


def has_real_core_fact_sources(r: GovernanceResult) -> tuple[bool, list[str]]:
    """判断对象是否具备快速晋升所需的真实核心事实源。"""
    required_files = ["plugin.py", "manifest.json"]
    required_dirs = list(HF_ENHANCED_REQUIRED_DIRS)

    if r.object_type == "plugin-enhanced-detector":
        required_files = list(HF_ENHANCED_REQUIRED_FILES)

    missing = [f for f in required_files if f not in r.files_present]
    missing.extend(d for d in required_dirs if d not in r.dirs_present)
    return len(missing) == 0, missing


def estimate_upgrade_effort(r: GovernanceResult, core_ready: bool) -> str:
    """估算升级到 HF 的工作量。"""
    if r.hf_upgrade_candidate != "yes":
        if r.governance_level == "HF":
            return "low"
        if r.governance_level == "STD" and core_ready:
            return "medium"
        return "high"

    scripts_count = len(r.scripts_present)
    if r.has_claude_commands and scripts_count >= 3:
        return "low"
    if scripts_count >= 1:
        return "medium"
    return "high"


def recommend_next_action(r: GovernanceResult, core_ready: bool, core_missing: list[str]) -> str:
    """基于当前对象目录事实给出具体下一步。"""
    if r.hf_upgrade_candidate == "yes":
        if not r.has_claude_commands:
            return "add/update .claude/commands"
        if "run_quality_gate.sh" not in r.scripts_present and "tests" in r.dirs_present:
            return "normalize run_quality_gate.sh"
        if ("README.md" in r.files_missing) or ("PROJECT_CARD.md" in r.files_missing):
            return "fill README / PROJECT_CARD mismatch"
        return "align 08_task_routing.md with scripts"

    if r.governance_level == "HF":
        return "align 08_task_routing.md with scripts"

    if r.governance_level == "STD" and r.object_type == "plugin-enhanced-detector" and not core_ready:
        if "tests" in core_missing or "configs" in core_missing:
            return "fill missing tests/configs core facts"
        if "README.md" in r.files_missing or "PROJECT_CARD.md" in r.files_missing:
            return "fill README / PROJECT_CARD mismatch"
        return "normalize run_quality_gate.sh"

    if r.governance_level == "STD":
        if not r.has_claude_commands:
            return "add/update .claude/commands"
        if "run_quality_gate.sh" not in r.scripts_present and "tests" in r.dirs_present:
            return "normalize run_quality_gate.sh"
        if r.has_agent_skills and r.scripts_present:
            return "align 08_task_routing.md with scripts"
        if "README.md" in r.files_missing or "PROJECT_CARD.md" in r.files_missing:
            return "fill README / PROJECT_CARD mismatch"
        return "stabilize core facts before HF promotion"

    if r.governance_level == "MIN":
        if not r.has_agent_skills:
            return "add .agent_skills/00~08"
        return "fill README / PROJECT_CARD mismatch"

    return "stabilize plugin core facts before promotion"


def evaluate_hf_upgrade_candidate(r: GovernanceResult) -> None:
    """基于当前对象目录事实固化 HF 快速晋升规则。"""
    core_ready, core_missing = has_real_core_fact_sources(r)

    if r.scope != "plugin":
        r.hf_upgrade_candidate = "no"
        r.hf_upgrade_reason = "当前规则只针对 plugin 对象评估 HF 快速晋升"
        r.estimated_upgrade_effort = estimate_upgrade_effort(r, core_ready)
        r.recommended_next_action = recommend_next_action(r, core_ready, core_missing)
        return

    if r.governance_level == "HF":
        r.hf_upgrade_candidate = "no"
        r.hf_upgrade_reason = "当前对象已经是 HF，无需作为 STD→HF 候选"
        r.estimated_upgrade_effort = estimate_upgrade_effort(r, core_ready)
        r.recommended_next_action = recommend_next_action(r, core_ready, core_missing)
        return

    if r.governance_level != "STD":
        r.hf_upgrade_candidate = "no"
        r.hf_upgrade_reason = f"当前治理等级为 {r.governance_level}，不属于 STD→HF 快速晋升阶段"
        r.estimated_upgrade_effort = estimate_upgrade_effort(r, core_ready)
        r.recommended_next_action = recommend_next_action(r, core_ready, core_missing)
        return

    if r.object_type != "plugin-enhanced-detector":
        r.hf_upgrade_candidate = "no"
        r.hf_upgrade_reason = (
            f"当前对象类型为 {r.object_type}，不适用 enhanced detector 型 HF 快速晋升规则"
        )
        r.estimated_upgrade_effort = estimate_upgrade_effort(r, core_ready)
        r.recommended_next_action = recommend_next_action(r, core_ready, core_missing)
        return

    if not r.has_agent_skills:
        r.hf_upgrade_candidate = "no"
        r.hf_upgrade_reason = "缺少 .agent_skills，尚未具备稳定治理知识层"
        r.estimated_upgrade_effort = estimate_upgrade_effort(r, core_ready)
        r.recommended_next_action = recommend_next_action(r, core_ready, core_missing)
        return

    if not core_ready:
        r.hf_upgrade_candidate = "no"
        r.hf_upgrade_reason = (
            "缺少 HF 快速晋升所需核心事实源: " + ", ".join(core_missing)
        )
        r.estimated_upgrade_effort = estimate_upgrade_effort(r, core_ready)
        r.recommended_next_action = recommend_next_action(r, core_ready, core_missing)
        return

    r.hf_upgrade_candidate = "yes"
    r.hf_upgrade_reason = (
        "STD + plugin-enhanced-detector + has_agent_skills=true + "
        "核心事实源齐备 (plugin.py / detector_enhanced.py / manifest.json / tests / configs)"
    )
    r.estimated_upgrade_effort = estimate_upgrade_effort(r, core_ready)
    r.recommended_next_action = recommend_next_action(r, core_ready, core_missing)


def scan_target(name: str, base: Path, scope: str) -> GovernanceResult:
    """扫描一个对象（root / ui / plugin）。"""
    expected_skills = ROOT_SKILLS if scope == "root" else PLUGIN_SKILLS
    r = GovernanceResult(name=name, scope=scope)

    r.has_agent_skills, r.agent_skills_files, r.agent_skills_missing = scan_agent_skills(
        base, expected_skills
    )
    r.has_claude_commands = (base / ".claude" / "commands").is_dir()
    r.scripts_present, r.scripts_missing = scan_scripts(base)
    r.files_present, r.files_missing, r.dirs_present, r.dirs_missing = scan_files_and_dirs(base)

    r.object_type = classify_object_type(r)
    r.governance_level = determine_governance_level(r)
    evaluate_hf_upgrade_candidate(r)
    return r


def scan_all(repo_root: Path) -> list[GovernanceResult]:
    """扫描全部对象。"""
    results = []

    # 1. root
    results.append(scan_target("root", repo_root, "root"))

    # 2. ui
    ui_dir = repo_root / "ui"
    if ui_dir.is_dir():
        results.append(scan_target("ui", ui_dir, "ui"))

    # 3. plugins
    plugins_dir = repo_root / "plugins"
    if plugins_dir.is_dir():
        for p in sorted(plugins_dir.iterdir()):
            if p.is_dir() and not p.name.startswith("_"):
                results.append(scan_target(p.name, p, "plugin"))

    return results


# ---------------------------------------------------------------------------
# 输出: 终端表格
# ---------------------------------------------------------------------------

def print_terminal_table(results: list[GovernanceResult]) -> str:
    """生成终端友好的摘要表格，同时返回字符串。"""
    header = (
        f"{'Name':<28} {'Scope':<8} {'Type':<28} {'Level':<5} "
        f"{'Skills':<8} {'Cmds':<5} {'Scripts':<10} {'HF↑':<4} {'Effort':<6} {'Files':<12} {'Dirs':<10}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        skills_str = (
            f"{len(r.agent_skills_files)}/{len(r.agent_skills_files) + len(r.agent_skills_missing)}"
            if r.has_agent_skills
            else "—"
        )
        cmds_str = "✓" if r.has_claude_commands else "—"
        scripts_str = f"{len(r.scripts_present)}/{len(r.scripts_present) + len(r.scripts_missing)}"
        hf_str = "✓" if r.hf_upgrade_candidate == "yes" else "—"
        files_str = f"{len(r.files_present)}/{len(r.files_present) + len(r.files_missing)}"
        dirs_str = f"{len(r.dirs_present)}/{len(r.dirs_present) + len(r.dirs_missing)}"

        lines.append(
            f"{r.name:<28} {r.scope:<8} {r.object_type:<28} {r.governance_level:<5} "
            f"{skills_str:<8} {cmds_str:<5} {scripts_str:<10} {hf_str:<4} {r.estimated_upgrade_effort:<6} {files_str:<12} {dirs_str:<10}"
        )

    lines.append(sep)
    output = "\n".join(lines)
    return output


# ---------------------------------------------------------------------------
# 输出: Markdown 报告
# ---------------------------------------------------------------------------

def generate_markdown(results: list[GovernanceResult]) -> str:
    """生成 Markdown 报告。"""
    lines: list[str] = []
    w = lines.append

    w("# DarkBreaker Agent Governance Report\n")
    w(f"> 自动生成 — 扫描时间基于当前目录状态\n")

    # --- 总体统计 ---
    w("## 1. 总体统计\n")
    total = len(results)
    by_level = {}
    by_type = {}
    for r in results:
        by_level.setdefault(r.governance_level, []).append(r.name)
        by_type.setdefault(r.object_type, []).append(r.name)
    hf_candidates = [r for r in results if r.hf_upgrade_candidate == "yes"]

    w(f"| 指标 | 值 |")
    w(f"|------|-----|")
    w(f"| 扫描对象总数 | {total} |")
    for lvl in ("HF", "STD", "MIN", "L0"):
        names = by_level.get(lvl, [])
        w(f"| {lvl} 等级数量 | {len(names)} |")
    w("")

    w("**按对象类型分布:**\n")
    w("| 类型 | 数量 | 成员 |")
    w("|------|------|------|")
    for t in sorted(by_type.keys()):
        members = ", ".join(by_type[t])
        w(f"| {t} | {len(by_type[t])} | {members} |")
    w("")

    # --- 治理等级表 ---
    w("## 2. 各对象治理等级表\n")
    w(
        "| 对象 | 范围 | 类型 | 等级 | "
        "agent_skills | commands | 脚本 | HF候选 | 工作量 | 下一步 | 关键文件 | 目录 |"
    )
    w("|------|------|------|------|-------------|----------|------|----------|--------|----------|----------|------|")
    for r in results:
        skills = (
            f"{len(r.agent_skills_files)}/{len(r.agent_skills_files) + len(r.agent_skills_missing)}"
            if r.has_agent_skills
            else "—"
        )
        cmds = "✓" if r.has_claude_commands else "—"
        scripts = f"{len(r.scripts_present)}/{len(r.scripts_present) + len(r.scripts_missing)}"
        hf = r.hf_upgrade_candidate
        files = f"{len(r.files_present)}/{len(r.files_present) + len(r.files_missing)}"
        dirs = f"{len(r.dirs_present)}/{len(r.dirs_present) + len(r.dirs_missing)}"
        w(
            f"| {r.name} | {r.scope} | {r.object_type} | **{r.governance_level}** | "
            f"{skills} | {cmds} | {scripts} | {hf} | {r.estimated_upgrade_effort} | {r.recommended_next_action} | {files} | {dirs} |"
        )
    w("")

    # --- 缺失项统计 ---
    w("## 3. 缺失项统计\n")
    # 按缺失类型汇总
    missing_skills: list[str] = []
    missing_commands: list[str] = []
    missing_scripts_map: dict[str, list[str]] = {}
    missing_files_map: dict[str, list[str]] = {}
    missing_dirs_map: dict[str, list[str]] = {}

    for r in results:
        if r.agent_skills_missing:
            missing_skills.append(f"{r.name} (缺 {', '.join(r.agent_skills_missing)})")
        if not r.has_claude_commands:
            missing_commands.append(r.name)
        for s in r.scripts_missing:
            missing_scripts_map.setdefault(s, []).append(r.name)
        for f in r.files_missing:
            missing_files_map.setdefault(f, []).append(r.name)
        for d in r.dirs_missing:
            missing_dirs_map.setdefault(d, []).append(r.name)

    w("### 3.1 缺失 agent_skills 编号\n")
    if missing_skills:
        for item in missing_skills:
            w(f"- {item}")
    else:
        w("全部齐全。")
    w("")

    w("### 3.2 缺失 .claude/commands/\n")
    if missing_commands:
        w(f"共 {len(missing_commands)} 个对象: {', '.join(missing_commands)}")
    else:
        w("全部具备。")
    w("")

    w("### 3.3 缺失质量门禁脚本\n")
    if missing_scripts_map:
        w("| 脚本 | 缺失对象数 | 缺失列表 |")
        w("|------|-----------|----------|")
        for s in QUALITY_SCRIPTS:
            objs = missing_scripts_map.get(s, [])
            if objs:
                w(f"| {s} | {len(objs)} | {', '.join(objs)} |")
    else:
        w("全部齐全。")
    w("")

    w("### 3.4 缺失关键文件\n")
    if missing_files_map:
        w("| 文件 | 缺失对象数 |")
        w("|------|-----------|")
        for f in DOC_FILES:
            objs = missing_files_map.get(f, [])
            if objs:
                w(f"| {f} | {len(objs)} |")
    w("")

    # --- HF 快速晋升规则 ---
    w("## 4. HF 候选快速晋升规则\n")
    w("### 4.1 固化规则\n")
    w("当前优先固化一条高价值筛选规则：\n")
    w("- `governance_level == STD`\n")
    w("- `object_type == plugin-enhanced-detector`\n")
    w("- `has_agent_skills == true`\n")
    w("- 且对象自身目录具备真实核心事实源：`plugin.py` / `detector_enhanced.py` / `manifest.json` / `tests/` / `configs/`\n")
    w("- 满足以上条件 -> 标记为 `HF upgrade candidate`\n")
    w("")

    w("### 4.2 HF 升级候选\n")
    if hf_candidates:
        w("| 对象 | 类型 | 工作量 | 推荐下一步 | 原因 |")
        w("|------|------|--------|------------|------|")
        for r in hf_candidates:
            w(
                f"| {r.name} | {r.object_type} | {r.estimated_upgrade_effort} | "
                f"{r.recommended_next_action} | {r.hf_upgrade_reason} |"
            )
    else:
        w("当前扫描结果中没有满足快速晋升规则的对象。")
    w("")

    std_candidates = [r for r in results if r.scope == "plugin" and r.governance_level == "STD"]
    std_not_ready = [r for r in std_candidates if r.hf_upgrade_candidate != "yes"]
    min_candidates = [r for r in results if r.governance_level == "MIN"]
    l0_candidates = [r for r in results if r.governance_level == "L0"]

    if std_not_ready:
        w("### 4.3 虽然是 STD，但当前不适合直接升 HF 的对象\n")
        w("| 对象 | 类型 | 当前不宜直升原因 | 建议下一步 |")
        w("|------|------|------------------|------------|")
        for r in std_not_ready:
            w(
                f"| {r.name} | {r.object_type} | {r.hf_upgrade_reason} | {r.recommended_next_action} |"
            )
        w("")

    # --- 推荐下一步动作 ---
    w("## 5. 推荐下一步动作\n")

    if min_candidates:
        w("### MIN → STD 升级候选\n")
        w("以下对象需要补齐 agent_skills 00~08:\n")
        for r in min_candidates:
            w(f"- **{r.name}**: 缺 skills {', '.join(r.agent_skills_missing)}")
        w("")

    if l0_candidates:
        w("### L0 占位态对象\n")
        w("以下对象处于占位状态，需要评估是否计划激活:\n")
        for r in l0_candidates:
            w(f"- **{r.name}** ({r.object_type})")
        w("")

    # 谁缺 README
    no_readme = [r.name for r in results if "README.md" in r.files_missing and r.scope == "plugin"]
    if no_readme:
        w("### 缺失 README.md 的插件\n")
        w(f"{', '.join(no_readme)}\n")

    # 谁缺 PROJECT_CARD
    no_card = [
        r.name for r in results if "PROJECT_CARD.md" in r.files_missing and r.scope == "plugin"
    ]
    if no_card:
        w("### 缺失 PROJECT_CARD.md 的插件\n")
        w(f"{', '.join(no_card)}\n")

    # --- 复用说明 ---
    w("## 6. 后续复用说明\n")
    w("### 可供 agent_task_router 复用的字段\n")
    w("- `object_type`: 直接作为路由分类依据，决定任务分发策略\n")
    w("- `governance_level`: 决定任务可接受的复杂度上限 (L0 不接受任何写操作)\n")
    w("- `has_agent_skills` + `agent_skills_files`: 判断是否具备上下文知识库\n")
    w("- `scripts_present`: 决定能否执行自动化质量门禁\n")
    w("")
    w("### 可供 sync_agent_commands.py 复用的字段\n")
    w("- `has_claude_commands`: 识别哪些对象需要同步 command 模板\n")
    w("- `governance_level == 'STD'` 的对象: 优先批量补齐 commands\n")
    w("- `scripts_present` / `scripts_missing`: 决定 command 中可引用的脚本\n")
    w("- `hf_upgrade_candidate` / `estimated_upgrade_effort`: 识别哪些对象值得优先补 command 与脚本\n")
    w("- `recommended_next_action`: 直接决定同步器先补 commands 还是先补脚本/文档\n")
    w("")
    w("### 为什么 enhanced detector 型插件是当前最优晋升对象\n")
    w("1. 这类对象通常已经有 `plugin.py + detector_enhanced.py + manifest.json` 这组稳定事实源，结构最适合模板化治理升级\n")
    w("2. 一旦同时具备 `tests/` 与 `configs/`，说明它已经有最基本的运行、验证与配置边界，补 HF 的收益高于重新补骨架\n")
    w("3. 它们从 STD 升到 HF 往往不需要重写算法，只需要把 commands、脚本和 task routing 对齐\n")
    w("")
    w("### 为什么这条规则对后续 sync_agent_commands.py / sync_plugin_template.py 有帮助\n")
    w("1. 规则只依赖扫描字段和对象自身目录事实，适合脚本稳定复用，不依赖人工记忆插件名字\n")
    w("2. `hf_upgrade_candidate` 可直接作为同步优先级过滤器，减少对不成熟对象误下发 commands/template 的风险\n")
    w("3. `recommended_next_action` 让同步器可以先做最值钱的一步：补 commands、补质量门禁脚本、或修正文档/卡片不一致\n")
    w("")
    w('### 如何判断"下一个最值得升级的插件"\n')
    w("1. 先看 `hf_upgrade_candidate == yes`\n")
    w("2. 再按 `estimated_upgrade_effort` 从 `low -> medium -> high` 排序\n")
    w("3. 同等条件下，优先选择 `recommended_next_action` 能直接由同步脚本完成的对象\n")
    w("4. 虽然是 STD 但缺少核心事实源、缺 tests/configs、或只是 service-integration 骨架的对象，暂不直升 HF\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 输出: JSON
# ---------------------------------------------------------------------------

def generate_json(results: list[GovernanceResult]) -> str:
    """生成 JSON 报告。"""
    data = {
        "summary": {
            "total": len(results),
            "by_level": {},
            "by_type": {},
            "hf_upgrade_candidates": [],
            "std_not_ready_for_hf": [],
        },
        "objects": [],
    }
    for r in results:
        data["summary"]["by_level"].setdefault(r.governance_level, []).append(r.name)
        data["summary"]["by_type"].setdefault(r.object_type, []).append(r.name)
        if r.hf_upgrade_candidate == "yes":
            data["summary"]["hf_upgrade_candidates"].append(r.name)
        elif r.scope == "plugin" and r.governance_level == "STD":
            data["summary"]["std_not_ready_for_hf"].append(r.name)
        data["objects"].append(asdict(r))

    return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def find_repo_root() -> Path:
    """从 CWD 或脚本位置查找仓库根。"""
    # 优先使用 CWD
    cwd = Path.cwd()
    if (cwd / "plugins").is_dir() and (cwd / "platform_core").is_dir():
        return cwd
    # 尝试脚本所在目录的上层
    script_dir = Path(__file__).resolve().parent.parent
    if (script_dir / "plugins").is_dir():
        return script_dir
    return cwd


def main():
    parser = argparse.ArgumentParser(description="DarkBreaker Agent Governance Scanner")
    parser.add_argument("--json", action="store_true", help="仅输出 JSON 报告到 docs/")
    parser.add_argument("--markdown-only", action="store_true", help="仅输出 Markdown 报告到 docs/")
    parser.add_argument("--stdout-only", action="store_true", help="仅终端输出，不写文件")
    parser.add_argument("--root", type=str, default=None, help="指定仓库根目录")
    args = parser.parse_args()

    repo_root = Path(args.root) if args.root else find_repo_root()
    if not (repo_root / "plugins").is_dir():
        print(f"错误: {repo_root} 不是有效的 DarkBreaker 仓库根目录", file=sys.stderr)
        sys.exit(1)

    results = scan_all(repo_root)

    # 终端表格 (除 --json / --markdown-only 外都输出)
    if not args.json and not args.markdown_only:
        table = print_terminal_table(results)
        print(table)
        print()

        # 快速摘要
        by_level = {}
        for r in results:
            by_level.setdefault(r.governance_level, []).append(r.name)
        print("治理等级分布:")
        for lvl in ("HF", "STD", "MIN", "L0"):
            names = by_level.get(lvl, [])
            if names:
                print(f"  {lvl}: {len(names)} — {', '.join(names)}")
        print()

        hf_candidates = [r.name for r in results if r.hf_upgrade_candidate == "yes"]
        if hf_candidates:
            print("HF 快速晋升候选:")
            print(f"  {len(hf_candidates)} — {', '.join(hf_candidates)}")
            print()

    # 写文件
    if not args.stdout_only:
        docs_dir = repo_root / "docs"
        docs_dir.mkdir(exist_ok=True)

        if not args.json:
            md_path = docs_dir / "agent_governance_report.md"
            md_path.write_text(generate_markdown(results), encoding="utf-8")
            print(f"Markdown 报告: {md_path}")

        if not args.markdown_only:
            json_path = docs_dir / "agent_governance_report.json"
            json_path.write_text(generate_json(results), encoding="utf-8")
            print(f"JSON 报告: {json_path}")


if __name__ == "__main__":
    main()
