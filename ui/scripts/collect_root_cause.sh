#!/usr/bin/env bash
# ============================================================
# collect_root_cause.sh — 复盘模板生成器
# 职责：输出统一复盘模板，供 UI 重构/修复后填写，
#       结果追加到 .agent_skills/04_quality_audit.md 和 07_learning_log.md。
# 用法：
#   ./scripts/collect_root_cause.sh                  # 输出到 stdout
#   ./scripts/collect_root_cause.sh > review.md      # 输出到文件
#   ./scripts/collect_root_cause.sh --append          # 追加到 agent_skills
# ============================================================

set -euo pipefail

UI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SKILLS_DIR="$UI_DIR/.agent_skills"
AUDIT_FILE="$SKILLS_DIR/04_quality_audit.md"
LEARNING_FILE="$SKILLS_DIR/07_learning_log.md"

TODAY=$(date '+%Y-%m-%d')

# ──────────────────────────────────────────────
# 收集当前项目状态信息
# ──────────────────────────────────────────────

# 模板文件数
TEMPLATE_COUNT=$(find "$UI_DIR/templates" -name "*.html" -not -name "*.bak" -not -name "*.backup*" 2>/dev/null | wc -l | tr -d ' ')
# JS 文件数
JS_COUNT=$(find "$UI_DIR/static/js" -name "*.js" -not -name "*.backup*" 2>/dev/null | wc -l | tr -d ' ')
# CSS 文件数
CSS_COUNT=$(find "$UI_DIR/static/css" -name "*.css" 2>/dev/null | wc -l | tr -d ' ')

# 超长文件列表
OVERLONG_FILES=""
while IFS= read -r f; do
    [ -z "$f" ] && continue
    LINES=$(wc -l < "$f" | tr -d ' ')
    if [ "$LINES" -gt 400 ]; then
        OVERLONG_FILES="${OVERLONG_FILES}\n  - $(basename "$f"): ${LINES} 行"
    fi
done < <(find "$UI_DIR/templates" -name "*.html" -not -name "*.bak" 2>/dev/null)

while IFS= read -r f; do
    [ -z "$f" ] && continue
    LINES=$(wc -l < "$f" | tr -d ' ')
    if [ "$LINES" -gt 500 ]; then
        OVERLONG_FILES="${OVERLONG_FILES}\n  - $(basename "$f"): ${LINES} 行"
    fi
done < <(find "$UI_DIR/static/js" -name "*.js" -not -name "*.backup*" 2>/dev/null)

# TODO/FIXME 残留数
TODO_COUNT=$(grep -rnE '(TODO|FIXME|HACK)' "$UI_DIR/templates" "$UI_DIR/static" \
    --include="*.html" --include="*.js" --include="*.css" 2>/dev/null | wc -l | tr -d ' ')

# ──────────────────────────────────────────────
# 输出复盘模板
# ──────────────────────────────────────────────

TEMPLATE=$(cat <<HEREDOC
## [${TODAY}] UI 复盘 — {{标题：简述本次变更范围}}

### 变更范围

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| {{文件路径}} | {{新增/修改/删除}} | {{一句话说明}} |

### 当前状态快照

- 模板文件: ${TEMPLATE_COUNT} 个
- JS 文件: ${JS_COUNT} 个
- CSS 文件: ${CSS_COUNT} 个
- TODO/FIXME 残留: ${TODO_COUNT} 处
- 超长文件: $([ -z "$OVERLONG_FILES" ] && echo "无" || echo -e "$OVERLONG_FILES")

### 问题分析

| # | 问题描述 | 所在文件:行号 | 违反规则编号 | 影响范围 |
|---|----------|--------------|-------------|---------|
| 1 | {{描述}} | {{file:line}} | {{F-001 等}} | {{影响}} |

### 根因分析

1. **直接原因**：{{为什么出了这个问题}}
2. **根本原因**：{{流程/规则/认知上哪里不足}}
3. **为什么没被拦住**：{{检查脚本/审查流程哪里漏了}}

### 反模式记录

- **反模式**：{{描述错误做法}}
- **正确做法**：{{描述应该怎么做}}
- **关联规则**：{{04_quality_audit 中的规则编号，若无则建议新增}}

### 经验提炼

1. {{可复用的经验 1}}
2. {{可复用的经验 2}}

### 规则更新建议

- [ ] 是否需要更新 04_quality_audit.md？规则编号：{{}}
- [ ] 是否需要更新 02_ui_contract.md？
- [ ] 是否需要更新 06_refactor_policy.md？
- [ ] 是否需要新增检查脚本逻辑？

---
HEREDOC
)

# ──────────────────────────────────────────────
# 输出 / 追加
# ──────────────────────────────────────────────

if [ "${1:-}" = "--append" ]; then
    echo "" >> "$LEARNING_FILE"
    echo "$TEMPLATE" >> "$LEARNING_FILE"
    echo "✔ 复盘模板已追加到 $(basename "$LEARNING_FILE")"

    # 同时在 audit 文件末尾添加占位
    echo "" >> "$AUDIT_FILE"
    echo "### [${TODAY}] 待填写复盘" >> "$AUDIT_FILE"
    echo "<!-- 请从 07_learning_log.md 同日期条目中提取规则更新 -->" >> "$AUDIT_FILE"
    echo "✔ 复盘占位已追加到 $(basename "$AUDIT_FILE")"
else
    echo "$TEMPLATE"
    echo ""
    echo "# ────────────────────────────────────────"
    echo "# 使用方式："
    echo "#   1. 复制上方模板，填写 {{}} 占位符"
    echo "#   2. 追加到 .agent_skills/07_learning_log.md"
    echo "#   3. 提取规则更新到 .agent_skills/04_quality_audit.md"
    echo "#   或直接运行: $0 --append"
    echo "# ────────────────────────────────────────"
fi
