#!/usr/bin/env bash
# ============================================================
# check_ui_contract.sh — UI 契约合规检查
# 职责：检查 ui/ 下模板和静态文件是否符合 .agent_skills/02_ui_contract.md
#        和 04_quality_audit.md 中的硬性规则。
# 检查项：
#   1. 内联 CSS（style= 属性）
#   2. 内联 JS（onclick/onload 等 on* 属性、<script> 内联块）
#   3. TODO / FIXME / alert('开发中') 残留
#   4. 页面模板是否使用 extra_css / extra_js block
#   5. 超长文件（模板 >400 行、JS >500 行、CSS >600 行）
# 输出：PASS / WARN / FAIL 逐项报告
# 依赖：grep（或 rg）、wc、bash 4+
# ============================================================

set -euo pipefail

UI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATES_DIR="$UI_DIR/templates"
STATIC_JS_DIR="$UI_DIR/static/js"
STATIC_CSS_DIR="$UI_DIR/static/css"

# 颜色
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

FAIL_COUNT=0
WARN_COUNT=0
PASS_COUNT=0

pass()  { PASS_COUNT=$((PASS_COUNT + 1));  echo -e "  ${GREEN}[PASS]${NC} $1"; }
warn()  { WARN_COUNT=$((WARN_COUNT + 1));  echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail()  { FAIL_COUNT=$((FAIL_COUNT + 1));  echo -e "  ${RED}[FAIL]${NC} $1"; }
header(){ echo -e "\n${CYAN}═══ $1 ═══${NC}"; }

# ──────────────────────────────────────────────
# 1. 内联 CSS 检查（style= 属性）
# ──────────────────────────────────────────────
header "1. 内联 CSS 检查（style= 属性）"

INLINE_CSS_HITS=$(grep -rn 'style="' "$TEMPLATES_DIR" --include="*.html" \
    | grep -v '{#' | grep -v '{%' | grep -v '<!--' \
    | grep -v 'id="nav-ws-status"' \
    || true)

if [ -z "$INLINE_CSS_HITS" ]; then
    pass "模板文件中未发现内联 style 属性"
else
    INLINE_CSS_COUNT=$(echo "$INLINE_CSS_HITS" | wc -l | tr -d ' ')
    fail "发现 ${INLINE_CSS_COUNT} 处内联 style 属性（违反 S-006）"
    echo "$INLINE_CSS_HITS" | head -10
    [ "$INLINE_CSS_COUNT" -gt 10 ] && echo "  ... 及更多（共 ${INLINE_CSS_COUNT} 处）"
fi

# ──────────────────────────────────────────────
# 2. 内联 JS 检查（on* 事件属性 + <script> 内联块）
# ──────────────────────────────────────────────
header "2. 内联 JS 检查"

# 2a. on* 事件属性
INLINE_EVENT_HITS=$(grep -rn -E '\s(onclick|onload|onchange|onsubmit|onerror|onmouseover|onkeydown|onkeyup|onfocus|onblur)=' \
    "$TEMPLATES_DIR" --include="*.html" || true)

if [ -z "$INLINE_EVENT_HITS" ]; then
    pass "未发现内联事件属性（onclick/onload 等）"
else
    EVENT_COUNT=$(echo "$INLINE_EVENT_HITS" | wc -l | tr -d ' ')
    fail "发现 ${EVENT_COUNT} 处内联事件属性"
    echo "$INLINE_EVENT_HITS" | head -10
fi

# 2b. <script> 内联块（排除 extra_js block 和 src= 引用）
INLINE_SCRIPT_HITS=$(grep -rn '<script' "$TEMPLATES_DIR" --include="*.html" \
    | grep -v 'src=' \
    | grep -v '{%' \
    | grep -v '<!--' \
    || true)

if [ -z "$INLINE_SCRIPT_HITS" ]; then
    pass "未发现非外部引用的 <script> 内联块"
else
    SCRIPT_COUNT=$(echo "$INLINE_SCRIPT_HITS" | wc -l | tr -d ' ')
    warn "发现 ${SCRIPT_COUNT} 处可能的内联 <script> 块（建议提取到 .js 文件）"
    echo "$INLINE_SCRIPT_HITS" | head -10
fi

# ──────────────────────────────────────────────
# 3. TODO / FIXME / alert 残留检查
# ──────────────────────────────────────────────
header "3. TODO / FIXME / alert 残留检查"

# 3a. TODO / FIXME
TODO_HITS=$(grep -rn -E '(TODO|FIXME|HACK|XXX)' \
    "$TEMPLATES_DIR" "$STATIC_JS_DIR" "$STATIC_CSS_DIR" \
    --include="*.html" --include="*.js" --include="*.css" \
    2>/dev/null || true)

if [ -z "$TODO_HITS" ]; then
    pass "未发现 TODO / FIXME / HACK 标记"
else
    TODO_COUNT=$(echo "$TODO_HITS" | wc -l | tr -d ' ')
    fail "发现 ${TODO_COUNT} 处 TODO/FIXME 残留（违反 H-001）"
    echo "$TODO_HITS" | head -10
fi

# 3b. alert('开发中') 或类似占位
ALERT_HITS=$(grep -rn -E "alert\s*\(\s*['\"]" \
    "$TEMPLATES_DIR" "$STATIC_JS_DIR" \
    --include="*.html" --include="*.js" \
    2>/dev/null || true)

if [ -z "$ALERT_HITS" ]; then
    pass "未发现 alert() 占位调用"
else
    ALERT_COUNT=$(echo "$ALERT_HITS" | wc -l | tr -d ' ')
    fail "发现 ${ALERT_COUNT} 处 alert() 调用（违反 H-002）"
    echo "$ALERT_HITS" | head -10
fi

# ──────────────────────────────────────────────
# 4. extra_css / extra_js block 使用检查
# ──────────────────────────────────────────────
header "4. extra_css / extra_js block 使用检查"

# 检查继承 base.html 的页面是否使用了 extra_css / extra_js
PAGE_FILES=$(find "$TEMPLATES_DIR/pages" -name "*.html" -not -name "*.bak" 2>/dev/null || true)

if [ -z "$PAGE_FILES" ]; then
    warn "未找到 templates/pages/ 下的页面文件"
else
    MISSING_EXTRA=()
    while IFS= read -r pf; do
        BASENAME=$(basename "$pf")
        HAS_EXTENDS=$(grep -c '{%.*extends' "$pf" 2>/dev/null || true)
        if [ "$HAS_EXTENDS" -gt 0 ]; then
            HAS_EXTRA_CSS=$(grep -c 'block extra_css' "$pf" 2>/dev/null || true)
            HAS_EXTRA_JS=$(grep -c 'block extra_js' "$pf" 2>/dev/null || true)
            if [ "$HAS_EXTRA_CSS" -eq 0 ] && [ "$HAS_EXTRA_JS" -eq 0 ]; then
                MISSING_EXTRA+=("$BASENAME")
            fi
        fi
    done <<< "$PAGE_FILES"

    if [ ${#MISSING_EXTRA[@]} -eq 0 ]; then
        pass "所有继承 base.html 的页面均使用了 extra_css 或 extra_js block"
    else
        warn "以下页面继承 base.html 但未使用 extra_css / extra_js block："
        for mf in "${MISSING_EXTRA[@]}"; do
            echo "    - $mf"
        done
    fi
fi

# ──────────────────────────────────────────────
# 5. 文件超长检查
# ──────────────────────────────────────────────
header "5. 文件超长检查"

TEMPLATE_LIMIT=400
JS_LIMIT=500
CSS_LIMIT=600

check_file_length() {
    local dir="$1"
    local ext="$2"
    local limit="$3"
    local label="$4"

    if [ ! -d "$dir" ]; then
        return
    fi

    OVERLONG=()
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        LINES=$(wc -l < "$f" | tr -d ' ')
        if [ "$LINES" -gt "$limit" ]; then
            OVERLONG+=("$(basename "$f"): ${LINES} 行（限制 ${limit}）")
        fi
    done < <(find "$dir" -name "*.$ext" -not -name "*.bak" -not -name "*.backup*" 2>/dev/null)

    if [ ${#OVERLONG[@]} -eq 0 ]; then
        pass "${label}文件均在 ${limit} 行限制内"
    else
        for ol in "${OVERLONG[@]}"; do
            fail "超长${label}文件: $ol（违反 F-001）"
        done
    fi
}

check_file_length "$TEMPLATES_DIR" "html" "$TEMPLATE_LIMIT" "模板"
check_file_length "$STATIC_JS_DIR" "js" "$JS_LIMIT" "JS"
check_file_length "$STATIC_CSS_DIR" "css" "$CSS_LIMIT" "CSS"

# ──────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────
header "汇总"

TOTAL=$((PASS_COUNT + WARN_COUNT + FAIL_COUNT))
echo -e "  检查项总计: ${TOTAL}"
echo -e "  ${GREEN}PASS: ${PASS_COUNT}${NC}  ${YELLOW}WARN: ${WARN_COUNT}${NC}  ${RED}FAIL: ${FAIL_COUNT}${NC}"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "\n  ${RED}✘ 契约检查未通过，请修复 FAIL 项后重新执行${NC}"
    exit 1
elif [ "$WARN_COUNT" -gt 0 ]; then
    echo -e "\n  ${YELLOW}⚠ 契约检查通过（有警告）${NC}"
    exit 0
else
    echo -e "\n  ${GREEN}✔ 契约检查全部通过${NC}"
    exit 0
fi
