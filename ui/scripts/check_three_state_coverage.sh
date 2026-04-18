#!/usr/bin/env bash
# ============================================================
# check_three_state_coverage.sh — 三态覆盖检查
# 职责：检查 UI 页面/组件是否覆盖 loading / empty / error 三态。
# 依据：04_quality_audit.md 三态覆盖规则
# 检查项：
#   1. 模板中是否存在 loading / empty / error 的显式文本或样式锚点
#   2. JS 中 fetch / async 数据加载是否有 catch
#   3. 页面/组件是否存在真实空态和错误态 DOM 节点
# ============================================================

set -euo pipefail

UI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATES_DIR="$UI_DIR/templates"
STATIC_JS_DIR="$UI_DIR/static/js"

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
# 1. 模板三态锚点检查
# ──────────────────────────────────────────────
header "1. 模板三态锚点检查（loading / empty / error）"

# 只检查页面模板（pages/ 下继承 base.html 的文件）
PAGE_FILES=$(find "$TEMPLATES_DIR/pages" -name "*.html" -not -name "*.bak" -not -name "*.backup*" 2>/dev/null || true)

if [ -z "$PAGE_FILES" ]; then
    warn "未找到 templates/pages/ 下的页面文件"
else
    while IFS= read -r pf; do
        [ -z "$pf" ] && continue
        BASENAME=$(basename "$pf")
        CONTENT=$(cat "$pf")

        # 检查 loading 态标识
        HAS_LOADING=$(echo "$CONTENT" | grep -cE '(loading|spinner|加载中|正在加载|data-state.*loading|class=.*loading|id=.*loading)' 2>/dev/null || true)
        # 检查 empty 态标识
        HAS_EMPTY=$(echo "$CONTENT" | grep -cE '(empty|no-data|暂无数据|无数据|data-state.*empty|class=.*empty|id=.*empty|没有.*记录)' 2>/dev/null || true)
        # 检查 error 态标识
        HAS_ERROR=$(echo "$CONTENT" | grep -cE '(error|alert-danger|加载失败|请求失败|data-state.*error|class=.*error|id=.*error|出错|异常)' 2>/dev/null || true)

        MISSING=()
        [ "$HAS_LOADING" -eq 0 ] && MISSING+=("loading")
        [ "$HAS_EMPTY" -eq 0 ]   && MISSING+=("empty")
        [ "$HAS_ERROR" -eq 0 ]   && MISSING+=("error")

        if [ ${#MISSING[@]} -eq 0 ]; then
            pass "$BASENAME — 三态锚点完整"
        elif [ ${#MISSING[@]} -le 1 ]; then
            warn "$BASENAME — 缺少: ${MISSING[*]}"
        else
            fail "$BASENAME — 缺少: ${MISSING[*]}"
        fi
    done <<< "$PAGE_FILES"
fi

# ──────────────────────────────────────────────
# 2. JS fetch / async 的 catch 覆盖检查
# ──────────────────────────────────────────────
header "2. JS fetch/async 的 catch 覆盖检查"

JS_FILES=$(find "$STATIC_JS_DIR" -name "*.js" -not -name "*.backup*" 2>/dev/null || true)

if [ -z "$JS_FILES" ]; then
    warn "未找到 static/js/ 下的 JS 文件"
else
    while IFS= read -r jf; do
        [ -z "$jf" ] && continue
        BASENAME=$(basename "$jf")

        # 统计 fetch 调用数
        FETCH_COUNT=$(grep -c 'fetch(' "$jf" 2>/dev/null || true)
        if [ "$FETCH_COUNT" -eq 0 ]; then
            continue  # 无 fetch 调用，跳过
        fi

        # 统计 .catch 或 try/catch 覆盖
        CATCH_COUNT=$(grep -cE '\.catch\s*\(|}\s*catch\s*\(' "$jf" 2>/dev/null || true)

        if [ "$CATCH_COUNT" -ge "$FETCH_COUNT" ]; then
            pass "$BASENAME — fetch:${FETCH_COUNT}, catch:${CATCH_COUNT}"
        elif [ "$CATCH_COUNT" -gt 0 ]; then
            warn "$BASENAME — fetch:${FETCH_COUNT}, catch:${CATCH_COUNT}（部分未覆盖）"
        else
            fail "$BASENAME — fetch:${FETCH_COUNT}, catch:0（无任何错误捕获）"
        fi
    done <<< "$JS_FILES"
fi

# ──────────────────────────────────────────────
# 3. 组件三态 DOM 节点检查
# ──────────────────────────────────────────────
header "3. 组件三态 DOM 节点检查"

COMPONENT_FILES=$(find "$TEMPLATES_DIR/components" -name "*.html" -not -name "*.bak" 2>/dev/null || true)

if [ -z "$COMPONENT_FILES" ]; then
    warn "未找到 templates/components/ 下的组件文件"
else
    CHECKED=0
    while IFS= read -r cf; do
        [ -z "$cf" ] && continue
        BASENAME=$(basename "$cf")
        CONTENT=$(cat "$cf")

        # 组件如果包含数据展示（table / chart / list），应有空态
        HAS_DATA_DISPLAY=$(echo "$CONTENT" | grep -cE '(<table|<canvas|<ul.*class|<div.*class.*(list|grid|chart|card))' 2>/dev/null || true)

        if [ "$HAS_DATA_DISPLAY" -eq 0 ]; then
            continue  # 纯控制组件，无需三态
        fi

        CHECKED=$((CHECKED + 1))
        HAS_EMPTY=$(echo "$CONTENT" | grep -cE '(empty|no-data|暂无|没有.*数据)' 2>/dev/null || true)
        HAS_ERROR=$(echo "$CONTENT" | grep -cE '(error|失败|出错|异常|alert-danger)' 2>/dev/null || true)

        MISSING=()
        [ "$HAS_EMPTY" -eq 0 ] && MISSING+=("empty")
        [ "$HAS_ERROR" -eq 0 ] && MISSING+=("error")

        if [ ${#MISSING[@]} -eq 0 ]; then
            pass "$BASENAME — 数据组件三态覆盖"
        else
            warn "$BASENAME — 数据组件缺少: ${MISSING[*]}"
        fi
    done <<< "$COMPONENT_FILES"

    [ "$CHECKED" -eq 0 ] && pass "无数据展示类组件需要检查"
fi

# ──────────────────────────────────────────────
# 4. JS 中 async 函数是否有 try-catch
# ──────────────────────────────────────────────
header "4. async 函数 try-catch 覆盖"

if [ -n "$JS_FILES" ]; then
    while IFS= read -r jf; do
        [ -z "$jf" ] && continue
        BASENAME=$(basename "$jf")

        ASYNC_COUNT=$(grep -c 'async ' "$jf" 2>/dev/null || true)
        [ "$ASYNC_COUNT" -eq 0 ] && continue

        TRY_COUNT=$(grep -c 'try\s*{' "$jf" 2>/dev/null || true)

        if [ "$TRY_COUNT" -ge "$ASYNC_COUNT" ]; then
            pass "$BASENAME — async:${ASYNC_COUNT}, try-catch:${TRY_COUNT}"
        elif [ "$TRY_COUNT" -gt 0 ]; then
            warn "$BASENAME — async:${ASYNC_COUNT}, try-catch:${TRY_COUNT}（部分未覆盖）"
        else
            fail "$BASENAME — async:${ASYNC_COUNT}, try-catch:0"
        fi
    done <<< "$JS_FILES"
fi

# ──────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────
header "汇总"

TOTAL=$((PASS_COUNT + WARN_COUNT + FAIL_COUNT))
echo -e "  检查项总计: ${TOTAL}"
echo -e "  ${GREEN}PASS: ${PASS_COUNT}${NC}  ${YELLOW}WARN: ${WARN_COUNT}${NC}  ${RED}FAIL: ${FAIL_COUNT}${NC}"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "\n  ${RED}✘ 三态覆盖检查未通过${NC}"
    exit 1
elif [ "$WARN_COUNT" -gt 0 ]; then
    echo -e "\n  ${YELLOW}⚠ 三态覆盖基本通过（有警告）${NC}"
    exit 0
else
    echo -e "\n  ${GREEN}✔ 三态覆盖检查全部通过${NC}"
    exit 0
fi
