#!/usr/bin/env bash
# ============================================================
# run_quality_gate.sh — UI 质量门禁（聚合脚本）
# 职责：依次执行所有检查脚本，汇总结果，输出最终 PASS/FAIL。
# 用法：
#   ./scripts/run_quality_gate.sh           # 执行全部检查
#   ./scripts/run_quality_gate.sh --quick   # 仅 contract 检查（快速模式）
# ============================================================

set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
UI_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"

# 颜色
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════════════╗"
echo "║        UI 质量门禁 — run_quality_gate.sh       ║"
echo "╚════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  工作目录: ${UI_DIR}"
echo -e "  执行时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

GATE_PASS=0
GATE_FAIL=0
GATE_WARN=0
RESULTS=()

run_check() {
    local name="$1"
    local script="$2"

    echo -e "${BOLD}────────────────────────────────────────────${NC}"
    echo -e "${BOLD}▶ ${name}${NC}"
    echo -e "${BOLD}────────────────────────────────────────────${NC}"

    if [ ! -f "$script" ]; then
        echo -e "  ${RED}[SKIP] 脚本不存在: $script${NC}"
        RESULTS+=("SKIP: $name")
        return
    fi

    if ! [ -x "$script" ]; then
        chmod +x "$script"
    fi

    set +e
    bash "$script"
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        # 检查输出中是否有 WARN
        RESULTS+=("PASS: $name")
        GATE_PASS=$((GATE_PASS + 1))
    else
        RESULTS+=("FAIL: $name")
        GATE_FAIL=$((GATE_FAIL + 1))
    fi

    echo ""
}

# ──────────────────────────────────────────────
# 执行检查
# ──────────────────────────────────────────────

# 1. UI 契约检查（必跑）
run_check "UI 契约合规检查" "$SCRIPTS_DIR/check_ui_contract.sh"

# 2. 三态覆盖检查（--quick 模式跳过）
if [ "${1:-}" != "--quick" ]; then
    run_check "三态覆盖检查" "$SCRIPTS_DIR/check_three_state_coverage.sh"
fi

# 3. Jinja2 模板语法检查（轻量级）
echo -e "${BOLD}────────────────────────────────────────────${NC}"
echo -e "${BOLD}▶ Jinja2 模板基础语法检查${NC}"
echo -e "${BOLD}────────────────────────────────────────────${NC}"

JINJA_ERRORS=0
while IFS= read -r tf; do
    [ -z "$tf" ] && continue
    # 检查未闭合的 {% %} 和 {{ }}
    OPEN_BLOCK=$(grep -c '{%' "$tf" 2>/dev/null || true)
    CLOSE_BLOCK=$(grep -c '%}' "$tf" 2>/dev/null || true)
    OPEN_VAR=$(grep -c '{{' "$tf" 2>/dev/null || true)
    CLOSE_VAR=$(grep -c '}}' "$tf" 2>/dev/null || true)

    if [ "$OPEN_BLOCK" -ne "$CLOSE_BLOCK" ]; then
        echo -e "  ${RED}[FAIL]${NC} $(basename "$tf") — {%..%} 标签不匹配: 开 ${OPEN_BLOCK} / 闭 ${CLOSE_BLOCK}"
        JINJA_ERRORS=$((JINJA_ERRORS + 1))
    fi
    if [ "$OPEN_VAR" -ne "$CLOSE_VAR" ]; then
        echo -e "  ${RED}[FAIL]${NC} $(basename "$tf") — {{..}} 变量不匹配: 开 ${OPEN_VAR} / 闭 ${CLOSE_VAR}"
        JINJA_ERRORS=$((JINJA_ERRORS + 1))
    fi
done < <(find "$UI_DIR/templates" -name "*.html" -not -name "*.bak" -not -name "*.backup*" 2>/dev/null)

if [ "$JINJA_ERRORS" -eq 0 ]; then
    echo -e "  ${GREEN}[PASS]${NC} 所有模板 Jinja2 标签匹配正常"
    RESULTS+=("PASS: Jinja2 语法检查")
    GATE_PASS=$((GATE_PASS + 1))
else
    RESULTS+=("FAIL: Jinja2 语法检查")
    GATE_FAIL=$((GATE_FAIL + 1))
fi

# 4. JS 语法检查（使用 node --check，如果 node 可用）
echo ""
echo -e "${BOLD}────────────────────────────────────────────${NC}"
echo -e "${BOLD}▶ JS 语法检查${NC}"
echo -e "${BOLD}────────────────────────────────────────────${NC}"

if command -v node &>/dev/null; then
    JS_ERRORS=0
    while IFS= read -r jf; do
        [ -z "$jf" ] && continue
        if ! node --check "$jf" 2>/dev/null; then
            echo -e "  ${RED}[FAIL]${NC} $(basename "$jf") — JS 语法错误"
            JS_ERRORS=$((JS_ERRORS + 1))
        fi
    done < <(find "$UI_DIR/static/js" -name "*.js" -not -name "*.backup*" 2>/dev/null)

    if [ "$JS_ERRORS" -eq 0 ]; then
        echo -e "  ${GREEN}[PASS]${NC} 所有 JS 文件语法检查通过"
        RESULTS+=("PASS: JS 语法检查")
        GATE_PASS=$((GATE_PASS + 1))
    else
        RESULTS+=("FAIL: JS 语法检查")
        GATE_FAIL=$((GATE_FAIL + 1))
    fi
else
    echo -e "  ${YELLOW}[SKIP]${NC} node 不可用，跳过 JS 语法检查"
    RESULTS+=("SKIP: JS 语法检查")
fi

# ──────────────────────────────────────────────
# 最终汇总
# ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════════════╗"
echo "║              质量门禁汇总报告                    ║"
echo "╚════════════════════════════════════════════════╝"
echo -e "${NC}"

for r in "${RESULTS[@]}"; do
    case "$r" in
        PASS:*) echo -e "  ${GREEN}✔${NC} ${r#PASS: }" ;;
        FAIL:*) echo -e "  ${RED}✘${NC} ${r#FAIL: }" ;;
        SKIP:*) echo -e "  ${YELLOW}⊘${NC} ${r#SKIP: }" ;;
    esac
done

echo ""
TOTAL=$((GATE_PASS + GATE_FAIL))
echo -e "  通过: ${GREEN}${GATE_PASS}${NC} / ${TOTAL}"

if [ "$GATE_FAIL" -gt 0 ]; then
    echo -e "\n  ${RED}${BOLD}✘ 质量门禁未通过 — 请修复 FAIL 项后重新执行${NC}"
    exit 1
else
    echo -e "\n  ${GREEN}${BOLD}✔ 质量门禁全部通过${NC}"
    exit 0
fi
