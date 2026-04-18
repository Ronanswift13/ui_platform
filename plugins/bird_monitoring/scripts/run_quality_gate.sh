#!/usr/bin/env bash
# bird_monitoring — 质量门入口（反模式扫描 + 测试 + 覆盖率）
#
# 用法:
#   scripts/run_quality_gate.sh [--no-coverage]
#
# 退出码:
#   0  全绿
#   1  反模式命中
#   2  pytest 失败
#   3  覆盖率不达标
#   4  环境缺失（pytest / ripgrep）
set -uo pipefail

NO_COVERAGE="0"
for arg in "$@"; do
    case "$arg" in
        --no-coverage) NO_COVERAGE="1" ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PLUGIN_DIR}/../.." && pwd)"

if ! python3 -m pytest --version >/dev/null 2>&1; then
    echo "[FAIL] pytest not available" >&2
    exit 4
fi

# 优先 ripgrep，回落到 grep -E（与 04_quality_audit.md 模式语法兼容）
SCAN_TOOL=""
if command -v rg >/dev/null 2>&1 && rg --version >/dev/null 2>&1; then
    SCAN_TOOL="rg"
elif command -v grep >/dev/null 2>&1; then
    SCAN_TOOL="grep"
else
    echo "[FAIL] 既无 rg 也无 grep" >&2
    exit 4
fi

echo "==> [1/3] 反模式扫描 (tool=${SCAN_TOOL})"
ANTIPATTERN_FAIL=0
scan_one() {
    local label="$1"; shift
    local pattern="$1"; shift
    local hits=""
    if [[ "${SCAN_TOOL}" == "rg" ]]; then
        hits="$(rg --color=never -n "${pattern}" "$@" 2>/dev/null || true)"
    else
        hits="$(grep -EHn "${pattern}" "$@" 2>/dev/null || true)"
    fi
    if [[ -n "${hits}" ]]; then
        echo "[FAIL] ${label} 命中:"
        echo "${hits}" | sed 's/^/    /'
        ANTIPATTERN_FAIL=1
    fi
}

cd "${PLUGIN_DIR}"

# 阻断级
scan_one "except: pass" 'except[^:]*:[[:space:]]*pass' plugin.py detector.py
scan_one "生产 print()" '\bprint\(' plugin.py detector.py
scan_one "驱离硬件耦合" 'requests\.post|urllib\.request|urlopen|serial\.|RPi\.GPIO|paho' plugin.py detector.py
scan_one "随机检测进生产" 'np\.random|random\.' plugin.py detector.py
scan_one "默认 sparrow 回退（return 形式）" 'return[[:space:]].*"sparrow"' plugin.py
# 高
scan_one "硬编码风险阈值（非 fallback）" 'RISK_THRESHOLDS\b' plugin.py

if [[ "${ANTIPATTERN_FAIL}" -ne 0 ]]; then
    exit 1
fi
echo "    ok"

cd "${REPO_ROOT}"

echo "==> [2/3] pytest"
if ! python3 -m pytest plugins/bird_monitoring/tests/ -q; then
    exit 2
fi

if [[ "${NO_COVERAGE}" == "1" ]]; then
    echo "==> [3/3] 覆盖率（已跳过）"
    exit 0
fi

if ! python3 -c "import coverage" >/dev/null 2>&1; then
    echo "==> [3/3] 覆盖率"
    echo "[WARN] coverage 未安装，跳过覆盖率门禁。安装: pip install coverage"
    exit 0
fi

echo "==> [3/3] 覆盖率（fail_under 见 .coveragerc）"
COVERAGE_RCFILE="plugins/bird_monitoring/.coveragerc" \
    python3 -m coverage run --rcfile=plugins/bird_monitoring/.coveragerc \
    -m pytest plugins/bird_monitoring/tests/ -q >/dev/null
if ! python3 -m coverage report --rcfile=plugins/bird_monitoring/.coveragerc; then
    exit 3
fi
