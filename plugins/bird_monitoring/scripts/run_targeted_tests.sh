#!/usr/bin/env bash
# bird_monitoring — 分层测试执行入口
#
# 用法:
#   scripts/run_targeted_tests.sh [layer]
#
# layer:
#   l0     仅纯逻辑（risk / quality / quality_tristate / preflight）
#   l1     仅集成（standalone / plugin_contract / directory_contract）
#   l2     仅 regression replay（合成 fixture）
#   all    全部（默认）
#
# 退出码:
#   0  全绿
#   非 0  pytest 退出码透传
set -euo pipefail

LAYER="${1:-all}"
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PLUGIN_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

L0_TESTS=(
    "plugins/bird_monitoring/tests/test_risk_assessment.py"
    "plugins/bird_monitoring/tests/test_quality_assessment.py"
    "plugins/bird_monitoring/tests/test_quality_tristate.py"
    "plugins/bird_monitoring/tests/test_real_dl_preflight.py"
)
L1_TESTS=(
    "plugins/bird_monitoring/tests/test_standalone.py"
    "plugins/bird_monitoring/tests/test_plugin_contract.py"
    "plugins/bird_monitoring/tests/test_directory_contract.py"
)
L2_TESTS=(
    "plugins/bird_monitoring/tests/test_replay_baseline.py"
)

case "${LAYER}" in
    l0)
        exec python3 -m pytest "${L0_TESTS[@]}" -q
        ;;
    l1)
        exec python3 -m pytest "${L1_TESTS[@]}" -q
        ;;
    l2)
        exec python3 -m pytest "${L2_TESTS[@]}" -q
        ;;
    all)
        exec python3 -m pytest plugins/bird_monitoring/tests/ -q
        ;;
    *)
        echo "unknown layer: ${LAYER}" >&2
        echo "usage: $0 [l0|l1|l2|all]" >&2
        exit 2
        ;;
esac
