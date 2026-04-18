#!/usr/bin/env bash
# bird_monitoring — 跨插件回归入口
#
# 用法:
#   scripts/run_regression_tests.sh [extra_pytest_args...]
#
# 默认覆盖 6 个对照插件，验证 bird_monitoring 不污染其他插件 import 路径。
#
# 退出码: 透传 pytest
set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PLUGIN_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PEER_PLUGINS=(
    "plugins/bird_monitoring/tests/"
    "plugins/busbar_inspection/tests/"
    "plugins/animal_detection/tests/"
    "plugins/fire_detection/tests/"
    "plugins/transformer_inspection/tests/"
    "plugins/switch_inspection/tests/"
)

# 仅对存在的目录入参，避免环境差异导致 pytest exit 4
EXISTING=()
for p in "${PEER_PLUGINS[@]}"; do
    if [[ -d "${p}" ]]; then
        EXISTING+=("${p}")
    fi
done

if [[ "${#EXISTING[@]}" -eq 0 ]]; then
    echo "[FAIL] 没有可跑的 peer plugin tests 目录" >&2
    exit 2
fi

exec python3 -m pytest "${EXISTING[@]}" -q "$@"
