#!/usr/bin/env bash
# ===========================================================================
# DarkBreaker Agent Task Router — Shell 入口
# ===========================================================================
#
# 用法:
#   ./scripts/agent_task_router.sh <target> <task> [options]
#
# 示例:
#   ./scripts/agent_task_router.sh ui audit
#   ./scripts/agent_task_router.sh plugins/busbar_inspection repair
#   ./scripts/agent_task_router.sh plugins/animal_detection implement --run --module detector
#   ./scripts/agent_task_router.sh plugins/action_event_monitoring upgrade --dry-run
#   ./scripts/agent_task_router.sh root audit --json
#   ./scripts/agent_task_router.sh plugins/radar implement
#
# 任务类型: implement / repair / audit / upgrade
# 选项:     --run / --dry-run / --json / --module <name> / --root <path>
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 查找 Python 3
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "错误: 未找到 python3 或 python" >&2
    exit 1
fi

# 简单帮助
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
    cat <<'USAGE'
DarkBreaker Agent Task Router

用法:
  ./scripts/agent_task_router.sh <target> <task> [options]

参数:
  target    目标路径: root / ui / plugins/<name>
  task      任务类型: implement / repair / audit / upgrade

选项:
  --run             实际执行脚本
  --dry-run         仅输出计划 (默认)
  --json            JSON 格式输出
  --module <name>   指定目标模块 (传给 targeted tests)
  --root <path>     指定仓库根目录

示例:
  ./scripts/agent_task_router.sh ui audit
  ./scripts/agent_task_router.sh plugins/busbar_inspection repair --run
  ./scripts/agent_task_router.sh plugins/animal_detection implement --module detector
  ./scripts/agent_task_router.sh plugins/radar implement
  ./scripts/agent_task_router.sh root audit --json
USAGE
    exit 0
fi

exec "$PYTHON" "$SCRIPT_DIR/agent_task_router.py" "$@"
