#!/bin/bash
# 根因分析素材收集
# 配合 prompts/root_cause_prompt.md 和 /repair 命令使用

PLUGIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PLUGIN_DIR"

echo "=== 根因分析素材收集 ==="
echo "时间: $(date)"
echo ""

# 1. 最近错误日志
echo "--- 最近错误日志 ---"
if [ -d "logs" ]; then
  for logfile in logs/indoor_fence/*.log logs/indoor_fence/*.jsonl logs/*.log; do
    if [ -f "$logfile" ]; then
      ERRORS=$(grep -h "ERROR\|CRITICAL\|FALLBACK\|error\|exception" "$logfile" 2>/dev/null | tail -20)
      if [ -n "$ERRORS" ]; then
        echo "[$logfile]"
        echo "$ERRORS"
        echo ""
      fi
    fi
  done
else
  echo "(logs/ 目录不存在)"
fi
echo ""

# 2. 降级事件统计
echo "--- 降级事件统计 ---"
if [ -d "logs" ]; then
  FALLBACKS=$(grep -rh "FALLBACK\|fallback" logs/ 2>/dev/null | sort | uniq -c | sort -rn | head -10)
  if [ -n "$FALLBACKS" ]; then
    echo "$FALLBACKS"
  else
    echo "(无降级事件)"
  fi
else
  echo "(logs/ 目录不存在)"
fi
echo ""

# 3. 最近测试失败
echo "--- 最近测试失败 ---"
pytest tests/ -q --tb=line 2>&1 | grep "FAILED\|ERROR" || echo "(无测试失败)"
echo ""

# 4. Git 最近变更
echo "--- Git 最近变更 ---"
git log --oneline -10 2>/dev/null || echo "(非 git 仓库)"
echo ""

# 5. 最近修改的文件
echo "--- 最近修改的文件 ---"
git diff --name-only 2>/dev/null | head -20 || echo "(无未提交修改)"
echo ""

echo "=== 收集完成 ==="
echo "请将以上信息用于根因分析 (参考 prompts/root_cause_prompt.md)"
