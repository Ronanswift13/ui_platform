#!/bin/bash
# AATF 质量闸门 - 合并前必须全部通过
# 对应 /audit 命令和 AATF 阶段 D
set -e

PLUGIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PLUGIN_DIR"

echo "=== AATF 质量闸门 ==="
echo "插件: indoor_fence"
echo "时间: $(date)"
echo ""

FAILURES=0

# 1. 代码风格
echo "[1/6] 代码风格检查..."
if command -v flake8 &>/dev/null || python3 -m flake8 --version &>/dev/null; then
  if python3 -m flake8 . --max-line-length=100 --exclude='__pycache__,*.pyc,.claude,.agent_skills,venv,.venv' 2>&1; then
    echo "  PASS"
  else
    echo "  FAIL"
    FAILURES=$((FAILURES+1))
  fi
else
  echo "  SKIP (flake8 未安装, 运行 pip install flake8)"
fi
echo ""

# 2. 类型检查
echo "[2/6] 类型检查..."
if command -v mypy &>/dev/null || python3 -m mypy --version &>/dev/null; then
  if python3 -m mypy . --ignore-missing-imports --exclude __pycache__ 2>&1; then
    echo "  PASS"
  else
    echo "  FAIL"
    FAILURES=$((FAILURES+1))
  fi
else
  echo "  SKIP (mypy 未安装, 运行 pip install mypy)"
fi
echo ""

# 3. 单元测试 + 覆盖率
# 使用 .coveragerc 配置排除入口脚本、demo 和遗留模块
# 当前阈值 70%: 硬件适配器代码需要复杂 mock，目标逐步提升到 80%
echo "[3/6] 测试 + 覆盖率 (>=70%)..."
if python3 -m pytest tests/ --cov=. --cov-config=.coveragerc --cov-fail-under=70 -q 2>&1; then
  echo "  PASS"
else
  echo "  FAIL"
  FAILURES=$((FAILURES+1))
fi
echo ""

# 4. 安全扫描
echo "[4/6] 安全扫描..."
if command -v bandit &>/dev/null; then
  if bandit -r . -ll -q --exclude __pycache__,tests,.claude,.agent_skills 2>/dev/null; then
    echo "  PASS"
  else
    echo "  WARN (非阻塞)"
  fi
else
  echo "  SKIP (bandit 未安装)"
fi
echo ""

# 5. 架构合规 (core/ 不得 import adapters/ 或 standalone/)
echo "[5/6] 架构合规检查..."
VIOLATIONS=$(grep -rn "from.*adapters\|import.*adapters\|from.*standalone\|import.*standalone" core/ 2>/dev/null | grep -v __pycache__ || true)
if [ -z "$VIOLATIONS" ]; then
  echo "  PASS"
else
  echo "  FAIL: core/ 依赖了 adapters/ 或 standalone/"
  echo "$VIOLATIONS"
  FAILURES=$((FAILURES+1))
fi
echo ""

# 6. 密钥扫描
echo "[6/6] 密钥扫描..."
SECRETS=$(grep -rn "password\s*=\|secret\s*=\|api_key\s*=\|token\s*=" --include="*.py" . 2>/dev/null | grep -v __pycache__ | grep -v test_ | grep -v "\.md" || true)
if [ -z "$SECRETS" ]; then
  echo "  PASS"
else
  echo "  WARN: 可能存在硬编码密钥"
  echo "$SECRETS"
fi
echo ""

# 总结
echo "================================"
if [ $FAILURES -eq 0 ]; then
  echo "质量闸门: 全部通过"
  exit 0
else
  echo "质量闸门: ${FAILURES} 项失败"
  exit 1
fi
