#!/bin/bash
# 运行回归测试
# 确保新修改不会破坏现有功能

set -e

echo "=== 回归测试套件 ==="
echo "开始时间: $(date)"
echo ""

# 1. 代码质量检查
echo "1/6 代码风格检查..."
flake8 indoor_fence/ --max-line-length=100 --exclude=__pycache__,*.pyc || {
  echo "✗ 代码风格检查失败"
  exit 1
}
echo "✓ 代码风格检查通过"
echo ""

# 2. 类型检查
echo "2/6 类型检查..."
mypy indoor_fence/ --ignore-missing-imports || {
  echo "✗ 类型检查失败"
  exit 1
}
echo "✓ 类型检查通过"
echo ""

# 3. 单元测试
echo "3/6 单元测试..."
pytest tests/ \
  --ignore=tests/test_integration.py \
  --cov=indoor_fence \
  --cov-fail-under=80 \
  -v || {
  echo "✗ 单元测试失败"
  exit 1
}
echo "✓ 单元测试通过"
echo ""

# 4. 集成测试
echo "4/6 集成测试..."
pytest tests/test_integration.py -v || {
  echo "✗ 集成测试失败"
  exit 1
}
echo "✓ 集成测试通过"
echo ""

# 5. 性能基准测试
echo "5/6 性能基准测试..."
if [ -f "tests/benchmark.py" ]; then
  python tests/benchmark.py || {
    echo "⚠ 性能测试失败（非阻塞）"
  }
else
  echo "⚠ 性能测试脚本不存在，跳过"
fi
echo ""

# 6. 安全扫描
echo "6/6 安全扫描..."
if command -v bandit &> /dev/null; then
  bandit -r indoor_fence/ -ll -q || {
    echo "⚠ 发现安全问题（非阻塞）"
  }
else
  echo "⚠ bandit 未安装，跳过安全扫描"
fi
echo ""

# 生成报告
echo "=== 回归测试完成 ==="
echo "结束时间: $(date)"
echo ""
echo "测试报告:"
echo "  - 覆盖率报告: htmlcov/index.html"
echo "  - 详细日志: pytest.log"
echo ""
echo "✓ 所有关键测试通过，可以合并代码"
