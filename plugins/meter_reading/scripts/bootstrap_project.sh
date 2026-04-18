#!/usr/bin/env bash
# bootstrap_project.sh — 初始化表计读数插件开发环境
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================="
echo " Meter Reading Plugin Bootstrap"
echo "=============================="

# 1. 检查 Python 版本
echo "[1/5] 检查 Python 版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    echo "ERROR: 需要 Python >= 3.10，当前版本: $PYTHON_VERSION"
    exit 1
fi
echo "  Python $PYTHON_VERSION ✓"

# 2. 安装依赖
echo "[2/5] 安装依赖..."
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
echo "  依赖安装完成 ✓"

# 3. 检查 SDK 版本
echo "[3/5] 检查 DarkBreaker SDK..."
python3 -c "
import importlib.metadata
ver = importlib.metadata.version('darkbreaker-sdk')
print(f'  darkbreaker-sdk {ver} ✓')
" 2>/dev/null || echo "  WARNING: darkbreaker-sdk 未安装 (Standalone 模式可跳过)"

# 4. 检查配置文件
echo "[4/5] 检查配置文件..."
if [ -f "$PROJECT_DIR/configs/default.yaml" ]; then
    echo "  configs/default.yaml ✓"
else
    echo "  ERROR: configs/default.yaml 缺失"
    exit 1
fi

# 5. 冒烟测试
echo "[5/5] 冒烟测试..."
cd "$PROJECT_DIR"
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from detector_enhanced import MeterReadingDetectorEnhanced
    print('  detector_enhanced 导入 ✓')
except ImportError as e:
    print(f'  WARNING: detector_enhanced 导入失败: {e}')

try:
    from plugin import MeterReadingPlugin
    print('  plugin 导入 ✓')
except ImportError as e:
    print(f'  WARNING: plugin 导入失败: {e}')
"

echo ""
echo "=============================="
echo " Bootstrap 完成"
echo " 项目路径: $PROJECT_DIR"
echo "=============================="
