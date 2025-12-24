#!/bin/bash
# 一键安装脚本

set -e

echo "=========================================="
echo "输变电激光星芒破夜绘明监测平台 - 安装脚本"
echo "=========================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 创建必要目录
echo "创建目录结构..."
mkdir -p data logs evidence/runs evidence/exports

# 初始化配置
if [ ! -f "configs/platform.yaml" ]; then
    echo "复制默认配置..."
    cp configs/platform.yaml.example configs/platform.yaml 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "安装完成!"
echo ""
echo "启动命令:"
echo "  source venv/bin/activate"
echo "  python run.py"
echo ""
echo "访问地址: http://127.0.0.1:8080"
echo "=========================================="
