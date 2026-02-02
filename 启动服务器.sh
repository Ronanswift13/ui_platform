#!/bin/bash

echo "=========================================="
echo "破夜绘明激光监测平台 - 启动服务器"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

echo "📋 检查环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 python3"
    echo "请先安装 Python 3"
    exit 1
fi

echo "✅ Python 版本: $(python3 --version)"
echo ""

echo "🚀 启动服务器..."
echo "访问地址: http://localhost:8080"
echo "数据导入: http://localhost:8080/data-import"
echo "API文档: http://localhost:8080/api/docs"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""
echo "=========================================="
echo ""

python3 run.py --port 8080
