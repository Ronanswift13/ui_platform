#!/bin/bash
# =============================================================================
# 模型部署脚本
# 破夜绘明激光监测平台
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}"
echo "=============================================="
echo "  破夜绘明激光监测平台 - 模型部署"
echo "=============================================="
echo -e "${NC}"

# 检查参数
if [ "$#" -lt 2 ]; then
    echo "使用方法: $0 <源目录> <目标主机>"
    echo "示例: $0 ./models user@windows-pc:/path/to/project/models"
    exit 1
fi

SRC_DIR=$1
DEST=$2

# 检查源目录
if [ ! -d "$SRC_DIR" ]; then
    echo -e "${RED}错误: 源目录不存在: $SRC_DIR${NC}"
    exit 1
fi

# 统计ONNX文件
ONNX_COUNT=$(find "$SRC_DIR" -name "*.onnx" | wc -l)
echo -e "${YELLOW}找到 $ONNX_COUNT 个ONNX模型${NC}"

# 列出模型
echo -e "\n模型列表:"
find "$SRC_DIR" -name "*.onnx" -exec ls -lh {} \;

# 确认
echo -e "\n${YELLOW}即将部署到: $DEST${NC}"
read -p "确认部署? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 部署
echo -e "\n${GREEN}开始部署...${NC}"

# 使用rsync或scp
if command -v rsync &> /dev/null; then
    rsync -avz --progress "$SRC_DIR/" "$DEST/"
else
    scp -r "$SRC_DIR/"* "$DEST/"
fi

echo -e "\n${GREEN}=============================================="
echo "  部署完成!"
echo "=============================================="
echo -e "${NC}"

echo -e "下一步:"
echo "  1. 在目标机器上运行验证脚本"
echo "  2. 检查模型加载是否正常"
echo "  3. 运行性能测试"
