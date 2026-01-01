#!/bin/bash
# =============================================================================
# 破夜绘明激光监测平台 - Mac训练快速启动脚本
# =============================================================================
#
# 使用方法:
#   chmod +x train_mac.sh
#   ./train_mac.sh [模式] [选项]
#
# 模式:
#   all       - 训练所有模型
#   demo      - 使用模拟数据快速演示
#   plugin    - 训练指定插件
#   export    - 仅导出ONNX
#   benchmark - 性能测试
#
# 示例:
#   ./train_mac.sh demo                           # 快速演示
#   ./train_mac.sh all                            # 训练所有模型
#   ./train_mac.sh plugin transformer             # 训练主变插件
#   ./train_mac.sh export                         # 导出ONNX
#
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=============================================="
echo "  破夜绘明激光监测平台 - Mac训练系统"
echo "=============================================="
echo -e "${NC}"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到Python3${NC}"
    exit 1
fi

# 检查PyTorch
python3 -c "import torch" 2>/dev/null || {
    echo -e "${YELLOW}安装PyTorch...${NC}"
    pip3 install torch torchvision torchaudio
}

# 检查其他依赖
echo -e "${BLUE}检查依赖...${NC}"
pip3 install -q numpy opencv-python onnx onnxruntime psutil 2>/dev/null || true

# 检测设备
echo -e "${BLUE}检测硬件...${NC}"
python3 -c "
import torch
import platform

print(f'系统: {platform.system()} ({platform.machine()})')
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ Apple Silicon MPS 可用')
elif torch.cuda.is_available():
    print(f'✅ NVIDIA CUDA 可用: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ 使用CPU训练')
"

# 模式选择
MODE=${1:-"demo"}
PLUGIN=${2:-""}

case $MODE in
    "demo")
        echo -e "${GREEN}运行演示模式 (使用模拟数据)...${NC}"
        python3 train_main.py --mode all --simulated --epochs 2
        ;;
    
    "all")
        echo -e "${GREEN}训练所有模型...${NC}"
        python3 train_main.py --mode all
        ;;
    
    "plugin")
        if [ -z "$PLUGIN" ]; then
            echo -e "${YELLOW}请指定插件名称: transformer, switch, busbar, capacitor, meter${NC}"
            exit 1
        fi
        echo -e "${GREEN}训练插件: $PLUGIN${NC}"
        python3 train_main.py --mode plugin --plugin $PLUGIN
        ;;
    
    "export")
        echo -e "${GREEN}导出ONNX模型...${NC}"
        python3 train_main.py --mode export
        ;;
    
    "benchmark")
        echo -e "${GREEN}性能测试...${NC}"
        python3 train_main.py --mode benchmark
        ;;
    
    "prepare")
        echo -e "${GREEN}准备数据目录...${NC}"
        python3 train_main.py --mode prepare
        ;;
    
    "info")
        echo -e "${GREEN}显示模型信息...${NC}"
        python3 train_main.py --mode info
        ;;
    
    *)
        echo -e "${YELLOW}未知模式: $MODE${NC}"
        echo "可用模式: demo, all, plugin, export, benchmark, prepare, info"
        exit 1
        ;;
esac

echo -e "${GREEN}"
echo "=============================================="
echo "  完成!"
echo "=============================================="
echo -e "${NC}"

# 显示下一步
echo -e "${BLUE}下一步操作:${NC}"
echo "  1. 检查 checkpoints/ 目录查看训练检查点"
echo "  2. 检查 models/ 目录查看ONNX模型"
echo "  3. 将 models/ 目录复制到Windows部署"
echo "  4. 运行 exports/validate_onnx_windows.py 验证"
