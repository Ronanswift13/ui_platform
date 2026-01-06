#!/bin/bash
# ==============================================================================
# AI巡检模型部署脚本
# 输变电站全自动AI巡检方案 - 模型文件部署工具
# ==============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
MODELS_DIR="${PROJECT_ROOT}/models"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   AI巡检模型部署工具${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "项目根目录: ${PROJECT_ROOT}"
echo -e "模型目录: ${MODELS_DIR}"
echo ""

# ==============================================================================
# 1. 创建目录结构
# ==============================================================================
create_directories() {
    echo -e "${YELLOW}[步骤1] 创建模型目录结构...${NC}"
    
    directories=(
        "${MODELS_DIR}/transformer"
        "${MODELS_DIR}/switch"
        "${MODELS_DIR}/busbar"
        "${MODELS_DIR}/capacitor"
        "${MODELS_DIR}/meter"
        "${MODELS_DIR}/common"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo -e "  ${GREEN}✓${NC} 创建目录: $dir"
        else
            echo -e "  ${GREEN}✓${NC} 目录已存在: $dir"
        fi
    done
    
    echo ""
}

# ==============================================================================
# 2. 模型文件清单
# ==============================================================================
declare -A MODELS=(
    # A组 - 主变巡视
    ["transformer/defect_yolov8n.onnx"]="YOLOv8n 主变缺陷检测模型 (640×640)"
    ["transformer/oil_segmentation_unet.onnx"]="U-Net 油位分割模型 (512×512)"
    ["transformer/silica_classifier.onnx"]="CNN 硅胶颜色分类模型 (224×224)"
    ["transformer/thermal_anomaly.onnx"]="热成像异常检测模型 (224×224)"
    
    # B组 - 开关间隔
    ["switch/switch_yolov8s.onnx"]="YOLOv8s 开关状态检测模型 (640×640)"
    ["switch/indicator_ocr.onnx"]="CRNN 指示牌OCR模型 (32×128)"
    
    # C组 - 母线巡视
    ["busbar/busbar_yolov8m.onnx"]="YOLOv8m 母线小目标检测模型 (1280×1280)"
    ["busbar/noise_classifier.onnx"]="环境干扰分类模型 (128×128)"
    
    # D组 - 电容器
    ["capacitor/capacitor_yolov8.onnx"]="YOLOv8 电容器检测模型 (640×640)"
    ["capacitor/rtdetr_intrusion.onnx"]="RT-DETR 入侵检测模型 (640×640)"
    
    # E组 - 表计读数
    ["meter/hrnet_keypoint.onnx"]="HRNet 表计关键点检测模型 (256×256)"
    ["meter/crnn_ocr.onnx"]="CRNN 数字OCR模型 (32×128)"
    ["meter/meter_classifier.onnx"]="表计类型分类模型 (224×224)"
    
    # 通用模型
    ["common/quality_assessor.onnx"]="图像质量评估模型 (224×224)"
    ["common/yolov8n_coco.onnx"]="YOLOv8n 通用检测模型 (640×640)"
)

# ==============================================================================
# 3. 检查模型文件
# ==============================================================================
check_models() {
    echo -e "${YELLOW}[步骤2] 检查模型文件...${NC}"
    echo ""
    
    local available=0
    local missing=0
    local total=${#MODELS[@]}
    
    for model_path in "${!MODELS[@]}"; do
        full_path="${MODELS_DIR}/${model_path}"
        description="${MODELS[$model_path]}"
        
        if [ -f "$full_path" ]; then
            size=$(du -h "$full_path" | cut -f1)
            echo -e "  ${GREEN}✓${NC} ${model_path} (${size})"
            echo -e "    ${description}"
            ((available++))
        else
            echo -e "  ${RED}✗${NC} ${model_path}"
            echo -e "    ${description}"
            ((missing++))
        fi
    done
    
    echo ""
    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "模型统计: ${GREEN}${available}${NC} 可用 / ${RED}${missing}${NC} 缺失 / ${total} 总计"
    echo -e "${BLUE}----------------------------------------${NC}"
    echo ""
    
    if [ $missing -gt 0 ]; then
        echo -e "${YELLOW}警告: 有 ${missing} 个模型文件缺失${NC}"
        echo -e "${YELLOW}增强检测器将回退到传统CV方法${NC}"
    fi
}

# ==============================================================================
# 4. 生成模型下载清单
# ==============================================================================
generate_download_list() {
    echo -e "${YELLOW}[步骤3] 生成模型下载清单...${NC}"
    
    local list_file="${PROJECT_ROOT}/models_download_list.txt"
    
    cat > "$list_file" << 'EOF'
# ==============================================================================
# AI巡检模型下载清单
# 输变电站全自动AI巡检方案
# ==============================================================================
#
# 使用说明:
# 1. 从模型训练团队获取以下模型文件
# 2. 或从公司模型仓库下载
# 3. 将模型文件放置到对应目录
#
# ==============================================================================

# A组 - 主变巡视模型
# ------------------
models/transformer/defect_yolov8n.onnx
  - 用途: 主变外观缺陷检测(油泄漏/锈蚀/破损/异物)
  - 输入: 640×640 RGB
  - 训练数据: 主变外观缺陷数据集
  - 预训练: YOLOv8n COCO

models/transformer/oil_segmentation_unet.onnx
  - 用途: 油枕油位分割
  - 输入: 512×512 RGB
  - 训练数据: 油枕图像分割数据集

models/transformer/silica_classifier.onnx
  - 用途: 呼吸器硅胶颜色分类
  - 输入: 224×224 RGB
  - 类别: blue/pink/white/unknown

models/transformer/thermal_anomaly.onnx
  - 用途: 热成像异常检测
  - 输入: 224×224 单通道/伪彩色

# B组 - 开关间隔模型
# ------------------
models/switch/switch_yolov8s.onnx
  - 用途: 开关分合状态检测
  - 输入: 640×640 RGB
  - 类别: breaker_open/closed, isolator_open/closed等

models/switch/indicator_ocr.onnx
  - 用途: 指示牌文字识别
  - 输入: 32×128 灰度
  - 字符集: 分合开关ONOFF...

# C组 - 母线巡视模型
# ------------------
models/busbar/busbar_yolov8m.onnx
  - 用途: 母线小目标缺陷检测
  - 输入: 1280×1280 RGB (高分辨率)
  - 支持切片推理

models/busbar/noise_classifier.onnx
  - 用途: 环境干扰过滤(鸟/飞虫/阴影)
  - 输入: 128×128 RGB

# D组 - 电容器模型
# ----------------
models/capacitor/capacitor_yolov8.onnx
  - 用途: 电容器单元检测
  - 输入: 640×640 RGB

models/capacitor/rtdetr_intrusion.onnx
  - 用途: 区域入侵检测
  - 输入: 640×640 RGB
  - 类别: person/vehicle/animal

# E组 - 表计读数模型
# ------------------
models/meter/hrnet_keypoint.onnx
  - 用途: 表计关键点检测
  - 输入: 256×256 RGB
  - 关键点: center/pointer_tip/scale_start等

models/meter/crnn_ocr.onnx
  - 用途: 数字表计OCR
  - 输入: 32×128 灰度

models/meter/meter_classifier.onnx
  - 用途: 表计类型分类
  - 输入: 224×224 RGB
  - 类别: sf6_pressure/oil_level等

# 通用模型
# --------
models/common/quality_assessor.onnx
  - 用途: 图像质量评估
  - 输入: 224×224 RGB
  - 输出: clarity/brightness/contrast

models/common/yolov8n_coco.onnx
  - 用途: 通用目标检测
  - 输入: 640×640 RGB
  - 预训练: COCO 80类
EOF

    echo -e "  ${GREEN}✓${NC} 清单已生成: ${list_file}"
    echo ""
}

# ==============================================================================
# 5. 创建占位模型 (用于测试)
# ==============================================================================
create_dummy_models() {
    echo -e "${YELLOW}[可选] 创建测试用占位模型...${NC}"
    
    read -p "是否创建占位模型用于测试? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "  跳过"
        return
    fi
    
    # 创建一个简单的Python脚本来生成占位ONNX模型
    python3 << 'PYTHON_SCRIPT'
import os
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("需要安装onnx库: pip install onnx")
    exit(1)

models_dir = os.environ.get('MODELS_DIR', 'models')

# 模型定义: (路径, 输入尺寸, 输出尺寸)
model_specs = [
    ("transformer/defect_yolov8n.onnx", [1, 3, 640, 640], [1, 84, 8400]),
    ("transformer/oil_segmentation_unet.onnx", [1, 3, 512, 512], [1, 1, 512, 512]),
    ("transformer/silica_classifier.onnx", [1, 3, 224, 224], [1, 4]),
    ("transformer/thermal_anomaly.onnx", [1, 3, 224, 224], [1, 224, 224]),
    ("switch/switch_yolov8s.onnx", [1, 3, 640, 640], [1, 84, 8400]),
    ("switch/indicator_ocr.onnx", [1, 1, 32, 128], [1, 32, 64]),
    ("busbar/busbar_yolov8m.onnx", [1, 3, 1280, 1280], [1, 84, 33600]),
    ("busbar/noise_classifier.onnx", [1, 3, 128, 128], [1, 5]),
    ("capacitor/capacitor_yolov8.onnx", [1, 3, 640, 640], [1, 84, 8400]),
    ("capacitor/rtdetr_intrusion.onnx", [1, 3, 640, 640], [1, 300, 84]),
    ("meter/hrnet_keypoint.onnx", [1, 3, 256, 256], [1, 9, 64, 64]),
    ("meter/crnn_ocr.onnx", [1, 1, 32, 128], [1, 32, 26]),
    ("meter/meter_classifier.onnx", [1, 3, 224, 224], [1, 9]),
    ("common/quality_assessor.onnx", [1, 3, 224, 224], [1, 3]),
    ("common/yolov8n_coco.onnx", [1, 3, 640, 640], [1, 84, 8400]),
]

for rel_path, input_shape, output_shape in model_specs:
    full_path = os.path.join(models_dir, rel_path)
    
    # 创建输入
    X = helper.make_tensor_value_info('images', TensorProto.FLOAT, input_shape)
    
    # 创建输出
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)
    
    # 创建一个简单的Identity节点
    node = helper.make_node('Identity', ['images'], ['output'])
    
    # 创建图
    graph = helper.make_graph([node], 'dummy_model', [X], [Y])
    
    # 创建模型
    model = helper.make_model(graph)
    model.opset_import[0].version = 11
    
    # 保存
    onnx.save(model, full_path)
    print(f"  ✓ 创建: {rel_path}")

print("\n占位模型创建完成!")
PYTHON_SCRIPT

    echo ""
}

# ==============================================================================
# 6. 验证Python环境
# ==============================================================================
check_python_env() {
    echo -e "${YELLOW}[步骤4] 验证Python环境...${NC}"
    
    # 检查Python
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version)
        echo -e "  ${GREEN}✓${NC} Python: ${python_version}"
    else
        echo -e "  ${RED}✗${NC} Python3 未安装"
        return 1
    fi
    
    # 检查必需的包
    packages=("numpy" "opencv-python" "onnxruntime" "pyyaml")
    
    for pkg in "${packages[@]}"; do
        if python3 -c "import ${pkg//-/_}" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} ${pkg}"
        else
            echo -e "  ${RED}✗${NC} ${pkg} (需要安装: pip install ${pkg})"
        fi
    done
    
    # 检查GPU支持
    echo ""
    echo -e "  GPU支持检查:"
    if python3 -c "import onnxruntime; print('CUDA' if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else 'CPU')" 2>/dev/null | grep -q "CUDA"; then
        echo -e "    ${GREEN}✓${NC} CUDA 可用"
    else
        echo -e "    ${YELLOW}!${NC} 仅CPU模式 (安装onnxruntime-gpu以启用GPU)"
    fi
    
    echo ""
}

# ==============================================================================
# 7. 生成启动脚本
# ==============================================================================
generate_startup_script() {
    echo -e "${YELLOW}[步骤5] 生成启动脚本...${NC}"
    
    local startup_script="${PROJECT_ROOT}/start_inspection.py"
    
    cat > "$startup_script" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
AI巡检系统启动脚本
输变电站全自动AI巡检方案

使用方式:
    python start_inspection.py
    
    # 或者在代码中导入
    from start_inspection import init_system, run_inspection
"""

import sys
import logging
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def init_system(config_path: str = "configs/models_config.yaml",
                preload: bool = True,
                enhanced: bool = True) -> bool:
    """
    初始化巡检系统
    
    Args:
        config_path: 模型配置文件路径
        preload: 是否预加载模型
        enhanced: 是否使用增强检测器
        
    Returns:
        bool: 是否成功
    """
    from platform_core.plugin_initializer import initialize_inspection_system
    
    results = initialize_inspection_system(
        config_path=config_path,
        preload_models=preload,
        enable_enhanced=enhanced
    )
    
    return all(results.values())


def run_inspection(plugin_id: str, frame, rois=None, context=None):
    """
    执行巡检
    
    Args:
        plugin_id: 插件ID (transformer_inspection, busbar_inspection等)
        frame: 输入图像 (numpy数组, BGR格式)
        rois: ROI列表
        context: 上下文信息
        
    Returns:
        Dict: 巡检结果
    """
    from platform_core.plugin_initializer import get_plugin_initializer
    
    initializer = get_plugin_initializer()
    return initializer.run_inspection(plugin_id, frame, rois, context)


def get_status():
    """获取系统状态"""
    from platform_core.plugin_initializer import get_plugin_initializer
    return get_plugin_initializer().get_status()


if __name__ == "__main__":
    print("=" * 60)
    print("AI巡检系统启动")
    print("=" * 60)
    
    # 初始化
    if init_system():
        print("\n系统初始化成功!")
        
        # 打印状态
        status = get_status()
        print(f"\n已加载插件: {len(status['plugins'])}")
        for pid, pinfo in status['plugins'].items():
            mode = "增强版" if pinfo['use_enhanced'] else "基础版"
            print(f"  - {pinfo['name']}: {mode}")
    else:
        print("\n系统初始化失败!")
        sys.exit(1)
PYTHON_SCRIPT

    chmod +x "$startup_script"
    echo -e "  ${GREEN}✓${NC} 启动脚本已生成: ${startup_script}"
    echo ""
}

# ==============================================================================
# 主流程
# ==============================================================================
main() {
    create_directories
    check_models
    generate_download_list
    check_python_env
    generate_startup_script
    
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}   部署完成!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "下一步操作:"
    echo -e "  1. 将模型文件放入 ${MODELS_DIR}/ 对应目录"
    echo -e "  2. 运行 ${GREEN}python start_inspection.py${NC} 启动系统"
    echo -e "  3. 或在代码中导入 ${GREEN}from start_inspection import init_system${NC}"
    echo ""
}

# 运行
main "$@"
