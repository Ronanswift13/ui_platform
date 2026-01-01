# 破夜绘明激光监测平台 - AI模型训练系统

## 📋 概述

本训练系统专为变电站全自动AI巡检设计，支持:
- **Mac M系列芯片 (MPS加速)** 开发训练
- **Windows (CUDA/TensorRT)** 部署推理
- **自动导出ONNX格式** 实现跨平台部署

### 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    训练流程图                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  公开数据集   │───▶│   预训练     │───▶│  基础模型    │       │
│  │ (500kV变电站)│    │  (Mac MPS)   │    │   (.pth)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                │                │
│                                                ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ 保山站数据    │───▶│   微调       │───▶│  最终模型    │       │
│  │ (多模态采集) │    │  (迁移学习)  │    │   (.pth)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                │                │
│                                                ▼                │
│                                          ┌──────────────┐       │
│                                          │  ONNX导出    │       │
│                                          │   (.onnx)    │       │
│                                          └──────────────┘       │
│                                                │                │
│                          ┌─────────────────────┼─────────────┐  │
│                          ▼                     ▼             ▼  │
│                    ┌──────────┐         ┌──────────┐  ┌───────┐│
│                    │ Windows  │         │  Linux   │  │  Mac  ││
│                    │ (部署)   │         │  (云端)  │  │ (测试)││
│                    └──────────┘         └──────────┘  └───────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ 目录结构

```
训练系统/
├── train_main.py              # 主训练入口
├── train_mac.sh               # Mac快速启动脚本
├── README.md                  # 本文档
│
├── ai_models/
│   └── training/              # 训练核心模块
│       ├── __init__.py        # 包初始化
│       ├── trainer.py         # 跨平台训练器
│       ├── datasets.py        # 数据集定义
│       ├── models.py          # 模型架构
│       └── exporters.py       # ONNX导出
│
├── configs/
│   └── training_config.yaml   # 训练配置文件
│
├── data/                      # 数据目录
│   ├── transformer/           # A组数据
│   ├── switch/                # B组数据
│   ├── busbar/                # C组数据
│   ├── capacitor/             # D组数据
│   └── meter/                 # E组数据
│
├── checkpoints/               # 训练检查点
│   ├── transformer/
│   ├── switch/
│   ├── busbar/
│   ├── capacitor/
│   └── meter/
│
├── models/                    # ONNX模型输出
│   ├── transformer/
│   │   ├── defect_yolov8n.onnx
│   │   ├── oil_unet.onnx
│   │   ├── silica_cnn.onnx
│   │   └── thermal_anomaly.onnx
│   ├── switch/
│   │   ├── switch_yolov8s.onnx
│   │   └── indicator_ocr.onnx
│   ├── busbar/
│   │   ├── busbar_yolov8m.onnx
│   │   └── noise_classifier.onnx
│   ├── capacitor/
│   │   ├── capacitor_yolov8.onnx
│   │   └── rtdetr_intrusion.onnx
│   └── meter/
│       ├── hrnet_keypoint.onnx
│       ├── crnn_ocr.onnx
│       └── meter_classifier.onnx
│
├── exports/                   # 导出工具
│   └── validate_onnx_windows.py
│
└── logs/                      # 训练日志
```

---

## 🚀 快速开始

### 1. 环境准备

#### Mac环境 (开发训练)

```bash
# 安装PyTorch (Apple Silicon)
pip install torch torchvision torchaudio

# 安装依赖
pip install numpy opencv-python onnx onnxruntime psutil pyyaml

# 可选: TensorBoard
pip install tensorboard
```

#### Windows环境 (部署推理)

```bash
# 安装ONNX Runtime GPU版
pip install onnxruntime-gpu

# 可选: TensorRT支持
pip install tensorrt
```

### 2. 数据准备

将数据按以下格式组织:

```
data/
├── transformer/
│   ├── defect/                # 缺陷检测 (COCO格式)
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── annotations/
│   │       ├── train.json
│   │       └── val.json
│   ├── silica/                # 硅胶分类 (目录格式)
│   │   ├── train/
│   │   │   ├── blue/
│   │   │   ├── pink/
│   │   │   └── white/
│   │   └── val/
│   └── thermal/               # 热成像分类
│       ├── train/
│       └── val/
...
```

### 3. 训练模型

#### 使用脚本 (推荐)

```bash
# 演示模式 (使用模拟数据)
./train_mac.sh demo

# 训练所有模型
./train_mac.sh all

# 训练单个插件
./train_mac.sh plugin transformer

# 导出ONNX
./train_mac.sh export
```

#### 使用Python

```bash
# 使用模拟数据快速测试
python train_main.py --mode all --simulated --epochs 5

# 训练主变巡视模型
python train_main.py --mode plugin --plugin transformer --epochs 50

# 训练单个模型
python train_main.py --mode model --plugin transformer --model defect_yolov8n

# 仅导出ONNX
python train_main.py --mode export

# 性能测试
python train_main.py --mode benchmark
```

### 4. 部署到Windows

```bash
# 1. 复制ONNX模型到Windows
scp -r models/ user@windows-pc:/path/to/project/

# 2. 在Windows上运行验证脚本
python exports/validate_onnx_windows.py
```

---

## 📊 模型列表

### A组 - 主变巡视

| 模型 | 类型 | 输入尺寸 | 用途 |
|------|------|----------|------|
| defect_yolov8n | 目标检测 | 640×640 | 外观缺陷检测 |
| oil_unet | 语义分割 | 512×512 | 油位分割 |
| silica_cnn | 分类 | 224×224 | 硅胶颜色分类 |
| thermal_anomaly | 分类 | 224×224 | 热成像异常 |

### B组 - 开关间隔

| 模型 | 类型 | 输入尺寸 | 用途 |
|------|------|----------|------|
| switch_yolov8s | 目标检测 | 640×640 | 开关状态检测 |
| indicator_ocr | OCR | 32×128 | 指示牌识别 |

### C组 - 母线巡视

| 模型 | 类型 | 输入尺寸 | 用途 |
|------|------|----------|------|
| busbar_yolov8m | 目标检测 | 1280×1280 | 小目标检测 |
| noise_classifier | 分类 | 128×128 | 干扰过滤 |

### D组 - 电容器

| 模型 | 类型 | 输入尺寸 | 用途 |
|------|------|----------|------|
| capacitor_yolov8 | 目标检测 | 640×640 | 结构检测 |
| rtdetr_intrusion | 目标检测 | 640×640 | 入侵检测 |

### E组 - 表计读数

| 模型 | 类型 | 输入尺寸 | 用途 |
|------|------|----------|------|
| hrnet_keypoint | 关键点 | 256×256 | 表盘关键点 |
| crnn_ocr | OCR | 32×128 | 数字识别 |
| meter_classifier | 分类 | 224×224 | 表计分类 |

---

## ⚙️ 配置说明

### 训练配置 (configs/training_config.yaml)

```yaml
global:
  default_epochs: 50          # 默认训练轮数
  default_batch_size: 16      # 默认批大小
  default_learning_rate: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  early_stopping: true
  patience: 10

mac_config:
  device: "mps"               # Apple Silicon加速
  precision: "float32"        # MPS使用FP32

windows_config:
  device: "cuda"
  precision: "float16"        # 使用FP16加速
  use_tensorrt: true
```

### 修改模型配置

编辑 `train_main.py` 中的 `PLUGIN_MODEL_CONFIGS`:

```python
"transformer": {
    "models": [
        {
            "name": "defect_yolov8n",
            "type": "detection",
            "input_size": (640, 640),
            "num_classes": 6,
            "epochs": 50,        # 修改训练轮数
            "batch_size": 16,    # 修改批大小
        },
        ...
    ]
}
```

---

## 🔧 API使用

### Python API

```python
from ai_models.training import (
    TrainingPipeline,
    CrossPlatformTrainer,
    TrainingConfig,
    create_dataloader,
    export_to_onnx,
    verify_onnx_model
)

# 方式1: 使用Pipeline
pipeline = TrainingPipeline()
pipeline.train_all(epochs=50)
pipeline.export_all_onnx()

# 方式2: 自定义训练
from ai_models.training.models import create_model

model = create_model(
    model_type="detection",
    model_name="defect_yolov8n",
    input_size=(640, 640),
    plugin_name="transformer",
    num_classes=6
)

config = TrainingConfig(
    model_name="transformer_defect",
    epochs=50,
    batch_size=16
)

trainer = CrossPlatformTrainer(model, config)
trainer.train(train_loader, val_loader)

# 导出ONNX
export_to_onnx(model, (3, 640, 640), "model.onnx")

# 验证
results = verify_onnx_model("model.onnx")
print(f"状态: {results['status']}")
```

### 命令行参数

```bash
python train_main.py [OPTIONS]

选项:
  --mode {all,plugin,model,export,benchmark,prepare,info}
                        运行模式
  --plugin {transformer,switch,busbar,capacitor,meter}
                        指定插件
  --model MODEL         指定模型名称
  --epochs N            训练轮数
  --batch-size N        批大小
  --simulated           使用模拟数据
  --data-dir PATH       数据目录
  --gpu                 强制使用GPU
  --cpu                 强制使用CPU
```

---

## 📈 训练监控

### TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir checkpoints/logs/tensorboard

# 浏览器访问
http://localhost:6006
```

### 训练日志

日志文件位于项目根目录:
```
training_20250101_120000.log
```

### 检查点

训练过程中自动保存:
```
checkpoints/
├── transformer/
│   ├── defect_yolov8n_best.pth   # 最佳模型
│   └── defect_yolov8n_final.pth  # 最终模型
```

---

## 🔄 跨平台部署流程

### 完整流程

```
Mac (开发)                    Windows (部署)
─────────────                 ──────────────
    │                              │
    ▼                              │
┌─────────────┐                    │
│  数据准备    │                    │
└──────┬──────┘                    │
       │                           │
       ▼                           │
┌─────────────┐                    │
│  模型训练    │                    │
│  (MPS加速)  │                    │
└──────┬──────┘                    │
       │                           │
       ▼                           │
┌─────────────┐                    │
│  导出ONNX   │                    │
└──────┬──────┘                    │
       │                           │
       │    ┌─────────────────┐    │
       └───▶│  复制ONNX文件    │───▶│
            └─────────────────┘    │
                                   ▼
                            ┌─────────────┐
                            │  验证推理    │
                            │ (CUDA加速)  │
                            └──────┬──────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │  集成部署    │
                            └─────────────┘
```

### 验证脚本

```python
# Windows上运行
python exports/validate_onnx_windows.py
```

输出示例:
```
验证模型: models/transformer/defect_yolov8n.onnx
  使用提供者: ['CUDAExecutionProvider', 'CPUExecutionProvider']
  ✅ 推理成功
  输入形状: [1, 3, 640, 640]
  输出形状: [(1, 6, 8400)]
  推理时间: 5.23 ms
  FPS: 191.2
```

---

## ❓ 常见问题

### 1. Mac MPS内存不足

```bash
# 减少batch size
python train_main.py --mode plugin --plugin busbar --batch-size 4
```

### 2. ONNX导出失败

检查:
- PyTorch版本 >= 2.0
- opset版本设置为17
- 模型中没有不支持的操作

### 3. Windows推理速度慢

确保:
- 安装了onnxruntime-gpu
- CUDA和cuDNN版本匹配
- 启用了TensorRT (如果可用)

### 4. 数据格式问题

参考 `train_main.py --mode prepare` 查看数据格式要求

---

## 📞 联系方式

如有问题，请联系破夜绘明团队。

---

## 📝 更新日志

### v1.0.0 (2025-01)
- 初始版本
- 支持5组插件共13个模型
- Mac MPS + Windows CUDA跨平台支持
- 自动ONNX导出和验证
