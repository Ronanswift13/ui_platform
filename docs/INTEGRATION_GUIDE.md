# 训练代码集成指南

## 📋 概述

本文档指导如何将训练代码集成到"破夜绘明激光监测平台"现有项目结构中。

---

## 🗂️ 文件映射

将以下文件复制到对应位置:

### 核心训练模块

| 源文件 | 目标位置 | 说明 |
|--------|----------|------|
| `ai_models/training/__init__.py` | `ai_models/training/__init__.py` | 包初始化 |
| `ai_models/training/trainer.py` | `ai_models/training/trainer.py` | 跨平台训练器 |
| `ai_models/training/datasets.py` | `ai_models/training/datasets.py` | 数据集定义 |
| `ai_models/training/models.py` | `ai_models/training/models.py` | 模型架构 |
| `ai_models/training/exporters.py` | `ai_models/training/exporters.py` | ONNX导出 |

### 入口和配置

| 源文件 | 目标位置 | 说明 |
|--------|----------|------|
| `train_main.py` | 项目根目录 | 主训练入口 |
| `train_mac.sh` | 项目根目录 | Mac快速启动脚本 |
| `configs/training_config.yaml` | `configs/training_config.yaml` | 训练配置 |
| `README.md` | `docs/TRAINING.md` | 训练文档 |

---

## 📁 完整目录结构

集成后的项目结构:

```
破夜绘明激光监测平台/
├── train_main.py              # ⭐ 新增: 训练入口
├── train_mac.sh               # ⭐ 新增: Mac启动脚本
├── run.py                     # 原有: 平台启动
│
├── ai_models/
│   ├── __init__.py
│   ├── integration.py         # 原有: 集成模块
│   │
│   ├── training/              # ⭐ 新增: 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py         # 跨平台训练器
│   │   ├── datasets.py        # 数据集
│   │   ├── models.py          # 模型定义
│   │   └── exporters.py       # ONNX导出
│   │
│   └── research/              # 原有: 研究模块
│       ├── graph_slam/
│       ├── uncertainty/
│       ├── compression/
│       └── few_shot/
│
├── plugins/                   # 原有: 巡检插件
│   ├── transformer_inspection/
│   ├── switch_inspection/
│   ├── busbar_inspection/
│   ├── capacitor_inspection/
│   └── meter_reading/
│
├── platform_core/             # 原有: 平台核心
│   ├── model_registry_manager.py
│   ├── plugin_initializer.py
│   └── ...
│
├── configs/
│   ├── models_config.yaml     # 原有: 模型配置
│   ├── enhanced_config.yaml   # 原有: 增强配置
│   └── training_config.yaml   # ⭐ 新增: 训练配置
│
├── models/                    # ONNX模型 (训练后生成)
│   ├── transformer/
│   ├── switch/
│   ├── busbar/
│   ├── capacitor/
│   └── meter/
│
├── checkpoints/               # ⭐ 新增: 训练检查点
│
├── data/                      # ⭐ 新增: 训练数据
│   ├── transformer/
│   ├── switch/
│   ├── busbar/
│   ├── capacitor/
│   └── meter/
│
├── docs/
│   ├── TRAINING.md            # ⭐ 新增: 训练文档
│   └── ...
│
└── ui/                        # 原有: 用户界面
```

---

## 🔧 集成步骤

### 步骤1: 创建目录

```bash
cd 破夜绘明激光监测平台

# 创建训练模块目录
mkdir -p ai_models/training

# 创建数据目录
mkdir -p data/{transformer,switch,busbar,capacitor,meter}

# 创建检查点目录
mkdir -p checkpoints/{transformer,switch,busbar,capacitor,meter}

# 创建模型输出目录
mkdir -p models/{transformer,switch,busbar,capacitor,meter,common}
```

### 步骤2: 复制文件

```bash
# 复制训练模块
cp /path/to/training_system/ai_models/training/*.py ai_models/training/

# 复制入口脚本
cp /path/to/training_system/train_main.py .
cp /path/to/training_system/train_mac.sh .
chmod +x train_mac.sh

# 复制配置
cp /path/to/training_system/configs/training_config.yaml configs/

# 复制文档
cp /path/to/training_system/README.md docs/TRAINING.md
```

### 步骤3: 安装依赖

```bash
# Mac
pip install torch torchvision torchaudio
pip install numpy opencv-python onnx onnxruntime psutil pyyaml

# 可选
pip install tensorboard onnxsim
```

### 步骤4: 验证安装

```bash
# 测试训练系统
python train_main.py --mode info

# 使用模拟数据快速测试
python train_main.py --mode all --simulated --epochs 2
```

---

## 🔗 与现有代码集成

### 与model_registry集成

训练完成后，生成的ONNX模型会自动放置到 `models/` 目录，与现有的 `model_registry_manager.py` 配置兼容。

现有配置 `configs/models_config.yaml`:
```yaml
transformer_inspection:
  defect_detector:
    model_id: "transformer_defect_yolov8n"
    model_path: "models/transformer/defect_yolov8n.onnx"  # 训练后生成
    ...
```

### 与插件集成

训练系统生成的模型直接被各插件的 `detector_enhanced.py` 使用:

```python
# plugins/transformer_inspection/detector_enhanced.py
class TransformerDetectorEnhanced:
    MODEL_IDS = {
        "defect": "transformer_defect_yolov8n",  # 对应训练的模型
        "oil": "transformer_oil_unet",
        "silica": "transformer_silica_cnn",
        "thermal": "transformer_thermal",
    }
```

---

## 📊 训练工作流

### 阶段1: 预训练 (公开数据)

```bash
# 1. 准备公开数据集
python train_main.py --mode prepare

# 2. 下载/组织数据到 data/ 目录

# 3. 使用公开数据预训练
python train_main.py --mode all --epochs 50
```

### 阶段2: 微调 (保山站数据)

```bash
# 1. 将保山站数据放入 data/baoshan/

# 2. 加载预训练模型进行微调
python train_main.py --mode all --data-dir data/baoshan --epochs 20
```

### 阶段3: 导出部署

```bash
# 1. 导出ONNX
python train_main.py --mode export

# 2. 验证ONNX
python train_main.py --mode benchmark

# 3. 复制到Windows部署
scp -r models/ windows-pc:/path/to/project/
```

---

## ⚠️ 注意事项

### 1. 模型命名一致性

训练代码中的模型名称必须与 `configs/models_config.yaml` 中的配置一致:

| 训练模型名 | 配置中的model_path |
|-----------|-------------------|
| defect_yolov8n | models/transformer/defect_yolov8n.onnx |
| switch_yolov8s | models/switch/switch_yolov8s.onnx |
| busbar_yolov8m | models/busbar/busbar_yolov8m.onnx |

### 2. 输入尺寸一致性

确保训练时的输入尺寸与部署配置一致:

```yaml
# training_config.yaml
defect_yolov8n:
  input_size: [640, 640]

# models_config.yaml
defect_detector:
  input_size: [640, 640]
```

### 3. 类别一致性

训练时的类别定义必须与检测器的类别映射一致:

```python
# train_main.py
"classes": ["oil_leak", "rust", "damage", ...]

# detector_enhanced.py
DEFECT_CLASSES = {
    0: DefectType.OIL_LEAK,
    1: DefectType.RUST,
    2: DefectType.DAMAGE,
    ...
}
```

---

## 🔄 更新模型流程

当需要更新已部署的模型时:

```bash
# 1. Mac上重新训练
python train_main.py --mode plugin --plugin transformer --epochs 50

# 2. 导出新的ONNX
python train_main.py --mode export --plugin transformer

# 3. 备份旧模型
mv models/transformer models/transformer_backup_$(date +%Y%m%d)

# 4. 部署新模型
scp -r models/transformer windows-pc:/path/to/project/models/

# 5. Windows上验证
python validate_onnx_windows.py

# 6. 重启服务
# (在Windows变电站电脑上)
```

---

## 📞 问题排查

### 训练相关

```bash
# 查看系统信息
python train_main.py --mode info

# 查看数据要求
python train_main.py --mode prepare
```

### 部署相关

```bash
# 验证ONNX模型
python -c "import onnx; model = onnx.load('models/transformer/defect_yolov8n.onnx'); onnx.checker.check_model(model); print('OK')"

# 测试ONNX Runtime
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

---

## 📝 相关文档

- [训练系统详细文档](docs/TRAINING.md)
- [部署指南](docs/DEPLOYMENT_GUIDE.md)
- [模型配置](configs/models_config.yaml)
- [架构说明](ARCHITECTURE.md)
