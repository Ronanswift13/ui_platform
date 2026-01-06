# AI巡检系统部署与集成指南

## 输变电站全自动AI巡检方案 - 深度学习模型集成

---

## 目录

1. [系统架构概述](#1-系统架构概述)
2. [解决方案总览](#2-解决方案总览)
3. [模型文件部署](#3-模型文件部署)
4. [模型注册中心配置](#4-模型注册中心配置)
5. [插件集成方式](#5-插件集成方式)
6. [启动和初始化](#6-启动和初始化)
7. [验证测试](#7-验证测试)
8. [故障排查](#8-故障排查)

---

## 1. 系统架构概述

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        应用层 (Application)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ 主变巡视  │   │ 开关间隔 │   │ 母线巡视  │  │ 电容/表计 │        │
│  │   (A组)  │  │   (B组)  │   │   (C组)  │  │ (D/E组)  │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
│       └─────────────┴──────┬──────┴─────────────┘               │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              插件初始化器 (PluginInitializer)            │   │
│  │    - 加载各插件的增强检测器                              │   │
│  │    - 注入 model_registry 到检测器                        │   │
│  │    - 管理插件生命周期                                    │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           模型注册中心 (ModelRegistryManager)            │   │
│  │    - 加载 models_config.yaml 配置                        │   │
│  │    - 管理所有 ONNX 模型的生命周期                        │   │
│  │    - 提供统一的推理接口                                  │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              推理引擎 (ONNXInferenceEngine)              │   │
│  │    - ONNX Runtime 推理                                   │   │
│  │    - 支持 CPU / CUDA / TensorRT                          │   │
│  │    - 模型预处理和后处理                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        模型层 (Models)                           │
│  models/                                                        │
│  ├── transformer/     # A组模型                                 │
│  │   ├── defect_yolov8n.onnx                                   │
│  │   ├── oil_segmentation_unet.onnx                            │
│  │   ├── silica_classifier.onnx                                │
│  │   └── thermal_anomaly.onnx                                  │
│  ├── switch/          # B组模型                                 │
│  ├── busbar/          # C组模型                                 │
│  ├── capacitor/       # D组模型                                 │
│  ├── meter/           # E组模型                                 │
│  └── common/          # 通用模型                                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| 模型配置 | `configs/models_config.yaml` | 定义所有模型路径、参数、类别 |
| 模型注册中心 | `platform_core/model_registry_manager.py` | 统一管理模型加载、缓存、推理 |
| 插件初始化器 | `platform_core/plugin_initializer.py` | 初始化插件并注入model_registry |
| 增强检测器 | `plugins/*/detector_enhanced.py` | 各插件的深度学习增强版检测器 |

---

## 2. 解决方案总览

针对您反馈的三个问题，解决方案如下：

### 问题1: 模型文件缺失

**解决方案**: 
- 创建 `models/` 目录结构
- 部署训练好的 ONNX 模型文件
- 配置 `models_config.yaml` 指定模型路径

```bash
# 运行部署脚本
bash scripts/deploy_models.sh
```

### 问题2: 回退机制已实现 ✓

所有增强检测器都已实现深度学习优先 + 传统CV回退的双路径机制：

```python
def detect(self, image):
    # 1. 尝试深度学习
    if self._use_deep_learning and self._model_registry:
        result = self._detect_by_deep_learning(image)
        if result:
            return result
    
    # 2. 回退到传统方法
    return self._detect_by_traditional(image)
```

### 问题3: model_registry 配置

**解决方案**:
- 使用 `ModelRegistryManager` 统一管理模型
- 通过 `PluginInitializer` 自动注入到各检测器
- 应用启动时调用初始化函数

```python
from platform_core.plugin_initializer import initialize_inspection_system

# 一行代码初始化所有组件
initialize_inspection_system()
```

---

## 3. 模型文件部署

### 3.1 目录结构

```
项目根目录/
├── models/
│   ├── transformer/          # A组 - 主变巡视
│   │   ├── defect_yolov8n.onnx
│   │   ├── oil_segmentation_unet.onnx
│   │   ├── silica_classifier.onnx
│   │   └── thermal_anomaly.onnx
│   │
│   ├── switch/               # B组 - 开关间隔
│   │   ├── switch_yolov8s.onnx
│   │   └── indicator_ocr.onnx
│   │
│   ├── busbar/               # C组 - 母线巡视
│   │   ├── busbar_yolov8m.onnx
│   │   └── noise_classifier.onnx
│   │
│   ├── capacitor/            # D组 - 电容器
│   │   ├── capacitor_yolov8.onnx
│   │   └── rtdetr_intrusion.onnx
│   │
│   ├── meter/                # E组 - 表计读数
│   │   ├── hrnet_keypoint.onnx
│   │   ├── crnn_ocr.onnx
│   │   └── meter_classifier.onnx
│   │
│   └── common/               # 通用模型
│       ├── quality_assessor.onnx
│       └── yolov8n_coco.onnx
│
├── configs/
│   └── models_config.yaml    # 模型配置文件
│
└── platform_core/
    ├── model_registry_manager.py
    └── plugin_initializer.py
```

### 3.2 模型文件获取

模型文件需要通过以下方式获取：

**方式1: 模型训练团队提供**
- 联系算法团队获取训练好的 ONNX 模型
- 确保模型输入输出格式与配置文件一致

**方式2: 自行训练导出**
```python
# PyTorch 导出 ONNX 示例
import torch

model = YourModel()
model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    opset_version=11,
    input_names=['images'],
    output_names=['output']
)
```

**方式3: 使用预训练模型**
```python
# 使用 ultralytics 导出 YOLOv8
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640)
```

### 3.3 模型文件清单

| 模型 | 路径 | 输入尺寸 | 用途 |
|------|------|----------|------|
| defect_yolov8n | transformer/ | 640×640 | 主变缺陷检测 |
| oil_unet | transformer/ | 512×512 | 油位分割 |
| silica_cnn | transformer/ | 224×224 | 硅胶分类 |
| thermal_anomaly | transformer/ | 224×224 | 热成像异常 |
| switch_yolov8s | switch/ | 640×640 | 开关状态 |
| indicator_ocr | switch/ | 32×128 | 指示牌OCR |
| busbar_yolov8m | busbar/ | 1280×1280 | 母线小目标 |
| noise_classifier | busbar/ | 128×128 | 干扰过滤 |
| capacitor_yolov8 | capacitor/ | 640×640 | 电容器检测 |
| rtdetr_intrusion | capacitor/ | 640×640 | 入侵检测 |
| hrnet_keypoint | meter/ | 256×256 | 关键点检测 |
| crnn_ocr | meter/ | 32×128 | 数字OCR |
| meter_classifier | meter/ | 224×224 | 表计分类 |

---

## 4. 模型注册中心配置

### 4.1 配置文件说明

`configs/models_config.yaml` 定义了所有模型的配置：

```yaml
# 全局配置
global:
  models_base_path: "models"      # 模型根目录
  default_backend: "onnx_cuda"    # 推理后端
  cuda_device_id: 0               # GPU设备ID
  enable_fp16: true               # 半精度加速
  warmup_iterations: 3            # 预热次数

# 各插件模型配置
transformer_inspection:
  defect_detector:
    model_id: "transformer_defect_yolov8n"
    model_path: "models/transformer/defect_yolov8n.onnx"
    model_type: "yolov8"
    input_size: [640, 640]
    classes:
      - "oil_leak"
      - "rust"
      - "damage"
    confidence_threshold: 0.5
    nms_threshold: 0.45
```

### 4.2 推理后端选择

| 后端 | 配置值 | 适用场景 |
|------|--------|----------|
| CPU | `onnx_cpu` | 无GPU环境、调试 |
| CUDA | `onnx_cuda` | 有NVIDIA GPU |
| TensorRT | `tensorrt` | 边缘部署、最高性能 |

### 4.3 修改配置

根据实际部署环境修改配置：

```yaml
# 边缘设备配置示例 (Jetson Xavier NX)
global:
  default_backend: "tensorrt"
  cuda_device_id: 0
  enable_fp16: true
  
# 服务器配置示例
global:
  default_backend: "onnx_cuda"
  cuda_device_id: 0
  enable_fp16: false
```

---

## 5. 插件集成方式

### 5.1 自动集成 (推荐)

使用 `PluginInitializer` 自动完成所有集成：

```python
from platform_core.plugin_initializer import initialize_inspection_system

# 初始化所有插件和模型
results = initialize_inspection_system(
    config_path="configs/models_config.yaml",
    preload_models=True,      # 预加载模型
    enable_enhanced=True      # 使用增强检测器
)

# 检查初始化结果
for plugin_id, success in results.items():
    print(f"{plugin_id}: {'成功' if success else '失败'}")
```

### 5.2 手动集成

如果需要更细粒度的控制：

```python
from platform_core.model_registry_manager import (
    ModelRegistryManager, 
    get_model_registry
)
from plugins.transformer_inspection.detector_enhanced import (
    TransformerDetectorEnhanced
)

# 1. 初始化模型注册中心
manager = ModelRegistryManager("configs/models_config.yaml")
manager.initialize()
registry = manager.get_registry()

# 2. 创建检测器并注入registry
detector = TransformerDetectorEnhanced(
    config={
        "confidence_threshold": 0.5,
        "use_deep_learning": True
    },
    model_registry=registry
)

# 3. 初始化检测器
detector.initialize()

# 4. 执行检测
result = detector.inspect(frame, rois, context)
```

### 5.3 检测器配置选项

各检测器支持的配置选项：

```python
# 主变检测器配置
transformer_config = {
    "confidence_threshold": 0.5,
    "use_deep_learning": True,
    "defect_confidence_threshold": 0.5,
    "oil_level_threshold": 0.5,
    "thermal_anomaly_threshold": 0.7,
}

# 母线检测器配置
busbar_config = {
    "use_deep_learning": True,
    "use_slicing": True,          # 切片推理
    "tile_size": 1280,            # 切片大小
    "overlap": 128,               # 重叠像素
    "small_target_threshold": 0.4,
}

# 表计检测器配置
meter_config = {
    "use_deep_learning": True,
    "max_retry": 3,               # 最大重试次数
    "perspective_correction": True,
}
```

---

## 6. 启动和初始化

### 6.1 完整启动流程

```python
#!/usr/bin/env python3
"""
AI巡检系统启动示例
"""

import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # 1. 初始化系统
    from platform_core.plugin_initializer import (
        initialize_inspection_system,
        get_plugin_initializer
    )
    
    results = initialize_inspection_system(
        config_path="configs/models_config.yaml",
        preload_models=True,
        enable_enhanced=True
    )
    
    if not all(results.values()):
        logging.error("部分组件初始化失败")
        return
    
    # 2. 获取初始化器
    initializer = get_plugin_initializer()
    
    # 3. 执行巡检
    import cv2
    import numpy as np
    
    # 加载测试图像
    frame = cv2.imread("test_image.jpg")
    
    # 定义ROI
    rois = [
        {"id": "roi_001", "type": "oil_gauge", "bbox": [100, 100, 200, 200]}
    ]
    
    # 执行主变巡检
    result = initializer.run_inspection(
        plugin_id="transformer_inspection",
        frame=frame,
        rois=rois,
        context={"task_id": "task_001"}
    )
    
    print(f"巡检结果: {result}")

if __name__ == "__main__":
    main()
```

### 6.2 FastAPI 集成示例

```python
from fastapi import FastAPI, UploadFile, File
from platform_core.plugin_initializer import (
    initialize_inspection_system,
    get_plugin_initializer
)
import cv2
import numpy as np

app = FastAPI()

@app.on_event("startup")
async def startup():
    """应用启动时初始化"""
    initialize_inspection_system()

@app.post("/api/inspect/{plugin_id}")
async def inspect(plugin_id: str, file: UploadFile = File(...)):
    """执行巡检"""
    # 读取图像
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 执行巡检
    initializer = get_plugin_initializer()
    result = initializer.run_inspection(plugin_id, frame)
    
    return result

@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    initializer = get_plugin_initializer()
    return initializer.get_status()
```

---

## 7. 验证测试

### 7.1 模型文件验证

```python
from platform_core.model_registry_manager import get_model_registry_manager

manager = get_model_registry_manager("configs/models_config.yaml")
manager.initialize()

# 检查所有模型状态
status = manager.check_models()
for model_id, info in status.items():
    icon = "✓" if info["exists"] else "✗"
    print(f"[{icon}] {model_id}: {info['path']}")
    if info["exists"]:
        print(f"    大小: {info['size_mb']} MB")
        print(f"    已加载: {info['loaded']}")
```

### 7.2 推理测试

```python
import cv2
import numpy as np
from platform_core.model_registry_manager import get_model_registry_manager

# 初始化
manager = get_model_registry_manager()
manager.initialize()
registry = manager.get_registry()

# 加载测试图像
image = cv2.imread("test.jpg")

# 测试推理
result = registry.infer("transformer_defect_yolov8n", image)

if result.success:
    print(f"推理成功!")
    print(f"  耗时: {result.inference_time_ms:.1f} ms")
    print(f"  输出形状: {result.metadata['output_shapes']}")
else:
    print(f"推理失败: {result.error_message}")
```

### 7.3 端到端测试

```python
from platform_core.plugin_initializer import (
    initialize_inspection_system,
    get_plugin_initializer
)
import cv2

# 初始化
initialize_inspection_system()
initializer = get_plugin_initializer()

# 测试各插件
test_plugins = [
    "transformer_inspection",
    "busbar_inspection", 
    "capacitor_inspection",
    "meter_reading"
]

image = cv2.imread("test.jpg")

for plugin_id in test_plugins:
    result = initializer.run_inspection(plugin_id, image)
    status = "✓" if result.get("success") else "✗"
    mode = "增强版" if result.get("use_enhanced") else "基础版"
    time_ms = result.get("processing_time_ms", 0)
    print(f"[{status}] {plugin_id}: {mode} ({time_ms:.0f}ms)")
```

---

## 8. 故障排查

### 8.1 常见问题

**问题: 模型加载失败**
```
错误: 模型文件不存在: models/transformer/defect_yolov8n.onnx
```

**解决**: 
- 检查模型文件是否在正确位置
- 检查 `models_config.yaml` 中的路径配置
- 运行 `bash scripts/deploy_models.sh` 创建目录结构

---

**问题: CUDA 不可用**
```
警告: 仅CPU模式 (安装onnxruntime-gpu以启用GPU)
```

**解决**:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

**问题: 增强检测器回退到传统方法**
```
日志: 深度学习推理失败，回退到传统方法
```

**解决**:
- 检查 `model_registry` 是否正确注入
- 检查模型文件是否存在
- 查看详细错误日志

---

### 8.2 日志级别配置

```python
import logging

# 调试模式 - 显示所有日志
logging.getLogger("platform_core").setLevel(logging.DEBUG)

# 生产模式 - 仅显示警告和错误
logging.getLogger("platform_core").setLevel(logging.WARNING)
```

### 8.3 性能监控

```python
# 获取模型统计信息
manager = get_model_registry_manager()
stats = manager.get_stats()

for model_id, model_stats in stats.get("models", {}).items():
    print(f"{model_id}:")
    print(f"  推理次数: {model_stats['inference_count']}")
    print(f"  平均耗时: {model_stats['avg_inference_time_ms']:.1f}ms")
```

---

## 附录: 文件清单

| 文件 | 说明 |
|------|------|
| `configs/models_config.yaml` | 模型配置文件 |
| `platform_core/model_registry_manager.py` | 模型注册中心管理器 |
| `platform_core/plugin_initializer.py` | 插件初始化器 |
| `scripts/deploy_models.sh` | 模型部署脚本 |
| `plugins/*/detector_enhanced.py` | 各插件增强检测器 |

---

*文档版本: 1.0*  
*更新日期: 2025-12-28*
