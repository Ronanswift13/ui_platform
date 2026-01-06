# 部署指南

## 概述

本指南说明如何将训练好的模型部署到Windows变电站工程电脑。

---

## 1. 模型文件结构

### 1.1 目录结构

```
models/
├── transformer/              # A组 - 主变巡视
│   ├── defect_yolov8n.onnx
│   ├── oil_segmentation_unet.onnx
│   ├── silica_classifier.onnx
│   └── thermal_anomaly.onnx
│
├── switch/                   # B组 - 开关间隔
│   ├── switch_yolov8s.onnx
│   └── indicator_ocr.onnx
│
├── busbar/                   # C组 - 母线巡视
│   ├── busbar_yolov8m.onnx
│   └── noise_classifier.onnx
│
├── capacitor/                # D组 - 电容器
│   ├── capacitor_yolov8.onnx
│   └── rtdetr_intrusion.onnx
│
├── meter/                    # E组 - 表计读数
│   ├── hrnet_keypoint.onnx
│   ├── crnn_ocr.onnx
│   └── meter_classifier.onnx
│
└── common/                   # 通用模型
    ├── quality_assessor.onnx
    └── yolov8n_coco.onnx
```

---

## 2. Windows环境配置

### 2.1 安装Python环境

```bash
# 安装Python 3.10+
# 推荐使用Anaconda

# 创建虚拟环境
conda create -n ai_inspect python=3.10
conda activate ai_inspect
```

### 2.2 安装ONNX Runtime GPU

```bash
# CUDA 11.x
pip install onnxruntime-gpu==1.16.0

# 或 CUDA 12.x
pip install onnxruntime-gpu==1.17.0
```

### 2.3 安装其他依赖

```bash
pip install numpy opencv-python pyyaml
```

---

## 3. 模型验证

### 3.1 验证脚本

```python
import onnxruntime as ort
import numpy as np

# 检查可用提供者
print(ort.get_available_providers())
# 预期输出: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# 加载模型
session = ort.InferenceSession(
    "models/transformer/defect_yolov8n.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 测试推理
test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
output = session.run(None, {'input': test_input})
print(f"输出形状: {output[0].shape}")
```

### 3.2 性能测试

```python
import time

# 预热
for _ in range(10):
    session.run(None, {'input': test_input})

# 测试
times = []
for _ in range(100):
    start = time.perf_counter()
    session.run(None, {'input': test_input})
    times.append((time.perf_counter() - start) * 1000)

print(f"平均推理时间: {np.mean(times):.2f} ms")
print(f"FPS: {1000 / np.mean(times):.1f}")
```

---

## 4. 集成到平台

### 4.1 修改配置文件

编辑 `configs/models_config.yaml`:

```yaml
global:
  models_base_path: "models"
  default_backend: "onnx_cuda"  # Windows使用CUDA
  cuda_device_id: 0
  enable_fp16: true
```

### 4.2 启动服务

```bash
python run.py --debug
```

访问: http://127.0.0.1:8080

---

## 5. 故障排除

### 5.1 CUDA不可用

检查:
- NVIDIA驱动版本
- CUDA Toolkit版本
- cuDNN版本

```bash
nvidia-smi
nvcc --version
```

### 5.2 模型加载失败

检查:
- ONNX文件是否完整
- opset版本兼容性

```python
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
```

### 5.3 推理速度慢

优化方法:
- 启用TensorRT
- 使用FP16精度
- 增加batch size

---

## 6. 联系支持

如遇问题，请联系破夜绘明团队。
