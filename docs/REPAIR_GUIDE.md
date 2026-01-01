# 修复指南 - 训练文件为空问题 & 误删文件恢复

## 📋 问题诊断

### 问题1: 训练生成的文件为空

**原因分析:**
1. 模拟数据集生成了但内容为空数组
2. 训练循环中某些依赖导入失败导致跳过
3. 检查点保存路径未正确创建
4. `ai_models/training/__init__.py` 被修改导致导入链断裂

**解决方案:** 使用修复版的 `train_main.py`

### 问题2: 误删文件

**需要恢复的文件:**
- `ui/__init__.py` - UI模块初始化
- `test_plugin_integration.py` - 集成测试
- `plugins/test_all_plugins.py` - 插件测试
- `cross_platform.py` - 跨平台脚本
- `docs/DEPLOYMENT_GUIDE.md` - 部署指南
- `scripts/deploy_models_remote.sh` - 部署脚本

**不需要恢复的文件 (按架构应删除):**
- `plugins/acoustic_monitoring/` - 应在 `ai_models/research/`
- `plugins/gas_detection/` - 同上
- `plugins/hyperspectral_detection/` - 同上
- `plugins/slam_mapping/` - 同上

---

## 🔧 修复步骤

### 步骤1: 恢复误删文件

```bash
cd 破夜绘明激光监测平台

# 恢复 ui/__init__.py
mkdir -p ui
cp restore/ui/__init__.py ui/

# 恢复测试文件
cp restore/test_plugin_integration.py .
cp restore/plugins/test_all_plugins.py plugins/

# 恢复跨平台脚本
cp restore/cross_platform.py .

# 恢复部署文档和脚本
mkdir -p docs scripts
cp restore/docs/DEPLOYMENT_GUIDE.md docs/
cp restore/scripts/deploy_models_remote.sh scripts/
chmod +x scripts/deploy_models_remote.sh
```

### 步骤2: 替换训练脚本

```bash
# 备份原有脚本
mv train_main.py train_main.py.bak

# 使用修复版
cp fixed_training/train_main.py .
```

### 步骤3: 创建必要目录

```bash
# 创建训练输出目录
mkdir -p checkpoints/{transformer,switch,busbar,capacitor,meter}
mkdir -p models/{transformer,switch,busbar,capacitor,meter,common}
mkdir -p logs
```

### 步骤4: 验证环境

```bash
# 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 检查MPS (Mac)
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 检查ONNX
python -c "import onnx; print('ONNX OK')"
python -c "import onnxruntime; print('ONNX Runtime OK')"
```

---

## 🚀 训练步骤

### 演示模式 (快速测试)

```bash
# 使用模拟数据训练3个epoch，验证流程
python train_main.py --mode demo --epochs 3
```

预期输出:
```
✅ PyTorch 2.x.x 已加载
✅ 使用 Apple Silicon MPS 加速
📁 创建目录: checkpoints
📁 创建目录: models
📁 创建目录: logs
...
Epoch 1/3 | Train Loss: 2.3xxx, Acc: xx.xx% | Val Loss: 2.3xxx, Acc: xx.xx%
Epoch 2/3 | Train Loss: 2.2xxx, Acc: xx.xx% | Val Loss: 2.2xxx, Acc: xx.xx%
Epoch 3/3 | Train Loss: 2.1xxx, Acc: xx.xx% | Val Loss: 2.1xxx, Acc: xx.xx%
💾 保存检查点: checkpoints/transformer/silica_cnn_best.pth (xxx KB)
✅ ONNX导出成功: models/transformer/silica_cnn.onnx (xxx KB)
```

### 训练单个插件

```bash
# 训练主变巡视 (A组)
python train_main.py --mode plugin --plugin transformer --epochs 30

# 训练开关间隔 (B组)
python train_main.py --mode plugin --plugin switch --epochs 30

# 训练母线巡视 (C组)
python train_main.py --mode plugin --plugin busbar --epochs 30

# 训练电容器 (D组)
python train_main.py --mode plugin --plugin capacitor --epochs 30

# 训练表计读数 (E组)
python train_main.py --mode plugin --plugin meter --epochs 30
```

### 训练所有模型

```bash
python train_main.py --mode all --epochs 50
```

---

## 📁 输出文件验证

训练完成后，检查以下目录:

### 检查点 (checkpoints/)

```bash
ls -la checkpoints/transformer/
# 应该看到:
# silica_cnn_best.pth (约500KB-2MB)
# silica_cnn_final.pth
# defect_yolov8n_best.pth
# ...
```

### ONNX模型 (models/)

```bash
ls -la models/transformer/
# 应该看到:
# silica_cnn.onnx (约500KB-2MB)
# defect_yolov8n.onnx
# ...
```

### 验证文件大小

```bash
# 检查是否为空文件
find checkpoints -name "*.pth" -size 0
find models -name "*.onnx" -size 0

# 如果上面命令有输出，说明有空文件
# 正常情况下不应该有任何输出
```

---

## 🔄 后续: 部署到Windows

### 1. 复制ONNX模型

```bash
# 方式1: 使用部署脚本
./scripts/deploy_models_remote.sh models user@windows-pc:/path/to/project/

# 方式2: 手动复制
scp -r models/* user@windows-pc:/path/to/project/models/
```

### 2. Windows验证

在Windows上运行:

```python
# validate_models.py
import onnxruntime as ort
import numpy as np
from pathlib import Path

models_dir = Path("models")

for onnx_file in models_dir.rglob("*.onnx"):
    print(f"验证: {onnx_file}")
    
    session = ort.InferenceSession(
        str(onnx_file),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    input_shape = session.get_inputs()[0].shape
    input_shape = [s if isinstance(s, int) else 1 for s in input_shape]
    
    test_input = np.random.randn(*input_shape).astype(np.float32)
    output = session.run(None, {session.get_inputs()[0].name: test_input})
    
    print(f"  ✅ 输出形状: {output[0].shape}")
```

---

## ❓ 常见问题

### Q: 训练时出现 "MPS backend out of memory"

A: 减少batch size
```bash
# 修改train_main.py中的batch_size，或使用更小的模型
```

### Q: ONNX导出失败

A: 检查opset版本兼容性
```python
# 尝试降低opset版本
torch.onnx.export(..., opset_version=14)  # 改为14
```

### Q: Windows推理速度慢

A: 
1. 确认使用了GPU: 检查 `ort.get_available_providers()`
2. 启用TensorRT
3. 使用FP16精度

---

## 📞 联系

如有问题，请查看日志文件 `training_*.log` 获取详细错误信息。
