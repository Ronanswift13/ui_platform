# 破夜绘明激光监测平台 - 问题解决方案总结

## 📋 问题汇总与解决方案

---

## 一、Pylance 类型错误处理

### 问题分析
从截图中看到，`extended_inference_engine.py` 存在 29 个 Pylance 错误，主要是：
- `无法将"dict[str, ndarray...]"类型转换...` - ONNX 推理返回类型不明确
- `无法访问类"SparseTensor/list/dict"的属性"squeeze"` - 类型检查器无法确定实际运行时类型

### 解决建议

**这些是静态类型检查警告，不影响程序运行。**

#### 方案一：快速修复（推荐）
在 VS Code 设置中调整 Pylance 严格程度：
```json
{
    "python.analysis.typeCheckingMode": "basic"
}
```

#### 方案二：代码修复
使用提供的 `fix_type_errors.py` 脚本，或手动修改：

```python
# 修改前
result.predictions = outputs[0].squeeze()

# 修改后
result.predictions = np.asarray(outputs[0]).squeeze()
```

#### 方案三：类型忽略注释
```python
result.predictions = outputs[0].squeeze()  # type: ignore[union-attr]
```

---

## 二、训练结果可用性检验

### 使用 `evaluate_training.py` 工具

```bash
# 评估单个模型
python evaluate_training.py --model models/switch/switch_yolov8s.onnx --type switch

# 评估所有模型
python evaluate_training.py --all

# 生成完整报告
python evaluate_training.py --report
```

### 检验项目

| 检验项 | 说明 | 工具 |
|--------|------|------|
| **推理性能** | 延迟、FPS、吞吐量 | 基准测试模块 |
| **检测精度** | mAP, Recall, Precision | 评估模块 |
| **稳定性** | 24h连续运行、内存泄漏 | 压力测试 |
| **模型加载** | ONNX 格式验证 | onnxruntime |

---

## 三、训练合格标准

根据项目文档，各插件的合格标准如下：

### A组 - 主变巡视 (transformer)
| 指标 | 最低要求 |
|------|----------|
| mAP@0.5 | ≥ 0.75 |
| Recall | ≥ 0.80 |
| Precision | ≥ 0.85 |
| 推理延迟 | ≤ 100ms |
| FPS | ≥ 10 |

### B组 - 开关间隔 (switch)
| 指标 | 最低要求 |
|------|----------|
| mAP@0.5 | ≥ 0.85 |
| 状态识别准确率 | ≥ 95% |
| 逻辑校验误报率 | ≤ 2% |
| 推理延迟 | ≤ 80ms |
| 单帧单ROI CPU | ≤ 300ms |

### C组 - 母线巡视 (busbar)
| 指标 | 最低要求 |
|------|----------|
| pin_missing Recall | ≥ 0.85 |
| pin_missing Precision | ≥ 0.85 |
| crack Recall | ≥ 0.70 |
| crack Precision | ≥ 0.80 |
| 4K图像推理(含切片) | ≤ 800ms (GPU) |

### D组 - 电容器 (capacitor)
| 指标 | 最低要求 |
|------|----------|
| mAP@0.5 | ≥ 0.80 |
| Recall | ≥ 0.85 |
| Precision | ≥ 0.85 |

### E组 - 表计读数 (meter)
| 指标 | 最低要求 |
|------|----------|
| 关键点 PCK@0.1 | ≥ 0.90 |
| OCR 准确率 | ≥ 0.95 |
| 读数误差(满量程%) | ≤ 2% |

### 通用稳定性标准
- 连续运行 **24小时** 不崩溃
- 内存增长 **≤ 100MB**
- 崩溃次数 **= 0**

---

## 四、训练依据

### 数据来源
1. **公开数据集** - 用于预训练基础模型
2. **现场采集数据** - 用于微调适配特定场景
3. **数据增强** - 提升模型泛化能力

### 评估依据
1. **验收数据集** - 独立测试集评估
2. **回放测试** - 使用历史视频/图片验证
3. **现场测试** - 实际变电站环境验证

---

## 五、全电压等级变电站适配方案

### 设计理念
管理员在系统中选择电压等级 → 系统自动匹配对应的模型库和设备配置

### 使用 `voltage_adapter_extended.py`

```python
from platform_core.voltage_adapter_extended import VoltageAdapterManager

# 初始化
manager = VoltageAdapterManager()

# 设置电压等级
manager.set_voltage_level("500kV_AC")  # 或 "220kV"、"35kV"

# 获取模型路径
model_path = manager.get_model_path("switch", "state_detection")
# 返回: models/ehv/500kV/switch/switch_state_500kv.onnx

# 获取设备配置
config = manager.get_equipment_config("switch")
# 返回包含角度参考值、开关类型等的配置

# 获取检测类别
classes = manager.get_detection_classes("busbar")
```

### 命令行使用

```bash
# 设置电压等级
python platform_core/voltage_adapter_extended.py --set 500kV_AC

# 查看当前配置
python platform_core/voltage_adapter_extended.py --show

# 导出配置
python platform_core/voltage_adapter_extended.py --export config_export.yaml
```

### API 集成

```python
from fastapi import FastAPI
from platform_core.voltage_adapter_extended import VoltageAdapterManager
from platform_core.voltage_api_extended import integrate_voltage_routes

app = FastAPI()
adapter = VoltageAdapterManager()
integrate_voltage_routes(app)

# API 端点:
# GET  /api/voltage/current       - 获取当前电压等级
# POST /api/voltage/set           - 设置电压等级
# GET  /api/voltage/config/{type} - 获取设备配置
# GET  /api/voltage/models        - 获取所有模型路径
```

### 220kV vs 500kV 主要差异

| 对比项 | 220kV | 500kV |
|--------|-------|-------|
| 主变容量 | 50-180 MVA | 500-1000 MVA |
| 母线高度 | ~8m | ~15m |
| 相间距 | 4.5m | 9.0m |
| 热成像阈值 | 60/75/85°C | 65/80/95°C |
| 特有检测项 | - | 套管裂纹、GIS位置、间隔棒损坏 |

---

## 六、训练数据获取方案

### 使用 `prepare_training_data.py`

```bash
# 列出所有可用数据集
python prepare_training_data.py --list

# 生成所有下载指南
python prepare_training_data.py --download-all-guides

# 下载 CPLID 数据集 (GitHub)
python prepare_training_data.py --download cplid

# 为开关插件准备500kV数据
python prepare_training_data.py --prepare switch --voltage 500kV

# 格式转换
python prepare_training_data.py --convert voc2coco --input data/raw --output data/coco
```

### 推荐数据集清单

| 数据集 | 图像数 | 格式 | 适用插件 | 获取方式 |
|--------|--------|------|----------|----------|
| CPLID | 848 | VOC | busbar | GitHub 直接下载 |
| 变电站缺陷检测 8000+ | 8307 | VOC/YOLO | all | CSDN 付费 |
| 真实巡检设备检测 | 7500 | YOLO | all | CSDN 付费 |
| 断路器分合闸 | 600 | YOLO | switch | 手动获取 |
| 控制柜面板状态 | 1800 | VOC | switch | 知乎汇总 |
| 指针式仪表 | 500 | VOC | meter | CSDN |
| 红外过热缺陷 | 1900 | VOC | transformer/switch | CSDN |
| 电力设备分割 | 2000 | COCO | all | CSDN |

### 数据集下载资源

**免费资源:**
- GitHub CPLID: https://github.com/InsulatorData/InsulatorDataSet
- 百度飞桨 AI Studio: https://aistudio.baidu.com/
- 公开绝缘子数据集整合: https://github.com/heitorcfelix/public-insulator-datasets

**付费/申请资源:**
- CSDN 数据集汇总: https://blog.csdn.net/DM_zx/article/details/129227962
- 知乎数据集汇总: https://zhuanlan.zhihu.com/p/484933022

---

## 七、下一步工作方向

### 短期 (1-2周)
1. ✅ 修复 Pylance 类型警告
2. 下载并整理训练数据集
3. 使用评估工具验证已训练模型
4. 配置 220kV/500kV 适配系统

### 中期 (1个月)
1. 补充训练数据，提升模型精度
2. 完成所有插件的模型训练
3. 进行系统集成测试
4. 现场验证测试

### 长期优化
1. 引入主动学习，持续改进模型
2. 添加多模态融合能力
3. 实现自适应阈值调整
4. 建立模型版本管理体系

---

## 八、文件清单

本次提供的工具文件：

| 文件 | 功能 |
|------|------|
| `fix_type_errors.py` | Pylance 类型错误修复工具 |
| `evaluate_training.py` | 训练结果评估工具 |
| `voltage_adapter_extended.py` | 全电压等级适配管理器 |
| `prepare_training_data.py` | 训练数据下载与准备工具 |
| `SOLUTION_SUMMARY.md` | 本解决方案总结文档 |

---

**如有问题，请随时联系。**
