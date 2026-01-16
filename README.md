# 输变电激光星芒破夜绘明监测平台

高智能化输变电站自主巡视与监测系统 V3.0

## 项目概述

本平台采用**分层 + 插件化 + 微服务**架构，将复杂系统拆解为高内聚、低耦合的独立功能模块，支持各博士生算法以"插件"方式直接导入集成。

### V3.0 新特性 (2025年更新)

- **深度学习增强**: 集成 YOLOv8-ViT、GL-TransLSTM、CNN-LSTM 等先进模型
- **异步架构**: 全面支持异步任务处理，提升并发性能
- **统一仪表盘**: 综合监控页面，按设备/区域展示实时状态
- **多模态融合**: 可见光+热成像融合检测能力
- **智能训练管道**: COCO/YOLO格式转换、自动标注、主动学习

### 核心特性

- **模块化**: 多业务模块独立开发、独立部署
- **标准化**: 统一输入输出接口（JSON）、统一文件结构
- **可调度**: 基于任务引擎的算法动态调用
- **可追溯**: 完整证据链（原图/ROI/结果/置信度/时间戳/模型版本）
- **可回放**: 确定性回放，结果可复现
- **深度学习**: 内置多种深度学习模型，支持GPU加速

## 快速开始

### 环境要求

- Python 3.10 或 3.11
- 支持 Windows / macOS / Linux
- 推荐 CUDA 11.8+ (GPU加速)

### 安装

```bash
# 克隆项目
git clone <https://github.com/Ronanswift13/ui_platform.git>
cd 破夜绘明激光监测平台

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装深度学习依赖 (可选)
pip install torch torchvision torchaudio
pip install ultralytics  # YOLOv8
```

### 启动平台

```bash
# 启动完整平台 (UI + API)
python run.py

# 仅启动API服务
python run.py --api --port 8000

# 调试模式
python run.py --debug --reload
```

访问地址: http://127.0.0.1:8080

## 项目结构

```
破夜绘明激光监测平台/
├── apps/                   # 应用入口
├── platform_core/          # 平台核心
│   ├── schema/             # 数据模型
│   ├── plugin_manager/     # 插件管理
│   │   ├── base.py         # 基础插件类
│   │   ├── enhanced_base.py # 增强异步插件基类 [V3.0]
│   │   └── dl_integration.py # 深度学习集成层 [V3.0]
│   ├── scheduler/          # 任务调度
│   ├── evidence/           # 证据链
│   ├── replay/             # 回放功能
│   ├── device_adapter/     # 设备适配
│   └── logging/            # 统一日志
├── ai_models/              # AI模型 [V3.0]
│   ├── deep_learning/      # 深度学习模型
│   │   ├── yolov8_vit.py   # YOLOv8-ViT目标检测
│   │   ├── gl_translstm.py # GL-TransLSTM气体预测
│   │   ├── acoustic_cnn_lstm.py # 声学CNN-LSTM
│   │   ├── deep_sort_tracker.py # DeepSORT跟踪
│   │   └── keypoint_detector.py # 关键点检测
│   └── training/           # 训练管道
│       └── data_pipeline.py # 数据处理管道
├── plugins/                # 插件目录
│   ├── transformer_inspection/  # A组: 主变巡视
│   ├── switch_inspection/       # B组: 开关间隔
│   ├── busbar_inspection/       # C组: 母线巡视
│   ├── capacitor_inspection/    # D组: 电容器
│   ├── meter_reading/           # E组: 表计读数
│   ├── indoor_fence/            # 室内电子围栏 [增强]
│   ├── gas_detection/           # 气体检测
│   ├── acoustic_monitoring/     # 声学监测
│   └── bird_monitoring/         # 鸟类监测
├── ui/                     # Web UI
│   ├── static/
│   │   ├── js/
│   │   │   ├── dashboard.js
│   │   │   └── unified_dashboard.js # 统一仪表盘 [V3.0]
│   │   └── css/
│   │       └── unified_dashboard.css
│   └── templates/
│       └── pages/
│           └── unified_dashboard.html # 统一仪表盘页面 [V3.0]
├── configs/                # 配置文件
├── evidence/               # 证据存储
├── logs/                   # 日志
├── tests/                  # 测试
└── docs/                   # 文档
```

## 功能模块

### 室外巡视模块

| 模块 | 负责组 | 状态 | V3.0 增强 |
|------|--------|------|-----------|
| 主变自主巡视 | A组 | ✅ 已集成 | YOLOv8-ViT 深度检测, 热成像融合 |
| 开关间隔巡视 | B组 | ✅ 已集成 | 多类型开关统一检测 |
| 母线自主巡视 | C组 | ✅ 已集成 | 远距小目标增强, FPN改进 |
| 电容器巡视 | D组 | ✅ 已集成 | 鼓包/渗漏检测增强 |
| 表计读数 | E组 | ✅ 已集成 | 关键点检测, 透视校正 |
| 鸟类监测 | - | ✅ 已集成 | DeepSORT轨迹跟踪 |

### 高级监测模块

| 模块 | 状态 | V3.0 增强 |
|------|------|-----------|
| 室内电子围栏 | ✅ 已集成 | 增强版多目标跟踪, DeepSORT支持 |
| 气体检测 | ✅ 已集成 | GL-TransLSTM 预测, 泄漏检测 |
| 声学监测 | ✅ 已集成 | CNN-LSTM 异常检测, 局放识别 |
| 多模态融合 | ✅ 已集成 | 可见光+热成像融合 |

## 深度学习模型 (V3.0)

### YOLOv8-ViT 目标检测

```python
from ai_models.deep_learning import YOLOv8ViTDetector, YOLOv8ViTConfig

config = YOLOv8ViTConfig(
    model_size='n',
    confidence_threshold=0.5,
    use_thermal_fusion=True
)
detector = YOLOv8ViTDetector(config)
detector.load()

result = detector.detect(image)
for det in result.detections:
    print(f"{det.class_name}: {det.confidence:.2f}")
```

### GL-TransLSTM 气体预测

```python
from ai_models.deep_learning import GLTransLSTM, GLTransLSTMConfig

config = GLTransLSTMConfig(input_dim=8, hidden_dim=128)
model = GLTransLSTM(config)
model.initialize()

prediction = model.predict(history_data, steps=6)
print(f"预测趋势: {prediction.trend}")
```

### CNN-LSTM 声学监测

```python
from ai_models.deep_learning import AcousticCNNLSTM, AcousticModelConfig

config = AcousticModelConfig(num_classes=5)
model = AcousticCNNLSTM(config)
model.load()

result = model.analyze(audio_features)
if result.has_anomaly:
    print(f"检测到异常: {result.anomaly_type}")
```

### 深度学习集成层

```python
from platform_core.plugin_manager.dl_integration import get_dl_integration

dl = get_dl_integration()

# 初始化模型
dl.init_model('yolov8_vit', model_size='n')
dl.init_model('gl_translstm', hidden_dim=128)

# 检测
detections = dl.detect(image)

# 气体预测
prediction = dl.predict_gas(history)

# 声学分析
analysis = dl.analyze_acoustic(features)
```

## 异步插件架构 (V3.0)

### 增强版基类

```python
from platform_core.plugin_manager.enhanced_base import EnhancedBasePlugin, TaskContext

class MyPlugin(EnhancedBasePlugin):

    def init(self, config: dict) -> bool:
        # 初始化深度学习模型
        self.init_deep_learning('yolov8_vit', model_size='n')
        return True

    def process(self, inputs: dict, context: TaskContext) -> UnifiedResult:
        # 使用深度学习检测
        detections = self.detect_with_dl(inputs['image'])

        return self.create_result(
            task_id=context.task_id,
            success=True,
            detections=detections
        )

    async def process_async(self, inputs: dict, context: TaskContext):
        # 异步处理
        return await super().process_async(inputs, context)
```

### 统一输出格式

```json
{
    "plugin_id": "my_plugin",
    "plugin_version": "1.0.0",
    "task_id": "xxx",
    "timestamp": "2025-01-16T10:00:00",
    "success": true,
    "status": "normal",
    "detections": [
        {
            "label": "oil_leak",
            "confidence": 0.92,
            "bbox": {"x": 0.1, "y": 0.2, "width": 0.1, "height": 0.1}
        }
    ],
    "alarms": [],
    "inference_time_ms": 45.2,
    "metadata": {}
}
```

## 数据训练管道 (V3.0)

### 格式转换

```python
from ai_models.training.data_pipeline import DataFormatConverter

converter = DataFormatConverter()

# COCO转YOLO
converter.coco_to_yolo('annotations.json', 'images/', 'yolo_output/')

# YOLO转COCO
converter.yolo_to_coco('yolo_labels/', 'images/', 'coco_output.json')
```

### 主动学习

```python
from ai_models.training.data_pipeline import ActiveLearningSampler

sampler = ActiveLearningSampler(strategy='uncertainty')
selected = sampler.select_samples(unlabeled_images, model, n_samples=100)
```

### 自动标注

```python
from ai_models.training.data_pipeline import AutoAnnotator

annotator = AutoAnnotator(detector=yolo_model, confidence_threshold=0.8)
annotations = annotator.annotate_batch(images)
```

## 统一仪表盘 (V3.0)

访问 `/unified-dashboard` 查看综合监控页面，包含:

- **设备概览**: 按类型分组的设备状态
- **健康趋势**: 各模块健康度实时图表
- **区域汇总**: 室外/室内/控制室状态
- **实时告警**: 分级告警列表
- **深度学习状态**: 模型加载和推理统计

## API 文档

启动平台后访问: http://127.0.0.1:8080/api/docs

### 主要接口

| 接口 | 方法 | 描述 |
|------|------|------|
| /api/health | GET | 健康检查 |
| /api/plugins | GET | 获取插件列表 |
| /api/tasks/run | POST | 运行任务 |
| /api/evidence/runs | GET | 获取证据记录 |
| /api/advanced/dl/stats | GET | 深度学习统计 |
| /api/advanced/modules/health | GET | 模块健康状态 |

## 验收标准

每个插件必须通过:

1. **可运行**: 按平台接口接入即跑
2. **可回放**: 给定回放数据,结果可复现
3. **可解释**: 输出 bbox/关键点/置信度/失败原因码
4. **可追溯**: 输出 model_version + code_hash
5. **可维护**: README + 配置样例 + 最小单测
6. **深度学习**: 支持DL模型集成 (V3.0新增)

## 技术栈

- **后端**: Python 3.10+, FastAPI, Pydantic
- **前端**: Bootstrap 5, Jinja2, Chart.js
- **视觉**: OpenCV, NumPy, Pillow
- **深度学习**: PyTorch, Ultralytics (YOLOv8)
- **日志**: Loguru
- **打包**: PyInstaller

## 文档

- [架构文档](docs/ARCHITECTURE.md)
- [插件开发指南](docs/plugins/)
- [API文档](docs/api/)
- [深度学习模型](ai_models/README.md)
- [升级计划](UPDATE.md)

## 更新日志

### V3.0.0 (2025-01)

- 新增 YOLOv8-ViT 深度学习目标检测
- 新增 GL-TransLSTM 气体浓度预测
- 新增 CNN-LSTM 声学异常检测
- 新增 DeepSORT 多目标跟踪
- 新增关键点检测表计读数
- 新增数据训练管道 (COCO/YOLO转换、主动学习)
- 新增异步插件架构支持
- 新增统一仪表盘前端
- 增强室内电子围栏多目标跟踪
- 深度学习模型集成层

### V2.0.0 (2024)

- 初始插件化架构
- 五大功能模块集成
- 证据链系统
- 回放功能

## License

Proprietary - All Rights Reserved
