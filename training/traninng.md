# TRAING_REFACTOR_PLAN.md

> DarkBreaker 训练库重构与独立训练调试 UI 方案初稿  
> 路径目标：`/Users/ronan/Desktop/DarkBreaker/training`

---

## 1. 文档目标

本方案用于将当前 `training` 目录从“面向少量视觉插件的初级训练脚本集合”，升级为一个：

1. 按插件能力分类组织的数据与训练平台  
2. 支持用户真实数据上传、校验、标准化、训练、评估、导出、注册的工程化训练库  
3. 可与 `plugins/` 下所有插件逐步打通的模型调度基础设施  
4. 自带一个与主平台 UI 独立的训练调试 UI，用于：
   - 数据上传
   - 上传进度查看
   - 批次管理
   - 数据校验结果查看
   - 训练任务启动与停止
   - 模型产物查看
   - 调试日志与错误排查

---

## 2. 当前现状与问题

### 2.1 当前已有内容

当前 `training/` 已具备以下基础：

- `training_api.py`
- `model_integration.py`
- `prepare_training_data.py`
- `data_aggregator.py`
- `dataset_downloader.py`
- `checkpoints/`
- `exports/`
- `logs/`

当前已有模型与训练产物主要集中在：

- busbar
- transformer
- capacitor
- switch
- meter

### 2.2 已发现的问题

#### 问题 1：training 与 plugins 的命名体系不一致

training 侧使用：
- `busbar`
- `transformer`
- `switch`
- `capacitor`
- `meter`

plugins 侧实际插件为：
- `busbar_inspection`
- `transformer_inspection`
- `switch_inspection`
- `capacitor_inspection`
- `meter_reading`
- 以及更多未纳入 training 的插件

这导致：

- 模型注册无法直接映射到插件
- 训练结果与插件加载关系不清晰
- 后续自动调度难以扩展

#### 问题 2：训练范式单一

当前训练逻辑基本围绕图像检测类任务构建，默认使用 YOLO 风格数据与模型思路。

但工程实际插件覆盖以下多种任务范式：

- 视觉目标检测
- 图像分类
- OCR / 读数识别
- 热像异常识别
- 高光谱谱空分析
- 声学时序异常检测
- 多变量时间序列异常检测
- 动作事件时序识别
- 多模态融合诊断
- 室内围栏/轨迹/空间感知
- SLAM/地图状态估计
- Radar 目标与轨迹判别

如果继续使用单一训练框架，将导致训练侧无法统一管理。

#### 问题 3：数据上传标准缺失

当前 training 体系没有统一的：

- 上传 manifest
- 任务类型 schema
- 标签格式标准
- 数据完整性校验
- 重复样本检测
- 数据质量审计
- 数据集版本管理

无法支持用户上传真实数据后直接纳入训练流水线。

#### 问题 4：路径与部署体系存在漂移

`model_integration.py` 中仍保留旧工程路径引用，需统一到当前 `DarkBreaker`。

#### 问题 5：训练 UI 缺失

当前缺少一个独立可视化入口用于：
- 真数据上传
- 训练批次管理
- 日志对比
- 调试和问题定位

---

## 3. 本次重构目标

### 3.1 总体目标

将 `training/` 升级为一个统一的 AI 训练库，满足以下能力：

1. 支持按插件分类、按任务范式组织训练流程
2. 支持用户上传真实训练数据
3. 支持自动校验、标准化入库、批次管理
4. 支持模型训练、评估、导出、注册
5. 支持插件侧后续按 `plugin_id` 调度模型
6. 提供独立训练调试 UI，仅服务训练相关工作流

### 3.2 范围边界

本阶段重点：

- 训练库重构
- 数据上传与管理
- 训练调试 UI
- 模型注册与插件映射基础

本阶段不重点处理：

- 主平台业务 UI 集成
- 插件侧在线推理调用链全部改造
- 主业务流程联动编排

这些作为下一阶段工作。

---

## 4. 插件分类与训练模块映射

### 4.1 视觉缺陷 / 状态识别类

插件：

- `busbar_inspection`
- `capacitor_inspection`
- `switch_inspection`
- `transformer_inspection`
- `animal_detection`
- `bird_monitoring`
- `fire_detection`
- `meter_reading`
- `temperature_monitoring`
- `thermal`
- `hyperspectral_detection`

训练模块：

- `visual_detection`
- `visual_classification`
- `ocr_reading`
- `thermal_analysis`
- `hyperspectral_analysis`

### 4.2 时序传感 / 数值异常类

插件：

- `acoustic_monitoring`
- `gas_detection`
- `device_monitoring`
- `action_event_monitoring`

训练模块：

- `acoustic_anomaly`
- `temporal_anomaly`
- `health_prediction`
- `event_sequence_recognition`

### 4.3 融合诊断类

插件：

- `multimodal_fusion`

训练模块：

- `multimodal_feature_fusion`
- `multimodal_decision_fusion`
- `hybrid_rule_model_diagnosis`

### 4.4 空间感知 / 安防类

插件：

- `indoor_fence`
- `slam_mapping`
- `radar`

训练模块：

- `fence_intrusion_detection`
- `trajectory_behavior_analysis`
- `slam_state_estimation`
- `radar_target_detection`
- `radar_track_classification`

---

## 5. 顶层目录重构方案

建议重构后的目录如下：

```text
training/
├── TRAING_REFACTOR_PLAN.md
├── __init__.py
├── api/
│   ├── app.py
│   ├── routes_training.py
│   ├── routes_upload.py
│   ├── routes_datasets.py
│   ├── routes_registry.py
│   ├── routes_debug.py
│   └── schemas.py
├── ui/
│   ├── app.py
│   ├── routes.py
│   ├── templates/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── icons/
│   └── components/
├── datasets/
│   ├── incoming/
│   ├── staging/
│   ├── standardized/
│   ├── visual_defect/
│   ├── temporal_anomaly/
│   ├── multimodal_fusion/
│   └── spatial_perception/
├── schemas/
│   ├── upload_manifest.schema.json
│   ├── detection.schema.json
│   ├── classification.schema.json
│   ├── ocr_reading.schema.json
│   ├── thermal.schema.json
│   ├── hyperspectral.schema.json
│   ├── temporal_series.schema.json
│   ├── event_sequence.schema.json
│   ├── multimodal_sample.schema.json
│   └── spatial_scene.schema.json
├── pipelines/
│   ├── ingestion/
│   ├── validation/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   ├── export/
│   └── registry/
├── registry/
│   ├── dataset_registry.json
│   ├── model_registry.json
│   ├── plugin_training_mapping.json
│   └── upload_batches.json
├── configs/
│   ├── plugins/
│   ├── task_types/
│   └── ui/
├── checkpoints/
├── exports/
├── adapters/
├── logs/
└── prompts/
    ├── batch01_visual_defect.md
    ├── batch02_temporal_anomaly.md
    ├── batch03_multimodal_fusion.md
    └── batch04_spatial_perception.md
```

---

## 6. 统一命名原则

### 6.1 统一使用 plugin_id

后续 training 内部统一使用 `plugin_id`，直接与 `plugins/*/manifest.json` 对齐。

示例：

- `busbar_inspection`
- `transformer_inspection`
- `meter_reading`
- `device_monitoring`

禁止继续在新代码中使用旧简写作为主标识：

- `busbar`
- `transformer`
- `switch`
- `capacitor`
- `meter`

旧命名仅作为兼容别名保留。

### 6.2 引入 task_type

统一任务类型字段：

- `detection`
- `segmentation`
- `classification`
- `ocr_reading`
- `thermal_anomaly`
- `hyperspectral_analysis`
- `acoustic_anomaly`
- `temporal_anomaly`
- `health_prediction`
- `event_sequence`
- `multimodal_fusion`
- `spatial_perception`
- `radar_tracking`

### 6.3 引入 modality

统一模态字段：

- `rgb`
- `thermal`
- `hyperspectral`
- `audio`
- `ultrasonic`
- `gas_timeseries`
- `device_metrics`
- `event_stream`
- `camera_track`
- `uwb`
- `imu`
- `lidar`
- `radar`

---

## 7. 数据上传标准

### 7.1 上传包结构

所有用户上传数据包应支持以下基本结构：

```text
upload_package/
├── manifest.json
├── samples/
├── labels/
├── metadata/
└── attachments/
```

### 7.2 manifest.json 必填字段

```json
{
  "dataset_name": "example_dataset_v1",
  "plugin_id": "transformer_inspection",
  "category_module": "visual_defect",
  "task_type": "detection",
  "modality": ["rgb"],
  "label_format": "yolo",
  "version": "v1",
  "source": "user_upload",
  "voltage_level": "HV_220kV",
  "split": {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
  }
}
```

### 7.3 上传后数据生命周期

1. 上传到 `datasets/incoming/`
2. 解包并登记批次
3. 放入 `datasets/staging/`
4. 校验 manifest 和标签
5. 通过标准化处理进入 `datasets/standardized/`
6. 更新 `dataset_registry.json`
7. 可用于训练任务创建

---

## 8. 数据校验机制

### 8.1 通用校验项

- 文件是否完整
- manifest 是否合法
- 标签文件是否缺失
- 样本数是否满足最小要求
- train/val/test 是否合理
- 类别映射是否一致
- 文件命名是否冲突
- 重复样本检测
- 明显坏数据过滤

### 8.2 视觉类专项校验

- 图片可读性
- 分辨率下限
- 标注框越界
- 标注类别越界
- OCR 标签文本为空
- 热像数据温度区间异常
- 高光谱谱段数量不一致

### 8.3 时序类专项校验

- 时间戳单调性
- 采样间隔异常
- 序列长度不足
- 关键特征列缺失
- 标签窗口对齐错误

### 8.4 多模态专项校验

- 模态缺失
- 模态时间不同步
- 同一 sample_id 对齐失败
- 某模态文件损坏

### 8.5 空间感知专项校验

- 坐标系定义缺失
- pose/trajectory 标签缺失
- 轨迹时间对齐问题
- 地图文件格式错误
- radar track 元数据缺失

---

## 9. 训练任务抽象

建议新增统一训练任务结构：

```json
{
  "task_id": "uuid",
  "plugin_id": "transformer_inspection",
  "task_type": "detection",
  "dataset_id": "dataset_xxx",
  "modality": ["rgb"],
  "model_family": "yolov8",
  "status": "created",
  "progress": 0,
  "batch_id": "upload_batch_xxx",
  "config_version": "v1",
  "created_at": "",
  "updated_at": ""
}
```

### 9.1 训练状态

- `created`
- `validating`
- `preparing`
- `queued`
- `training`
- `evaluating`
- `exporting`
- `registering`
- `completed`
- `failed`
- `cancelled`

---

## 10. 模型注册方案

### 10.1 model_registry.json 核心字段

- `model_id`
- `plugin_id`
- `task_type`
- `modality`
- `version`
- `dataset_id`
- `metrics`
- `export_formats`
- `checkpoint_path`
- `export_path`
- `preprocess_config`
- `postprocess_config`
- `class_mapping`
- `runtime_compatibility`
- `created_at`

### 10.2 plugin_training_mapping.json 核心用途

用于描述：

- 某插件允许哪些 task_type
- 默认模型类型是什么
- 默认加载哪个模型版本
- 是否允许多模型并存
- 是否支持热更新

---

## 11. 与 plugins 的映射策略

### 11.1 从 plugins 动态发现训练能力

后续 training 不再手写固定插件列表，而是动态扫描：

- `/Users/ronan/Desktop/DarkBreaker/plugins/*/manifest.json`

抽取：
- `id`
- `name`
- `capabilities`
- `models_required`
- `config_schema`
- `input_schema`
- `output_schema`

### 11.2 插件模型查找方式

插件后续统一按以下键查询模型：

- `plugin_id`
- `task_type`
- `version`（可选）
- `runtime`（可选）

### 11.3 错误判别

必须支持以下错误分类：

- 模型不存在
- 插件与模型 task_type 不匹配
- 模态不匹配
- 标签映射不一致
- 版本不兼容
- preprocess 配置缺失
- 模型文件损坏
- 导出格式不支持

---

## 12. 独立训练调试 UI 设计目标

### 12.1 UI 定位

该 UI 与主平台业务 UI 相互独立，仅用于：

- 训练数据上传
- 批次查看
- 训练任务管理
- 调试与日志
- 模型产物查看
- 数据标准化结果检查

### 12.2 启动方式

建议独立启动：

- 不依赖主平台完整运行
- 可单独运行在本地端口
- 作为训练调试控制台

示例：
- API：`127.0.0.1:8081`
- UI：`127.0.0.1:8091`

### 12.3 UI 核心页面

#### 页面 1：总览 Dashboard
- 最近上传批次
- 当前训练任务
- 最近失败任务
- 最近导出模型
- 各分类插件覆盖情况

#### 页面 2：数据上传
- 拖拽上传
- 上传 manifest 预览
- plugin_id / task_type 自动识别
- 上传进度条
- 批次备注
- 解包与校验日志

#### 页面 3：批次管理
- 批次列表
- 状态筛选
- 查看批次详情
- 查看校验结果
- 触发标准化处理
- 触发训练任务创建

#### 页面 4：训练任务
- 训练任务列表
- 进度
- 当前 epoch
- loss / metric 曲线
- 停止/重试
- 训练日志查看

#### 页面 5：模型注册表
- 按 plugin_id 查看模型
- 版本切换
- 指标对比
- 导出格式查看
- 与插件兼容性查看

#### 页面 6：调试中心
- 错误列表
- 路径配置
- schema 校验错误
- 数据异常预警
- API 健康状态

---

## 13. UI 技术方案建议

### 13.1 风格参考

参考 `plugins/` 下已有 standalone UI 架构：

- `standalone/app.py`
- `templates/`
- `static/`
- 简洁、工程化、面板式布局
- 不追求业务大一统首页
- 优先可用性与调试效率

### 13.2 推荐技术路线

后端：
- FastAPI

前端：
- Jinja2 模板 + 原生 JS / HTMX / 少量 ECharts
或
- 若已有统一偏好，可保留轻量方案，不强制引入复杂前端框架

建议原则：
- 轻依赖
- 易维护
- 易单独启动
- 适合本地调试

### 13.3 独立性要求

- 独立入口
- 独立模板
- 独立 static 资源
- 不污染主平台 UI 导航
- 仅调用 training 下 API

---

## 14. 分批实施计划

### 批次 1：基础设施重构
目标：
- 命名统一
- registry 建立
- schema 建立
- API 拆分
- 动态扫描 plugins manifest

交付：
- `registry/`
- `schemas/`
- `api/`
- `plugin_training_mapping.json`

### 批次 2：视觉类优先接入
目标：
- 接入 visual_defect 全套上传与训练
- 支持真实图片数据上传
- 支持 meter/thermal/hyperspectral 的任务分流

交付：
- 视觉类数据上传
- 视觉类批次管理
- 视觉类训练任务管理

### 批次 3：独立训练调试 UI
目标：
- 上传页面
- 批次页面
- 训练任务页面
- 注册表页面
- 调试页面

交付：
- `ui/`
- 独立启动脚本
- API 联通

### 批次 4：时序异常类接入
目标：
- acoustic / gas / device / action_event

### 批次 5：multimodal_fusion 接入

### 批次 6：spatial_perception 接入
目标：
- indoor_fence
- slam_mapping
- radar

---

## 15. 验收标准

### 15.1 第一阶段验收
- 可扫描所有插件 manifest
- 可统一列出 plugin_id
- 可上传一个真实数据包
- 可完成 manifest 校验与批次登记
- 可在 UI 查看上传进度和批次状态

### 15.2 第二阶段验收
- 至少 1 个视觉类插件可完成：
  上传 → 校验 → 标准化 → 启动训练 → 查看结果 → 注册模型

### 15.3 第三阶段验收
- 独立训练 UI 可单独启动
- 页面独立于主平台
- 可查看训练日志、批次详情、失败原因

---

## 16. 风险与注意事项

### 16.1 不要一次性强删旧逻辑
现有 `training_api.py` / `model_integration.py` 应优先做兼容改造，不宜一开始彻底推倒。

### 16.2 先做 registry 和 schema
若先写 UI、后补 schema，会导致后续接口频繁返工。

### 16.3 thermal / radar 允许占位实现
对占位态插件，优先建立：
- 数据结构
- 接口
- 注册机制
- UI 入口
后续再补算法实现。

### 16.4 meter_reading 不应混入普通 detection
必须单独当 OCR / 读数任务处理。

### 16.5 device_monitoring 不应走视觉流水线
应单独走时序/健康预测流水线。

---

## 17. 下一步建议

建议优先执行顺序：

1. 输出详细的第一批重构 prompt
2. 先完成 `TRAING_REFACTOR_PLAN.md`
3. 再实现 `api/ + registry/ + schemas/`
4. 再实现独立训练调试 UI
5. 最后逐类接入训练能力

---
