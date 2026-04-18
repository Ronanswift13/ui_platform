# Temporal Training Refactor

## 目标

本次重构将 `/Users/ronan/Desktop/DarkBreaker/training` 从“默认所有任务都是 YOLO 图像检测”升级为“视觉任务 + 时序/数值异常任务”的统一训练库。第一阶段目标是可工程交付:

- 上传包有统一 schema，可在进入训练前做结构校验。
- `plugin_id + task_type` 可以路由到正确的数据目录、预处理器和训练器。
- 时序任务导出遵循统一 bundle 规范，后续可直接接入插件侧 resolver。
- 保持视觉链路兼容，避免对已上线的图像插件造成回归。

## 目录重构

新增时序训练域：

```text
training/
├── temporal_anomaly/
│   ├── __init__.py
│   ├── task_profiles.py
│   └── data_contract.py
├── pipelines/
│   ├── preprocessing/
│   │   └── temporal_preprocessor.py
│   └── training/
│       └── temporal_trainer.py
├── plugin_configs/
│   ├── acoustic_monitoring.yaml
│   ├── gas_detection.yaml
│   ├── device_monitoring.yaml
│   └── action_event_monitoring.yaml
└── datasets/
    ├── visual_defect/
    └── temporal_anomaly/
        └── {plugin_id}/{task_type}/{version}/
```

时序数据不再落到 `datasets/visual_defect/`，统一进入 `datasets/temporal_anomaly/`。视觉任务保持原目录不变。

## 任务划分

新增 5 个 task_type：

- `acoustic_time_frequency_anomaly`
  原始音频与时频视图联合异常检测。
- `multivariate_sensor_anomaly`
  多变量数值时序异常检测，适合气体与设备运行监测。
- `equipment_health_prediction`
  健康指数、异常分数、故障预测联合建模。
- `health_index_calibration`
  设备健康指数校准，用于把原始运行指标稳定映射为 `health_index`。
- `action_event_sequence_recognition`
  带时间戳事件序列识别和异常动作链分析。

## 统一 Schema 设计

`DatasetManifest` 新增以下时序字段：

- `num_samples`
- `sample_rate_hz`
- `window_size`
- `stride`
- `sequence_length`
- `prediction_horizon`
- `timestamp_field`
- `timestamp_unit`
- `entity_id_field`
- `sequence_id_field`
- `feature_views`
- `sensor_columns`
- `event_types`
- `temporal_schema`
- `target_schema`

其中 `temporal_schema` 是统一输入契约：

- `sources`
  描述原始输入源，支持 `waveform / spectrogram / timeseries / event_sequence / labels / metadata`
- `feature_views`
  描述预计算特征视图，例如 `mel_spectrogram`、`log_mel`、`rolling_stats`
- `targets`
  描述监督目标，例如故障等级、未来 24h 故障标签、健康分数

`target_schema` 用于描述输出层，特别用于 `equipment_health_prediction`。

### acoustic_monitoring 双路径方案

`acoustic_monitoring` 明确支持两条输入路径：

- 原始音频路径：`waveforms/*.wav|*.flac`
- 时频图路径：`spectrograms/*.npy|*.npz`

推荐 manifest：

- `input_modality: audio_waveform+spectrogram`
- `feature_views: [raw_waveform, mel_spectrogram, log_mel]`

工程上允许 3 种模式：

- 仅波形：先落地最快，适合端到端 Transformer / 1D CNN
- 仅时频图：便于沿用 2D CNN / temporal CNN
- 双路径：主路径走 waveform，辅路径走 mel/log-mel，训练时融合，部署时也可只保留主模型

### device_monitoring 三输出层

`device_monitoring` 的 `equipment_health_prediction` 必须显式定义三个输出层：

- `health_index`
  0-100 的健康指数回归头，用于健康看板和分层阈值。
- `anomaly_score`
  0-1 的异常分数头，用于异常检测排序和告警置信度。
- `predicted_failure`
  故障预测头，可输出未来时窗故障概率、风险等级或剩余寿命分桶。

当前代码已在 `validate_temporal_contract()` 中对这三个输出层做必填校验。

## 数据上传标准

### acoustic_time_frequency_anomaly

```text
dataset_root/
├── manifest.json
├── waveforms/
├── spectrograms/        # 可选，双路径推荐
├── labels/              # 可选，监督训练时提供
└── metadata/
```

### multivariate_sensor_anomaly / equipment_health_prediction

```text
dataset_root/
├── manifest.json
├── timeseries/
├── labels/              # health / failure / anomaly 监督目标
└── metadata/
```

### action_event_sequence_recognition

```text
dataset_root/
├── manifest.json
├── events/
├── labels/
└── metadata/
```

上传格式建议：

- 声学：`wav_dir / flac_dir / npy_waveform / mel_npz / dual_audio_views`
- 时序：`csv_series / parquet_series / jsonl_series / npz_series`
- 事件：`jsonl_events / csv_events / parquet_events`

预处理输出统一为：

- `train/`
- `val/`
- `test/`
- `dataset_config.json`
- `samples.jsonl`

## 训练范式设计

统一支持 5 类范式：

- `autoencoder`
  适合无监督异常检测，优先用于 `device_monitoring`、`gas_detection`
- `lstm/gru`
  适合短中期预测、设备健康趋势建模
- `temporal_cnn`
  适合固定窗口特征提取与低时延部署
- `transformer_encoder`
  适合长依赖、事件序列和双路径声学建模
- `hybrid_statistical_ml`
  统计阈值、EWMA、季节分解等基线与 ML 模型融合，用于冷启动和回退策略

`TemporalTrainer` 当前实现了统一训练计划输出，后续可以把真实训练后端挂到对应 `paradigm` 分支上。

## 训练与评估指标

### acoustic_monitoring

- `auc_roc`
- `auc_pr`
- `f1`
- `false_alarm_rate`
- `latency_ms`

### gas_detection / device_monitoring

- `anomaly_auc`
- `health_mae`
- `health_rmse`
- `failure_f1`
- `failure_recall`
- `lead_time_hours`

### action_event_monitoring

- `sequence_accuracy`
- `macro_f1`
- `event_recall`
- `edit_distance`

## 模型导出与插件调度映射

统一导出目录：

```text
training/exports/{plugin_id}/{task_type}/{version}/
├── model.onnx
├── bundle.json
├── label_map.json           # 可选
├── preprocess.yaml|json     # 可选
└── postprocess.yaml|json    # 可选
```

`ModelExporter.export_bundle()` 已支持写出该结构。

关键映射如下：

- `acoustic_monitoring`
  - `audio_anomaly_transformer` -> `acoustic_time_frequency_anomaly`
  - `ultrasonic_pd_detector` -> `acoustic_time_frequency_anomaly`
- `gas_detection`
  - `sf6_forecast` -> `equipment_health_prediction`
  - `multi_gas_forecast` -> `equipment_health_prediction`
  - `equipment_health_trend` -> `multivariate_sensor_anomaly`
- `device_monitoring`
  - `device_anomaly_autoencoder` -> `multivariate_sensor_anomaly`
  - `failure_predictor` -> `equipment_health_prediction`
- `action_event_monitoring`
  - `action_sequence_encoder` -> `action_event_sequence_recognition`

其中 `device_monitoring manifest.json` 现有的 `models_required` 已通过 registry 中的 `model_roles` 保持兼容。

## training_api.py 改造方案

### 现状

`training_api.py` 是典型 v1 兼容接口，依赖：

- `voltage_level + plugin`
- 固定检查点 `best.pt`
- 视觉检测假设

### 本次改动

- 保持原有接口不破坏
- 新增 `/api/training/capabilities`
  用于显式告知调用方：
  - v1 只保证旧视觉接口
  - 时序/数值异常训练请走 `/v2`
  - 新调度键是 `plugin_id + task_type + version`

### 推荐迁移路线

1. 旧业务继续调用 `training_api.py`
2. 新业务统一迁移到 `training_api_v2.py`
3. 训练完成后统一导出 bundle 到 `exports/{plugin_id}/{task_type}/{version}`
4. 插件侧后续通过统一 resolver 加载 bundle

## 当前代码落地范围

已落地：

- 时序 task profile 与数据契约
- manifest/schema 扩展
- registry 映射
- temporal preprocessor / trainer 骨架
- router 与 v2 API 支持
- bundle 导出规范

下一阶段建议：

- 为 `TemporalTrainer` 接入真实 PyTorch / Lightning / sklearn 后端
- 将 `UploadValidator` 扩到时序文件质量检测
- 在插件侧接入统一 temporal model resolver
