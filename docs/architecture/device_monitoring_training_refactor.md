# Device Monitoring Training Refactor

## 范围

本方案聚焦两个目录：

- `/Users/ronan/Desktop/DarkBreaker/plugins/device_monitoring`
- `/Users/ronan/Desktop/DarkBreaker/training`

目标不是单纯补算法名词，而是给出可直接落到上传、训练、导出、插件接入的工程方案。

## 现状约束

`device_monitoring` 当前插件输入已经稳定：

- `device_id`
- `device_type`
- `device_name`
- `metrics.cpu_temp`
- `metrics.cpu_usage`
- `metrics.memory_usage`
- `metrics.disk_usage`
- `metrics.network_quality`
- `metrics.uptime_hours`
- `metrics.error_count`
- `metrics.last_heartbeat`

当前插件输出已经稳定：

- `health_index`
- `anomaly_score`
- `predicted_failure`
- `recommendations`

当前 detector 是规则 + 统计基线：

- `health_index` 由规则权重扣分得到
- `anomaly_score` 由历史健康分数 z-score 得到
- `predicted_failure` 由健康趋势线性拟合得到

因此重构策略不能一步替换，而应采用“三模型可选 + 规则回退”的渐进方案。

## 任务拆分

`device_monitoring` 专项训练拆为 3 个独立任务：

### 1. 无监督异常检测

- task_type: `multivariate_sensor_anomaly`
- 目标：学习正常运行行为，输出连续异常分数
- 推荐范式：`autoencoder`
- 主模型角色：`device_anomaly_autoencoder`
- 输出字段：`anomaly_score`

### 2. 监督式故障预测

- task_type: `equipment_health_prediction`
- 目标：基于过去窗口预测未来故障风险、故障类别或剩余时间
- 推荐范式：`lstm` 或 `transformer_encoder`
- 主模型角色：`failure_predictor`
- 输出字段：`predicted_failure`

### 3. 健康指数校准

- task_type: `health_index_calibration`
- 目标：把原始设备指标映射为稳定、可解释、可跨设备类型比较的 `health_index`
- 推荐范式：`hybrid_statistical_ml`
- 主模型角色：`health_calibrator`
- 输出字段：`health_index`

说明：

- `recommendations` 不建议直接端到端训练成自由文本。
- 第一阶段由规则引擎根据 `health_index / anomaly_score / predicted_failure / threshold hits / device_type` 组合生成。

## 训练数据格式

### 统一主键

所有样本建议使用以下主键：

- `device_id`
- `device_type`
- `timestamp`
- `sequence_id` 或 `window_id`

### 核心指标列

最低特征集合：

- `cpu_temp`
- `cpu_usage`
- `memory_usage`
- `disk_usage`
- `network_quality`
- `uptime_hours`
- `error_count`
- `last_heartbeat_delta_sec`

建议扩展列：

- `restart_count_24h`
- `packet_loss_rate`
- `latency_ms`
- `fan_speed`
- `power_voltage`
- `power_cycle_count`
- `ambient_temp`

### 训练标签拆分

#### 无监督异常检测

- 可无标签训练
- 若有标签，可补充：
  - `is_anomaly`
  - `alarm_level`
  - `anomaly_type`

#### 监督式故障预测

- 必须提供故障标签
- 至少一种：
  - `failure_within_24h`
  - `failure_within_72h`
  - `failure_type`
  - `failure_time`

#### 健康指数校准

- 必须提供 `health_index`
- 来源可以是：
  - 专家评分
  - 维保验收结果映射
  - SLA 得分
  - 规则引擎旧版健康分数，经人工抽检修正

## User Upload 数据包格式

### A. 单设备时序

适用于单台设备长序列建模。

```text
dataset_root/
├── manifest.json
├── timeseries/
│   └── device_cam_01.csv
├── labels/
│   ├── alarms.jsonl
│   ├── failures.jsonl
│   └── health_targets.jsonl
└── metadata/
    └── device_catalog.json
```

`device_cam_01.csv` 推荐列：

- `timestamp`
- `device_id`
- `device_type`
- `cpu_temp`
- `cpu_usage`
- `memory_usage`
- `disk_usage`
- `network_quality`
- `uptime_hours`
- `error_count`
- `last_heartbeat_delta_sec`

### B. 多设备批量数据

适用于批量上传多设备样本，训练时按 `device_id` 分组切窗。

```text
dataset_root/
├── manifest.json
├── timeseries/
│   ├── batch_2026_04_part_01.csv
│   └── batch_2026_04_part_02.csv
├── labels/
│   ├── alarms.jsonl
│   ├── failures.jsonl
│   └── health_targets.jsonl
└── metadata/
    ├── device_catalog.json
    └── feature_dictionary.json
```

批量表必须包含：

- `device_id`
- `device_type`
- `timestamp`

### C. 告警标签

推荐 `labels/alarms.jsonl`：

```json
{"device_id":"cam_01","timestamp":"2026-04-01T10:30:00Z","alarm_level":"warning","alarm_type":"cpu_temp_high","is_anomaly":1}
```

用途：

- 异常检测弱监督评估
- 告警级别对齐
- 生成 `recommendations` 的回放验证

### D. 故障发生时间标签

推荐 `labels/failures.jsonl`：

```json
{"device_id":"cam_01","failure_time":"2026-04-03T18:00:00Z","failure_type":"camera_offline","severity":"alarm"}
```

用途：

- 构建 `failure_within_{h}` 标签
- 评估提前量 `lead_time_hours`
- 校验 `predicted_failure` 的命中率和漏报率

## Manifest 设计

`device_monitoring` 建议使用以下 task_type：

- `multivariate_sensor_anomaly`
- `equipment_health_prediction`
- `health_index_calibration`

关键 manifest 字段：

- `sequence_length`
- `prediction_horizon`
- `sensor_columns`
- `temporal_schema`
- `target_schema`

`equipment_health_prediction` 的 `target_schema.outputs` 必须包含：

- `health_index`
- `anomaly_score`
- `predicted_failure`

`health_index_calibration` 至少包含：

- `health_index`

## 模型产出到插件输出字段的映射

### 字段映射

| 插件字段 | 来源任务 | 来源模型 | 说明 |
|---|---|---|---|
| `health_index` | `health_index_calibration` | `health_calibrator` | 输出 0-100 分数 |
| `anomaly_score` | `multivariate_sensor_anomaly` | `device_anomaly_autoencoder` | 输出 0-1 异常分数 |
| `predicted_failure` | `equipment_health_prediction` | `failure_predictor` | 输出未来故障概率、类别、剩余时间 |
| `recommendations` | 规则引擎 | 非直接训练 | 基于上面三个输出和阈值生成 |

### 推荐插件侧聚合顺序

1. 读取 `health_calibrator`，生成 `health_index`
2. 读取 `device_anomaly_autoencoder`，生成 `anomaly_score`
3. 读取 `failure_predictor`，生成 `predicted_failure`
4. 将三者送入 `recommendation_engine`
5. 若任一模型缺失，则使用已有规则/统计回退

### `predicted_failure` 建议结构

```json
{
  "predicted": true,
  "failure_type": "camera_offline",
  "failure_probability": 0.82,
  "hours_to_failure": 18.5,
  "horizon_hours": 24
}
```

### `recommendations` 生成建议

由规则模板生成，而不是由模型直接自由输出：

- `health_index < 30` -> `立即人工巡检`
- `anomaly_score > 0.8` -> `检查近24小时错误日志`
- `failure_probability > 0.7` -> `提前备件 / 创建工单`
- `device_type == camera && network_quality < 50` -> `检查交换机端口和网线`

## 错误判别逻辑

建议插件和训练侧共用错误码。

### 1. 特征缺失

- code: `FEATURE_MISSING`
- 触发条件：
  - 必填指标列缺失
  - `device_id / timestamp / device_type` 缺失
  - 某模型要求的输入特征未提供
- 处理：
  - 插件侧拒绝调用对应模型
  - 回退到规则引擎
  - 返回缺失字段列表

### 2. 指标分布漂移

- code: `FEATURE_DRIFT`
- 触发条件：
  - 在线输入分布与训练集基线差异过大
  - 例如均值漂移、方差膨胀、缺测率异常、PSI 超阈值
- 建议阈值：
  - `psi > 0.2` -> warning
  - `psi > 0.3` -> reject / retrain candidate
- 处理：
  - 输出 `drift_detected=true`
  - `recommendations` 附加 `建议重新标定或重训`

### 3. 模型不匹配设备类型

- code: `DEVICE_TYPE_MISMATCH`
- 触发条件：
  - 模型 bundle 中声明 `supported_device_types`
  - 输入 `device_type` 不在允许列表
- 处理：
  - 不加载该模型
  - 回退到同类通用模型或规则

### 4. 输入时间窗口不足

- code: `WINDOW_INSUFFICIENT`
- 触发条件：
  - 模型要求 `sequence_length=336`
  - 实际仅收到 120 个时间步
- 处理：
  - 异常检测可降级使用短窗模型或统计基线
  - 故障预测直接返回不可判定
  - 健康校准可使用单点 / 短窗近似

## 模块设计

### training 侧

- `task_type=multivariate_sensor_anomaly`
  - preprocessor: `TemporalPreprocessor`
  - trainer: `TemporalTrainer`
  - export role: `device_anomaly_autoencoder`
- `task_type=equipment_health_prediction`
  - preprocessor: `TemporalPreprocessor`
  - trainer: `TemporalTrainer`
  - export role: `failure_predictor`
- `task_type=health_index_calibration`
  - preprocessor: `TemporalPreprocessor`
  - trainer: `TemporalTrainer`
  - export role: `health_calibrator`

### plugin 侧

建议新增 3 个内部组件：

- `DeviceFeatureAdapter`
  负责把 `device_readings` 转成统一张量 / 窗口输入
- `DeviceModelOrchestrator`
  负责按模型角色加载 `health_calibrator / anomaly_autoencoder / failure_predictor`
- `RecommendationEngine`
  负责把模型输出翻译成 `recommendations`、工单和告警

## 评估指标

### 无监督异常检测

- `anomaly_auc`
- `f1@threshold`
- `false_positive_rate`
- `alert_precision`

### 监督式故障预测

- `failure_f1`
- `failure_recall`
- `lead_time_hours`
- `topk_failure_type_accuracy`

### 健康指数校准

- `health_mae`
- `health_rmse`
- `health_rank_corr`
- `calibration_mae`

### 插件级联调指标

- `ticket_precision`
- `critical_alarm_recall`
- `recommendation_acceptance_rate`

## 过渡实施步骤

### Phase 1

- 保留现有规则引擎为主
- 训练侧支持 3 个 task_type
- 上传包按新 manifest 校验
- manifest 新增可选模型 `health_calibrator`

### Phase 2

- 插件新增 `DeviceModelOrchestrator`
- 先接 `device_anomaly_autoencoder`
- `health_index` 仍使用规则分数

### Phase 3

- 接入 `failure_predictor`
- 输出标准化 `predicted_failure`
- 工单生成逻辑引用模型概率

### Phase 4

- 接入 `health_calibrator`
- 用模型输出覆盖规则 `health_index`
- 规则分数降级为 fallback

### Phase 5

- 加入分布漂移监控
- 建立重训触发条件
- 形成闭环：上传 -> 训练 -> 导出 -> 插件加载 -> 在线监控 -> 回流数据

## 已落地到仓库的相关位置

- [plugins/device_monitoring/manifest.json](/Users/ronan/Desktop/DarkBreaker/plugins/device_monitoring/manifest.json)
- [plugins/device_monitoring/plugin.py](/Users/ronan/Desktop/DarkBreaker/plugins/device_monitoring/plugin.py)
- [plugins/device_monitoring/detector.py](/Users/ronan/Desktop/DarkBreaker/plugins/device_monitoring/detector.py)
- [training/plugin_configs/device_monitoring.yaml](/Users/ronan/Desktop/DarkBreaker/training/plugin_configs/device_monitoring.yaml)
- [training/registry/plugin_training_mapping.json](/Users/ronan/Desktop/DarkBreaker/training/registry/plugin_training_mapping.json)
- [training/temporal_anomaly/task_profiles.py](/Users/ronan/Desktop/DarkBreaker/training/temporal_anomaly/task_profiles.py)
- [training/examples/device_monitoring_upload/README.md](/Users/ronan/Desktop/DarkBreaker/training/examples/device_monitoring_upload/README.md)
