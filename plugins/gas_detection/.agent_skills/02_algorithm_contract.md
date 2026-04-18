# 02_algorithm_contract

本插件更准确地说是“气体监测服务编排契约 + 趋势预测后端契约”，但仍沿用 `02_algorithm_contract` 文件名以保持统一。

## 1. 输入契约

### 1.1 `init(config_or_registry=None)`

支持三种真实模式：

1. `None`
   - 直接使用 `GasDetectionConfig()` 默认值
2. `dict`
   - `_parse_config()` 会解析 sampling/window/thresholds/runtime/model/alarm_rules/upgrade_placeholders
   - `history_length`、`prediction_horizon`、`sample_interval_seconds` 等趋势字段已落到 `GasDetectionConfig`
3. 其他对象
   - 被视为 `model_registry`

### 1.2 `process(inputs)`

当前真实输入：

```python
{
    "device_id": str,            # 可缺省，默认 "unknown"
    "timestamp": float,          # 可缺省，默认 time.time()
    "gas_readings": {
        "SF6": float,
        "H2": float,
        "CO": float,
        "C2H2": float,
        "CH4": float,
        "C2H4": float,
        "C2H6": float,
    },
    "environmental": {
        "temperature": float,
        "humidity": float,
        "pressure": float,
    }
}
```

约束：

1. `gas_readings` 为空时直接返回失败。
2. `device_id` 和 `timestamp` 缺失时会自动补默认值。
3. 非 `GasType.get_all()` 中的气体会被历史缓冲忽略。

### 1.3 `manifest.json` 声明契约

平台声明层给出的最低要求是：

- `required = ["device_id", "gas_readings"]`
- 输出层声明有 `success`、`status`、`gas_status`、`predictions`、`leak_detection`、`alarms`

注意：

1. 当前代码同时返回统一壳 `status/label/value/confidence/metadata/results` 与历史兼容字段 `overall_status/gas_status/gas_levels`。
2. 当前成功返回还包含 `trend_analysis`、`trend_diagnosis`、`anomaly_events`、`abnormal_intervals`、`reason_codes`、`recommendations`、`health_index`。

## 2. 输出契约

### 2.1 `process()` 成功返回（当前真实结构）

```python
{
    "success": True,
    "status": str,
    "label": str,
    "value": dict,
    "confidence": float,
    "metadata": dict,
    "results": list,            # 非图像虚拟 RecognitionResult，virtual_roi=True
    "device_id": str,
    "timestamp": float,
    "overall_status": str,       # normal | attention | warning | alarm | critical
    "gas_status": {
        gas: {
            "value": float,
            "unit": "ppm",
            "level": str,
            "status": str,
            "threshold": {
                "attention": float,
                "warning": float,
                "alarm": float,
                "critical": float,
            },
            "percentage_of_alarm": float,
        }
    },
    "gas_levels": dict,          # 当前与 gas_status 重复
    "predictions": dict,
    "trend_analysis": dict,
    "trend_diagnosis": dict,
    "anomaly_events": list[dict],
    "abnormal_intervals": list[dict],
    "reason_codes": list[str],
    "leak_detection": dict,
    "alarms": list[dict],
    "recommendations": list[str],
    "health_index": float,
}
```

### 2.2 失败返回

当前真实失败壳：

```python
{"success": False, "error": "..."}
```

可见失败场景：

1. 插件未初始化
2. 缺少 `gas_readings`
3. 预测或分析路径内部异常被 `process()` 外层捕获并转换为可解释失败壳

失败壳也保持统一 metadata、trend_diagnosis、model_info/placeholders，不静默吞掉缺失输入。

### 2.3 `healthcheck()`

当前真实返回：

```python
HealthStatus(
    healthy=self._is_initialized,
    message="OK" if self._is_initialized else "未初始化"
)
```

## 3. 配置与依赖契约

### 3.1 当前真实配置来源

| 配置项 | 当前来源 | 备注 |
|---|---|---|
| 气体阈值 | `GasDetectionConfig.thresholds` 默认值，或 `_parse_config(thresholds)` | 当前唯一真正可外部覆盖的主要配置 |
| 历史缓冲长度 | `GasDetectionConfig.history_buffer_size` | 默认 1000 |
| 主输出历史窗口 | `GasDetectionConfig.history_length` | 默认 24，决定趋势预测最小样本数 |
| 预测步长 | `GasDetectionConfig.prediction_horizon` | 默认 24 |
| 采样间隔 | `GasDetectionConfig.sample_interval_seconds` | 默认 3600 |
| 泄漏检测窗口 | `GasDetectionConfig.leak_detection_window` | 默认 10 |
| 泄漏速率阈值 | `GasDetectionConfig.leak_rate_threshold` | 默认 5.0 |
| 模型 ID 映射 | `GasDetectionConfig.model_ids` | 当前默认键为 `sf6_forecast` / `multi_gas_forecast` / `health_trend` |

### 3.2 当前已对齐的配置项

以下字段已在 YAML、dataclass、解析逻辑、测试中形成闭环：

- `history_length`
- `prediction_horizon`
- `sample_interval_seconds`
- `leak_confidence_threshold`

`create_standalone()` 仍保留 SDK 兜底能力，但当前事实源是 `configs/default.yaml`，不再依赖“缺文件返回空字典”的隐式行为。

### 3.3 模型依赖契约

1. 第一优先：`ai_models.deep_learning.gl_translstm`
2. 第二优先：`model_registry.infer(...)`
3. 最后回退：传统线性外推 / 简单速率法

补充事实：

1. `manifest.json`、YAML 与 `GasDetectionConfig.model_ids` 保持兼容键：`sf6_forecast` / `multi_gas_forecast` / `equipment_health_trend`，并保留 `lstm` / `transformer` 别名。
2. `predictor.set_model_registry()` 可继续使用深度模型；模型不可用时回退传统趋势预测。

## 4. 降级策略

| 场景 | 当前真实行为 |
|---|---|
| `configs/default.yaml` 缺失 | SDK 返回 `{}`，插件退回 `GasDetectionConfig()` 默认值 |
| 预测器导入失败 | `_predictor=None`，`_predict_trends()` 回退到静态 `"available": False` 或简化结果 |
| 分析器导入失败 | `_analyzer=None`，`trend_analysis.available=False` 且 reason 可观测 |
| 历史样本少于 24 | `predictions = {"available": False, "reason": "历史数据不足", ...}` |
| 传统泄漏检测样本不足 | 返回 `{"detected": False, "reason": "数据不足"}` |
| GL-TransLSTM 不可用 | 预测器尝试后备深度学习或传统方法 |
| 当前默认配置下历史样本达到 24 | 进入 predictor + analyzer 主链路，传统预测可用时返回 `available=True` |

## 5. 已验证的最小事实链路

1. `GasDetectionPlugin().init()` 返回 `True`
2. 单条样本 `process()` 返回 `success=True`
3. `create_standalone()` 返回实例
4. `demo/run_demo.py` 可跑通
5. 24 条样本回放触发趋势预测后，`predictions` 与 `trend_analysis` 纳入主输出合同

因此当前合同应写成：

- 基础阈值告警：可用
- 历史缓冲：可用
- 趋势预测：可用，模型不可用时回退传统方法并标注 fallback
- 趋势分析：`GasDataAnalyzer.analyze_trends` 已接入主输出
- DGA 详细分析：工具层存在，尚未接入主输出
