# 02_algorithm_contract

本文件更准确地说是“多模态 service/integration contract”，但沿用 `02_algorithm_contract` 文件名以保持统一。

## 1. 输入契约

### 1.1 `init(config_or_registry=None)`

支持三种真实模式：

1. `None`
   - 使用默认 `MultimodalConfig()`
2. `dict`
   - 当前 `_parse_config()` 只解析：
     - `fusion_strategy`
     - `alarm_threshold`
     - `modality_weights`
   - `max_history_length`、`modality_dims` 等当前不会被真正消费
3. 其他对象
   - 被保存到 `_model_registry`
   - 当前主链路并未真正使用它

### 1.2 `process(inputs)`

当前代码真实消费的输入：

```python
{
    "device_id": str,                 # 可缺省，默认 "unknown"
    "modalities": dict,               # 必填；为空时报错
    "fusion_strategy": str,           # 可缺省，默认取 config.fusion_strategy
}
```

模态数据当前支持两种真实路径：

1. 已注册模态插件
   - 调用 `registered_plugin.process(data)`
2. 未注册模态插件
   - 直接把输入包成：
   ```python
   {"success": True, "data": data, "confidence": ...}
   ```

### 1.3 manifest 声明契约 vs 当前实现

manifest `input_schema` 还声明了：

- `timestamp`
- `pre_processed`
- `modality_results`

但当前 `plugin.py::process()` 实际并未消费这些字段。

## 2. 输出契约

### 2.1 `process()` 成功返回（当前基础链路）

```python
{
    "success": True,
    "device_id": str,
    "timestamp": float,
    "overall_status": str,
    "confidence": float,
    "modality_contributions": dict[str, float],
    "fused_detections": list,
    "diagnostic_report": dict,
    "recommendations": list[str],
    "alarms": list[dict],
}
```

### 2.2 增强引擎成功时的附加字段

如果 `_process_with_enhanced_engine()` 真正成功，还可能返回：

- `fault_chain`
- `bayesian_inference`
- `engine = "enhanced"`

### 2.3 失败返回

当前真实失败壳：

```python
{"success": False, "error": "..."}
```

已确认的常见失败：

1. 未初始化
2. 缺少 `modalities`
3. 内部异常

## 3. 当前真实融合链路

### 3.1 模态处理

`_process_modalities()` 当前行为：

1. 若模态已注册插件：
   - 调用 `plugin.process(data)`
2. 否则：
   - 直接把输入视为已经处理好的模态结果

### 3.2 增强引擎优先

`init()` 成功后，如果 `fusion_engine_enhanced.py` 可导入，则默认：

- `self.enhanced_engine` 初始化
- `self.strategy_manager` 初始化
- `_use_enhanced_engine = True`

### 3.3 基础回退

增强引擎失败后，当前会回退到基础融合：

- `early`
- `late`
- `attention`
- `hybrid`

## 4. 已核验的降级逻辑

| 场景 | 当前真实行为 |
|---|---|
| `configs/default.yaml` 缺失 | `create_standalone()` 仍可成功，回到默认 `MultimodalConfig` |
| 未初始化 | `process()` 返回 `{"success": False, "error": "插件未初始化"}` |
| 缺少 `modalities` | `process()` 返回 `{"success": False, "error": "缺少模态数据"}` |
| 增强引擎处理失败 | 记录 warning，关闭 `_use_enhanced_engine`，递归回退到基础融合 |
| 未注册模态插件 | 直接使用输入数据作为模态结果 |
| 模态插件 `process()` 抛异常 | 该模态标为 `success=False`，其余模态继续处理 |

## 5. 当前已确认的脆弱点

1. 常见 dict/status 输入会让增强引擎失败，例如：
   - `{"status": "normal", "confidence": 0.9}`
2. 原因是增强引擎前的数据格式转换会尝试把字典值转成数值数组，而状态字符串会导致 `float()` 失败。
3. 即使提供数值 `features`，当不同模态 feature 长度不一致时，增强引擎仍可能因数组形状不齐失败。

这说明当前真实合同应写成：

- 增强引擎：默认尝试启用，但对常见输入并不稳
- 基础融合：当前更接近实际稳定链路

## 6. 配置与依赖契约

### 6.1 当前真实配置来源

| 配置项 | 当前来源 | 备注 |
|---|---|---|
| 融合策略 | `MultimodalConfig.fusion_strategy` | 默认 `hybrid` |
| 历史长度 | dataclass 默认 100 | dict 配置当前不能覆盖 |
| 告警阈值 | `alarm_threshold` | 当前配置可覆盖，但主链路基本未实际使用 |
| 模态权重 | `modality_weights` | 当前可覆盖并参与基础融合贡献计算 |

### 6.2 manifest 与实现的当前错位

manifest `default_config` 还声明：

- `max_history_length`
- `modality_dims`

但当前 `_parse_config()` 不消费：

- `max_history_length`
- `modality_dims`

### 6.3 外部依赖 / 模型依赖

1. `requirements.txt` 当前依赖：
   - `darkbreaker-sdk`
   - `numpy`
   - `opencv-python`
   - `pydantic`
   - `pyyaml`
2. manifest 额外声明：
   - `onnxruntime`
3. manifest 还声明了外部模态插件依赖：
   - `acoustic_monitoring`
   - `gas_detection`
   - `hyperspectral_detection`
4. 当前代码支持“注册模态插件”，但默认并不会自动发现或加载这些插件。

## 7. 已验证的最小事实链路

1. `MultimodalFusionPlugin().init()` 返回 `True`
2. `process({})` 返回缺模态错误
3. demo 回放可执行
4. 未注册模态插件时，基础融合仍可返回成功
5. 增强引擎默认会尝试工作，但 demo 这类常见输入会回退到基础融合

因此当前合同应表述为：

- 这是模态结果编排与融合插件
- 当前稳定入口是基础融合回退链路
- 增强引擎可用性依赖更严格的数值化输入规范
