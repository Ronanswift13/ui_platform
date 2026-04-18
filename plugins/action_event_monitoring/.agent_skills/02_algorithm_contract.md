# 02_algorithm_contract

本插件更准确地说是“事件归一化与分析编排契约”，但仍沿用 `02_algorithm_contract` 文件名以保持统一。

## 1. 输入契约

### 1.1 `init(config)`

- `config` 可为空；为空时当前实现会使用空字典，而不是自动从 `configs/default.yaml` 读取。
- 推荐调用方先自行加载 `configs/default.yaml` 后再传入。
- `create_standalone(config=None)` 会从 `configs/default.yaml` 加载默认配置并调用 `init()`。

### 1.2 `process(input_data)`

支持三种真实输入模式：

1. 统一事件壳
   - `{"device_id": "...", "events": [ ... ]}`
2. 信号变位壳
   - `signal_changes` 或 `state_change_events`
3. 协议采集壳
   - `protocol_ingested_data`
4. 单条完整事件 dict
   - 条件：含 `event_id` 和 `action_type`
5. 批量事件
   - 结构：`{"events": [ ... ]}`

### 1.3 原始信号归一化规则

1. `action_type` 为空时，尝试从 `action_desc` 归一化。
2. `protection_type` 为空时，尝试从 `action_desc` 归一化。
3. `value_numeric` 缺失时，尝试从 `value_after` 或 `value` 解析浮点值。
4. `voltage_level` / `cabinet_id` 缺失时，尝试通过 `DeviceCorrelationService` 补充。

## 2. 输出契约

### 2.1 `process()` 返回结构

当前真实返回：

```python
{
  "success": bool,
  "status": str,
  "label": str,
  "value": dict | None,
  "confidence": float,
  "metadata": dict,
  "results": list,              # 非图像虚拟 ROI RecognitionResult
  "plugin_name": str,
  "task_type": "event_monitoring",
  "summary": dict,
  "anomaly_events": list[dict],
  "abnormal_intervals": list[dict],
  "severity": str,
  "reason_codes": list[str],
  "recommended_actions": list[str],
  "trend_diagnosis": dict,
  "evidence": list[dict],
  "review_required": bool,
  "model_info": dict,
  "placeholders": dict,
  "stored_event_ids": list[str],
  "analysis_triggered": bool,
  "analysis_result": dict | None,
}
```

说明：

- 当前不是简单 `{success, data, message}` 壳，而是 B 类统一事件监测输出壳。
- 当前也没有本地 `candidate_events` 或 `manual_review_status` 字段输出。
- `results[0].metadata.virtual_roi=True`，`roi_id` 使用 signal/device/channel id。

### 2.2 `get_status()` 返回结构

当前真实返回：

- `plugin_id`
- `plugin_name`
- `version`
- `initialized`
- `running`
- `stats`
- `store_stats`

### 2.3 根因分析结果

当 `analysis_triggered=True` 时，`analysis_result` 来自 `platform_core.root_cause_service`。
本地实测该结果至少可能包含：

- `trace_id`
- `root_cause_category`
- `root_cause_reason`
- `confidence`
- `probabilities`
- `evidence_chain`
- `counter_evidence`
- `evidence_gaps`
- `evidence_sufficiency`
- `manual_review_items`
- `next_actions`

这说明“证据不足时降级”是当前真实链路的一部分，但“人工复核 API”不是当前本地实现的一部分。

## 3. 状态与行为契约

1. `init()` 成功后，`get_status()["initialized"] == True`。
2. `start()` 只表示插件进入运行态，不等于协议已连接成功。
3. `stop()` 将 `running` 置回 `False`。
4. `shutdown()` 会调用 `stop()` 并重置初始化状态。
5. `process()` 出现异常时返回：
   - `{"success": False, "error": "..."}`

## 4. 动作链分析触发契约

分析是否触发取决于：

- `analysis.auto_analyze_on_protection`
- `analysis.auto_analyze_on_breaker`

当前行为：

1. 若事件类型属于保护动作，且 `auto_analyze_on_protection=true` -> 触发分析。
2. 若事件类型属于断路器分合，且 `auto_analyze_on_breaker=true` -> 触发分析。

## 5. 当前已确认的实现边界

1. 本地没有：
   - REST API
   - CandidateEvent 本地输出接口
   - 人工复核状态流转接口
   - timeline 查询接口
   - UI/dashboard/cockpit 前端入口接线
2. 历史经验文档中提到的这些能力，只能作为“未来演进方向”，不能作为当前合同。

## 6. 已验证的最小事实链路

在当前仓库环境下，以下链路已验证可执行：

1. 导入 `ActionEventMonitoringPlugin`
2. 加载 `configs/default.yaml`
3. `init(config)` 返回 `True`
4. `process(sample_event)` 返回 `success=True`
5. `get_status()` 统计随事件处理而变化
6. `Plugin` 别名、`demo/run_demo.py`、`__main__.py`、`run_standalone.py` 可导入
7. standalone smoke route 可提交模拟事件
