# 02_algorithm_contract

## 1. 输入契约

### plugin.detect() 输入
```python
device_readings: List[Dict]   # 设备遥测数据列表
context: Optional[Dict]       # 含 task_id 等上下文（可选）
```

每条 `device_reading` 结构：
```python
{
    "device_id": str,
    "device_name": str,
    "device_type": str,           # camera | lidar | temperature_sensor | ...
    "metrics": {
        "cpu_temp": float,        # °C
        "cpu_usage": float,       # 0-100 %
        "memory_usage": float,    # 0-100 %
        "network_quality": float, # 0-100 %
        "error_count": int,       # 当前周期错误数
        "last_heartbeat": float,  # Unix timestamp
        "uptime_hours": float,
    },
    "status": str,                # online | standby | warning | error | offline
}
```

- `device_readings` 为空列表 → 返回空汇总，`success=True`。
- `metrics` 中字段缺失 → `DeviceHealthCalculator` 使用安全默认值（如 `cpu_temp` 默认 40）。

### plugin.scan_devices() — 无外部输入
从 `managed_devices` 配置生成模拟读数，调用 `detect()`。仅用于演示/standalone 模式。

## 2. 输出契约

### plugin.detect() 输出
```python
{
    "success": bool,
    "status": str,                     # "normal" | "warning" | "alarm"
    "devices": List[{
        "device_id": str,
        "health_index": float,         # [0, 100]
        "anomaly_score": float,        # [0, 1]
        "status": str,                 # online | standby | warning | error | offline
        "issues": List[str],
        "recommendations": List[str],
        "predicted_failure": Optional[Dict],
    }],
    "summary": {
        "total_devices": int,
        "online_count": int,
        "warning_count": int,
        "error_count": int,
        "offline_count": int,
        "avg_health": float,
        "critical_devices": List[str],
    },
    "alarms": List[Dict],
    "maintenance_tickets": List[Dict],
}
```

## 3. 健康指数计算

**模型**：加权扣分制，基准 100 分。

```
score = 100
对每项指标:
  if 指标超过 alarm 阈值: score -= weight × 100
  elif 指标超过 warning 阈值: score -= weight × 50 (或按比例)
health_index = clamp(score, 0, 100)
```

| 指标 | 权重（默认） | 来源 |
|------|-------------|------|
| cpu_temp | 0.15 | `health_weights.cpu_temp` |
| cpu_usage | 0.15 | `health_weights.cpu_usage` |
| memory_usage | 0.10 | `health_weights.memory_usage` |
| network_quality | 0.20 | `health_weights.network_quality` |
| error_rate | 0.20 | `health_weights.error_rate` |
| uptime | 0.20 | `health_weights.uptime` |

## 4. 阈值来源

| 阈值 | 来源 | 配置路径 |
|------|------|----------|
| 健康预警 / 告警 / 严重 | YAML | `thresholds.health_warning/alarm/critical` |
| CPU 温度预警 / 告警 | YAML | `thresholds.cpu_temp_warning/alarm` |
| 内存预警 | YAML | `thresholds.memory_warning_percent` |
| 心跳超时 | YAML | `thresholds.heartbeat_timeout_sec` |
| 错误率预警 | YAML | `thresholds.error_rate_warning` |
| 工单优先级映射 | YAML | `maintenance.ticket_priority_map` |

**注意**：`cpu_usage > 90` / `> 70` 在 detector.py 中硬编码（约第 93-99 行），未走 config。

## 5. 降级策略

| 场景 | 降级行为 |
|------|----------|
| 插件未初始化 | `detect()` 返回 `success=False, error="插件未初始化"` |
| device_readings 为空 | 返回空汇总，`success=True` |
| 单设备 metrics 字段缺失 | 使用安全默认值，不中断其他设备处理 |
| 故障预测模型不可用 | 跳过预测，`predicted_failure=None` |
| 异常检测历史不足 | `anomaly_score` 基于统计方法降级 |
