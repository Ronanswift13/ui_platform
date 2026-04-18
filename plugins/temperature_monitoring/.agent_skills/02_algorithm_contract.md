# 02_algorithm_contract

## 1. 输入契约

### plugin.detect() 输入
```python
thermal_frame: Optional[np.ndarray]      # 热成像帧（可选）
sensor_readings: Optional[List[Dict]]    # 传感器阵列读数（可选）
context: Optional[Dict]                  # 含 task_id 等上下文
```

- 两者均缺失 → 自动生成模拟热力图（`_simulate_heatmap()`）。
- `thermal_frame` 可为 H×W（单通道）或 H×W×3（取第一通道）。值 > 200 时自动归一化到 15-75°C。
- `sensor_readings` 每条含 `{sensor_id, position: {row, col}, temperature, timestamp}`。

### detector.detect() 输入
- `thermal_frame`: `Optional[np.ndarray]`
- `sensor_readings`: `Optional[List[Dict]]`
- 内部统一转换为 2D `np.ndarray` 温度矩阵（heatmap）。

## 2. 输出契约

### plugin.detect() 输出
```python
{
    "success": bool,
    "status": str,                  # "normal" | "warning" | "alarm" | "critical"
    "heatmap": List[List[float]],   # 2D 温度矩阵 (.tolist())
    "max_temp": float,              # °C
    "min_temp": float,
    "avg_temp": float,
    "hotspots": List[{
        "zone_id": str,
        "zone_name": str,
        "center": (float, float),   # 归一化坐标 [0,1]
        "temperature": float,       # °C
        "area_ratio": float,        # [0, 1]
        "severity": str,            # "normal" | "warning" | "alarm" | "critical"
    }],
    "trend": {
        "current": float,
        "avg_1min": float,
        "avg_5min": float,
        "rise_rate": float,         # °C/min
        "direction": str,           # "rising" | "stable" | "falling"
        "predicted_30min": Optional[float],
    },
    "linkage_events": List[Dict],
    "alarms": List[Alarm],
    "inference_time_ms": float,
}
```

## 3. 阈值来源

| 阈值 | 来源 | 配置路径 |
|------|------|----------|
| 正常上限 | YAML | `thresholds.normal_max` (45°C) |
| 预警 | YAML | `thresholds.warning` (55°C) |
| 报警 | YAML | `thresholds.alarm` (70°C) |
| 紧急 | YAML | `thresholds.critical` (85°C) |
| 温升速率预警 | YAML | `thresholds.rise_rate_warning` (2.0°C/min) |
| 温升速率报警 | YAML | `thresholds.rise_rate_alarm` (5.0°C/min) |
| z-score 异常门限 | YAML | `hotspot.z_score_threshold` (2.5) |
| 最小热点面积 | YAML | `hotspot.min_area_ratio` (0.02) |
| 区域阈值偏移 | YAML | `zones[*].threshold_offset` |

## 4. 降级策略

| 场景 | 降级行为 |
|------|----------|
| thermal_frame 和 sensor_readings 均缺失 | 生成模拟热力图，标记 metadata |
| cv2 不可用 | 跳过轮廓分析，简化热点检测（阈值 mask + 均值中心） |
| 预测模型不可用 | 回退到 linear 线性外推 |
| 历史数据不足 | 趋势中 avg_1min/avg_5min 使用当前值填充 |
| 单次检测异常 | 不中断后续检测，记 warning 日志 |
