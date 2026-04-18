# 02_algorithm_contract

## 1. 输入契约

### plugin.detect() 输入
```python
frame: np.ndarray              # H×W×3 BGR uint8（必须）
thermal_frame: np.ndarray      # H×W 或 H×W×1 热成像（可选）
sensor_data: Optional[Dict]    # 传感器读数（可选）
context: Optional[Dict]        # 含 task_id 等上下文
```

`sensor_data` 结构：
```python
{
    "smoke_concentration": float,  # 烟雾浓度 (%)
    "temperature": float,          # °C
    "co_concentration": float,     # ppm
    "humidity": float,             # %
}
```

- `frame` 为必须输入；`thermal_frame` 和 `sensor_data` 可选，缺失时跳过对应融合维度。

### detector.detect() 输入
- `frame`: `np.ndarray`, H×W×3, BGR, uint8。
- 模型输入尺寸由 `config.model.input_size` 控制（默认 640×640）。

## 2. 输出契约

### plugin.detect() 输出
```python
{
    "success": bool,
    "status": str,                      # "normal" | "warning" | "alarm" | "critical"
    "detections": List[{
        "type": str,                    # "fire" | "smoke" | "spark" | "ember" | "hot_spot"
        "bbox": {x, y, width, height},  # 归一化 [0,1]
        "confidence": float,            # [0, 1]
        "zone_id": str,
        "area_ratio": float,
        "spread_rate": float,
        "track_id": int,
    }],
    "fire_level": str,                  # FireLevel 枚举值
    "fusion_confidence": float,         # [0, 1]
    "sensor_status": Dict,
    "suppression_actions": List[Dict],
    "evacuation_routes": List[Dict],
    "alarms": List[Alarm],
    "inference_time_ms": float,
}
```

## 3. 检测链路

```
frame → YOLO 推理 → raw detections (fire/smoke/spark/ember)
    → NMS → FireTracker.update() → tracked detections + spread_rate
    → zone 匹配 → detections with zone_id

[thermal_frame] → 热成像异常检测 → hot_spot detections

[sensor_data] → SensorReading → 传感器置信度

传感器融合:
    fusion = visual_weight × visual_conf
           + smoke_sensor_weight × smoke_conf
           + thermal_sensor_weight × thermal_conf
    方法: weighted_ds | weighted_avg | bayesian

火灾等级:
    fire_level = f(fusion_confidence, max_area_ratio, detection_types)
    阈值来自 config.fire_level.*
```

## 4. 阈值来源

| 阈值 | 来源 | 配置路径 |
|------|------|----------|
| 检测置信度 | YAML | `detection.confidence_threshold` |
| NMS | YAML | `detection.nms_threshold` |
| 火焰面积报警 | YAML | `detection.fire_area_alarm_ratio` |
| 烟雾浓度报警 | YAML | `detection.smoke_density_alarm` |
| 融合权重 | YAML | `sensor_fusion.*_weight` |
| CO 报警 | YAML | `sensor_fusion.co_alarm_ppm` |
| 温度预警/报警 | YAML | `sensor_fusion.temperature_warning/alarm_celsius` |
| 火灾等级分界 | YAML | `fire_level.smoldering/small/medium/large/critical.*` |
| 灭火触发等级 | YAML | `suppression.trigger_level` |

## 5. 降级策略

| 场景 | 降级行为 |
|------|----------|
| ONNX 模型不可用 | 跳过 YOLO 推理，仅依赖传感器数据 |
| cv2 不可用 | 跳过图像预处理，返回空检测列表 |
| thermal_frame 缺失 | 跳过热成像融合维度 |
| sensor_data 缺失 | 跳过传感器融合维度，仅用视觉置信度 |
| detector 初始化失败 | plugin 进入降级模式（`_detector.initialize()` 返回 False），仍可响应 healthcheck |
| 单帧异常 | 不中断后续帧处理，记 warning 日志 |
