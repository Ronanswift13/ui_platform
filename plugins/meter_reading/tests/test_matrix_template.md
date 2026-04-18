# Test Matrix — 表计读数插件

## 测试矩阵总览

### 模拟表读数测试矩阵

| 用例ID | 表计类型 | 输入条件 | 期望读数 | 精度要求 | 置信度要求 | 级别 |
|--------|----------|----------|----------|----------|------------|------|
| A-001 | pressure_gauge | 正面拍摄, 指针中位 | 0.5 MPa | ±2% FS | ≥ 0.8 | L0 |
| A-002 | pressure_gauge | 正面拍摄, 指针零位 | 0.0 MPa | ±2% FS | ≥ 0.8 | L0 |
| A-003 | pressure_gauge | 正面拍摄, 指针满量程 | 1.6 MPa | ±2% FS | ≥ 0.8 | L0 |
| A-004 | temperature_gauge | 正面拍摄, 中位 | 50 °C | ±2% FS | ≥ 0.8 | L0 |
| A-005 | pressure_gauge | 倾斜 15° | ~0.5 MPa | ±3% FS | ≥ 0.6 | L2 |
| A-006 | pressure_gauge | 倾斜 30° | ~0.5 MPa | ±5% FS | ≥ 0.5 | L2 |
| A-007 | pressure_gauge | 倾斜 45° (极限) | ~0.5 MPa | ±5% FS | ≥ 0.5 | L2 |
| A-008 | pressure_gauge | 低光照 | — | ±3% FS | ≥ 0.5 | L2 |
| A-009 | pressure_gauge | 反光/眩光 | — | ±5% FS | ≥ 0.5 | L2 |
| A-010 | pressure_gauge | 模糊/低分辨率 | — | — | flag MANUAL | L2 |

### 数字表读数测试矩阵

| 用例ID | 表计类型 | 输入条件 | 期望文本 | 精度要求 | 级别 |
|--------|----------|----------|----------|----------|------|
| D-001 | digital_display | 清晰 LCD | "123.4" | 完全匹配 | L0 |
| D-002 | digital_display | 多位小数 | "0.001" | 完全匹配 | L0 |
| D-003 | seven_segment | 标准七段 | "456" | 完全匹配 | L0 |
| D-004 | digital_display | 低对比度 | — | 字符级≥95% | L2 |
| D-005 | digital_display | 部分遮挡 | — | — | L2 |

### LED 指示灯测试矩阵

| 用例ID | 输入条件 | 期望状态 | 精度要求 | 级别 |
|--------|----------|----------|----------|------|
| L-001 | 红灯亮 | Red (1) | 100% | L0 |
| L-002 | 绿灯亮 | Green (2) | 100% | L0 |
| L-003 | 黄灯亮 | Yellow (3) | 100% | L0 |
| L-004 | 灯灭 | Off (0) | 100% | L0 |
| L-005 | 多灯同框 | 逐个识别 | ≥99% | L2 |
| L-006 | 环境干扰光 | 正确状态 | ≥95% | L2 |

### Fallback 链路测试矩阵

| 用例ID | 模拟条件 | 期望行为 | 级别 |
|--------|----------|----------|------|
| F-001 | HRNet 模型缺失 | 降级到 HoughCircle | L1 |
| F-002 | 圆检测失败 | 降级到 HoughLine | L1 |
| F-003 | 所有检测失败 | 返回 FAILED + confidence=0 | L1 |
| F-004 | OCR 模型缺失 | 返回 FAILED | L1 |
| F-005 | 重试 3 次仍失败 | 标记 NEED_MANUAL_REVIEW | L1 |

### 性能测试矩阵

| 用例ID | 测试项 | 目标 | 方法 | 级别 |
|--------|--------|------|------|------|
| P-001 | 单帧延迟 (模拟表) | ≤ 500ms CPU | 100帧P95 | L1 |
| P-002 | 单帧延迟 (数字表) | ≤ 300ms CPU | 100帧P95 | L1 |
| P-003 | 单帧延迟 (LED) | ≤ 100ms CPU | 100帧P95 | L1 |
| P-004 | 内存占用 | ≤ 512MB | 持续推理1000帧 | L1 |

## Fixture 文件命名规范
```
tests/fixtures/{type}/{subtype}/{condition}_{index}.{ext}
示例:
  tests/fixtures/analog/pressure/normal_001.jpg
  tests/fixtures/analog/pressure/tilted30_001.jpg
  tests/fixtures/digital/lcd/clear_001.jpg
  tests/fixtures/led/red_on_001.jpg
```

## 标注文件格式 (labels.json)
```json
{
  "analog/pressure/normal_001.jpg": {
    "meter_type": "pressure_gauge",
    "expected_value": 0.8,
    "unit": "MPa",
    "range_min": 0.0,
    "range_max": 1.6,
    "conditions": ["normal", "front_view"]
  }
}
```
