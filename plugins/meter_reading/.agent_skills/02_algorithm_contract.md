# 02_algorithm_contract

## 1. 固定母版规则（跨插件通用）

1. 输入必须先校验，再进入推理。
2. 单个 ROI 独立产出单个结果，不允许跨 ROI 合并读数。
3. 只能有一条主链路，降级链必须显式、可观测、可测试。
4. 输出 `confidence` 必须落在 `[0, 1]`，异常值必须先清洗。
5. 失败或复核必须可解释，不能靠“幻象默认值”伪造成功结果。

## 2. 输入契约（meter_reading 专属）

### 2.1 帧输入

- `frame` 类型：`np.ndarray`
- 形状：`H x W x 3`
- dtype：`uint8`
- 色彩空间：BGR

非法输入处理：

- 空数组、`None`、非三通道、非 `uint8`、维度非法 -> `FAILED`
- metadata 必须带 `error_code`、`pipeline_stage=input_validation`

### 2.2 ROI 输入

- 接受 `x/y/w/h` 或 `x/y/width/height`
- 坐标必须归一化到 `[0, 1]`
- 裁剪后宽高必须都 >= 4 像素

非法 ROI 处理：

- 负坐标、越界、宽高 <= 0、像素过小 -> `FAILED`
- 不做“clamp 后继续成功”的静默纠偏

### 2.3 表计类型

`MeterType` 必须严格属于以下 9 类：

- `pressure_gauge`
- `temperature_gauge`
- `oil_level_gauge`
- `sf6_density_gauge`
- `digital_display`
- `led_indicator`
- `ammeter`
- `voltmeter`
- `seven_segment`

## 3. 主状态机（当前实现单一真相）

```text
S0 输入校验
  ├─ 非法 -> FAILED
  └─ 合法 -> S1

S1 ROI裁剪（可选） + 预处理
  └─ S2 类型分发

S2 类型分发
  ├─ 模拟表 -> S3A
  ├─ 数字表/七段码 -> S3B
  ├─ LED -> S3C
  └─ 未知类型 -> NEED_MANUAL_REVIEW

S3A 模拟表
  ├─ 关键点主链/降级链 -> 透视校正 -> 指针角度 -> 量程映射
  ├─ 角度越界 / 倾角超限 / 量程缺失 -> NEED_MANUAL_REVIEW
  └─ 结果有效 -> 置信度判定

S3B 数字表
  ├─ OCR -> `_clean_ocr_text`
  ├─ 非法串 -> NEED_MANUAL_REVIEW
  └─ 合法数值 -> 置信度判定

S3C LED
  ├─ HSV 统计 + 可分离性校验
  ├─ 颜色不确定 / 低饱和 / 色簇混叠 -> NEED_MANUAL_REVIEW
  └─ 颜色明确 -> 置信度判定
```

### 3.1 模拟表降级链契约

`fallback_level` 的当前语义固定为：

- `0`：HRNet / 深度学习关键点成功
- `1`：降级到 `HoughCircle`
- `2`：降级到 `HoughLine`
- `3`：全部关键点检测失败
- `-1`：尚未进入模拟表有效链路（如输入校验失败、未知类型）

## 4. 输出契约

### 4.1 RecognitionResult / MeterReading 契约

所有成功、失败、复核结果都必须包含：

- `meter_type`
- `reading_status`
- `pipeline_stage`
- `fallback_level`
- `timestamp_ms`
- `runtime_mode`
- `review_status`
- `failure_reason`

全链路结果还会追加：

- `processing_time_ms`
- 若超出 `performance.max_reading_time_ms`，追加 `latency_violation=true`

`runtime_mode` 当前取值：

- `real_dl`
- `traditional_fallback`
- `simulation`
- `not_applicable`

`review_status` 当前取值：

- `clear`
- `manual_review_required`
- `failed`

### 4.2 模拟表额外字段

- `pointer_angle_deg`
- `angle_range_deg`
- `range_min`
- `range_max`
- `tilt_deg`（若可得）
- `tilt_exceeded=true`（若视角超限）

### 4.3 数字表额外字段

- `ocr_text_raw`

### 4.4 LED 额外字段

- `hsv_stats`
- `color_class`

### 4.5 告警契约（plugin.postprocess）

- `*_failed` -> `AlarmLevel.INFO`
- `need_manual_review=true` -> `AlarmLevel.INFO`
- 告警规则命中 -> `AlarmLevel.WARNING`

## 5. meter_reading 特有硬约束

1. `ReadingStatus` 只能有三态，不得回引 `LOW_CONFIDENCE`。
2. 模拟表角度必须落在 `pointer_detection.angle_range` 内；越界直接进入 `NEED_MANUAL_REVIEW`。
3. `METER_RANGES` 缺失时必须进入 `NEED_MANUAL_REVIEW`，不得回退到 `0~100` 这类幻象默认量程。
4. `_clean_ocr_text()` 必须拒绝空串、多小数点、中间负号、无有效字符。
5. LED 颜色编码固定为：`off=0.0`、`red=1.0`、`green=2.0`、`yellow=3.0`，同时 `unit` 保存颜色名。
6. `need_manual_review=true` 与 `reading_status=need_manual_review` 必须同步。
7. `reload_config()` 当前只热加载 `confidence_threshold` 与 `manual_review_threshold`；非法值必须保持旧配置，不得半更新。

## 6. 配置契约

### 6.1 YAML 管理的参数

- `inference.*`
- `fallback.*`
- `preprocessing.*`
- `pointer_detection.*`
- `led_detection.*`
- `performance.*`

### 6.2 代码注册表管理的参数

- `detector_enhanced.py::METER_RANGES`

说明：

- 阈值与流程参数优先 YAML。
- 基线量程目前仍是代码注册表；若后续迁移到 YAML，必须先改本文件，再改实现和测试。

## 7. 必测映射（改动即需回归）

- 输入校验：`tests/test_input_validation.py`
- 模拟表链路：`tests/test_analog_meter.py`
- 数字/OCR：`tests/test_digital_ocr.py`
- LED：`tests/test_led_indicator.py`
- 置信度与状态集：`tests/test_confidence.py`
- 输出 schema / metadata：`tests/test_output_structure.py`
- 插件集成：`tests/test_plugin_integration.py`
