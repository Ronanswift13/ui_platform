# Design Notes

本目录记录 animal_detection 插件的关键设计决策与技术方案。

## 热成像验证设计

### 背景
变电站内摄像头可能因光斑、阴影、电缆晃动等产生假阳性检测。热成像验证利用红外图像的温差来区分活体与非活体目标。

### 方案
```
可见光检测框
    ↓ bbox 坐标映射到红外帧
红外 ROI 温度提取
    ↓ mean(ROI) - mean(背景)
温差比较 >= heat_diff_threshold_c (默认 2.0°C)
    ├─ 通过 → is_thermal_validated = True
    └─ 不通过 → is_thermal_validated = False (可选过滤)
```

### 关键设计决策

**DR-001: 热验证为可选功能**
- 决定：热成像验证默认关闭（`thermal.enabled: false`）
- 原因：不是所有部署场景都有双光摄像头。单摄像头部署必须正常工作
- 影响：验证器返回 `is_thermal_validated=False` + `thermal_diff_c=None`，不影响检测结果

**DR-002: 背景温度自适应更新**
- 决定：使用指数移动平均（EMA）更新背景温度模型
- 参数：`background_update_rate = 0.01`（慢速更新，避免短时干扰）
- 原因：室温会随时间和季节变化，固定基线会导致漂移

**DR-003: 温差比较而非绝对温度**
- 决定：使用 ROI 与周围背景的温差，而非 ROI 绝对温度
- 原因：活体温度因动物种类差异大（蛇为变温动物），绝对温度判定不可靠
- 特殊情况：变温动物（蛇、昆虫）在高温环境下温差可能极小，热验证结果仅为辅助参考

**DR-004: 分离校验与过滤**
- 决定：`validate_detections()` 只标记状态，`filter_non_thermal()` 执行过滤
- 原因：上游可能需要保留未通过热验证的检测（例如，仅标记为低置信度而非丢弃）

---

## 驱离控制设计

### 背景
检测到动物入侵后，系统可自动触发声光驱离设备。驱离效果需要跟踪和评估。

### 方案
```
跟踪器检测到超时轨迹 (stay > threshold)
    ↓
驱离控制器查询动物类别策略
    ↓
触发声音/灯光/组合驱离
    ↓ 等待 duration_s + 5s 评估窗口
评估结果
    ├─ 轨迹消失 → SUCCESS
    ├─ 轨迹存在但移动 → PARTIAL
    ├─ 轨迹不动 → FAILED → 重试
    └─ 重试次数 >= escalation_threshold → ESCALATED → 通知人工
```

### 关键设计决策

**DR-005: 驱离为可选功能**
- 决定：驱离功能默认关闭（`deterrent.enabled: false`）
- 原因：许多部署场景无物理驱离设备。启用需确认设备适配器已注入
- 影响：禁用时 `evaluate_and_trigger()` 返回空列表

**DR-006: 按动物类型差异化驱离策略**
- 决定：在 `DETERRENT_STRATEGY` 字典中为不同动物定义频率和方式
- 策略：
  - 鼠：超声波 20kHz（人耳不可闻）
  - 蛇：低频振动 300Hz + 灯光（蛇对地面振动敏感）
  - 猫/狗：高频音 16-18kHz + 灯光
  - 鸟：15kHz 声波（无灯光，避免光污染影响飞行）
- 原因：不同动物对声光刺激的敏感频率不同

**DR-007: 冷却时间防止过度驱离**
- 决定：同一轨迹两次驱离之间必须间隔 `cooldown_s`（默认 60 秒）
- 原因：过度频繁的声光刺激可能导致动物习惯化（适应性失效），也避免噪音扰民

**DR-008: 升级机制**
- 决定：驱离重试次数达到 `escalation_threshold`（默认 3 次）后停止自动驱离，转为人工通知
- 原因：如果驱离多次失败，说明当前策略无效，需要人工介入（如物理清理、通道封堵）
- 实现：通过 `set_escalation_callback()` 注入通知逻辑，保持 core 层无外部依赖

**DR-009: 灯光适配器跨插件复用**
- 决定：复用 `indoor_fence` 的 `LightAdapter` 接口
- 风险：跨插件依赖可能因 indoor_fence 变更而断裂
- 缓解：通过 try/except 包裹，灯光失败不阻塞驱离流程
- 后续：考虑将 LightAdapter 接口提升到 SDK 公共层

---

## 数据流全景

```
摄像头帧 (BGR ndarray)
    │
    ▼
detector.detect()          ← configs/default.yaml
    │ List[AnimalDetectionResult]
    ▼
tracker.update()           ← tracking config
    │ List[AnimalDetectionResult] + track_id + stay_duration
    ▼
thermal_validator.validate()  ← thermal config (可选)
    │ is_thermal_validated + thermal_diff_c
    ▼
event_schema.build_intrusion_event()
    │ AnimalEvent
    ▼
deterrent.evaluate_and_trigger()  ← deterrent config (可选)
    │ List[DeterrentAction]
    ▼
plugin.emit_event()
    │ → SDK 事件总线
    ▼
statistics.record_invasion()
    │ → 日/周报表
```

---

*新的设计决策应追加到本文件，使用 DR-NNN 编号。*
