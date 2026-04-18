# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. **层级方向固定**：plugin(接口层) → detector(算法层) → config；禁止反向依赖。
2. **算法层纯业务**：detector 不得依赖 SDK schema；SDK 适配留在 plugin.py。
3. **standalone 隔离**：`standalone/` 仅做运行与展示，不承载算法决策。
4. **配置单一来源**：运行参数只从 YAML 注入，不在算法主链写死阈值。
5. **循环依赖禁止**：任意模块间不得形成循环 import。

## 2. 本项目差异规则（fire_detection）

### 2.1 目录改动权限

- **允许直接修改**：`tests/`、`.agent_skills/`、`scripts/`
- **允许但需契约同步**：`plugin.py`、`detector.py`、`configs/default.yaml`
- **禁止修改**：`manifest.json` 的 `id/entrypoint/plugin_class`

### 2.2 依赖方向

```
plugin.py ──→ detector.py ──→ (numpy, cv2, onnxruntime)
    │              │
    │              ├── FireDetector       (主检测器：YOLO 推理 + 融合 + 评估)
    │              ├── FireTracker        (跟踪器：IoU 匹配 + 扩散分析)
    │              ├── FireType/Level     (枚举)
    │              ├── SensorReading      (传感器数据类)
    │              └── FireAssessment     (综合评估数据类)
    │
    └──→ darkbreaker_sdk.interfaces / schemas
```

- `detector.py` 依赖 `cv2`（可选）和 `onnxruntime`（可选），缺失时走降级路径。
- `plugin.py` 通过 `from .detector import FireDetector` 直接导入。

### 2.3 配置流向

```
configs/default.yaml
    ↓ (plugin.py: init → load_plugin_config)
config dict
    ↓ (传入 FireDetector)
detector 从 config dict 读取:
    detection.*          → 检测阈值
    sensor_fusion.*      → 融合权重
    fire_level.*         → 等级判定阈值
    suppression.*        → 灭火触发（plugin 层使用）
    zones.*              → 区域定义
```

### 2.4 安全分层

灭火联动（`suppression`）和疏散（`evacuation`）配置由 **plugin 层** 消费并触发动作，**detector 层** 只输出评估结果，不直接控制物理设备。这一分层是关键安全边界。
