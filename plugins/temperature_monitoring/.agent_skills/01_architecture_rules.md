# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. **层级方向固定**：plugin(接口层) → detector(算法层) → config；禁止反向依赖。
2. **算法层纯业务**：detector 不得依赖 SDK schema；SDK 适配留在 plugin.py。
3. **standalone 隔离**：`standalone/` 仅做运行与展示，不承载算法决策。
4. **配置单一来源**：运行参数只从 YAML 注入，不在算法主链写死阈值。
5. **循环依赖禁止**：任意模块间不得形成循环 import。

## 2. 本项目差异规则（temperature_monitoring）

### 2.1 目录改动权限

- **允许直接修改**：`tests/`、`.agent_skills/`、`scripts/`
- **允许但需契约同步**：`plugin.py`、`detector.py`、`configs/default.yaml`
- **禁止修改**：`manifest.json` 的 `id/entrypoint/plugin_class`

### 2.2 依赖方向

```
plugin.py ──→ detector.py ──→ (numpy, cv2[可选])
    │              │
    │              ├── TemperatureDetector  (主检测器)
    │              ├── Hotspot / TempTrend / TemperatureResult (数据类)
    │              └── 历史缓冲 + 趋势预测
    │
    └──→ darkbreaker_sdk.interfaces / schemas
```

- `detector.py` 依赖 `cv2`（可选），缺失时走简化热点检测路径。
- `plugin.py` 通过 `from .detector import TemperatureDetector` 直接导入。

### 2.3 配置流向

```
configs/default.yaml
    ↓ (plugin.py: init → load_plugin_config)
config dict
    ↓ (传入 TemperatureDetector)
detector 从 config dict 读取:
    sensor.*        → 传感器参数
    thresholds.*    → 温度分级阈值
    hotspot.*       → 热点检测参数
    prediction.*    → 预测配置
    zones.*         → 区域定义
    linkage.*       → 联动配置
```

所有温度阈值（normal_max/warning/alarm/critical）和检测参数（z_score_threshold/min_area_ratio）均从 YAML 读取，detector 构造时解析一次。
