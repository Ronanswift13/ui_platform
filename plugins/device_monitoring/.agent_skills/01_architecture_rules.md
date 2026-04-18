# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. **层级方向固定**：plugin(接口层) → detector(算法层) → config；禁止反向依赖。
2. **算法层纯业务**：detector 不得依赖 SDK schema；SDK 适配留在 plugin.py。
3. **standalone 隔离**：`standalone/` 仅做运行与展示，不承载算法决策。
4. **配置单一来源**：运行参数只从 YAML 注入，不在算法主链写死阈值。
5. **循环依赖禁止**：任意模块间不得形成循环 import。

## 2. 本项目差异规则（device_monitoring）

### 2.1 目录改动权限

- **允许直接修改**：`tests/`、`.agent_skills/`、`scripts/`
- **允许但需契约同步**：`plugin.py`、`detector.py`、`configs/default.yaml`
- **禁止修改**：`manifest.json` 的 `id/entrypoint/plugin_class`

### 2.2 依赖方向

```
plugin.py ──→ detector.py ──→ (numpy only)
    │              │
    │              ├── DeviceHealthCalculator (内部类，纯计算)
    │              ├── DeviceMonitorDetector  (主检测器)
    │              └── MaintenanceTicket      (数据类)
    │
    └──→ darkbreaker_sdk.interfaces (HealthStatus)
```

- `detector.py` 只依赖 `numpy`，无 SDK 依赖。
- `plugin.py` 通过 `from .detector import DeviceMonitorDetector` 直接导入。

### 2.3 配置流向

```
configs/default.yaml
    ↓ (plugin.py: init → load_plugin_config)
config dict
    ↓ (传入 DeviceMonitorDetector)
detector 从 config dict 读取 health_weights / thresholds / prediction / maintenance
    ↓ (传入 DeviceHealthCalculator)
calculator 读取 weights + thresholds
```

所有健康指数权重（`cpu_temp: 0.15` 等）和告警阈值（`health_warning: 70` 等）均来自 YAML。

### 2.4 已知架构注意点

- `detector.py` 中 `DeviceHealthCalculator.calculate()` 的 CPU 使用率阈值已从 `thresholds.cpu_usage_warning_percent` / `thresholds.cpu_usage_alarm_percent` 读取；后续不得退回硬编码。
- `plugin.py` 的 `scan_devices()` 生成随机指标用于演示，非生产数据路径。
