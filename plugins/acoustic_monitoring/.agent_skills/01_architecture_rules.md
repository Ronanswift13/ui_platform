# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. **层级方向固定**：plugin(接口层) -> detector(算法层) -> config；禁止反向依赖。
2. **算法层纯业务**：detector/analyzer 不得依赖 SDK schema；SDK 适配留在 plugin.py。
3. **standalone 隔离**：`standalone/` 仅做运行与展示，不承载算法决策。
4. **配置单一来源**：运行参数只从 YAML 注入，不在算法主链写死阈值。
5. **循环依赖禁止**：任意模块间不得形成循环 import。

## 2. 本项目差异规则（acoustic_monitoring）

### 2.1 目录改动权限

- **允许直接修改**：`tests/`、`.agent_skills/`、`scripts/`
- **允许但需契约同步**：`plugin.py`、`detector.py`、`analyzer.py`、`configs/default.yaml`
- **禁止修改**：`manifest.json` 的 `id/entrypoint/plugin_class`

### 2.2 依赖方向

```
plugin.py ──→ detector.py ──→ (numpy only)
    │
    └──→ analyzer.py ──→ (numpy only)
    │
    └──→ darkbreaker_sdk.interfaces / schemas
```

- `detector.py` 和 `analyzer.py` 互不依赖。
- `standalone/audio_manager.py` 只依赖 `plugin` 实例，不直接调用 detector/analyzer。

### 2.3 配置流向

```
configs/default.yaml
    ↓ (plugin.py: _load_config)
AcousticConfig dataclass
    ↓ (传入 detector / analyzer)
算法层读取 config 属性
```

所有检测阈值（PD、corona、bearing、transformer、mechanical）均在 `AcousticConfig` 中定义，由 YAML 覆盖。
