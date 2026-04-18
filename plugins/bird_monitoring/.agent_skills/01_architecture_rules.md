# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. **层级方向固定**：接口层(plugin) -> 算法层(detector) -> 配置；禁止反向依赖。
2. **算法层纯业务**：detector 不得依赖 SDK schema；SDK 适配留在 plugin.py。
3. **standalone 隔离**：`standalone/` 仅做运行与展示，不承载算法决策。
4. **配置单一来源**：运行参数只从 YAML 注入，不在算法主链写死阈值。
5. **循环依赖禁止**：任意模块间不得形成循环 import。

## 2. 本项目差异规则（bird_monitoring）

### 2.1 目录改动权限

- **允许直接修改**：`tests/`、`.agent_skills/`、`scripts/`（待新建）
- **允许但需契约同步**：`plugin.py`、`detector.py`、`experimental/advanced_bird_detector.py`、`configs/default.yaml`
- **禁止修改**：`manifest.json` 的 `id/entrypoint/plugin_class`

### 2.2 依赖方向

```
plugin.py ──→ detector.py::BirdDetector ──→ (numpy, cv2, onnxruntime)
    │
    └──→ darkbreaker_sdk.interfaces / schemas

experimental/demo:
experimental/advanced_bird_detector.py ──→ (torch, numpy random)
detector.py::BirdDetectorEnhanced ──→ legacy enhanced code
```

- `plugin.py` 通过 `_load_detector_class()` 只加载 `BirdDetector`。
- `BirdDetectorEnhanced` 和 `experimental/advanced_bird_detector.py` 不得自动进入生产主链。
- `detector.py` 和 `experimental/advanced_bird_detector.py` **互不依赖**。

### 2.3 配置流向

```
configs/default.yaml
    ↓ (plugin.py: init → load_plugin_config)
config dict
    ↓ (传入 BirdDetector)
算法层从 config dict 读取
```

### 2.4 已知架构问题（2026-04-16 更新）

**已修复**
- ✅ `RISK_THRESHOLDS` 双源问题：现在全部从 YAML 读取，类常量只做 fallback。
- ✅ `print()` 污染：17 处清零，统一 logger。
- ✅ 无测试 / 无 runtime_mode 可观测：见 03_test_strategy & 02_algorithm_contract。
- ✅ 生产加载器固定为 `BirdDetector`，不再优先 `BirdDetectorEnhanced`。
- ✅ detector 硬件驱离执行入口已阻断，仅保留建议语义。
- ✅ `advanced_bird_detector.py` 顶层文件已降为兼容 shim，真实概念验证代码迁入 `experimental/`。

**待解决**
- `BirdDetectorEnhanced` 仍在 `detector.py` 中，建议后续移入 `experimental/` 或加显式 opt-in。
- `BIRD_DATABASE` 仍硬编码在 plugin.py（外迁 YAML 需人工确认，不阻塞）。

### 2.5 插件 → 硬件的强约束（新增）

本插件**禁止**出现以下代码：
- `requests.post(...)` 驱离控制器 URL
- `urllib.request.urlopen(...)` / HTTP 驱离控制
- `serial.Serial(...)` / `pyserial` 调用
- 任何 GPIO / PWM 库调用
- 任何 MQTT 发布到驱离主题

驱离语义边界：插件只输出 `deterrent_suggestion` JSON（action / methods / reason）。当前插件不得决定、触发或模拟执行真实硬件动作。
