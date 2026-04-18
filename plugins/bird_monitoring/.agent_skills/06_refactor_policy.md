# 06_refactor_policy

> 最后更新：2026-04-16（目录收敛 + 职责分层）

## 1. 固定母版规则

1. 先证明问题，再重构。
2. 结构重构与行为变更分离。
3. 保持 `infer / postprocess / healthcheck / create_standalone` 公共契约兼容。
4. 没有测试支撑的重构不得进入生产主链。
5. simulation、experimental、demo 必须显式隔离。

## 2. 当前文件职责

| 文件 / 对象 | 职责 | 分类 | 约束 |
|-------------|------|------|------|
| `plugin.py` | SDK 适配、质量门、风险编排、告警、metadata | production | 不承载 ONNX 后处理细节；不触发硬件 |
| `detector.py::BirdDetector` | 生产检测器入口；ONNX/session/runtime truth；无模型时空检测 | production | `_simulate_detection()` 必须返回 `[]` |
| `detector.py::BirdDetectorEnhanced` | 旧增强检测器代码 | experimental/legacy | 不得被 plugin 自动加载；不得作为完成能力宣传 |
| `experimental/advanced_bird_detector.py` | PyTorch 概念验证、轨迹预测 demo | experimental/demo | 含随机生成检测，禁止生产 import |
| `advanced_bird_detector.py` | 顶层兼容 shim | compatibility | 只转发旧 import，不承载算法 |
| `standalone/app.py` | 真实插件 runner + `/api/simulator/*` 路由挂载 | standalone | 真实 `/api/detect` 与仿真路由隔离 |
| `standalone/bird_simulator.py` | UI 演示仿真器 | standalone/demo | 所有输出必须标记 `standalone_simulation` |
| `run_standalone.py` / `__main__.py` | 外部兼容启动入口 | compatibility launcher | 不得静默删除或改语义 |
| `main.py` | train / infer CLI 工具 | CLI | 不是 standalone server |
| `demo/` / `notebooks/` | 演示与调试 | demo | 输出不得回流生产契约 |
| `docs/` / `prompts/` | 升级说明与 agent prompt 槽位 | governance scaffold | 不宣称未验证能力 |
| `tests/replay/` / `tests/regression/` | replay/regression 槽位规划 | blocked/planned | 无真实图片前不得作为精度基线 |

## 3. 目录分层规则

```text
production runtime:
  plugin.py
  detector.py::BirdDetector

standalone demo:
  standalone/app.py
  standalone/templates/
  standalone/static/
  standalone/bird_simulator.py

compatibility launchers:
  run_standalone.py
  __main__.py
  advanced_bird_detector.py

experimental/demo:
  experimental/
  demo/
  notebooks/

tests and governance:
  tests/
  docs/
  prompts/
  .agent_skills/
```

## 4. 已完成的低风险收敛

- `plugin.py` 检测器加载固定为 `BirdDetector`，不再自动优先 `BirdDetectorEnhanced`。
- 输入质量门改为真正阻断极小 / 过暗 / 模糊等问题图像。
- `quality_failed` 与 `error` 结果补齐 `training_placeholders`。
- `quality_failed` 告警级别与契约统一为 WARNING。
- `trigger_deterrent()` 保留兼容入口但恒返回 `False`。
- `RepelController` 禁止加载/执行设备命令，检测器不再访问 HTTP/Modbus 控制通道。
- healthcheck 补充模型路径、模型存在性、ONNX session、real model 状态。
- 测试矩阵从 39 扩到 44。
- `experimental/advanced_bird_detector.py` 从顶层迁入 `experimental/`，顶层保留兼容 shim。
- `standalone/static/`、`docs/`、`prompts/`、`tests/regression/` 建立显式目录占位。
- 目录职责契约测试加入测试矩阵，插件内验证提升到 58 passed。

## 5. 允许的低风险重构

- 将 `BirdDetectorEnhanced` 移入 `experimental/` 并修正 import。
- 将 `BIRD_DATABASE` 外部化到 `configs/bird_species.yaml`。
- 补充 real_dl preflight 只读校验字段。
- 增加真实 fixture/replay 测试，不改生产输出 schema。
- 补充 docs/prompts 中的升级记录或 agent 执行模板。

## 6. 高风险重构（需人工确认）

- 启用真实 ONNX 模型并允许 `runtime_mode=real_dl`。
- 修改风险评分公式或默认阈值。
- 修改 9 种 label 语义。
- 将 standalone 仿真结果接入真实 `/api/detect` 或 `plugin.infer()`。
- 接入任何硬件驱离控制。

## 7. 禁止重构

- 不得把 `experimental/advanced_bird_detector.py` 的随机检测接入 `plugin.infer()`。
- 不得让 simulation 输出 `bird_safe / unknown_bird / review_required` 等“检测到鸟”结果。
- 不得把 `quality_failed / review_required / unknown_bird` 合并。
- 不得用 UI、README、manifest 宣称 blocked/experimental 能力已 verified。
- 不得新增 `urllib.request`、`requests.post`、`serial`、GPIO、MQTT 等硬件控制路径。

## 8. 契约兼容矩阵

| 变更 | 是否 breaking | 必须同步 |
|------|---------------|----------|
| 新增 metadata 字段 | 否 | tests（若必填）、README、PROJECT_CARD |
| 删除 metadata 字段 | 是 | 禁止静默删除 |
| 新增 runtime_mode | 否 | tests + 02_algorithm_contract |
| 新增 label | 否但需审查 | label 白名单 + postprocess + README + PROJECT_CARD |
| 删除/改名 label | 是 | 需人工确认 |
| 启用 real_dl | 否 | preflight、fixture、healthcheck truth、learning log |
