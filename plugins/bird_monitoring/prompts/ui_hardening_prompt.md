# Prompt: 独立 UI 稳固化

## 使用场景

当 `standalone/templates/bird_monitoring.html` 需要扩展但 runtime_mode 边界
容易被破坏时，用本 prompt 指导 agent 做最小增量。

## 任务目标

在 `plugins/bird_monitoring/standalone/` 范围内：

1. 所有 UI 输出必须显式呈现 `runtime_mode` 徽章；徽章颜色按：
   - `simulation` → 灰
   - `standalone_simulation` → 黄（仅 /api/simulator/*）
   - `real_dl` → 绿
   - `traditional_fallback` → 橙
2. 真实上传 (`/api/detect`) 返回结果必须展示：
   - 每条检测的 label / confidence / risk_level
   - `input_quality` 的 `status`、`clarity_score`、`brightness_score`
   - `training_placeholders` 的 JSON 下载按钮
3. 仿真演示页 (`/api/simulator/*`) 必须带 "STANDALONE SIMULATION" 水印与
   `runtime_mode=standalone_simulation` 徽章。
4. 不得在 UI 层伪造 bbox 或隐藏 `quality_failed` 结果。

## 强制约束

- 仅修改 `plugins/bird_monitoring/standalone/`。
- 静态资源只能放 `standalone/static/`；不得写入站点级 `evidence/`。
- 不新增网络外调（CDN 或外部 fetch）。
- 模板中 runtime_mode 字段必须同时出现在 HTML 源码与 JS 状态中，便于
  `tests/test_standalone.py::TestStandaloneAssetsExist` 检查。

## 验收清单

- [ ] 三类 runtime_mode 的徽章视觉可区分
- [ ] 质量门 status 在上传结果中可见
- [ ] training_placeholders 可作为 JSON 下载
- [ ] `pytest plugins/bird_monitoring/tests/test_standalone.py` 通过
- [ ] 仿真页仍显式标记 `standalone_simulation`
