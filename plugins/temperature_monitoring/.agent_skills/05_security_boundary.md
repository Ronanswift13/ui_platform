# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：只读/只改当前插件目录。
2. 高风险操作先确认：系统配置、跨仓库批量改动。
3. 日志/测试样本不落敏感信息。
4. 破坏性命令默认禁止。

## 2. 本项目文件边界

- **允许自动改动**：
  - `plugins/temperature_monitoring/.agent_skills/**`
  - `plugins/temperature_monitoring/tests/**`
  - `plugins/temperature_monitoring/scripts/**`
- **需人工确认后改动**：
  - `plugin.py`、`detector.py`
  - `configs/default.yaml`
  - `standalone/**`
- **禁止改动**：
  - `manifest.json` 的 `id/entrypoint/plugin_class`
  - `darkbreaker_sdk/**`

## 3. 特殊安全关注

- **跨模块联动**：`linkage.fire_detection_enabled` / `fence_plugin_enabled` 会触发其他插件的联动事件。修改联动逻辑需确认下游插件兼容。
- **通风控制**：`linkage.ventilation_control` 默认 `false`。启用涉及物理设备控制，必须人工确认。
- **数据存储**：`data/results.db` 仅存温度统计摘要，不存储原始热成像帧。

## 4. 依赖安全

- 核心依赖：`numpy`，`opencv-python`（可选）。
- LSTM 预测为可选模型（`models/temperature/temp_lstm_predictor.onnx`），默认走 linear 方法。
- 不允许引入需外部网络连接的运行时依赖。
