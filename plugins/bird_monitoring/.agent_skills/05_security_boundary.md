# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：只读/只改当前插件目录。
2. 高风险操作先确认：系统配置、跨仓库批量改动。
3. 日志/测试样本不落敏感信息。
4. 破坏性命令默认禁止。

## 2. 本项目文件边界

- **允许自动改动**：
  - `plugins/bird_monitoring/.agent_skills/**`
  - `plugins/bird_monitoring/tests/**`
  - `plugins/bird_monitoring/docs/**`
  - `plugins/bird_monitoring/prompts/**`
  - `plugins/bird_monitoring/scripts/**`（待新建）
- **需人工确认后改动**：
  - `plugin.py`、`detector.py`、`experimental/advanced_bird_detector.py`
  - `configs/default.yaml`
  - `standalone/**`
- **禁止改动**：
  - `manifest.json` 的 `id/entrypoint/plugin_class`
  - `darkbreaker_sdk/**`

## 3. 特殊安全关注

- **驱鸟设备控制**：当前为 blocked。`RepelController` 不得触发物理设备；插件只输出 `deterrent_suggestion`。
- **外部 API**：`configs/default.yaml` 中 `bird_database.online_api` 配置了外部接口（默认关闭）。启用外部 API 需人工确认。
- **数据存储**：`data/results.db` 仅存检测结果摘要，不应存储原始图像。

## 4. 依赖安全

- 核心依赖：`numpy`, `opencv-python`, `onnxruntime`（无网络请求）。
- 可选依赖：`torch`（`experimental/advanced_bird_detector.py`）。
- 不允许引入需要外部网络连接的运行时依赖。
