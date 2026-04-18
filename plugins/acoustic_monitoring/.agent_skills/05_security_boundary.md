# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：只读/只改当前插件目录。
2. 高风险操作先确认：系统配置、跨仓库批量改动。
3. 日志/测试样本不落敏感信息。
4. 破坏性命令默认禁止。

## 2. 本项目文件边界

- **允许自动改动**：
  - `plugins/acoustic_monitoring/.agent_skills/**`
  - `plugins/acoustic_monitoring/tests/**`
  - `plugins/acoustic_monitoring/scripts/**`
- **需人工确认后改动**：
  - `plugin.py`、`detector.py`、`analyzer.py`
  - `configs/default.yaml`
  - `standalone/**`
- **禁止改动**：
  - `manifest.json` 的 `id/entrypoint/plugin_class`
  - `darkbreaker_sdk/**`（上游 SDK）

## 3. 数据安全

- 音频数据不得持久化到未授权路径。
- `data/results.db` 仅存储检测结果摘要，不存储原始波形。
- WebSocket 广播的波形为降采样后的可视化数据，非原始采样。

## 4. 依赖安全

- 核心依赖仅 `numpy` + `onnxruntime`，无网络请求依赖。
- 不允许引入需要外部网络连接的依赖。
