# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：只读/只改当前插件目录。
2. 高风险操作先确认：系统配置、跨仓库批量改动。
3. 日志/测试样本不落敏感信息。
4. 破坏性命令默认禁止。

## 2. 本项目文件边界

- **允许自动改动**：
  - `plugins/device_monitoring/.agent_skills/**`
  - `plugins/device_monitoring/tests/**`
  - `plugins/device_monitoring/scripts/**`
- **需人工确认后改动**：
  - `plugin.py`、`detector.py`
  - `configs/default.yaml`
  - `standalone/**`
- **禁止改动**：
  - `manifest.json` 的 `id/entrypoint/plugin_class`
  - `darkbreaker_sdk/**`

## 3. 特殊安全关注

- **协议接入**：`configs/default.yaml` 中 `protocols.snmp_enabled` / `modbus_enabled` 默认关闭。启用 SNMP/Modbus 涉及网络扫描，必须人工确认。
- **工单通知**：`maintenance.notification_channels` 含 `email`，发送外部通知需确认配置。
- **数据存储**：`data/results.db` 仅存检测结果摘要，不存储设备凭证或敏感配置。

## 4. 依赖安全

- 核心依赖仅 `numpy`，无网络请求依赖。
- SNMP/Modbus 协议库为可选依赖，当前未在 `requirements.txt` 中声明。
