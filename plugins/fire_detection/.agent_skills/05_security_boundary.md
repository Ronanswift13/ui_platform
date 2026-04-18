# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：只读/只改当前插件目录。
2. 高风险操作先确认：系统配置、跨仓库批量改动。
3. 日志/测试样本不落敏感信息。
4. 破坏性命令默认禁止。

## 2. 本项目文件边界

- **允许自动改动**：
  - `plugins/fire_detection/.agent_skills/**`
  - `plugins/fire_detection/tests/**`
  - `plugins/fire_detection/scripts/**`
- **需人工确认后改动**：
  - `plugin.py`、`detector.py`
  - `configs/default.yaml`
  - `standalone/**`
- **禁止改动**：
  - `manifest.json` 的 `id/entrypoint/plugin_class`
  - `darkbreaker_sdk/**`

## 3. 物理设备安全（本插件核心安全边界）

本插件可触发物理设备动作，这是所有 DarkBreaker 插件中安全等级最高的之一：

| 动作 | 配置开关 | 默认值 | 安全约束 |
|------|----------|--------|----------|
| 自动喷淋 | `suppression.auto_sprinkler_enabled` | **false** | 必须保持默认关闭，启用需人工确认 |
| 自动断电 | `suppression.auto_power_cutoff` | **false** | 必须保持默认关闭，启用需人工确认 |
| 声光报警 | `suppression.alarm_sound/light_enabled` | true | 低风险，可自动启用 |
| 灭火触发等级 | `suppression.trigger_level` | "alarm" | 调低需人工确认 |

**硬约束**：任何修改灭火联动逻辑或配置的改动必须人工确认。detector 层不得直接控制物理设备。

## 4. 数据安全

- `data/results.db` 仅存检测结果摘要。
- `history.snapshot_on_alarm` 报警截图存储需确保不泄露敏感区域画面。
- 视频剪辑（`video_clip_seconds`）应存储在受控路径。

## 5. 依赖安全

- 核心依赖：`numpy`, `opencv-python`, `onnxruntime`（无网络请求）。
- 不允许引入需外部网络连接的运行时依赖。
