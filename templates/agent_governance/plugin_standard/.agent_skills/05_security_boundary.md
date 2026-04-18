# 05 安全边界 — {{PLUGIN_DISPLAY_NAME}}

## 文件边界

- 可读写: `plugins/{{PLUGIN_NAME}}/` 内所有文件
- 只读: `platform_core/`、`darkbreaker_sdk/`
- 禁止: 其他插件目录

## 行为边界

- 不访问外部网络
- 不持久化原始数据到未授权目录
- 日志中不输出敏感标识符
