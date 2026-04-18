# 07 — 配置层级体系

## 配置层级 (高优先级覆盖低优先级)

```
环境变量 / CLI 参数
    ↓ 覆盖
configs/platform.yaml          # 平台全局配置
    ↓ 覆盖
configs/plugins_config.yaml    # 插件注册 + 插件级参数
    ↓ 覆盖
plugins/<name>/configs/default.yaml  # 插件默认配置
```

## 关键配置文件

| 文件 | 职责 |
|---|---|
| `configs/platform.yaml` | 平台端口/日志/全局开关 |
| `configs/plugins_config.yaml` | 插件启用列表 + 加载顺序 + 各插件参数 |
| `configs/enhanced_config.yaml` | 增强功能配置 |
| `configs/indoor_config.yaml` | 室内围栏专用配置 |
| `configs/models_config.yaml` | 模型注册表 |
| `configs/training_config.yaml` | 训练管道配置 |
| `configs/station_zones.yaml` | 站区分区定义 |
| `configs/voltage_config.yaml` | 电压等级配置 |
| `configs/devices/` | 设备适配配置 |
| `configs/rules/` | 规则引擎配置 |
| `configs/sites/` | 站点配置 |
| `configs/tasks/` | 任务模板 |

## 配置格式

- 全部使用 **YAML** 格式
- 编码: UTF-8
- 注释: `#` 行注释，说明每个参数含义
- 禁止在配置文件中硬编码密钥/凭证

## 新增配置的规则

1. 全局参数 → 加到 `configs/platform.yaml`
2. 插件参数 → 先加到 `plugins/<name>/configs/default.yaml`，再在 `plugins_config.yaml` 中覆盖
3. 新配置文件 → 必须在本文件中登记
