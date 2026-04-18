# 01 架构规则 — {{PLUGIN_DISPLAY_NAME}}

## 架构不变量

1. `plugin.py` 仅做 SDK 适配与编排，不承载核心算法。
2. `{{DETECTOR_FILE}}` 不得依赖 `darkbreaker_sdk` 或 `standalone`。
3. 所有配置参数必须从 `configs/default.yaml` 加载，不得硬编码到推理路径。
4. 日志使用 `logging` 模块，生产路径禁止 `print()`。
5. 异常不得静默吞没（禁止 `except: pass` 和 `except Exception: pass`）。

## 模块权限矩阵

| 模块 | 可读 | 可写 | 禁止依赖 |
|------|------|------|----------|
| plugin.py | configs, {{DETECTOR_FILE}} | 无 | standalone |
| {{DETECTOR_FILE}} | configs | 无 | darkbreaker_sdk, standalone |
| standalone/ | plugin.py, configs | templates | {{DETECTOR_FILE}} 内部 |
| tests/ | 所有模块 | fixtures/ | 生产配置 |

## 反模式检查命令

```bash
# 检查 detector 是否引入了禁止依赖
rg -n "darkbreaker_sdk|standalone" {{DETECTOR_FILE}}

# 检查静默异常
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py {{DETECTOR_FILE}}

# 检查生产路径 print
rg -n "\bprint\(" plugin.py {{DETECTOR_FILE}}
```

<!-- BUSINESS: 补充本插件专属的层级边界规则（如 ROI 隔离、编解码分离等） -->
