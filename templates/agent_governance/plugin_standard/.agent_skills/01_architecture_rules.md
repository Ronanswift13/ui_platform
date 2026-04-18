# 01 架构规则 — {{PLUGIN_DISPLAY_NAME}}

## 架构不变量

1. `plugin.py` 仅做 SDK 适配与编排，不承载核心算法。
2. 所有配置参数必须从 `configs/default.yaml` 加载，不得硬编码。
3. 日志使用 `logging` 模块，禁止 `print()`。
4. 异常不得静默吞没。

## 反模式检查

```bash
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py
rg -n "\bprint\(" plugin.py
```

<!-- BUSINESS: 补充本插件专属架构规则 -->
