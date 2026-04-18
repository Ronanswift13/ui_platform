# 04 质量审计 — {{PLUGIN_DISPLAY_NAME}}

## 审计规则

- [ ] 无静默异常
- [ ] 无生产路径 `print()`
- [ ] 配置从 `configs/default.yaml` 加载
- [ ] 测试存在且可运行

## 审计命令

```bash
python -m pytest tests/ -q
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py
rg -n "\bprint\(" plugin.py
```

<!-- BUSINESS: 补充本插件专属审计项 -->
