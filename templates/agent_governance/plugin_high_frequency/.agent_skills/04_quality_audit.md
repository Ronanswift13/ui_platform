# 04 质量审计 — {{PLUGIN_DISPLAY_NAME}}

## 审计规则

### 高优先级（阻断交付）

- [ ] `{{DETECTOR_FILE}}` 不依赖 SDK / standalone
- [ ] 无静默异常（`except: pass`）
- [ ] 无生产路径 `print()`
- [ ] 所有配置从 `configs/default.yaml` 加载
- [ ] 测试覆盖率达标

### 中优先级（需记录风险）

- [ ] 降级链路有测试覆盖
- [ ] 日志级别合理（无过度 DEBUG）
- [ ] configs/default.yaml 有注释说明

### 低优先级（建议改进）

- [ ] 文档与代码一致
- [ ] 测试命名规范

## 审计命令

```bash
# 一键审计
./scripts/run_quality_gate.sh

# 架构反模式快速扫描
rg -n "darkbreaker_sdk|standalone" {{DETECTOR_FILE}}
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py {{DETECTOR_FILE}}
rg -n "\bprint\(" plugin.py {{DETECTOR_FILE}}
```

<!-- BUSINESS: 补充本插件专属的审计检查项 -->
