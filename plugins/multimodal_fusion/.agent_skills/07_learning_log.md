# 07_learning_log

用于记录当前插件已核验的经验、暴露出的契约问题，以及后续修复时必须记住的教训。

## Entry Template

- Date:
- Context:
- Symptom:
- Root cause:
- Fix:
- Prevention:
- Follow-up:

## Entries

- Date: 2026-04-09
- Context: standalone 默认配置回退核验
- Symptom: 插件目录没有 `configs/default.yaml`，但 `create_standalone()` 仍然成功
- Root cause: SDK 配置加载缺文件时返回空字典，实例回到默认 `MultimodalConfig`
- Fix: 在 skill 中明确写成“默认配置兜底”，而不是“已有配置文件”
- Prevention: 若需要可调配置，先补真实 YAML，再验证 `_parse_config()`
- Follow-up: 未来补 YAML 时必须同时检查 `max_history_length` / `modality_dims`

---

- Date: 2026-04-09
- Context: 增强引擎回放
- Symptom: demo 和常见 `{status, confidence}` 输入下，增强引擎会失败并回退到基础融合
- Root cause: 当前模态数据转 `features` 的逻辑会把包含状态字符串的 dict 值尝试转成数值数组
- Fix: 当前 skill 合同中把增强引擎降级为“优先尝试但不稳定”
- Prevention: 增强引擎必须有明确的输入规范和回归测试
- Follow-up: 修复后补 `tests/test_enhanced_engine_fallback.py`

---

- Date: 2026-04-09
- Context: 配置契约核验
- Symptom: 传入 `max_history_length=5` 后，config 仍保留默认 100
- Root cause: `_parse_config()` 当前不消费该字段
- Fix: 在当前 skill 中把 manifest 默认配置与实现错位明确化
- Prevention: manifest 中出现的默认配置项，必须有代码消费证据
- Follow-up: 修复后补 `tests/test_config_contract.py`

---

- Date: 2026-04-09
- Context: 多实现并存核验
- Symptom: 目录内同时存在 `plugin.py`、`fusion_engine.py`、`fusion_engine_enhanced.py`、`plugin_v4_bayesian.py`
- Root cause: 当前目录承载了现行实现、增强引擎和候选未来实现
- Fix: 当前 skill 体系中把 `plugin_v4_bayesian.py` 降级为“并行实现/演进方向”
- Prevention: 只有 manifest 指向的实现，才能写成当前现状
- Follow-up: 若未来切主实现，必须同步更新 `00/02/04/08`
