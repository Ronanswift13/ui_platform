# 07_learning_log

用于记录当前插件已核验的经验、已暴露的运行边界，以及后续修复时必须记住的教训。

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
- Symptom: 插件目录没有 `configs/default.yaml`，但 `create_standalone()` 仍能成功
- Root cause: 当前独立启动路径会回退到默认 `HyperspectralConfig`
- Fix: 在 skill 中把这写成“默认配置回退”，而不是“已有配置文件”
- Prevention: 若要让配置真正可调，优先补真实 `configs/default.yaml`
- Follow-up: 新增 YAML 后，需验证 `_parse_config()` 是否真的消费这些键

---

- Date: 2026-04-09
- Context: manifest 默认配置契约核验
- Symptom: 传入 `spectral_bands` / `wavelength_start_nm` / `wavelength_end_nm` 等 manifest 风格键时，配置对象几乎不变
- Root cause: `_parse_config()` 只读取 `wavelength_range` 和 `num_bands`
- Fix: 在当前 skill 合同里明确标注 manifest 键名漂移
- Prevention: manifest 中出现的默认配置项，必须有代码消费证据
- Follow-up: 修复后要补 `tests/test_config_contract.py`

---

- Date: 2026-04-09
- Context: 光谱维度推断回放
- Symptom: `(32,32,224)` 与 `(224,32,32)` 等 3D 输入会产出 32 长度光谱，而不是 224
- Root cause: 当前 band 轴判断依赖 `shape[0] < shape[2]` 的简化启发式
- Fix: 后续应显式定义输入形状契约，或增加可靠的 band 轴参数/判断逻辑
- Prevention: 任何高维数组处理都要有形状回归测试
- Follow-up: 将该问题固定到 sanity 脚本与形状测试中

---

- Date: 2026-04-09
- Context: 入口与能力文案核验
- Symptom: `__main__.py` usage 提到了 `run_standalone.py`，但文件不存在；`analysis_type` 和 `_model_registry` 存在却未生效
- Root cause: 文案和预留字段超前于当前实现
- Fix: 在 skill 中把这些内容降级为“当前未落地/未生效”
- Prevention: 只有接入主链路的字段和入口，才能写成现状
- Follow-up: 后续若落地真实分支或新增脚本，再同步更新 `02/04/08`
