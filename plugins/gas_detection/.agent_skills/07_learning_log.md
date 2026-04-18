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

- Date: 2026-04-17
- Context: B 类时序传感插件主链路一致性收口，回灌 predictor/analyzer/plugin 调用关系与趋势输出合同。
- Symptom: skill 仍记录 `configs/default.yaml` 缺失、tests/scripts 为空、24 样本趋势会因 `prediction_horizon` 缺失失败；当前代码已补配置、测试、sanity 脚本，并把 predictor + analyzer 纳入 `process()` 主输出。
- Root cause: 配置与趋势修复先落地，agent skill 没有同步，导致后续 agent 可能重复修复已闭合的问题。
- Fix: 更新 `00/02/03/08`，明确 `predictions`、`trend_analysis`、`trend_diagnosis` 是当前主合同；24 样本路径应返回传统预测/分析结果而不是失败。
- Prevention: 后续改动模型配置、history window 或 analyzer 接入时，必须同步 `test_trend_contract.py`、`configs/default.yaml` 和 `02_algorithm_contract.md`。
- Follow-up: DGA 详细分析和 `data/results.db` 写入仍是第二阶段入口，不得写成当前已落地能力。

---

- Date: 2026-04-09
- Context: `create_standalone()` 与独立服务入口核验
- Symptom: 历史阶段插件目录没有 `configs/default.yaml`，但 `create_standalone()` 仍能成功
- Root cause: `darkbreaker_sdk.utils.load_plugin_config()` 在文件缺失时返回空字典，插件随后回退到 `GasDetectionConfig()` 默认值
- Fix: 已在 2026-04-17 补齐真实 `configs/default.yaml` 并纳入 config contract test；该条保留为历史教训
- Prevention: 未来调阈值、历史窗口、预测窗口，必须同步 YAML、dataclass、解析逻辑和测试
- Follow-up: 已验证 `create_standalone()` 可消费默认配置；后续关注新增配置字段消费证据

---

- Date: 2026-04-09
- Context: 24 样本历史回放
- Symptom: 历史阶段同一设备累计 24 条小时级样本后，`process()` 返回失败
- Root cause: `predictor.py` 访问 `self.config.prediction_horizon`，但当前 `GasDetectionConfig` 和 `_parse_config()` 没有完整承接该字段
- Fix: 已在 2026-04-17 对齐 dataclass、manifest、YAML、解析逻辑，并由 `tests/test_trend_contract.py` 回归锁住
- Prevention: 对所有 manifest 默认配置项，必须有代码消费证据
- Follow-up: 继续审查阈值对象访问方式和新增模型字段兼容性

---

- Date: 2026-04-09
- Context: 功能边界核验
- Symptom: 目录里有 `analyzer.py` 和 `data/results.db`，容易让人误以为 DGA 已接入、结果已持久化
- Root cause: 代码资产存在，但历史阶段 `process()` 主链路没有实际消费分析器，也未见写库逻辑
- Fix: 2026-04-17 已将 `GasDataAnalyzer.analyze_trends` 接入 `trend_analysis` 主输出；`data/results.db` 写入和 DGA 详细分析仍保持预留能力
- Prevention: 只有出现在主调用链里的模块，才能写进“当前输出合同”
- Follow-up: 若未来接入 DGA 详细分析或写库，再同步更新 `02/04/08`

---

- Date: 2026-04-09
- Context: 元数据一致性核验
- Symptom: `manifest.json.version` 与 `plugin.py.PLUGIN_VERSION` 不一致
- Root cause: 元数据维护未同步
- Fix: 后续版本变更时统一更新声明层和实现层
- Prevention: 把版本一致性纳入 sanity 检查
- Follow-up: 建议在首个 sanity 脚本中加入版本一致性检查
