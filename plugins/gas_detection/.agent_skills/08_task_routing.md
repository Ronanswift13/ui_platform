# 08_task_routing

本文件定义 `gas_detection` 当前的标准治理基线路由。

## 0. 当前路由前提

当前插件有：

1. `plugin.py`
2. `predictor.py`
3. `analyzer.py`
4. `standalone/`
5. `demo/run_demo.py`
6. `configs/default.yaml`
7. `tests/test_config_contract.py`
8. `tests/test_process_contract.py`
9. `tests/test_trend_contract.py`
10. `tests/test_standalone.py`
11. `scripts/run_sanity_checks.sh`

当前没有 targeted/regression/coverage 门禁，不得伪造这些能力。

## 1. 通用前置（所有任务共享）

### 必读

1. `plugin.py`
2. `predictor.py`
3. `manifest.json`
4. `.agent_skills/00_project_context.md`
5. `.agent_skills/01_architecture_rules.md`

### 阻断检查

- 若任务默认假设存在 `detector.py` -> 先停止并纠正
- 若任务要改 `process()` 契约 -> 需要平台侧确认
- 若任务要把 standalone 暴露到真实网络 -> 需要人工确认

## 2. implement（功能实现）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/03_test_strategy.md`

### 执行顺序

1. 先判断改动属于：
   - 阈值/配置
   - 预测/泄漏检测
   - standalone 服务壳
   - 测试/脚本治理
2. 若是代码改动，先补最小验证
3. 若涉及预测路径，必须覆盖“24 样本历史”
4. 若涉及配置，必须同步 `configs/default.yaml` + `_parse_config()` + config contract test
5. 运行 `cd plugins/gas_detection && ./scripts/run_sanity_checks.sh`

## 3. repair（缺陷修复）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/06_refactor_policy.md`
4. `.agent_skills/07_learning_log.md`

### 执行顺序

1. 先定位故障属于哪条链路：
   - import
   - init
   - sample process
   - 24 样本预测
   - standalone 启动
2. 最小因果修复
3. 运行 `cd plugins/gas_detection && ./scripts/run_sanity_checks.sh`
4. 追加回归测试或脚本检查

## 4. audit（质量审计）

### 必读

1. 通用前置
2. `.agent_skills/04_quality_audit.md`
3. `.agent_skills/05_security_boundary.md`

### 执行顺序

1. 运行 `04` 中的 `rg` 审计命令
2. 运行 `cd plugins/gas_detection && ./scripts/run_sanity_checks.sh`
3. 视需要复现 24 样本趋势输出合同
4. 输出分级结论：
   - 阻断
   - 高风险
   - 建议

## 5. upgrade（治理升级）

当目标是把插件从“标准治理基线”升级到“可持续开发级”时，优先顺序固定为：

1. 已完成 `scripts/run_sanity_checks.sh`
2. 已完成 process/config/trend/standalone tests
3. 已补 `configs/default.yaml`
4. 已修复 24 样本趋势配置契约断裂
5. 后续再考虑 DGA 详细分析、写库、服务增强

当前仍不要伪装成“已有完整质量门禁”的插件，因为 targeted/regression/coverage 门禁尚未建立。

## 6. 快速参考

| 任务 | 最小验证命令 |
|---|---|
| implement | `cd plugins/gas_detection && ./scripts/run_sanity_checks.sh` |
| repair | `cd plugins/gas_detection && ./scripts/run_sanity_checks.sh` |
| audit | `rg` 审计 + `cd plugins/gas_detection && ./scripts/run_sanity_checks.sh` |

## 7. 后续最值得补的脚本

当前 `scripts/run_sanity_checks.sh` 已存在。后续建议文件：

- `scripts/run_targeted_tests.sh`
- `scripts/run_quality_gate.sh`

建议包含：

1. import / py_compile 检查
2. process/trend/config 快速合同
3. 反模式扫描
4. 版本一致性检查
5. 24 样本预测链路检查

当前不建议先补复杂 regression 脚本，因为主链路合同仍应保持小而稳定。
