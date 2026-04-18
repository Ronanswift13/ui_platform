# 08_task_routing

本文件定义 `multimodal_fusion` 当前的最小任务路由。

## 0. 当前路由前提

当前插件有：

1. `plugin.py`
2. `fusion_engine.py`
3. `fusion_engine_enhanced.py`
4. `plugin_v4_bayesian.py`
5. `standalone/`
6. `demo/run_demo.py`

当前插件没有：

1. `configs/default.yaml`
2. `run_standalone.py`
3. 自动化测试用例
4. 任何现成脚本

因此当前任务应走“最小治理级”路线：先锁住当前主实现合同和回退行为，再扩展治理。

## 1. 通用前置（所有任务共享）

### 必读

1. `plugin.py`
2. `manifest.json`
3. `demo/run_demo.py`
4. `.agent_skills/00_project_context.md`
5. `.agent_skills/01_architecture_rules.md`

### 阻断检查

- 若任务默认假设 `plugin_v4_bayesian.py` 已接入 manifest -> 先停止并纠正
- 若任务默认假设 `pre_processed` / `modality_results` 当前已生效 -> 先停止并核实
- 若任务默认假设存在 `run_standalone.py` -> 先停止并核实
- 若任务要改 `process()` 返回合同 -> 需要平台侧确认

## 2. implement（功能实现）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/03_test_strategy.md`

### 执行顺序

1. 先判断改动属于：
   - 融合合同/输入规范
   - 增强引擎修复
   - 外部模态插件接入
   - 测试/脚本治理
2. 若改动 `plugin.py` 或增强引擎，先补最小验证
3. 若涉及 `plugin_v4_bayesian.py`，必须先确认只是参考还是要切主链
4. 若涉及 manifest，核对 schema 是否与实现同步

## 3. repair（缺陷修复）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/06_refactor_policy.md`
4. `.agent_skills/07_learning_log.md`

### 执行顺序

1. 先定位故障属于哪条链路：
   - init / healthcheck
   - process 基础融合
   - 增强引擎回退
   - 模态插件注册
   - demo / standalone
2. 最小因果修复
3. 运行 demo 回放
4. 追加回归测试或 sanity 检查

## 4. audit（质量审计）

### 必读

1. 通用前置
2. `.agent_skills/04_quality_audit.md`
3. `.agent_skills/05_security_boundary.md`

### 执行顺序

1. 运行 `04` 中的审计命令
2. 运行 demo 回放
3. 视需要复现增强引擎失败与回退
4. 输出分级结论：
   - 阻断
   - 高风险
   - 建议

## 5. upgrade（治理升级）

当目标是把插件从“最小治理级”升级到“可持续开发级”时，优先顺序固定为：

1. 新建 `scripts/run_sanity_checks.sh`
2. 新建 `tests/test_plugin_process.py`
3. 对齐 `_parse_config()` 与 manifest 默认配置
4. 修复增强引擎输入规范
5. 再决定是否切换到贝叶斯主链路

在前四项完成前，不要伪装成“融合合同已经稳定”的插件。

## 6. 快速参考

| 任务 | 最小验证命令 |
|---|---|
| implement | `python3 -m plugins.multimodal_fusion.demo.run_demo` |
| repair | demo 回放 + 增强引擎回退复现 |
| audit | `rg` 审计 + demo 回放 |

## 7. 后续最值得补的第一个脚本

建议文件：

`scripts/run_sanity_checks.sh`

建议包含：

1. `init()` 检查
2. 缺模态报错检查
3. 基础 `process()` 检查
4. demo 回放检查
5. 增强引擎失败回退检查
6. 缺失 `run_standalone.py` / `configs/default.yaml` 提醒

当前不建议先补复杂 regression 或 benchmark 脚本，因为最核心的合同和回退行为还没有锁住。
