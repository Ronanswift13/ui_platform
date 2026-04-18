# 08_task_routing

本文件定义 `hyperspectral_detection` 当前的最小任务路由。

## 0. 当前路由前提

当前插件有：

1. `plugin.py`
2. `standalone/`
3. `demo/run_demo.py`

当前插件没有：

1. `configs/default.yaml`
2. `run_standalone.py`
3. 自动化测试用例
4. 任何现成脚本

因此所有任务都应从“最小治理级”路线出发，先验证真实入口，再扩展治理。

## 1. 通用前置（所有任务共享）

### 必读

1. `plugin.py`
2. `manifest.json`
3. `requirements.txt`
4. `.agent_skills/00_project_context.md`
5. `.agent_skills/01_architecture_rules.md`

### 阻断检查

- 若任务默认假设存在 `detector.py` -> 先停止并纠正
- 若任务默认假设存在 `configs/default.yaml` -> 先停止并核实
- 若任务默认假设 `run_standalone.py` 可执行 -> 先停止并核实
- 若任务要改 `process()` 契约 -> 需要平台侧确认

## 2. implement（功能实现）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/03_test_strategy.md`

### 执行顺序

1. 先判断改动属于：
   - 配置契约
   - 光谱处理
   - 占位能力接入
   - standalone 服务壳
   - 测试/脚本治理
2. 若是代码改动，先补最小验证
3. 若涉及高光谱数组处理，必须覆盖形状回归
4. 若涉及配置，优先对齐 manifest 与 `_parse_config()`

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
   - simulated process
   - spectrum 形状处理
   - standalone 启动
2. 最小因果修复
3. 运行 demo 回放
4. 追加回归测试或脚本检查

## 4. audit（质量审计）

### 必读

1. 通用前置
2. `.agent_skills/04_quality_audit.md`
3. `.agent_skills/05_security_boundary.md`

### 执行顺序

1. 运行 `04` 中的审计命令
2. 运行 demo 回放
3. 视需要复现形状误判
4. 输出分级结论：
   - 阻断
   - 高风险
   - 建议

## 5. upgrade（治理升级）

当目标是把插件从“最小治理级”升级到“可持续开发级”时，优先顺序固定为：

1. 新建 `scripts/run_sanity_checks.sh`
2. 新建 `tests/test_plugin_process.py`
3. 补 `configs/default.yaml`
4. 对齐配置契约
5. 修 band 轴处理
6. 再考虑真实模型/PCA/分割能力

在前五项完成前，不要伪装成“已有完整质量门禁”的插件。

## 6. 快速参考

| 任务 | 最小验证命令 |
|---|---|
| implement | `python3 -m plugins.hyperspectral_detection.demo.run_demo` |
| repair | demo 回放 + 形状复现 |
| audit | `rg` 审计 + demo 回放 |

## 7. 后续最值得补的第一个脚本

建议文件：

`scripts/run_sanity_checks.sh`

建议包含：

1. import 检查
2. `init()` 检查
3. `process()` 模拟回退检查
4. demo 回放检查
5. 缺失 `run_standalone.py`/`configs/default.yaml` 的显式提醒
6. band 轴形状回归检查

当前不建议先补复杂 regression / quality gate 脚本，因为基础事实校验仍然缺失。
