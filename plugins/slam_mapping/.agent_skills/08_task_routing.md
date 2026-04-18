# 08_task_routing

本文件定义 `slam_mapping` 当前的最小任务路由。

## 0. 当前路由前提

当前插件有：

1. `plugin.py`
2. `semantic_slam_plugin.py`
3. `standalone/`
4. `run_standalone.py`
5. `demo/run_demo.py`
6. `tests/test_standalone.py`

当前插件没有：

1. `configs/default.yaml`
2. `scripts/`

因此当前任务应走“基础治理”路线：先锁住主实现合同，再扩展治理。

## 1. 通用前置（所有任务共享）

### 必读

1. `plugin.py`
2. `manifest.json`
3. `tests/test_standalone.py`
4. `.agent_skills/00_project_context.md`
5. `.agent_skills/01_architecture_rules.md`

### 阻断检查

- 若任务默认假设存在标准 `process()` -> 先停止并核实
- 若任务默认假设 `semantic_slam_plugin.py` 已接入 manifest -> 先停止并纠正
- 若任务要改 `process_point_cloud()` 返回结构 -> 需要平台侧确认
- 若任务要启用地图导出到生产路径 -> 需要人工确认

## 2. implement（功能实现）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/03_test_strategy.md`

### 执行顺序

1. 先判断改动属于：
   - 配置/生命周期合同
   - 点云算法链路
   - 地图/路径/沉降服务接口
   - 测试/脚本治理
2. 若改动 `plugin.py`，先补最小验证
3. 若涉及 `semantic_slam_plugin.py`，必须先确认是否只是参考，不得偷偷切主链
4. 若涉及导出或路径规划，补反例测试

## 3. repair（缺陷修复）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/06_refactor_policy.md`
4. `.agent_skills/07_learning_log.md`

### 执行顺序

1. 先定位故障属于哪条链路：
   - 初始化/健康检查
   - process_point_cloud
   - 路径规划
   - 沉降监测
   - demo / standalone
2. 最小因果修复
3. 运行 smoke 测试 + demo
4. 追加回归测试或 sanity 检查

## 4. audit（质量审计）

### 必读

1. 通用前置
2. `.agent_skills/04_quality_audit.md`
3. `.agent_skills/05_security_boundary.md`

### 执行顺序

1. 运行 `04` 中的契约复现命令
2. 运行 smoke 测试
3. 运行 demo 回放
4. 输出分级结论：
   - 阻断
   - 高风险
   - 建议

## 5. upgrade（治理升级）

当目标是把插件从“基础治理”升级到“可持续开发级”时，优先顺序固定为：

1. 新建 `scripts/run_sanity_checks.sh`
2. 新建 `tests/test_plugin_contract.py`
3. 修复 `init()` / `healthcheck()` / `shutdown()`
4. 再考虑统一 `process()` 包装层
5. 最后再考虑切换到 `semantic_slam_plugin.py`

在前四项完成前，不要伪装成“契约已经稳定”的插件。

## 6. 快速参考

| 任务 | 最小验证命令 |
|---|---|
| implement | `python3 -m pytest plugins/slam_mapping/tests/test_standalone.py -q` |
| repair | smoke + demo + 初始化边界复现 |
| audit | 契约复现 + smoke + demo |

## 7. 后续最值得补的第一个脚本

建议文件：

`scripts/run_sanity_checks.sh`

建议包含：

1. smoke 测试
2. demo 回放
3. `process_point_cloud()` 最小样本检查
4. `init(dict)` 配置误判检查
5. `shutdown()` 后健康检查一致性检查

当前不建议先补复杂 regression / benchmark 脚本，因为最核心的合同问题还没有锁住。
