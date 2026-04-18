# 08_task_routing

本文件定义 `action_event_monitoring` 当前的标准治理基线路由。

## 0. 当前路由前提

当前插件已有：

1. `tests/`
2. `standalone/`
3. `requirements.txt`
4. `demo/run_demo.py`
5. `__main__.py` / `run_standalone.py`
6. `Plugin = ActionEventMonitoringPlugin` 标准别名
7. `scripts/run_sanity_checks.sh`
8. 全局 integration standalone 清单接入
9. installer 分类/端口映射：`monitoring` / `8097`

当前插件没有：

1. `.claude/commands/`
2. targeted/regression/coverage 质量门禁脚本
3. UI/dashboard/cockpit 前端统一入口
4. 真实协议服务依赖下的 smoke

核心验证入口为：

- `./scripts/run_sanity_checks.sh`
- `python3 -m pytest tests -q`

## 1. 通用前置（所有任务共享）

### 必读

1. `manifest.json`
2. `plugin.py`
3. `configs/default.yaml`
4. `.agent_skills/00_project_context.md`
5. `.agent_skills/01_architecture_rules.md`

### 阻断检查

- 若任务声称“已有 CandidateEvent/人工复核/API/UI/cockpit 接口” -> 先停止并核实
- 若任务涉及真实协议联调 -> 需要人工提供环境
- 若任务要把插件升级为 `BasePlugin` 或新增 API 层 -> 需要人工确认

## 2. implement（功能实现）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/03_test_strategy.md`

### 执行顺序

1. 明确改动点属于：
   - 生命周期
   - 事件归一化
   - 协议初始化/回调
   - 分析触发
   - 配置/拓扑
2. 实施最小改动
3. 若改动 `plugin.py` 或 `configs/`，运行 `./scripts/run_sanity_checks.sh`
4. 若改动输出合同、entrypoints 或 standalone，运行 `python3 -m pytest tests -q`
5. 若新增能力超出当前结构（API/UI/cockpit/真实协议 smoke），必须先在结果中明确这是“升级路线”

## 3. repair（缺陷修复）

### 必读

1. 通用前置
2. `.agent_skills/02_algorithm_contract.md`
3. `.agent_skills/06_refactor_policy.md`
4. `.agent_skills/07_learning_log.md`

### 执行顺序

1. 先定位故障属于哪条最小链路：
   - import
   - init
   - start
   - process
   - topology load
2. 最小修复
3. 运行 `./scripts/run_sanity_checks.sh`
4. 必要时运行 `python3 -m pytest tests -q`
5. 把经验回写到 `07_learning_log.md`

## 4. audit（质量审计）

### 必读

1. 通用前置
2. `.agent_skills/04_quality_audit.md`
3. `.agent_skills/05_security_boundary.md`

### 执行顺序

1. 运行 `./scripts/run_sanity_checks.sh`
2. 运行 `python3 -m pytest tests -q`
3. 审查 `plugin.py` 是否继续扩散职责
4. 审查配置是否包含真实凭据
5. 审查历史文档是否又开始伪造未落地能力
6. 审查全局 installer/integration 是否仍包含 `action_event_monitoring`

## 5. upgrade（治理升级）

当目标是把插件从“标准治理基线”升级到“高频开发级”时，优先顺序固定为：

1. 已补 `tests/`
2. 已补本地 `standalone/` 与 smoke
3. 已补本地 entrypoints 与 demo
4. 已补全局 installer/integration 接线
5. 后续补 `scripts/run_targeted_tests.sh`
6. 后续补 `.claude/commands/`
7. 获得授权后再考虑 UI/cockpit、本地 API、CandidateEvent/人工复核接口

在真实协议 smoke、UI/cockpit 接线和候选事件/复核接口未完成前，不要伪装成高频开发级插件。
