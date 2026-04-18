# 06_refactor_policy

## 1. 当前重构原则

1. 先修合同和状态一致性，再做算法升级。
2. 先补最小脚本与测试，再考虑切换到语义 SLAM。
3. 当前以 `plugin.py` 主链路为准，不跨越到 `semantic_slam_plugin.py`。
4. 任何涉及 `process_point_cloud()` 的改动都必须有回归验证。

## 2. 允许的低风险重构

1. 补齐 `.agent_skills/00~08`
2. 新建 `scripts/run_sanity_checks.sh`
3. 新增 `tests/test_plugin_contract.py`
4. 修复 demo 字段漂移
5. 同步 `requirements.txt` 与 manifest 依赖

## 3. 中风险重构（建议先补测试）

1. 修复 `init()` 的配置/registry 语义
2. 修复 `shutdown()` / `healthcheck()` / `_is_initialized`
3. 为 `process_point_cloud()` 增加初始化保护
4. 增加统一 `process()` 包装层
5. 让 `data/results.db` 真正接入持久化前先抽象存储层

## 4. 高风险重构（需人工确认）

1. 切换 manifest 到 `semantic_slam_plugin.py`
2. 改写点云处理核心算法或路径规划合同
3. 接入真实语义模型/GPU 推理
4. 开启地图导出到生产路径
5. 改变 standalone 对外暴露方式

## 5. 当前强制流程

1. 先读 `00/01/02/03`
2. 明确改动属于：
   - 文档治理
   - 测试/脚本治理
   - 合同修复
   - 算法实现
3. 若改动 `plugin.py` / `manifest.json`
   - 至少跑 smoke 测试
   - 至少跑 demo 回放
   - 至少复现初始化/健康检查边界
