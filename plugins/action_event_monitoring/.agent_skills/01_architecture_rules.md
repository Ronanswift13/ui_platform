# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. 事实源优先级固定：`plugin.py` > `manifest.json` > `configs/*.yaml` > 历史经验文档。
2. 未落地能力不得写成既有能力。
3. 配置与拓扑必须保持本地文件驱动，不在主逻辑中硬编码站点台账。
4. 协议适配与业务编排要分层，即使当前仍在单文件中，也不能继续把职责揉得更乱。

## 2. 当前架构事实（action_event_monitoring）

1. 当前插件是“单文件编排器”架构：
   - `plugin.py` 同时承担生命周期管理、事件归一化、拓扑加载、协议回调、分析触发、状态输出。
2. `manifest.json` 已声明 `entrypoint`、`plugin_class`、`config_file` 与 `standalone` 元数据。
3. `configs/default.yaml` 提供协议、订阅、分析、告警参数。
4. `configs/topology/*.yaml` 提供厂站和信号点映射。
5. 当前已有本地 standalone app 与 smoke route；没有 UI/dashboard/cockpit 前端统一入口，也没有 CandidateEvent/人工复核 REST API。

## 3. 当前模块职责（必须保持）

1. `init()`：
   - 创建 `ActionEventStore`
   - 创建 `DeviceCorrelationService`
   - 创建 `ActionSequenceAnalyzer`
   - 创建 `RootCauseService`
   - 加载拓扑文件
2. `start()`：
   - 标记运行状态
   - 在配置了协议类型时尝试初始化协议适配器
3. `process()`：
   - 归一化输入
   - 存储事件
   - 根据配置决定是否触发动作链和根因分析
4. `stop()/shutdown()`：
   - 关闭协议连接
   - 结束运行状态
5. `get_status()`：
   - 输出插件状态和存储统计

## 4. 当前架构红线

1. 不得在 skill 中把 standalone smoke 描述成完整 Web/API/前端平台接线。
2. 不得把 `07_learning_log.md` 中提到的 CandidateEvent / 复核 API 当成当前本地已存在实现。
3. 不得把当前 tests 描述成 coverage / regression 质量门。
4. 不得继续扩大 `plugin.py` 的职责范围而不补对应合同测试。
5. 不得把协议写操作或控制指令接入到当前插件中；当前角色应保持“采集与分析触发”。

## 5. 当前可接受的最小演进方向

1. 维护 `scripts/run_sanity_checks.sh`
2. 维护 config/process/entrypoints/standalone 合同测试
3. 将协议回调和事件归一化逻辑逐步拆为私有 helper 或独立模块
4. 把“协议未连通但 start 成功”的行为明确化或修正
5. 获得 UI 目录授权后，再接 dashboard/cockpit 入口

## 6. 高风险改动（需人工确认）

1. 改动生命周期方法名或语义
2. 改动 `process()` 的输入模式
3. 改动分析触发条件
4. 改动默认协议配置或实际外部依赖
5. 引入新的 REST API、UI/cockpit 入口或外部网络暴露策略
