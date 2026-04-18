# 06_refactor_policy

## 1. 固定母版规则

1. 先证明问题，再重构。
2. 结构重构与行为变更分离提交。
3. 保持公共契约兼容。
4. 没有测试支撑的重构不允许进入主干。

## 2. 允许的低风险重构

- 将 `detector.py` 中 `cpu_usage > 90` / `> 70` 硬编码提取为 config 读取。
- 将 `print()` 迁移为结构化日志。
- 补充类型注解。
- 拆分 `DeviceMonitorDetector` 中过长方法，但保持 `detect()` 输入输出不变。

## 3. 高风险重构（需人工确认）

- 修改 `DeviceHealthCalculator` 的评分模型（权重结构、扣分逻辑）。
- 修改 `detect()` 输出 schema。
- 引入 SNMP/Modbus 协议实现（涉及网络访问）。
- 修改工单优先级映射或触发条件。

## 4. 当前不建议的重构

- 不需要拆分 `detector.py` 为多文件 — 当前 `DeviceHealthCalculator` + `DeviceMonitorDetector` 内聚度高。
- 不需要引入 `config_adapter.py` — 配置为扁平 dict 传递，足够简单。
- 不建议在无测试基线前修改健康指数计算逻辑。
