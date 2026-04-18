# 06_refactor_policy

## 1. 固定母版规则

1. 先证明问题，再重构。
2. 结构重构与行为变更分离提交。
3. 保持公共契约兼容。
4. 没有测试支撑的重构不允许进入主干。

## 2. 允许的低风险重构

- 将 `print()` 迁移为结构化日志。
- 补充类型注解。
- 将 `_process_thermal()` 中硬编码的 15-75°C 映射范围提取到配置。
- 拆分 `TemperatureDetector` 中过长方法，但保持 `detect()` 输入输出不变。

## 3. 高风险重构（需人工确认）

- 修改 `detect()` 输出 schema（`TemperatureResult` 结构）。
- 修改 z-score 热点检测算法。
- 修改温度阈值分级逻辑。
- 修改联动事件触发条件。
- 切换预测方法（linear → LSTM/ARIMA）。

## 4. 当前不建议的重构

- 不应在无测试基线前修改热点检测或趋势分析逻辑。
- 不需要拆分 `detector.py` 为多文件 — 当前 `TemperatureDetector` 内聚度高，数据类简洁。
- 不需要引入 `config_adapter.py` — 配置为嵌套 dict，detector 构造时解析，足够清晰。
