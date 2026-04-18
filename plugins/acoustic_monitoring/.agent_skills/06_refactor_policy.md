# 06_refactor_policy

## 1. 固定母版规则

1. 先证明问题，再重构。
2. 结构重构与行为变更分离提交。
3. 保持公共契约兼容。
4. 没有测试支撑的重构不允许进入主干。

## 2. 允许的低风险重构

- 将 `print()` 迁移为结构化日志。
- 拆分 `detector.py` 中过长的检测方法，但保持输入输出不变。
- 抽取 `AcousticConfig` 的 YAML 加载逻辑为独立方法。
- 补充类型注解。

## 3. 高风险重构（需人工确认）

- 拆分 `detector.py` 为多个检测器模块（改变目录结构）。
- 修改 `process()` 输出 schema。
- 引入新的第三方依赖（如 librosa、scipy）。
- 修改 `AcousticAnomalyType` 枚举值。

## 4. 当前不建议的重构

- 不需要引入 `config_adapter.py` 或 `reason_code_mapper.py` — 当前配置链路足够简单，dataclass 直映射即可。
- 不需要拆分 `plugin.py` 中的 mock 音频生成 — 功能内聚且仅用于演示。
