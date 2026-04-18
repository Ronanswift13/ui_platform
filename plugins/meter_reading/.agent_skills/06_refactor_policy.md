# 06_refactor_policy

## 1. 固定母版规则

1. 先证明问题，再重构。
2. 行为变更与结构整理尽量分离。
3. 公共契约保持兼容。
4. 没有测试支撑的重构不进入主干。

## 2. 本项目允许的低风险重构

1. 调整 `.agent_skills/`、`.claude/commands/`、`scripts/` 的结构与话术。
2. 提取 detector / plugin 内部辅助函数，但不改变输入输出契约。
3. 统一 metadata builder、测试夹具、脚本参数解析。
4. 为 `standalone/` 增加 smoke 入口或测试桩，不改产品行为。

## 3. 高风险重构（需人工确认）

1. 调整 `MeterType` 集合或 `METER_RANGES` 业务量程。
2. 改变 `RecognitionResult.label/value` 语义。
3. 改变 `postprocess()` 的告警级别或文案口径。
4. 大规模重排 `standalone/`、`manifest.json` 或插件对外接口。
5. 一次性改动超过 5 个生产代码文件。

## 4. 强制执行流程

1. 先读取 `08_task_routing.md` 对应任务路由。
2. 先补或更新对应测试。
3. 运行 `./scripts/run_targeted_tests.sh <module>`。
4. 若触及生产代码或配置，再运行 `./scripts/run_regression_tests.sh`。
5. 若契约、测试重点或经验发生变化，同步更新 `02 / 03 / 07`。

## 5. 回滚触发条件

1. 三态状态集或 metadata 必填字段被破坏。
2. 模拟表降级链或 LED / OCR 语义发生意外漂移。
3. targeted / regression 任一失败。
4. 审计发现把“空回归目录”误报成“已完成回归”。
