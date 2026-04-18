# 06 重构策略 — {{PLUGIN_DISPLAY_NAME}}

## 允许的重构类型

| 类型 | 条件 | 验证 |
|------|------|------|
| 提取函数 | 函数 > 50 行 | targeted tests 通过 |
| 重命名 | 命名不符合规范 | targeted + regression 通过 |
| 移动模块 | 违反架构分层 | 全量回归通过 |
| 接口变更 | 需人工确认 | 全量回归 + 手工验证 |

## 执行流程

1. 明确重构范围与动机（引用规则编号）
2. 先写保护测试（确认当前行为）
3. 执行重构
4. 运行 `./scripts/run_targeted_tests.sh <module>`
5. 运行 `./scripts/run_regression_tests.sh`
6. 更新 `07_learning_log.md`

## 回滚触发条件

- regression 测试失败
- 改动超出声明范围
- 引入新的架构违规
