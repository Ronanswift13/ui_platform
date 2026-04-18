# 03 测试策略 — {{PLUGIN_DISPLAY_NAME}}

## 测试分层

| 层级 | 范围 | 入口 | 频率 |
|------|------|------|------|
| L0 | 模块级单元测试 | `./scripts/run_targeted_tests.sh <module>` | 每次改动 |
| L1 | 契约/集成测试 | `./scripts/run_targeted_tests.sh all` | 每次改动 |
| L2 | 全量回归 | `./scripts/run_regression_tests.sh` | 提交前 |
| L3 | 质量闸门 | `./scripts/run_quality_gate.sh` | 交付前 |

## 覆盖率目标

- 核心算法模块 (`{{DETECTOR_FILE}}`): ≥ 80%
- 插件适配层 (`plugin.py`): ≥ 70%
- 整体: ≥ 70%

## 测试模块映射

```bash
# 查看可用模块
./scripts/run_targeted_tests.sh --help
```

<!-- BUSINESS: 补充本插件的测试模块与对应测试文件映射 -->

## 失败条件

- 任何 L0/L1 测试失败 → 阻断提交
- L2 回归失败 → 阻断合并
- L3 质量闸门失败 → 阻断交付
