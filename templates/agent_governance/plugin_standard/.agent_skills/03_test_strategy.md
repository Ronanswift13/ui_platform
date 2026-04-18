# 03 测试策略 — {{PLUGIN_DISPLAY_NAME}}

## 测试分层

| 层级 | 范围 | 入口 |
|------|------|------|
| L0 | 单元测试 | `python -m pytest tests/ -q` |
| L1 | 质量闸门 | `./scripts/run_quality_gate.sh`（如可用） |

## 覆盖率目标

- 整体: ≥ 60%

## 升级路径

补齐以下脚本可升级到 HF:
- `scripts/run_targeted_tests.sh`
- `scripts/run_regression_tests.sh`
- `scripts/run_quality_gate.sh`
- `scripts/collect_root_cause.sh`
