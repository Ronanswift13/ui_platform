# 08 任务路由 — {{PLUGIN_DISPLAY_NAME}}

## 当前治理前提

- 治理等级: MIN（最小治理）
- 验证入口: `./scripts/run_sanity_checks.sh`（如存在）或 `python -m pytest tests/ -q`
- 脚本体系: 仅 sanity check

## 共享前置

1. `.agent_skills/00_project_context.md`

## implement / repair / audit / upgrade

### 执行顺序
1. 读取 `00_project_context.md` 确认当前状态
2. 执行最小验证: `./scripts/run_sanity_checks.sh`
3. 若任务声称已有 tests / API / 完整接口 → 先停止并核实

### 回写
- `.agent_skills/07_learning_log.md`（如存在）

## 阻断条件

- 若任务涉及不存在的文件或接口 → 停止并报告
- 若声称通过了不存在的测试 → 结论无效

## 升级路径

1. 补齐 tests/ 目录
2. 补齐 scripts/run_targeted_tests.sh
3. 补齐 `.claude/commands/`
4. 提升到 STD 级
