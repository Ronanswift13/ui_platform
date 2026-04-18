# 06_refactor_policy

## 1. 当前重构原则

1. 先补最小治理，再做大拆分。
2. 先保留生命周期和输入输出契约，再调整内部结构。
3. 任何重构都至少要跑 sanity 脚本；触及合同输出时还要跑 pytest。
4. 历史设计意图与当前本地实现不一致时，以当前可执行实现为准。

## 2. 允许的低风险重构

1. 补齐 `.agent_skills/00~08`
2. 维护最小 `scripts/run_sanity_checks.sh`
3. 提取 `plugin.py` 内部 helper，降低单文件复杂度
4. 维护已有 `tests/`，为未来协议/拓扑测试预留更清晰的私有方法边界

## 3. 高风险重构（需人工确认）

1. 改动 `init/start/process/stop/shutdown/get_status` 语义
2. 将插件迁移到 `BasePlugin` 或新平台接口
3. 改动协议初始化行为
4. 改动分析触发逻辑
5. 新增本地 API/复核/候选事件能力
6. 修改全局 installer/integration 端口或 UI/cockpit 接线

## 4. 当前强制流程

1. 先读取 `00/01/02/03`
2. 明确改动是“文档治理”“最小脚本”“行为实现”中的哪一类
3. 若改动 `plugin.py` 或 `configs/`，必须运行 `./scripts/run_sanity_checks.sh`
4. 若改动输出合同、entrypoints 或 standalone，必须运行 `python3 -m pytest tests -q`
5. 若发现历史笔记与当前实现冲突，必须先在 skill 中纠偏，再继续扩散
