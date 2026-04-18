# 00_project_context

## 0. 输入完备性审计（本轮真实状态）

| 输入项 | 状态 | 来源 | 处置规则 |
|---|---|---|---|
| `PROJECT_CARD.md` | **可用** | 当前插件目录 | 业务目标与验收指标的权威来源 |
| 现有项目目录结构 | **可用** | 当前仓库文件树 | 作为规则落地的目录边界 |
| 历史决策记录 | **目录已建** | `docs/decision_records/` 已存在 | 重大决策需补 ADR 文件 |
| 可迁移 `.agent_skills/` 规则 | **可用** | `plugins/indoor_fence/.agent_skills`、`plugins/meter_reading/.agent_skills` | 抽取固定母版规则并按本项目差异化落地 |

## 1. 固定母版规则（跨插件统一）

1. **接口契约冻结**：插件必须实现 `init/infer/postprocess/healthcheck`，且签名与 SDK 一致，不允许私自改签名。
2. **配置优先**：阈值、NMS、切片参数、质量门禁参数必须来自配置，不允许在推理主路径新增硬编码业务阈值。
3. **降级可观测**：每条降级路径必须输出可审计字段（`failure_reason` + `metadata`）。
4. **输出可校验**：`bbox`、`confidence`、`label` 必须满足 schema 与值域约束。
5. **测试分层**：至少包含 L0（单测）、L1（集成）、L2（回归）分层执行入口。
6. **质量门禁自动化**：提交前必须可一键执行回归门禁脚本。

## 2. 本项目差异规则（busbar_inspection 专属）

1. **业务目标固定**：聚焦母线场景远距小目标（销钉缺失、裂纹、异物）与环境干扰过滤。
2. **ROI 批处理固定**：单帧允许多 ROI；每个 ROI 独立产出结果，不得跨 ROI 复用 bbox。
3. **原因码固定**：必须输出原因码用于复核/二次抓拍决策。
4. **变焦建议固定**：必须输出建议动作与建议倍率，不允许只给“检测失败”而不给动作建议。
5. **4K 兼容固定**：高分辨率输入默认允许切片处理，切片参数由配置控制。

## 3. 当前目录与职责边界

```
busbar_inspection/
├── plugin.py                   # SDK 适配层（输入输出编排、告警生成）
├── detector_enhanced.py        # 算法层（质量门禁、检测、NMS、变焦建议）
├── config_adapter.py           # 配置映射层（嵌套 YAML -> 统一键）
├── reason_code_mapper.py       # 原因码映射层（内部码 -> 外部码）
├── configs/default.yaml        # 运行参数来源
├── tests/                      # 自动化测试（契约 + 回归）
├── scripts/                    # 门禁脚本（targeted/regression/quality/root_cause）
├── .agent_skills/              # AI 代理规则体系（00~08）
├── .claude/commands/           # 任务命令定义（implement/repair/audit/bootstrap/propagate）
├── docs/                       # 决策记录与设计笔记
└── manifest.json               # 插件注册信息
```

## 4. AI 自动闭环 vs 人工确认边界

### 4.1 可由 AI 自动闭环

- 规则文档维护（`.agent_skills/*.md`）
- 测试脚本维护（`scripts/run_targeted_tests.sh`、`scripts/run_regression_tests.sh`）
- 测试用例补齐与执行
- 配置键路径一致性检查与修复
- 质量审计与反模式扫描

### 4.2 必须人工确认

- `PROJECT_CARD.md` 业务口径（验收指标、上线范围、非功能目标）
- 缺陷标签最终清单（是否保留 `corrosion/flashover/broken_strand/...`）
- 现场告警等级策略（ERROR/WARNING 的业务映射）
- 变焦动作接入平台执行策略（插件仅建议，平台是否执行）

## 5. 硬约束（执行时必须满足）

1. Python >= 3.9。
2. `configs/default.yaml` 必须存在且可解析。
3. `plugin.py` 必须可被 `Plugin.create_standalone()` 正常实例化。
4. `infer()` 输出 `RecognitionResult` 列表；任何单 ROI 异常不得导致整帧崩溃。
5. 不允许访问外部网络；不允许持久化原始图像到未授权路径。

## 6. 可执行校验命令

```bash
# 1) 配置可解析
python -c "import yaml; yaml.safe_load(open('configs/default.yaml', 'r', encoding='utf-8'))"

# 2) 插件可导入
python -c "from plugins.busbar_inspection.plugin import Plugin; print(Plugin.__name__)"

# 3) 基础单测
python -m pytest tests/test_standalone.py -q

# 4) 快速依赖扫描（禁止缺失核心文件）
rg --files | rg '^(plugin\.py|detector_enhanced\.py|manifest\.json|configs/default\.yaml)$'
```

## 7. 失败模式与处理

| 失败模式 | 触发信号 | 处理动作 |
|---|---|---|
| 配置路径不一致 | 调整 YAML 后行为不变 | 按 `02_algorithm_contract.md` 的配置映射契约修复 |
| 原因码语义错配 | `failure_reason` 有值但文案为“未知原因” | 统一原因码字典并补测试 |
| 质量门禁误判 | 高质量图被大量 `quality_failed` | 回放样本并校正阈值来源 |
| 切片后框偏移 | bbox 越界或误报激增 | 检查 remap/NMS 契约并补边界测试 |
