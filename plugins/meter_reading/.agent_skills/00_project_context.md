# 00_project_context

## 1. 输入完备性审计（当前真实状态）

| 输入项 | 状态 | 来源 | 处置规则 |
|---|---|---|---|
| `PROJECT_CARD.md` | 可用 | 当前插件目录 | 业务目标、验收指标、禁止事项的权威来源 |
| `CLAUDE.md` | 可用，但含历史话术 | 当前插件目录 | 作为固定指令来源，执行细节以 `08_task_routing.md` 为准 |
| `plugin.py` | 可用 | 当前插件目录 | SDK 接口层单一真相来源 |
| `detector_enhanced.py` | 可用 | 当前插件目录 | 算法链路、状态机、热加载、metadata 的单一真相来源 |
| `configs/default.yaml` | 可用 | 当前插件目录 | 阈值与运行参数来源 |
| `tests/` | 可用 | 当前插件目录 | 当前有 7 个 Python 测试文件，`run_targeted_tests.sh all` 可收集 98 个用例 |
| `tests/regression/` | 目录存在但为空 | 当前插件目录 | 只能显式报告“未执行数据集回归”，不得宣称已覆盖 L2 |
| `tests/fixtures/` | 目录存在但为空 | 当前插件目录 | 回归样本集尚未就位，脚本必须输出可审计的 skip |
| `standalone/` | 可用 | 当前插件目录 | Web/模拟器运行入口，不承载业务判定 |
| `docs/decision_records/` | 仅模板 | 当前插件目录 | 重大契约或目录决策变更时才补 ADR |

## 2. 固定母版规则（跨插件通用）

1. 插件接口契约冻结：`init / infer / postprocess / healthcheck` 的职责与签名不得随意漂移。
2. 执行证据优先：命令、脚本、审计结论必须基于真实命令输出，不能“纸面通过”。
3. 测试分层固定：必须区分 targeted、regression、quality gate；空目录必须显式 skip。
4. 层级依赖固定：接口层 -> 算法层 -> 配置层；`standalone/` 只做运行与展示。
5. 规则单点归档：任务入口看 `08_task_routing.md`，不要在多个 skill 文件重复写完整流程。

## 3. 本插件特有事实（meter_reading 专属）

1. 业务目标固定为单帧表计读数，覆盖 9 种表计类型：6 种模拟表、2 种数字表、1 种 LED 指示灯。
2. 模拟表链路的可靠性核心是 `HRNet -> HoughCircle -> HoughLine` 三级降级，且必须输出 `fallback_level`。
3. `plugin.py` 负责 ROI 提取、`RecognitionResult` 组装、告警生成、健康统计；`detector_enhanced.py` 不直接依赖 SDK schema。
4. 当前阈值来自 `configs/default.yaml`，但模拟表基础量程仍由 `detector_enhanced.py` 中的 `METER_RANGES` 注册表维护；若变更量程，需要同步契约与测试。
5. 当前热加载只覆盖 `confidence_threshold` 与 `manual_review_threshold` 两个参数，不能假定全量配置都支持热更新。
6. 当前仓库没有独立的 `standalone` 自动化测试，也没有真实标定样本回归集；这两项属于明确的测试缺口，不是“默认已覆盖”。

## 4. 当前目录与职责边界

```text
meter_reading/
├── plugin.py                   # SDK 接口层：init/infer/postprocess/healthcheck
├── detector_enhanced.py        # 算法层：输入校验、预处理、三条识别链路、热加载
├── configs/default.yaml        # 阈值、预处理、fallback、LED、性能参数
├── standalone/                 # Web 服务、视频流、模拟器
├── tests/                      # 契约/单测/集成测试
├── scripts/                    # targeted/regression/quality/root_cause 入口
├── .agent_skills/              # 00~08 规则体系
├── .claude/commands/           # implement/repair/audit 等命令入口
├── docs/decision_records/      # ADR（当前只有模板）
└── manifest.json               # 插件注册信息
```

## 5. AI 自动闭环 vs 人工确认

### 可由 AI 自动闭环

- 维护 `.agent_skills/*.md`、`.claude/commands/*.md`、`scripts/*.sh`
- 补齐或调整现有测试文件与测试分组
- 校验配置路径、metadata 契约、状态机契约
- 执行 targeted / regression / quality gate 并输出证据
- 在 `07_learning_log.md` 中回写经验与回灌建议

### 必须人工确认

- 新增或删除表计类型
- 调整 `METER_RANGES` 中的业务量程与单位
- 修改 `manifest.json` 核心字段（`id` / `entrypoint` / `plugin_class`）
- 引入真实回归样本集、上线阈值或告警业务口径
- 明显改变 standalone 的产品交互或对外接口

## 6. 可执行检查命令

```bash
# 1) 配置可解析
python3 -c "import yaml; yaml.safe_load(open('configs/default.yaml', 'r', encoding='utf-8'))"

# 2) 快速验证当前测试入口
./scripts/run_targeted_tests.sh all

# 3) 检查 detector 层未反向依赖 SDK / standalone
rg -n "darkbreaker_sdk|standalone" detector_enhanced.py

# 4) 检查三态/metadata 锚点
rg -n "ReadingStatus|fallback_level|timestamp_ms|need_manual_review" plugin.py detector_enhanced.py tests
```

## 7. 当前已知风险（执行时必须显式说明）

1. `tests/regression/` 与 `tests/fixtures/` 为空，当前只能做“代码级回归”，不能做“数据集级回归”。
2. `METER_RANGES` 仍是代码注册表，不是 YAML 配置；量程相关改动需要额外谨慎。
3. standalone 功能已有代码，但缺少专门的自动化回归入口，涉及该目录时必须补充 smoke 或人工验证说明。
