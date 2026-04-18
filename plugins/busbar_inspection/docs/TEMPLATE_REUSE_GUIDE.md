# 可复制模板说明

本文档说明 busbar_inspection 中哪些内容可作为通用母版供其他插件复用，哪些是 busbar 业务专属。

---

## 1. 可直接复制的通用母版（结构级）

以下文件/目录的**结构和框架**可直接复用，但需替换业务内容：

### 1.1 `.agent_skills/` 体系

| 文件 | 通用部分 | 需替换的业务内容 |
|---|---|---|
| `00_project_context.md` | 输入完备性审计表、固定母版规则、AI 闭环边界、失败模式表 | §2 差异规则、§3 目录职责、§7 失败模式具体条目 |
| `01_architecture_rules.md` | §1 固定母版规则、§3 反模式拦截、§4 校验命令模板 | §2.2 依赖方向图、§2.3 模块职责分配 |
| `02_algorithm_contract.md` | §1 通用算法契约、§2.3 状态机结构 | §2.1 输入契约具体字段、§2.2 输出字段集合、§2.4 原因码字典、§2.5 变焦契约 |
| `03_test_strategy.md` | §1 固定母版规则、§2.1 分层定义、脚本职责说明 | §2.2 覆盖目标数字、§5 必测矩阵 |
| `04_quality_audit.md` | §1 零容忍项、§3 审计命令模板、§4 结论模板 | §2 差异审计规则、§6 测试陷阱（部分通用） |
| `05_security_boundary.md` | 全文通用（仅需替换文件路径） | §2.1 文件边界中的具体路径 |
| `06_refactor_policy.md` | 全文通用 | §2.1/§2.2 的具体模块名 |
| `07_learning_log.md` | Entry Template、可复用审查清单 | 具体 Entries |
| `08_task_routing.md` | 通用前置、四类路由的结构和执行流程 | 各路由中的具体文件名和模块名 |

### 1.2 `.claude/commands/` 体系

| 文件 | 可复用程度 |
|---|---|
| `implement.md` | **完全通用** — 仅需确认文件读取列表中的路径 |
| `repair.md` | **完全通用** — 同上 |
| `audit.md` | **完全通用** — 同上 |
| `bootstrap.md` | **完全通用** — 同上 |
| `propagate.md` | **完全通用** — 同上 |

### 1.3 `scripts/` 体系

| 脚本 | 可复用程度 | 需替换项 |
|---|---|---|
| `run_targeted_tests.sh` | **结构通用** | 模块名映射表（case 语句中的模块→测试文件映射） |
| `run_regression_tests.sh` | **完全通用** | 仅需确认 `--cov=plugins.PLUGIN_NAME` 路径 |
| `run_quality_gate.sh` | **结构通用** | `PLUGIN_NAME` 变量、架构检查中的生产文件列表 |
| `collect_root_cause.sh` | **完全通用** | 无需修改 |
| `bootstrap_project.sh` | **完全通用** | 无需修改（已参数化） |

### 1.4 配置文件

| 文件 | 可复用程度 |
|---|---|
| `.coveragerc` | **结构通用** — 替换 source 路径和 omit 列表 |
| `CLAUDE.md` | **结构通用** — §0-§2, §4-§6 通用；§3 替换为目标插件差异指令 |
| `PROJECT_CARD.md` | **结构通用** — 所有字段需替换 |

---

## 2. busbar 业务专属（不可盲目复制）

以下内容是 busbar_inspection 的业务逻辑，其他插件不应照搬：

| 内容 | 所在文件 | 说明 |
|---|---|---|
| 原因码字典（101-301） | `02_algorithm_contract.md` §2.4, `reason_code_mapper.py` | 每个插件有自己的原因码体系 |
| 变焦建议契约 | `02_algorithm_contract.md` §2.5 | 仅适用于 PTZ 场景 |
| 缺陷标签集合 | `02_algorithm_contract.md` §2.2 | pin_missing/crack/foreign_object 是母线专属 |
| 切片检测参数 | `configs/default.yaml` tiling 段 | 仅适用于远距小目标场景 |
| 告警等级映射 | `02_algorithm_contract.md` §2.2.2 | 每个插件有不同的告警策略 |
| 质量门禁检测顺序 | `04_quality_audit.md` §6.1 | 其他插件可能有不同的质量评估优先级 |
| 4K/切片相关测试陷阱 | `04_quality_audit.md` §6.1-§6.4 | 陷阱本身有参考价值，但具体阈值和检测顺序因插件而异 |

---

## 3. 新插件接入步骤（推荐）

```bash
# 1) 用 bootstrap 脚本复制模板
./scripts/bootstrap_project.sh ../new_plugin_name new_plugin_name

# 2) 必须立即填写的文件
#    - PROJECT_CARD.md（业务目标、验收指标）
#    - .agent_skills/00_project_context.md §2（差异规则）
#    - .agent_skills/02_algorithm_contract.md §2（差异契约）
#    - configs/default.yaml（运行参数）

# 3) 必须调整的脚本
#    - scripts/run_targeted_tests.sh 中的模块映射表
#    - scripts/run_quality_gate.sh 中的架构检查文件列表
#    - .coveragerc 中的 source 路径

# 4) 验证
./scripts/run_targeted_tests.sh all
```

---

## 4. 原则

1. **只抽结构，不抽业务** — 通用母版定义"必须有什么"，不定义"值是什么"。
2. **差异写在差异段** — 每个 skill 文件的 §1 是固定母版，§2 是项目差异，新插件只改 §2。
3. **脚本参数化** — 脚本通过变量（`PLUGIN_NAME`、模块映射）适配不同插件，不靠复制粘贴。
4. **不强制同构** — 并非所有插件都需要变焦建议、切片检测、原因码映射；不适用的契约段落直接删除。
