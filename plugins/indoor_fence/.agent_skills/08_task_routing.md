# 08_task_routing — 任务路由表

本文件定义 indoor_fence 的标准任务路径：先明确当前任务属于主插件运行时还是 V3 演练链路，再按任务类型读取文件、执行脚本、回写知识。
所有 `/implement`、`/repair`、`/audit` 命令必须先加载本文件。

---

## 通用前置（所有任务类型共享）

**必读（按序）：**
1. `PROJECT_CARD.md` — 确认当前业务目标、API 面与验收门槛
2. `CLAUDE.md` — 确认固定流程与脚本入口
3. `.agent_skills/00_project_context.md` — 判断主链路 / V3 链路
4. `.agent_skills/01_architecture_rules.md` — 确认落点与依赖边界

**阻断检查：**
- 若任务会修改 `manifest.json` 的 `id` / `entrypoint` / `plugin_class` / 核心 `capabilities` → 先暂停并报告
- 若任务会改变 `RecognitionResult.label`、`/api/indoor-fence/*` 路径、`zone.yaml` / scenario JSON 结构 → 先列出受影响测试与文档
- 若无法判断变更属于主插件运行时还是 V3 演练链路 → 先写出路由判断，再开始实现

---

## implement（功能 / 规则 / 配置 / standalone 实现）

### 必读
| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | 通用前置 | 确认边界 |
| 2 | `.agent_skills/02_algorithm_contract.md` | 确认输出、fallback、回滚契约 |
| 3 | `.agent_skills/03_test_strategy.md` | 选择最近测试与 targeted 模块 |
| 4 | `configs/default.yaml` | 当前配置基线 |
| 5 | `standalone/configs/zone.yaml` / `configs/scenarios/*.json`（按需） | 区域或场景资产 |

### 执行顺序
1. **先判定落点**：主插件运行时 / V3 演练链路 / standalone surface / config assets
2. **先写测试**：最近测试文件必须先补或先改
3. **最小实现**：新增检测能力优先放 `detection/*`；新的 standalone 能力只经 `get_standalone_routes()`
4. **targeted 验证**：`./scripts/run_targeted_tests.sh <module>`
5. **regression 验证**：`./scripts/run_regression_tests.sh`
6. **quality gate（按需）**：涉及跨层、路由、配置结构时运行 `./scripts/run_quality_gate.sh`

### 回写文件
| 文件 | 条件 |
|------|------|
| `tests/test_*.py` | 必须：新增或更新最近测试 |
| 实现模块 | 必须：最小改动 |
| `configs/default.yaml` / `standalone/configs/zone.yaml` / `configs/scenarios/*.json` | 若契约或资产变化 |
| `.agent_skills/04_quality_audit.md` | 若提炼出新的通用审计规则 |
| `.agent_skills/07_learning_log.md` | 若遇到非显然约束、回滚、fallback 或跨层问题 |

---

## repair（缺陷修复）

### 必读
| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | 通用前置 | 确认边界 |
| 2 | `.agent_skills/02_algorithm_contract.md` | 确认期望行为 |
| 3 | `.agent_skills/06_refactor_policy.md` | 保持最小因果修复 |
| 4 | `.agent_skills/07_learning_log.md` | 查找是否已有同类问题 |
| 5 | `.agent_skills/04_quality_audit.md` | 识别是否属于已知反模式 |

### 执行顺序
1. **记录现象**：现象、复现路径、首次失败边界
2. **根因收集**：必要时先跑 `./scripts/collect_root_cause.sh`
3. **先写失败测试**：必须先把 bug 固定在最近测试中
4. **最小因果修复**：只改当前根因，不顺带重构其他链路
5. **targeted 验证**：`./scripts/run_targeted_tests.sh <module>`
6. **regression 验证**：`./scripts/run_regression_tests.sh`
7. **知识回灌**：必须追加 `07_learning_log.md`；若形成共性规则，再更新 `04_quality_audit.md`

### 回写文件
| 文件 | 条件 |
|------|------|
| `tests/test_*.py` | 必须：失败测试转绿 |
| 缺陷所在模块 | 必须：最小修复 |
| `.agent_skills/07_learning_log.md` | **必须**：追加根因条目 |
| `.agent_skills/04_quality_audit.md` | 若根因可推广为通用反模式 |

---

## audit（质量审计）

### 必读
| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | 通用前置 | 确认边界 |
| 2 | `.agent_skills/04_quality_audit.md` | 审计清单与分级规则 |
| 3 | `.agent_skills/05_security_boundary.md` | 安全边界 |
| 4 | `.coveragerc` | 覆盖率排除与门槛解释 |
| 5 | `.agent_skills/07_learning_log.md` | 识别是否复发 |

### 执行顺序
1. **执行质量闸门**：`./scripts/run_quality_gate.sh`
2. **人工审计**：检查 route surface、config rollback、fallback、template / MJPEG / scenario / zone 一致性
3. **分级输出**：`BLOCKER` / `HIGH_RISK` / `DEBT`
4. **知识回灌**：若审计发现新共性规则或系统性问题，可更新 `04_quality_audit.md` 与 `07_learning_log.md`，但不默认修生产代码

### 回写文件
| 文件 | 条件 |
|------|------|
| 对话中的审计报告 | 必须：含命令、证据、分级结论 |
| `.agent_skills/04_quality_audit.md` | 若发现 checklist 缺口 |
| `.agent_skills/07_learning_log.md` | 若发现复发模式或系统性问题 |

**关键约束：** 未给出实际执行命令与结果时，不得宣称通过。

---

## upgrade（依赖 / 契约 / schema 升级）

### 必读
| 顺序 | 文件 | 目的 |
|------|------|------|
| 1 | 通用前置 | 确认边界 |
| 2 | `manifest.json` | 检查对外契约 |
| 3 | `requirements.txt` | 检查依赖版本 |
| 4 | `.agent_skills/02_algorithm_contract.md` | 确认输出变化 |
| 5 | `.agent_skills/03_test_strategy.md` | 确认覆盖范围 |

### 执行顺序
1. **影响评估**：主插件运行时 / V3 演练链路 / standalone surface 分开列影响面
2. **先更新测试**：先更新受影响期望值
3. **逐层适配**：契约层 -> 算法 / 适配器 -> standalone -> plugin
4. **targeted 验证**：`./scripts/run_targeted_tests.sh all`
5. **regression 验证**：`./scripts/run_regression_tests.sh`
6. **quality gate**：`./scripts/run_quality_gate.sh`
7. **知识回灌**：同步更新 `04_quality_audit.md` / `07_learning_log.md`

### 回写文件
| 文件 | 条件 |
|------|------|
| `tests/test_*.py` | 必须：适配升级契约 |
| 受影响实现模块 | 必须 |
| `manifest.json` / `requirements.txt` | 若对外契约或依赖变化 |
| `configs/default.yaml` | 若默认配置变化 |
| `.agent_skills/02_algorithm_contract.md` | 若输出/降级契约变化 |
| `.agent_skills/04_quality_audit.md` | 若新增审计规则 |
| `.agent_skills/07_learning_log.md` | 必须：记录升级风险 |

---

## 快速参考：任务 -> 脚本映射

| 任务 | 最小验证 | 完整验证 | 必要回写 |
|------|----------|----------|----------|
| implement | `run_targeted_tests.sh <module>` | `run_regression_tests.sh` | `04` / `07`（按条件） |
| repair | `run_targeted_tests.sh <module>` | `run_regression_tests.sh` | `07` 必须，`04` 按条件 |
| audit | `run_quality_gate.sh` | 人工审计 | `04` / `07`（按条件） |
| upgrade | `run_targeted_tests.sh all` | `run_regression_tests.sh` + `run_quality_gate.sh` | `04` + `07` |

### targeted 模块名
- `plugin`
- `adapters`
- `detection`
- `fusion`
- `logic`
- `standalone`
- `integration`
- `all`

