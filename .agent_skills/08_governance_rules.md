# 08 — 治理规则与升级条件

## 治理层级定义

| 层级 | 适用范围 | 治理文件 |
|---|---|---|
| **根级** | 整个仓库 | `PROJECT_CARD.md` + `CLAUDE.md` + `.agent_skills/00-08` |
| **插件级 L3** | 成熟插件 | 四件套 + 可选 `.agent_skills/` |
| **插件级 L0** | 占位插件 | 仅 `README.md` (占位说明) |

## 升级路径

### L0 (占位态) → L1 (骨架态)

**前提**: 有人承诺维护此插件

需新增:
- [ ] `plugin.py` — 继承 `EnhancedBasePlugin`
- [ ] `manifest.json` — name / version / description / dependencies

### L1 → L2 (基本态)

需新增 (至少一项):
- [ ] `configs/default.yaml`
- [ ] `tests/test_plugin.py`

### L2 → L3 (完整态)

四件套全部补齐:
- [ ] `plugin.py` ✓
- [ ] `manifest.json` ✓
- [ ] `configs/default.yaml` ✓
- [ ] `tests/test_plugin.py` ✓ 且 pytest 通过

### L3 → L3+ (skill 化)

**前提**: L3 已稳定运行，有跨模块交互需求

可新增:
- [ ] `.agent_skills/` 子目录 (仅限已有足够事实源的插件)
- [ ] 插件级 `CLAUDE.md`

## 治理红线

1. **不得跳级** — L0 不能直接升 L3，必须逐级
2. **不得空壳 skill** — 没有事实源的目录禁止生成 `.agent_skills/`
3. **不得破坏证据链** — `evidence/` 目录只增不删
4. **不得静默改接口** — `platform_core/schema/` 的变更必须通知所有插件维护者
5. **占位态不阻塞** — L0 插件不参与 CI 测试、不出现在 enabled_plugins 列表

## 定期审查

建议每月审查一次插件矩阵 (`.agent_skills/02_plugin_registry.md`)：
- 是否有 L0 插件已具备升级条件?
- 是否有 L3 插件退化 (测试失败)?
- 是否有新插件需要登记?
