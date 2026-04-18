# 08 — 任务路由协议（Task Routing Protocol）

> 本文件定义 Agent 接到 UI 任务时的**读取顺序**和**执行顺序**。
> 目标：减少重复 prompt、避免漏读规则、确保每次执行可追溯。

---

## 一、任务类型定义

| 类型 | 关键词 | 典型场景 |
|------|--------|----------|
| **implement** | 新页面 / 重构 / 新增交互 / 新组件 | 新建 cockpit 页面、拆分 outdoor_center |
| **repair** | bug 修复 / 样式修复 / 逻辑修复 | 三态缺失、fetch 无 catch、布局溢出 |
| **audit** | 提测审计 / 质量检查 / 上线前检查 | 跑质量门禁、检查合规项 |
| **upgrade** | 长期结构升级 / 经验回灌 / 规则迭代 | 更新 learning_log、补充 quality_audit 规则 |

---

## 二、读取顺序（Read Order）

Agent 在开始任务前，**必须按以下顺序读取**对应文件。未列出的文件视为"按需读取"。

### implement（新建 / 重构）

```
1. CLAUDE.md                        ← 角色 + 固定戒律 + 工作流
2. 00_project_context.md            ← 技术栈 + 路由表
3. 01_architecture_rules.md         ← 分层规则 + 主题变量
4. 02_ui_contract.md                ← 完整交付契约
5. 05_security_boundary.md          ← 安全红线
6. 04_quality_audit.md（仅回顾段）   ← 历史踩坑，避免重蹈
7. 目标页面/组件源码                  ← 理解现状
```

### repair（Bug 修复）

```
1. CLAUDE.md                        ← 角色 + 固定戒律
2. 04_quality_audit.md              ← 定位违规编号
3. 03_test_strategy.md              ← 验证手段
4. 目标文件源码                      ← 定位问题
5. 07_learning_log.md（最近 2 条）   ← 是否已知问题
```

### audit（提测审计）

```
1. CLAUDE.md                        ← 角色
2. 04_quality_audit.md              ← 完整审计清单
3. 03_test_strategy.md              ← 测试层级
4. 02_ui_contract.md（验收段）       ← 交付标准
5. 执行脚本：
   → scripts/run_quality_gate.sh    ← 一键跑完所有检查
```

### upgrade（经验回灌 / 规则升级）

```
1. 07_learning_log.md               ← 现有沉淀
2. 04_quality_audit.md              ← 现有规则
3. 06_refactor_policy.md            ← 重构触发条件
4. 执行脚本：
   → scripts/collect_root_cause.sh  ← 生成复盘模板
```

---

## 三、执行顺序（Execution Order）

### implement

```
Step 1  读取上述文件（不写代码）
Step 2  列出交付清单（页面 / 组件 / JS / CSS / 路由）
Step 3  与用户确认清单
Step 4  按 02_ui_contract 交付（三态、响应式、无内联）
Step 5  自查：scripts/check_ui_contract.sh + scripts/check_three_state_coverage.sh
Step 6  若有问题 → 修复 → 重跑自查
Step 7  提交前跑 scripts/run_quality_gate.sh
```

### repair

```
Step 1  读取上述文件
Step 2  定位问题 → 关联 04_quality_audit 规则编号
Step 3  最小化修复（不做额外重构）
Step 4  自查：scripts/check_ui_contract.sh
Step 5  补写 07_learning_log.md（若属新类型问题）
```

### audit

```
Step 1  读取上述文件
Step 2  执行 scripts/run_quality_gate.sh
Step 3  逐条确认输出（PASS / FAIL / WARN）
Step 4  输出审计报告给用户
Step 5  若有 FAIL → 转入 repair 流程
```

### upgrade

```
Step 1  读取上述文件
Step 2  执行 scripts/collect_root_cause.sh 生成模板
Step 3  填写复盘内容
Step 4  追加到 04_quality_audit.md 和 07_learning_log.md
Step 5  检查是否需要更新 02_ui_contract.md 或 06_refactor_policy.md
```

---

## 四、脚本清单

| 脚本 | 职责 | 典型调用场景 |
|------|------|-------------|
| `scripts/run_quality_gate.sh` | 聚合门禁：依次调用下列脚本并汇总 | implement 完成后、audit 全量检查 |
| `scripts/check_ui_contract.sh` | 检查内联 CSS/JS、TODO/FIXME、extra_css/js block、文件超长 | 每次代码变更后 |
| `scripts/check_three_state_coverage.sh` | 检查 loading/empty/error 三态覆盖 | implement / repair 后 |
| `scripts/collect_root_cause.sh` | 输出复盘模板到 stdout | upgrade / 重构完成后 |

---

## 五、Token 节约策略

1. **按任务类型裁剪读取**：repair 不需要读 02_ui_contract 全文，只读 04 + 03 + 源码。
2. **脚本替代重复 prompt**：质量检查由脚本输出结论，Agent 不需要逐条人肉检查。
3. **分层读取**：先读 CLAUDE.md 获取戒律，再按任务类型读取最少必要文件。
4. **复盘模板化**：collect_root_cause.sh 输出固定格式，Agent 填空即可，不需要从头构造。
