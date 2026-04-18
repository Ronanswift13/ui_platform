# AATF 可复用模板 (Adaptive Agentic Template Framework)

> 本文档记录 indoor_fence 插件中验证的跨项目可复用资产，供其他 DarkBreaker 插件迁移使用。

## 目录

1. [AATF 框架概述](#1-aatf-框架概述)
2. [标准目录结构](#2-标准目录结构)
3. [Skill 文档格式规范](#3-skill-文档格式规范)
4. [测试流程规范](#4-测试流程规范)
5. [Root Cause 回灌流程](#5-root-cause-回灌流程)
6. [质量闸门规范](#6-质量闸门规范)
7. [受控扩散修复策略](#7-受控扩散修复策略)
8. [AI 角色分工规范](#8-ai-角色分工规范)
9. [敏感文件权限边界](#9-敏感文件权限边界)
10. [通用 Prompt 结构](#10-通用-prompt-结构)

---

## 1. AATF 框架概述

### 7 阶段工作流

| 阶段 | 名称 | 命令 | 说明 | 输出物 |
|------|------|------|------|--------|
| A | 起盘 | `/bootstrap` | 初始化环境，加载上下文，记录基准 | 环境验证报告 + 测试基准 |
| B+C | 建模+实现 | `/implement` | 确认约束 → TDD → 实现 → 验证 | 功能代码 + 测试 |
| D | 质量闸门 | `/audit` | 静态分析 + 测试 + 架构合规 + 安全扫描 | 质量报告 |
| E | 回灌 | `/repair` | 根因分析 → 修复 → 经验记录 | Bug 修复 + 学习日志 |
| F | 受控扩散 | `/propagate` | 扫描同类问题 → 分批修复 → 回归验证 | 批量修复 + 回归报告 |
| G | 跨项目迁移 | 见 `prompts/cross_project_reuse_prompt.md` | 复制到其他插件 | 迁移清单 + 定制指南 |

### 核心原则

1. **最小化修改**: 只改必须改的代码
2. **降级优先**: 所有故障场景必须有降级方案
3. **测试驱动**: 修改前先写测试
4. **文档同步**: 接口变更必须更新文档
5. **经验回灌**: 每次修复后记录到学习日志

---

## 2. 标准目录结构

### 通用插件目录结构

```
<plugin_name>/
├── .agent_skills/          # AI 技能文件（8 个标准文件）
│   ├── 00_project_context.md
│   ├── 01_architecture_rules.md
│   ├── 02_algorithm_contract.md
│   ├── 03_test_strategy.md
│   ├── 04_quality_audit.md
│   ├── 05_security_boundary.md
│   ├── 06_refactor_policy.md
│   └── 07_learning_log.md
├── .claude/
│   └── commands/           # AATF 工作流命令
│       ├── bootstrap.md
│       ├── implement.md
│       ├── repair.md
│       ├── audit.md
│       └── propagate.md
├── adapters/               # 硬件适配层
│   ├── base_adapter.py
│   └── <sensor>_adapter.py
├── configs/                # 配置文件
│   ├── default.yaml
│   └── scenarios/
├── core/                   # 核心算法模块（禁止依赖 adapters/standalone）
│   ├── fusion/
│   ├── rules/
│   └── tracking/
├── detection/              # 行为检测（ML 模型 + 规则降级）
├── standalone/             # 独立运行支持（Web UI/录制/回放/训练）
│   ├── templates/
│   └── static/
├── tests/                  # 单元测试 + 集成测试
│   ├── fixtures/
│   └── test_*.py
├── docs/                   # 文档
│   ├── plans/
│   └── decision_records/
├── scripts/                # 自动化脚本
│   ├── run_quality_gate.sh
│   ├── run_regression_tests.sh
│   ├── run_targeted_tests.sh
│   └── collect_root_cause.sh
├── prompts/                # AI 提示词
│   ├── root_cause_prompt.md
│   ├── terminal_execute_prompt.md
│   ├── web_modeling_prompt.md
│   └── cross_project_reuse_prompt.md
├── protocols.py            # 数据协议定义（Pydantic）
├── plugin.py               # DarkBreaker 插件接口（冻结）
├── run_standalone.py       # 独立运行入口
├── CLAUDE.md               # Claude 开发指南（项目入口）
├── PROJECT_CARD.md         # 项目卡片（9 字段格式）
├── .coveragerc             # 覆盖率配置
└── requirements.txt        # 依赖清单

```

### 目录职责说明

| 目录 | 职责 | 依赖方向 | 可选/必需 |
|------|------|----------|----------|
| `core/` | 核心算法，纯函数，无 I/O | 不依赖其他模块 | 必需 |
| `adapters/` | 硬件适配，I/O 操作，降级逻辑 | 依赖 `protocols.py` | 必需 |
| `detection/` | 行为检测，ML 推理 + 规则降级 | 依赖 `core/` | 可选 |
| `standalone/` | Web UI，录制回放，训练管道 | 依赖所有模块 | 可选 |
| `tests/` | 测试代码，Mock 数据 | 依赖所有模块 | 必需 |
| `.agent_skills/` | AI 技能文件，开发指南 | 无代码依赖 | 必需 |
| `.claude/commands/` | AATF 工作流命令 | 无代码依赖 | 必需 |
| `scripts/` | 自动化脚本，质量门禁 | 无代码依赖 | 必需 |
| `prompts/` | AI 提示词，诊断框架 | 无代码依赖 | 必需 |

---

## 3. Skill 文档格式规范

### 统一 8 段格式

所有 `.agent_skills/*.md` 文件必须遵循以下 8 段格式：

```markdown
# <编号> <标题>

## 1. When to use
在以下场景查阅本文件：
- [场景 1]
- [场景 2]
- [场景 3]

## 2. Inputs
- [输入 1]
- [输入 2]
- [输入 3]

## 3. Outputs
- [输出 1]
- [输出 2]
- [输出 3]

## 4. Hard Constraints
- [约束 1]
- [约束 2]
- [约束 3]

## 5. Algorithm / Logic Contract
[核心算法、流程、规则的详细说明]

## 6. Validation Rules
```bash
# 验证命令
[命令 1]
[命令 2]
```

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| [故障 1] | [影响] | [处理方案] |
| [故障 2] | [影响] | [处理方案] |

## 8. Required Tests
- [测试 1]
- [测试 2]
- [测试 3]
```

### 8 个标准 Skill 文件

| 文件 | 用途 | 何时查阅 |
|------|------|---------|
| `00_project_context.md` | 项目结构和依赖 | 初次了解项目 |
| `01_architecture_rules.md` | 架构约束和分层 | 修改 import 或文件结构 |
| `02_algorithm_contract.md` | 算法降级契约 | 修改核心算法 |
| `03_test_strategy.md` | 测试规范和边界用例 | 添加或修改测试 |
| `04_quality_audit.md` | 质量检查和反模式 | 提交前审查 |
| `05_security_boundary.md` | 安全边界 | 涉及 I/O、网络、配置 |
| `06_refactor_policy.md` | 受控重构策略 | 跨文件修改 |
| `07_learning_log.md` | 经验回灌知识库 | 修复 bug 后记录 |

---

## 4. 测试流程规范

### Mock 策略

**硬件 Mock**
- 传感器缺失: 使用模拟器生成合成数据
- 串口不可用: 自动降级到模拟器模式
- 设备异常: 返回默认值（如零加速度/零角速度）

**模型 Mock**
- ML 模型缺失: 使用规则引擎降级
- 推理失败: 返回 `UNKNOWN` 状态，不阻塞主流程

**测试 Mock**
- 使用 `pytest.fixture` 提供标准化 mock 数据
- 所有 adapter 实现统一接口，便于替换

### 边界用例分类

| 类别 | 示例用例 | 测试目标 |
|------|---------|---------|
| **数据质量边界** | 零噪声、高噪声、异常值、时间戳乱序 | 滤波鲁棒性 |
| **信号丢失边界** | 丢失 1 帧、4 帧、5 帧、100 帧 | 降级策略 |
| **运动模式边界** | 静止、慢走、快跑、急停、急转弯 | 状态识别 |
| **系统资源边界** | 高频数据、内存不足、磁盘满、CPU 过载 | 资源管理 |

### 覆盖率要求

| 模块 | 最低覆盖率 | 说明 |
|------|-----------|------|
| 整体 | 80% | 质量闸门阈值 |
| `core/fusion/` | 90% | 核心算法，高要求 |
| `core/rules/` | 85% | 规则引擎，高要求 |
| `adapters/` | 80% | 硬件适配，需复杂 mock |
| `detection/` | 85% | 行为检测，高要求 |
| `standalone/` | 70% | Web UI，低优先级 |

### 测试命令

```bash
# 运行测试 + 覆盖率
pytest tests/ --cov=. --cov-fail-under=80 --cov-report=html -v

# 模块针对性测试
./scripts/run_targeted_tests.sh <module_name>

# 回归测试套件
./scripts/run_regression_tests.sh
```

---

## 5. Root Cause 回灌流程

### 4 步分析流程

```
1. 收集症状 ─→ 错误日志 + 异常堆栈 + 系统状态 + 复现步骤
2. 假设生成 ─→ 基于症状生成可能的根因假设，按概率排序
3. 验证假设 ─→ 通过日志分析、代码审查、测试复现验证假设
4. 定位根因 ─→ 确定最可能的根本原因并提供修复方案
```

### 常见故障模式分类

| 故障类型 | 典型症状 | 高概率根因 |
|---------|---------|-----------|
| **定位精度下降** | 位置抖动、轨迹不连续、误报告警 | UWB 信号质量差 (40%)、EKF 发散 (30%)、IMU 漂移 (20%) |
| **系统崩溃** | 进程异常退出、无响应、内存溢出 | 未捕获异常 (50%)、资源泄漏 (30%)、死锁 (15%) |
| **性能下降** | 延迟增加、CPU 占用高、帧率下降 | 循环中重复计算 (40%)、不必要的深拷贝 (25%)、阻塞 I/O (20%) |
| **数据异常** | 传感器数据为 NaN、位置跳变、行为检测错误 | 串口数据损坏 (35%)、单位转换错误 (30%)、边界条件未处理 (25%) |

### 分析模板

```markdown
## 故障报告

**症状描述**
[用户报告的现象]

**复现步骤**
1. [步骤 1]
2. [步骤 2]
3. [观察到的错误]

**环境信息**
- 系统版本: [版本号]
- Python 版本: [版本号]
- 依赖版本: [关键依赖]

**日志片段**
```
[相关日志]
```

**根因假设**
1. [假设 1] (概率 X%)
   - 验证方法: [如何验证]
   - 预期结果: [如果假设正确会看到什么]

2. [假设 2] (概率 Y%)
   ...

**验证结果**
- [假设 1]: ✓ 已验证 / ✗ 已排除
- [假设 2]: ...

**根本原因**
[确定的根因]

**修复方案**
- 短期: [临时解决方案]
- 长期: [根本性修复]

**预防措施**
- 添加测试: [测试用例]
- 更新文档: [文档位置]
- 改进监控: [监控指标]
```

### 经验回灌规则

每次修复后必须将 root cause 追加到 `.agent_skills/07_learning_log.md`：

```markdown
## [日期] [故障类型]

**症状**: [简短描述]
**根因**: [根本原因]
**修复**: [修复方案]
**预防**: [预防措施]
```

---

## 6. 质量闸门规范

### 6 步检查流程

```bash
#!/bin/bash
# AATF 质量闸门 - 合并前必须全部通过

# 1. 代码风格检查
flake8 . --max-line-length=100 --exclude='__pycache__,*.pyc,.claude,.agent_skills,venv,.venv'

# 2. 类型检查
mypy . --ignore-missing-imports --exclude __pycache__

# 3. 单元测试 + 覆盖率 (>=80%)
pytest tests/ --cov=. --cov-config=.coveragerc --cov-fail-under=80 -q

# 4. 安全扫描
bandit -r . -ll -q --exclude __pycache__,tests,.claude,.agent_skills

# 5. 架构合规 (core/ 不得 import adapters/ 或 standalone/)
grep -rn "from.*adapters\|import.*adapters\|from.*standalone\|import.*standalone" core/ | grep -v __pycache__

# 6. 密钥扫描
grep -rn "password\s*=\|secret\s*=\|api_key\s*=\|token\s*=" --include="*.py" . | grep -v __pycache__ | grep -v test_ | grep -v "\.md"
```

### 阈值要求

| 检查项 | 阈值 | 阻塞级别 |
|--------|------|---------|
| 代码风格 | 0 错误 | 阻塞 |
| 类型检查 | 0 错误 | 阻塞 |
| 测试覆盖率 | >= 80% | 阻塞 |
| 安全扫描 | 0 高危 | 警告 |
| 架构合规 | 0 违规 | 阻塞 |
| 密钥扫描 | 0 泄漏 | 警告 |

### 一键执行

```bash
./scripts/run_quality_gate.sh
```

---

## 7. 受控扩散修复策略

### 扩散修复流程

```
1. 扫描 ─→ grep -rn "旧接口名" --include="*.py" . > 影响清单
2. 排序 ─→ 按依赖方向: core/ → detection/ → adapters/ → standalone/ → tests/
3. 分批 ─→ 每批 <= 3 个文件
4. 执行 ─→ 修改 → pytest tests/ -v --tb=short → 通过则继续
5. 验证 ─→ scripts/run_regression_tests.sh
6. 记录 ─→ docs/decision_records/
```

### 依赖方向排序规则

修改顺序必须从被依赖方开始，向依赖方传播：

```
protocols.py → core/fusion/ → core/rules/ → core/tracking/
    → detection/ → adapters/ → standalone/ → plugin.py(只读)
```

### 硬约束

1. 禁止单次提交修改超过 5 个文件
2. 每批修改后必须运行 `pytest tests/` 全部通过才能继续下一批
3. `protocols.py` 修改必须保持向后兼容（新增字段用 `Optional` + 默认值）
4. `plugin.py` 接口禁止修改（DarkBreaker SDK 契约）
5. 不得引入循环依赖
6. 重构期间禁止同时进行功能开发
7. 重构前后测试数量不得减少
8. 重构前后覆盖率不得下降超过 2%

### 向后兼容策略

```python
# protocols.py 新增字段示例
@dataclass
class SensorData:
    x: float
    y: float
    z: float = 0.0            # 已有字段
    confidence: float = 1.0    # 新增字段，默认值保持兼容
```

---

## 8. AI 角色分工规范

### 5 个 AATF 命令对应的 AI 角色

| 命令 | 角色 | 职责 | 输入 | 输出 |
|------|------|------|------|------|
| `/bootstrap` | 环境初始化专家 | 验证环境、加载上下文、记录基准 | 项目路径 | 环境报告 + 测试基准 |
| `/implement` | 功能实现工程师 | TDD 开发、实现功能、补充测试 | 需求描述 | 功能代码 + 测试 |
| `/repair` | 故障诊断专家 | 根因分析、修复 bug、记录经验 | 故障报告 | 修复代码 + 学习日志 |
| `/audit` | 质量审计员 | 静态分析、测试验证、合规检查 | 代码变更 | 质量报告 |
| `/propagate` | 重构协调员 | 扫描影响、分批修复、回归验证 | 修改目标 | 批量修复 + 回归报告 |

### 角色协作流程

```
/bootstrap → /implement → /audit → /repair (如有问题) → /propagate (如需扩散) → /audit (最终验证)
```

### 角色切换规则

- 同一个会话中，AI 可以根据用户指令切换角色
- 每个角色有独立的提示词文件（`.claude/commands/*.md`）
- 角色切换时，AI 必须重新加载对应的提示词和上下文

---

## 9. 敏感文件权限边界

### 9 条硬约束

1. **禁止在代码或日志中硬编码密钥、令牌、密码**
2. **禁止新增联网依赖**（除非 PROJECT_CARD.md 明确批准）
3. **串口设备路径必须来自配置文件**，禁止硬编码 `/dev/ttyUSB*`
4. **日志不得输出原始传感器二进制数据**（可能含设备指纹）
5. **配置文件中不得存储认证凭据**
6. **Web UI 不得暴露文件系统路径**
7. **训练数据目录禁止包含生产环境录制数据**
8. **模型文件路径不得使用 `..` 遍历父目录**
9. **所有 `open()` 调用必须使用 `with` 语句**

### 安全检查规则

```bash
# 检查硬编码密钥
grep -rn "password\|secret\|api_key\|token" --include="*.py" . | grep -v __pycache__ | grep -v test_

# 检查配置文件中的凭据
grep -rn "password\|secret\|credential" configs/

# 安全扫描
bandit -r . -ll -q --exclude __pycache__,tests

# 检查 open() 是否都用了 with
grep -rn "open(" --include="*.py" . | grep -v "with " | grep -v __pycache__ | grep -v test_

# 检查 yaml.load (应使用 safe_load)
grep -rn "yaml\.load(" --include="*.py" . | grep -v safe_load | grep -v __pycache__
```

### 文件操作安全

- **串口通信**: 超时必须设置 `timeout <= 1s`，防止线程阻塞
- **网络通信**: Web 路由中所有用户输入必须经 pydantic 验证
- **文件操作**: 所有文件操作使用 `with` 语句
- **配置加载**: `yaml.safe_load()` 替代 `yaml.load()`
- **日志管理**: 使用 `RotatingFileHandler`，防止磁盘耗尽

---

## 10. 通用 Prompt 结构

### 标准 Prompt 格式

所有 `prompts/*.md` 文件必须遵循以下结构：

```markdown
# <提示词标题>

## 角色定义
你是一个 [角色描述]，负责 [职责描述]。

## [主要流程/分析框架]

### 1. [步骤 1]
[详细说明]

### 2. [步骤 2]
[详细说明]

### 3. [步骤 3]
[详细说明]

## [常见场景/故障模式]

### A. [场景 1]

**症状**
- [症状 1]
- [症状 2]

**可能根因**
1. **[根因 1]** (概率 X%)
   - 验证: [验证方法]
   - 日志: [日志命令]
   - 修复: [修复方案]

2. **[根因 2]** (概率 Y%)
   ...

## [输出模板/工具]

```[格式]
[模板内容]
```

## 参考文档
- [文档 1]
- [文档 2]
```

### 4 个标准 Prompt 文件

| 文件 | 用途 | 何时使用 |
|------|------|---------|
| `root_cause_prompt.md` | 故障根因分析诊断框架 | 修复 bug 时 |
| `terminal_execute_prompt.md` | 终端操作指南 | 执行命令时 |
| `web_modeling_prompt.md` | Web UI 建模规格 | 开发 Web UI 时 |
| `cross_project_reuse_prompt.md` | 跨项目迁移指南 | 迁移到其他插件时 |

---

## 迁移清单

### 可直接复用的文件（通用，不需修改）

| 文件 | 原因 |
|------|------|
| `.agent_skills/04_quality_audit.md` | 反模式清单通用 |
| `.agent_skills/05_security_boundary.md` | 安全规则通用 |
| `.agent_skills/06_refactor_policy.md` | 重构策略通用 |
| `.agent_skills/07_learning_log.md` | 经验回灌模板通用 |
| `.claude/commands/*` | 工作流命令通用 |
| `scripts/run_quality_gate.sh` | 质量门禁通用 |
| `scripts/collect_root_cause.sh` | 根因收集通用 |
| `prompts/root_cause_prompt.md` | 诊断框架通用（需改故障模式） |
| `docs/decision_records/000-template.md` | ADR 模板通用 |

### 需要定制的文件（每个项目不同）

| 文件 | 定制内容 |
|------|---------|
| `PROJECT_CARD.md` | 9 个字段全部重写: 项目名称、类型、输入源、输出目标、约束、验收标准、禁止事项、参考物、当前任务 |
| `.agent_skills/00_project_context.md` | 目录结构、依赖列表、Mock 策略、配置加载优先级 |
| `.agent_skills/01_architecture_rules.md` | 层级图、依赖方向、禁止修改列表 |
| `.agent_skills/02_algorithm_contract.md` | 降级策略、算法公式、阈值参数 |
| `.agent_skills/03_test_strategy.md` | Mock 构造方法、边界用例清单、覆盖率目标 |
| `CLAUDE.md` | 项目概述、启动命令、常见任务 |
| `configs/default.yaml` | 全部参数 |
| `scripts/run_targeted_tests.sh` | 模块名和测试文件映射 |
| `prompts/terminal_execute_prompt.md` | 项目特定命令 |
| `prompts/web_modeling_prompt.md` | Web UI 规格 |

---

## 迁移步骤

### 1. 复制通用文件

```bash
# 复制 skill 文件
cp indoor_fence/.agent_skills/04_quality_audit.md  target_plugin/.agent_skills/
cp indoor_fence/.agent_skills/05_security_boundary.md  target_plugin/.agent_skills/
cp indoor_fence/.agent_skills/06_refactor_policy.md  target_plugin/.agent_skills/
cp indoor_fence/.agent_skills/07_learning_log.md  target_plugin/.agent_skills/

# 复制命令文件
cp -r indoor_fence/.claude/commands/  target_plugin/.claude/

# 复制脚本
cp indoor_fence/scripts/run_quality_gate.sh  target_plugin/scripts/
cp indoor_fence/scripts/collect_root_cause.sh  target_plugin/scripts/

# 复制文档模板
cp -r indoor_fence/docs/decision_records/  target_plugin/docs/

# 复制 prompt 文件
cp indoor_fence/prompts/root_cause_prompt.md  target_plugin/prompts/
cp indoor_fence/prompts/cross_project_reuse_prompt.md  target_plugin/prompts/

# 复制覆盖率配置
cp indoor_fence/.coveragerc  target_plugin/
```

### 2. 创建 PROJECT_CARD.md

使用 9 字段格式，填写目标插件的具体内容：

```markdown
# PROJECT_CARD: <Plugin Name>

## 1. 项目名称
[项目名称和简短描述]

## 2. 项目类型
[plugin_new / plugin_update / plugin_refactor]

## 3. 输入源
- [输入源 1]: [协议/频率/格式]
- [输入源 2]: [协议/频率/格式]

## 4. 输出目标
- [输出 1]: [格式/频率]
- [输出 2]: [格式/频率]

## 5. 关键约束
- [约束 1]
- [约束 2]

## 6. 验收标准
- [标准 1]
- [标准 2]

## 7. 禁止事项
- [禁止 1]
- [禁止 2]

## 8. 已知参考物
- [参考 1]
- [参考 2]

## 9. 当前任务
[由使用者在每轮任务开始时填写]
```

### 3. 创建 00-03 技能文件

使用 8 段格式模板，填入目标插件的具体内容：

- `00_project_context.md`: 目录结构、依赖列表、Mock 策略
- `01_architecture_rules.md`: 层级图、依赖方向、禁止修改列表
- `02_algorithm_contract.md`: 降级策略、算法公式、阈值参数
- `03_test_strategy.md`: Mock 构造方法、边界用例清单、覆盖率目标

### 4. 创建 CLAUDE.md

参考 indoor_fence 的结构，修改项目概述和启动命令：

```markdown
# Claude 开发指南 - <Plugin Name>

## 项目概述
[项目简介]

## AATF 工作流 (7 阶段)
[复制 AATF 表格]

## 开发原则
[复制开发原则]

## 启动命令
```bash
[项目特定的启动命令]
```

## 技能文件索引
[复制技能文件索引表格]

## 命令索引
[复制命令索引表格]

## 脚本索引
[复制脚本索引表格]

## 提示词索引
[复制提示词索引表格]

## 常见任务
[项目特定的常见任务]

## 禁止事项
[项目特定的禁止事项]

## 强制工作流
[复制强制工作流]

## 提交前检查
```bash
./scripts/run_quality_gate.sh
```
```

### 5. 调整 scripts/run_targeted_tests.sh

修改模块名和测试文件映射：

```bash
#!/bin/bash
MODULE=$1

case $MODULE in
  fusion)
    pytest tests/test_fusion*.py -v
    ;;
  adapters)
    pytest tests/test_*_adapter.py -v
    ;;
  detection)
    pytest tests/test_detection*.py -v
    ;;
  rules)
    pytest tests/test_rules*.py -v
    ;;
  *)
    echo "未知模块: $MODULE"
    echo "可用模块: fusion, adapters, detection, rules"
    exit 1
    ;;
esac
```

### 6. 运行 /bootstrap 验证

确认环境和测试基准：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 运行质量门禁
./scripts/run_quality_gate.sh
```

---

## 验证清单

迁移完成后，使用以下清单验证：

- [ ] 所有通用文件已复制
- [ ] PROJECT_CARD.md 已创建并填写完整
- [ ] 00-03 技能文件已创建并填写完整
- [ ] CLAUDE.md 已创建并更新项目特定内容
- [ ] scripts/run_targeted_tests.sh 已调整模块映射
- [ ] `pip install -r requirements.txt` 成功
- [ ] `pytest tests/` 全部通过
- [ ] `./scripts/run_quality_gate.sh` 全部通过
- [ ] 目录结构符合标准模板
- [ ] 依赖方向符合架构规则

---

## 附录：PROJECT_CARD.md 9 字段详解

| 字段 | 说明 | 示例 |
|------|------|------|
| 1. 项目名称 | 插件名称和简短描述 | `indoor_fence - 室内围栏监控插件` |
| 2. 项目类型 | 新建/迭代/重构 | `plugin_update（v2.1 → v3.0）` |
| 3. 输入源 | 数据来源、协议、频率、格式 | `UWB 定位: 串口/网络协议，10Hz` |
| 4. 输出目标 | 输出内容、格式、频率 | `电子围栏告警: 进入/越界/停留事件` |
| 5. 关键约束 | 架构约束、性能约束、兼容性约束 | `core/ 禁止依赖 adapters/` |
| 6. 验收标准 | 测试覆盖率、质量门禁、功能验收 | `pytest --cov-fail-under=80 通过` |
| 7. 禁止事项 | 禁止修改的接口、禁止的操作 | `禁止修改 plugin.py 接口签名` |
| 8. 已知参考物 | 设计文档、测试矩阵、决策记录 | `docs/plans/v3-design.md` |
| 9. 当前任务 | 每轮任务开始时填写 | `修复 EKF 发散后重置不生效的 bug` |

---

## 附录：AATF 命令快速参考

| 命令 | 用途 | 何时使用 |
|------|------|---------|
| `/bootstrap` | 初始化开发环境 | 首次接触项目 |
| `/implement` | 功能开发闭环 | 添加新功能 |
| `/repair` | Bug 修复 + 经验回灌 | 修复 bug |
| `/audit` | 质量门禁检查 | 提交前审查 |
| `/propagate` | 跨文件批量修复 | 重构或批量修改 |

---

## 版本历史

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v1.0 | 2026-03-11 | 初始版本，基于 indoor_fence 插件提炼 |

---

## 维护说明

本文档应当：
- 每次 AATF 框架升级时更新
- 每次发现新的可复用模式时补充
- 每次迁移到新插件后，根据反馈优化

本文档不应当：
- 包含项目特定的业务逻辑
- 包含硬编码的配置参数
- 包含过时的或未验证的规则

---

**END OF TEMPLATE**
