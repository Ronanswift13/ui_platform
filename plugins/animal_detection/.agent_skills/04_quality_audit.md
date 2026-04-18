# 04 Quality Audit

## 代码质量检查清单

### 每次提交前
- [ ] 无 TODO/FIXME/HACK 残留
- [ ] 无硬编码魔数 (阈值走配置)
- [ ] 类型注解完整 (public 方法)
- [ ] 异常处理: 捕获具体异常，不吞 traceback
- [ ] 日志: 关键路径有 logger.info/warning

### 算法变更时
- [ ] 回归测试通过
- [ ] 精度不劣化 (与基线比对)
- [ ] 降级链路仍可用 (模型缺失/热成像缺失/跟踪器异常)
- [ ] 性能不劣化 (P95 ≤ 100ms 单帧检测)

### 配置变更时
- [ ] default.yaml schema 一致性
- [ ] 新参数有默认值
- [ ] 参数范围有效性校验

## 指标监控

| 指标 | 阈值 | 采集方式 |
|------|------|----------|
| 单帧检测延迟 P95 | ≤ 100ms | 内置计时 (detector.get_stats) |
| 召回率 | ≥ 85% | 测试集评估 |
| 精确率 | ≥ 80% | 测试集评估 |
| 误报率 | < 5% | 24h 连续运行 |
| 内存占用 | ≤ 512MB | psutil 采样 |

---

## 根因分析与普适审查规则

> 以下规则源自 meter_reading 插件的 V3.0/V3.1 迭代和 animal_detection 自身的质量闸门回顾，
> 经验证为跨插件通用的算法质量审查规则。每条规则以 QR-N 编号。

### RC-1: 算法代码中的隐式阈值脱管

**核心 root cause**:
算法层在初始化阶段正确地从配置中读取了 `confidence_threshold`、`nms_threshold` 等顶层参数，但在推理中间环节直接写入字面量（如 bbox_clamped 时 `score_i *= 0.8`），绕过了配置值。根本原因是配置粒度不足——YAML 只定义了顶层门限，未覆盖中间环节的策略参数。

**在 animal_detection 中的体现**:
- `detector.py:430` — bbox 越界时置信度衰减系数 `0.8` 为硬编码
- 置信度阈值按动物类型不同应有差异化（合约定义: 鼠 0.5, 蛇 0.6, 鸟 0.4），但当前统一使用 `confidence_threshold`

**普适审查规则 QR-1**:
> 凡是在非初始化路径中出现的 `0.x` 浮点字面量，若其语义为"质量评分 / 阈值 / 权重 / 衰减系数"，则必须在配置中有对应条目。审查时用 `grep -rn '[^a-zA-Z_]0\.[0-9]' --include='*.py'` 扫描，逐条判定来源。

---

### RC-2: 工程骨架缺失导致合约与实现脱节

**核心 root cause**:
项目在算法演进到一定规模时，仍无算法合约文档、无测试文件、无质量门禁脚本。合约隐含在开发者脑中，从未被显式记录。

**普适审查规则 QR-2**:
> 任何算法模块的代码行数超过 300 行时，必须存在同级的算法合约文档（定义输入/输出/不变量/异常兜底），且合约中的每条硬约束必须有对应的自动化测试。无合约的算法代码视为"未定义行为"，不允许进入主干。

---

### RC-3: 降级链路的行为规范仅靠代码注释传递

**核心 root cause**:
animal_detection 的降级策略（ONNX 加载失败 → 空检测、热成像不可用 → 跳过验证、跟踪器异常 → 重置状态）分散在多个模块中，无集中定义，无独立测试验证降级行为。

**普适审查规则 QR-3**:
> 凡存在 A → B → C 级联降级的逻辑，必须满足三条：(1) 合约中有降级策略表且标注每级的触发条件和行为；(2) 每级降级有独立测试用例模拟故障场景；(3) 输出 metadata 中携带降级标记供外部审计。

---

### RC-4: 配置文件与算法层的参数路径不对齐

**核心 root cause**:
`configs/default.yaml` 中定义了嵌套结构（如 `inference.confidence_threshold`），但算法层直接通过构造函数参数接收。中间的映射关系（在 plugin.py 中）未被测试覆盖。一旦 YAML 结构调整，算法层可能静默使用默认值。

**普适审查规则 QR-4**:
> 凡是用 `dict.get(key, default)` 模式从配置中读取参数的代码，必须有对应的测试验证：(1) YAML 中有该键时读到的是 YAML 值；(2) YAML 中无该键时读到的是声明的默认值且该默认值与文档一致。静默 fallback 到默认值是配置系统最常见的隐性 bug 来源。

---

### RC-5: 置信度语义在不同链路中含义漂移

**核心 root cause**:
animal_detection 中 `confidence` 来源包括：YOLO 模型输出的 class_score、热验证后的调整值、bbox 越界后的衰减值。不同来源的 confidence 语义不一致——YOLO 输出 0.7 与热验证后 0.7 代表不同的可信程度，但下游统一用同一阈值切分。

**普适审查规则 QR-5**:
> 凡是多条处理链路共享同一个质量评分字段（confidence / score / quality），必须在合约中定义统一的计算语义。不同链路不可用不同的公式各自赋值后直接写入同一字段。若短期无法统一，则 metadata 中必须携带 `confidence_source` 标签，供下游区分来源。

---

### RC-6: 系统 Python 运行缺依赖 — venv 未激活

**核心 root cause**:
`run_standalone.py` 未检查当前解释器是否在虚拟环境中。当用户用系统 Python（如 `/opt/homebrew/bin/python3.13`）运行时，缺少 numpy 等依赖，立即抛出 `ModuleNotFoundError`。macOS 受 PEP 668 保护，系统级 `pip install` 被拒绝。

**当前状态**: 已在 `run_standalone.py` 顶部增加 venv guard（2026-03-25 修复）。

**普适审查规则 QR-6**:
> 所有插件的 `run_standalone.py` 必须包含 venv 自动激活守卫。守卫逻辑必须在任何业务 import 之前执行。检测方法：`grep -L 'sys.base_prefix' plugins/*/run_standalone.py` 找出缺少守卫的入口文件。

---

### RC-7: SDK `__init__.py` 顶层 import 重量级依赖

**核心 root cause**:
SDK 的 `__init__.py` 在模块顶层 `import numpy`，导致任何 `from darkbreaker_sdk import ...` 都会触发 numpy 加载。在缺少 numpy 的轻量级环境中（CLI 工具、类型检查），SDK 不可导入。

**普适审查规则 QR-7**:
> `__init__.py` 和被其直接或间接 import 的模块，顶层不得引入 C 扩展依赖（numpy, opencv, onnxruntime 等）。如需使用，应在方法体内延迟导入或放入独立子模块。

---

### RC-8: 枚举状态集与合约不一致

**核心 root cause**:
枚举定义可能在不更新合约的情况下被扩展，导致下游 `if/elif` 分支有未覆盖的状态。

**在 animal_detection 中的检查点**:
- `AnimalClass` 枚举必须与合约定义的动物类型严格一致（当前 8 类）
- `EventType` 枚举必须与合约定义的事件类型严格一致（当前 10 类）
- `RiskLevel` 枚举必须与合约定义的风险等级严格一致（当前 4 级）

**普适审查规则 QR-8**:
> 凡是合约中定义了有限状态集的枚举，代码中的枚举成员数量和值域必须与合约严格一致。测试中必须包含 `len(Enum)` 断言和值集合断言。新增枚举成员必须先更新合约，再更新代码，最后更新所有消费该枚举的 `if/elif` 分支。

---

### RC-9: 越界值被静默截断而非显式拒绝

**核心 root cause**:
代码中用 `max(0, min(1, x))` 截断连续值，使得越界值仍然输出看似合法的结果。在 animal_detection 中，bbox 归一化坐标的截断（detector.py:419-422）虽已加 warning 和置信度衰减，但仍然输出结果而非拒绝。

**普适审查规则 QR-9**:
> 凡是在算法链路中出现 `max(0, min(1, x))`、`np.clip`、`clamp` 等截断操作的代码，必须在截断前有显式的合法域判定分支。若原始值超出物理/合约定义的合法域，应当记录警告并降低置信度或进入拒绝分支，不得截断后继续输出高置信度结果。

---

### RC-10: 数据缺失时用幻象默认值掩盖空洞

**核心 root cause**:
查找/匹配方法返回"兜底默认值"而非 `None` / 异常，使错误数据进入下游。

**在 animal_detection 中的检查点**:
- `class_map.get(cls_id, AnimalClass.OTHER)` — 未知类别 ID 映射为 OTHER 而非标记为异常。当前可接受（OTHER 有对应的处置流程），但应在 metadata 中记录 `original_class_id`。
- `ANIMAL_RISK_MAP.get(d.animal_class, RiskLevel.MEDIUM)` — 未知动物类别默认为 MEDIUM 风险。

**普适审查规则 QR-10**:
> 凡是从映射表/配置/注册表中查找数据的方法，当查找目标不存在时，必须返回 `None` 或抛出特定异常，不得返回构造的"合理默认值"。如因兼容性必须有默认值，则 metadata 中记录 `fallback_used: true` 及原始查找键。

---

### RC-11: 浮点质量信号未在赋值入口做合法性清洗

**核心 root cause**:
`confidence` 字段可能因模型输出 NaN、除零、极端值等产生超出 `[0, 1]` 的值。

**当前状态**: 已实现 `_sanitize_confidence()` 纯函数（event_schema.py:23, detector.py:32），在赋值入口调用。有独立测试 `test_confidence_sanitization.py` 覆盖 NaN/Inf/负数/超1 四种边界。

**普适审查规则 QR-11**:
> 凡是输出 schema 中定义了有限定义域的浮点字段（如 confidence ∈ [0,1]），代码中每个赋值入口必须经过清洗函数。清洗函数必须是独立纯函数，有独立单元测试覆盖 NaN / Inf / 负数 / 超上界四种边界。审查时搜索 `\.confidence =` / `result.confidence =` 等赋值语句，检查右侧是否经过清洗。

---

### RC-12: 输出 metadata 字段与合约定义脱节

**核心 root cause**:
输出结构的 metadata 字段未被测试锁定，缺失的字段从未被发现。

**在 animal_detection 中的检查点**:
- `AnimalDetectionResult.to_dict()` 必须包含合约定义的所有字段
- `AnimalEvent.to_dict()` 必须包含统一事件契约的所有核心字段
- 已有 `test_event_schema_contract.py` 锁定字段集

**普适审查规则 QR-12**:
> 凡是合约中定义了必填字段的输出结构（尤其是 `Dict[str, Any]` 类型的 metadata），每个必填字段必须有测试断言锁定其存在性。metadata 的构造必须集中在 builder 函数中，不允许在业务逻辑中零散赋值。

---

### RC-13: 输入清洗规则必须有独立纯函数 + 独立测试

**核心 root cause**:
输入清洗逻辑内嵌在业务方法中，无法独立测试和复用。

**在 animal_detection 中的检查点**:
- 图像帧尺寸验证（最大 4096x4096）应提取为独立函数
- ROI 坐标裁剪逻辑（detector.py:205-212）应提取为独立函数
- `_sanitize_confidence()` 已满足此规则 ✅

**普适审查规则 QR-13**:
> 凡是合约中定义了输入清洗/校验规则的链路，清洗逻辑必须提取为独立的纯函数，不得内嵌在业务方法中。该函数必须有独立的参数化测试。审查时检查业务方法中是否仍有内联的校验逻辑。

---

### RC-14: 架构文档描述了未实现的行为

**核心 root cause**:
文档描述与代码实现不一致，制造虚假的安全感。

**在 animal_detection 中的检查点**:
- 合约描述了热验证流程，但 `thermal.enabled: false` 时无实际调用 — 需确保文档标注此为可选功能
- 合约描述了驱离控制，但 `deterrent.enabled: false` 时同理
- ByteTrack 跟踪流程在合约中描述，需确认 `core/tracker.py` 实现完整

**普适审查规则 QR-14**:
> 凡是合约或架构文档中描述的流水线步骤、处理阶段、功能模块，代码中必须有可定位的对应实现，且至少有一个测试验证该实现被调用或产生了预期效果。文档中描述但代码中未实现的行为比没有文档更危险。

---

## 审查规则汇总表

| 规则 ID | 规则名称 | 检测方法 | 适用阶段 |
|---------|----------|----------|----------|
| QR-1 | 中间环节无脱管阈值 | `grep -rn '0\.[0-9]' *.py` + 逐条判定来源 | 每次提交 |
| QR-2 | 算法合约覆盖度 | 代码行数 > 300 → 检查同级合约 + 测试映射 | 每次新模块 |
| QR-3 | 降级链路可审计 | 检查降级策略表 + 降级测试 + metadata 标记 | 链路变更时 |
| QR-4 | 配置路径对齐 | YAML 键 → plugin 映射 → 算法层读取三者一致 | 配置变更时 |
| QR-5 | 质量评分语义一致 | 同一 confidence 字段的不同赋值路径必须语义对齐 | 算法变更时 |
| QR-6 | 运行入口环境自愈 | run_standalone.py 必须含 venv 自动激活守卫 | 新插件创建时 |
| QR-7 | 顶层 import 依赖可控 | `__init__.py` 顶层不引入 C 扩展依赖 | 每次提交 |
| QR-8 | 枚举状态严格等于合约定义 | `len(Enum)` 断言 + 值集合断言 | 枚举变更时 |
| QR-9 | 越界值显式处理不得静默截断 | 搜索 `max(0, min(` / `clip` / `clamp` + 判定语义 | 算法变更时 |
| QR-10 | 数据缺失不得用幻象默认值掩盖 | 搜索 fallback 默认值 + 检查 metadata 标记 | 算法变更时 |
| QR-11 | 浮点质量信号必须在赋值入口清洗 | 搜索 `confidence =` 赋值点 + 检查清洗守卫 | 每次提交 |
| QR-12 | 输出 schema 字段必须由合约驱动测试锁定 | 必填字段有 `assert key in` 测试 | 合约变更时 |
| QR-13 | 输入清洗规则必须有独立纯函数 + 独立测试 | 清洗逻辑不得内嵌在业务方法中 | 新增链路时 |
| QR-14 | 文档描述的行为必须有代码实现和测试锚点 | 文档功能声明 → `grep` 检查对应实现 | 文档变更时 |

---

## 执行命令

```bash
# 运行完整质量闸门
./scripts/run_quality_gate.sh

# 运行针对性测试
./scripts/run_targeted_tests.sh

# 查看审计日志
cat logs/quality_audit.jsonl | jq .
```

## 审计日志格式

```json
{
  "timestamp": "2026-03-26T10:00:00Z",
  "commit": "abc1234",
  "author": "developer",
  "quality_gate": {
    "flake8": "PASS",
    "tests": "PASS",
    "coverage": "72%",
    "architecture": "PASS",
    "security": "PASS"
  },
  "result": "PASS"
}
```
