# Engineering Baseline Playbook — "模型缺席时的工程契约完备"

> **来源**：bird_monitoring 插件 2026-04-16 「最小治理 → 标准治理」转换
> **适用**：任何处于 DarkBreaker「低数据 / 弱模型 / 强工程治理」阶段的插件
> **哲学**：没有模型不是不交付的理由。工程契约、降级路径、复核闭环、训练占位先行。

---

## 0. 判定是否适用本 Playbook

若你的插件满足以下任一条件，建议按本 Playbook 升级：
- [ ] tests/ 为空或 <10 个测试
- [ ] 生产代码存在 `print()`
- [ ] 业务阈值硬编码在类常量中
- [ ] 未知输入 / 未识别类别会"默认"到某个具体值（如默认 sparrow / 默认正常）
- [ ] 异常场景可能导致插件 crash
- [ ] runtime_mode 在外部不可读
- [ ] manifest.json 没有 `verified / experimental / blocked` 三分
- [ ] 插件"声称"控制硬件（声学 / 激光 / 电机 / 继电器）

---

## 1. 七步转换法（按顺序执行）

### Step 1 — 能力三分（1 小时）
把所有 `capabilities` 分桶到：
- **verified**：有测试 + 有 fixture 证明；现在就能交付
- **experimental**：代码存在但测试不足
- **blocked**：代码存在但阻塞于真实模型 / 真实硬件 / 真实数据

写入 `manifest.json`（新增三个字段），同步 `PROJECT_CARD.md § Phase 2 能力三分表`。

### Step 2 — 语义纠偏（1 小时）
扫描插件中"控制类"名词：
- `_control` / `_trigger` / `_execute` → 若实际不对接硬件，必须改为 `_suggestion` / `_advisory` / `_hint`
- 在 README 明示："本插件不控制任何真实硬件，仅输出建议"
- 在 manifest `blocked_capabilities` 加 `*_hardware_control`

### Step 3 — 未知 / 未识别回退（30 分钟）
找出所有"默认到某具体值"的地方（伪数据风险）：
- ❌ 未命中物种 → 默认 sparrow
- ❌ OCR 失败 → 默认 "0"
- ❌ 分类失败 → 默认 "normal"

全部改为：
- ✅ 返回 `unknown_*` / `review_required`
- ✅ 置信度归 0
- ✅ 对未知做"保守评估"（对业务最安全的方向）

### Step 4 — 质量门（1 小时）
在算法层（detector）新增 `assess_image_quality(image)` 返回：
```python
{
    "is_valid": bool,
    "issues": List[str],          # 中文描述，便于前端展示
    "brightness_score": float,    # 灰度均值归一
    "clarity_score": float,       # Laplacian 方差
    "overall_score": float,       # 0-100
}
```

在 plugin 层新增 `_assess_input_quality(frame, roi)` 做第一道闸。质量不达标 → 返回 `label="quality_failed"` 结果，**不进入推理**。

阈值从 `config.quality.*` 读取：
```yaml
quality:
  min_dimension: 64
  clarity_threshold: 50.0
  review_confidence_threshold: 0.5
  species_confidence_threshold: 0.6   # 若有分类链路
```

### Step 5 — 复核路径（1 小时）
置信度低于阈值 → 标记 `review_required`，不当作正常结果。
- 在 Detection dataclass 加 `review_required: bool` + `review_reason: str`
- 在 plugin.infer 中判断 `confidence < config.quality.review_confidence_threshold`
- 在 postprocess 中：`review_required` → `AlarmLevel.WARNING` 告警
- 新增 label `review_required`（记得更新 test 白名单）

### Step 6 — 训练占位（30 分钟）
为未来模型迭代预留数据闭环，每条 result 附：
```python
metadata["training_placeholders"] = {
    "hard_negative_candidate": bool,
    "hard_positive_candidate": bool,
    "suggested_label_for_dataset": str,
    "annotation_status": "pending" | "reviewed" | ...,
    "model_placeholder": str,  # 说明当前用的是 simulation / fallback
}
```

把它抽成 `_make_training_placeholders()` 工具方法。

### Step 7 — 可观测 runtime_mode（30 分钟）
- detector 层暴露 `runtime_mode` property: `"simulation" | "real_dl" | "traditional_fallback"`
- plugin.healthcheck().details 含 `runtime_mode`
- 每条 result 的 metadata 含 `runtime_mode`
- 测试断言 `runtime_mode in ("simulation", "traditional_fallback", "real_dl")`

---

## 2. 测试矩阵模板（移植自 bird_monitoring）

| 文件 | 维度 | 最少测试数 | 必覆盖 |
|---|---|---:|---|
| test_standalone.py | 生命周期 | 10 | create_standalone / healthcheck / infer / cleanup / code_hash / 未初始化 infer 返回 error/9000 |
| test_quality_assessment.py | 输入质量 | 5 | 合法 / None / 空 / 极小 / 过暗 |
| test_risk_assessment.py 或 test_core_logic.py | 业务核心 | 10 | 边界 + 未知回退 + 保守评估 |
| test_plugin_contract.py | 结果结构 + 告警 | 10 | 必需字段 / training_placeholders / runtime_mode / 告警分级 / 无告警分支 |

**conftest.py 必备夹具**：
- `plugin_dir`
- `default_config`（含 quality 段）
- `plugin_instance`（create_standalone）
- `sample_frame`（合法尺寸）
- `tiny_frame` / `dark_frame`（质量门必败）
- `make_context` 工厂
- `make_roi` 工厂（含 name / component_id / roi_type）

**跨插件 import 防御**：所有 test 必须使用 `from plugins.<plugin_id>.plugin import ...` 完整路径，避免 pytest 并行解析冲突。

---

## 3. 禁止行为（跨插件零容忍）

| 行为 | 原因 | 替代 |
|---|---|---|
| 生产代码 `print()` | 日志污染 | `logger.info/warning/error` |
| `except: pass` | 静默失败 | 捕获 + 日志 + 返回 error 结果 |
| 业务阈值硬编码 | 无法运营调优 | `config.<section>.<key>` |
| 未识别 → 默认具体值 | 伪数据 | `unknown_*` / `review_required` |
| simulation 返回合成数据 | 伪数据 | 返回 `[]` |
| 非硬件插件对接硬件接口 | 语义错位 | 输出 `*_suggestion` JSON |
| infer 前未检查初始化 | 静默崩溃 | 返回 `label="error", failure_reason="9000"` |
| 新增 label 不更新测试白名单 | 破坏契约 | 改测试同时改实现 |

---

## 4. 准出清单（Definition of Done）

提交前必须全绿：

- [ ] 插件内 pytest 全绿
- [ ] 跨插件 pytest（至少 6 个插件）无退化
- [ ] `rg '\bprint\(' plugin.py detector.py` → 0 命中
- [ ] `rg 'except.*pass' *.py` → 0 命中
- [ ] manifest.json 含 `verified_capabilities / experimental_capabilities / blocked_capabilities / runtime_mode_support / current_known_limits`
- [ ] README.md 存在且含「运行模式 / 输入输出契约 / label 表 / 已知边界」
- [ ] PROJECT_CARD.md 含 Phase 2 能力三分表
- [ ] `.agent_skills/07_learning_log.md` 追加本次变更 Entry（Symptom / Root cause / Fix / Prevention / Validation）
- [ ] 每条 result.metadata 含 `runtime_mode` + `training_placeholders`
- [ ] healthcheck 暴露 `runtime_mode`
- [ ] 未初始化 infer 返回 `error/9000`

---

## 5. 9 种 label 参考（bird_monitoring 实战）

可作为其他插件 label 设计的参考模板（视场景裁剪）：

| label | 含义 | 典型告警等级 |
|-------|------|----------|
| `no_<target>` | 未检测到目标 | — |
| `<target>_safe` | 检测到但风险低 | — |
| `<target>_warning` | 警戒 | — |
| `<target>_danger` | 危险 | ERROR |
| `<target>_critical` | 紧急 | CRITICAL |
| `review_required` | 需人工复核（低置信度） | WARNING |
| `unknown_<target>` | 未识别子类 | WARNING |
| `quality_failed` | 输入质量不达标 | WARNING |
| `error` | 插件未初始化 / 运行异常 | — |

---

## 6. 耗时参考

bird_monitoring 实际转换耗时约一个工作日（含：
- Phase 1 审计与去冗余：~1h
- plugin.py 重写：~2h
- detector.py 改造：~30min
- 测试 4 个文件：~2h
- manifest / PROJECT_CARD / README：~1h
- 跨插件回归 + 修导入冲突：~30min

建议每个插件按同一节奏执行，避免"小修小补累积技术债"。

---

## 7. 参考实现

- plugin.py：`plugins/bird_monitoring/plugin.py`
- detector.py：`plugins/bird_monitoring/detector.py`（含 `assess_image_quality` / `runtime_mode` property）
- 配置：`plugins/bird_monitoring/configs/default.yaml`
- 测试：`plugins/bird_monitoring/tests/`（4 文件 / 39 测试）
- 文档：`plugins/bird_monitoring/README.md`、`PROJECT_CARD.md`、`.agent_skills/07_learning_log.md`
