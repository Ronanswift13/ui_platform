# 04_quality_audit

## 1. 固定母版规则（质量审计零容忍项）

1. 禁止 `except: pass`。
2. 禁止在生产代码新增 `print()`。
3. 禁止硬编码业务阈值。
4. 禁止输出越界 `bbox` 或非法 `confidence`。
5. 禁止提交未覆盖的新逻辑分支。

## 2. 本项目差异审计规则（busbar_inspection）

### 2.1 高优先级审计项

1. **配置路径一致性**：YAML 嵌套键必须被算法读取，禁止静默 fallback。
2. **原因码一致性**：`plugin.py` 与 `detector_enhanced.py` 的原因码必须可映射。
3. **质量门禁可解释性**：`quality_failed` 必须带 `failure_reason` 与 `suggested_action`。
4. **切片坐标正确性**：remap 后 bbox 必须归一化且不越界。
5. **告警等级稳定性**：标签到 `AlarmLevel` 的映射必须固定且可测试。
6. **Runtime Truth 前置**：未完成 runtime truth 收口前，不得直接做精度优化。
7. **Tri-state 前置**：未完成 quality gate tri-state 前，不得进入 real_dl 验证。
8. **Simulator 校准职责**：simulator 必须校准真实 `check_quality_gate()`，不得只做视觉演示。
9. **ONNX Preflight 前置**：未完成 ONNX preflight 验证前，不得宣称 real_dl 已跑通。
10. **Label Compatibility Gate**：未通过 `label_contract.py` 兼容性门禁的模型，不得接入生产链路。
11. **Session != Delivery**：session 建立成功不等于可交付精度提升，审查结论只能写“接入链路已验证”。
12. **Triplet Provenance Gate**：无可追溯来源的 ONNX 不得通过猜测 sidecar 补齐 manifest/class_map。
13. **No Asset Switch During Packaging**：资产补齐阶段不得越权做默认路径切换。
14. **Triplet != Default Eligible**：三件套自洽不等于 `default_eligible`，仍需后续替代资格判定。

### 2.2 中优先级审计项

1. 动态加载检测器失败时必须抛明确信息。
2. `healthcheck` 必须包含关键运行计数。
3. standalone 启动脚本必须保持可独立运行。

## 3. 审计命令（可直接执行）

```bash
# A. 反模式扫描
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py detector_enhanced.py
rg -n "\bprint\(" plugin.py detector_enhanced.py
rg -n "TODO|FIXME|HACK" plugin.py detector_enhanced.py tests

# B. 基础质量门禁
python -m pytest tests -q

# C. 契约专项（新增后必须通过）
python -m pytest tests/test_config_contract.py tests/test_reason_code_contract.py -q

# D. 类型与风格（工具存在则执行）
python -m pip show mypy >/dev/null 2>&1 && mypy . --ignore-missing-imports || true
python -m pip show flake8 >/dev/null 2>&1 && flake8 . --max-line-length=100 --exclude=__pycache__,*.pyc || true
```

## 4. 审计结论模板（AI 输出必须使用）

```text
[AUDIT_RESULT] PASS|FAIL
[BLOCKERS] <阻断项列表，空则写 NONE>
[HIGH_RISK] <高风险项列表，空则写 NONE>
[ACTION_ITEMS] <必须执行的修复动作，按优先级排序>
```

## 5. AI 自动闭环 / 人工确认

### 可自动闭环

- 反模式替换（print -> logger、裸 except -> 具体异常）
- 补齐契约测试
- 生成审计报告与阻断清单

### 必须人工确认

- 告警等级是否符合运维值班策略
- 变焦建议动作是否与 PTZ 调度策略一致
- 质量门禁阈值是否符合现场光照条件

## 6. 测试实现常见陷阱与审查规则

### 6.1 测试数据构造陷阱

**Root Cause**: 测试图像缺乏足够纹理导致质量评分异常

**触发条件**:
- 使用纯色或低噪声图像测试质量门禁
- 质量评分算法依赖边缘/梯度特征（如拉普拉斯方差）
- 测试期望触发特定质量失败类型（过曝/欠曝/低对比），但实际先触发模糊检测

**错误症状**:
```python
# 错误：纯色图像拉普拉斯方差为0，总是触发模糊检测
overexp_image = np.ones((480, 640, 3), dtype=np.uint8) * 250
result = detector.check_quality_gate(overexp_image)
assert result.status == QualityGateStatus.FAIL_OVEREXPOSED  # 失败！实际是 FAIL_BLUR
```

**正确范式**:
```python
# 正确：添加足够纹理，确保清晰度评分通过，再触发目标质量问题
def _create_textured_image(base_value, noise_range=20):
    """创建具有纹理的图像，避免纯色导致的低清晰度"""
    image = np.ones((480, 640, 3), dtype=np.uint8) * base_value
    noise = np.random.randint(-noise_range, noise_range, (480, 640, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image

overexp_image = _create_textured_image(240, noise_range=30)  # 高亮度 + 足够纹理
```

**审查规则**:
1. 质量门禁测试必须考虑检测顺序：模糊 -> 亮度 -> 对比度 -> 遮挡
2. 测试图像必须有足够纹理（拉普拉斯方差 > clarity_threshold）才能测试后续质量项
3. 使用 `np.random` 生成纹理时，噪声范围需根据目标亮度调整（亮度越极端，需要越大噪声）

### 6.2 配置读取与测试环境不一致

**Root Cause**: 生产代码从顶层配置键读取，测试提供嵌套配置结构

**触发条件**:
- 生产代码直接读取 `config.get('clarity_threshold')`
- 测试 fixture 提供嵌套结构 `{'quality': {'blur_thr': 0.35}}`
- 配置适配器尚未集成到生产代码

**错误症状**:
```python
# 测试 fixture 提供嵌套配置
config = {'quality': {'blur_thr': 0.35}}
detector = BusbarDetectorEnhanced(config)

# 生产代码读取顶层键，回退到默认值 0.5
self._clarity_threshold = config.get("clarity_threshold", 0.5)  # 得到 0.5，不是 0.35
```

**正确范式**:
```python
# 测试必须提供生产代码实际读取的配置键
config = {
    'quality': {'blur_thr': 0.35},  # 保留嵌套结构（未来集成用）
    'clarity_threshold': 0.35,       # 当前生产代码读取的顶层键
    'brightness_range': (0.2, 0.8),
    'contrast_threshold': 0.3,
}
```

**审查规则**:
1. 测试配置必须与生产代码读取路径一致，不能假设配置适配器已集成
2. 新增配置适配器时，先写测试验证映射正确性，再集成到生产代码
3. 配置适配器测试必须覆盖：完整映射、部分映射、缺失键回退默认值

### 6.3 覆盖率门槛与模块集成状态不匹配

**Root Cause**: 新增模块有完整单元测试但未被生产代码导入，导致覆盖率统计为0%

**触发条件**:
- 创建新模块（如 config_adapter.py）并编写完整单元测试
- 新模块尚未集成到生产代码（plugin.py/detector_enhanced.py）
- 回归测试脚本设置固定覆盖率门槛（如 70%）

**错误症状**:
```bash
# 新模块测试100%通过，但覆盖率统计显示0%
config_adapter.py: 0% (31/31 miss)
reason_code_mapper.py: 0% (19/19 miss)

# 总体覆盖率低于门槛
FAIL Required test coverage of 70% not reached. Total coverage: 38.15%
```

**正确范式**:
```ini
# .coveragerc - 临时排除未集成模块
[run]
omit =
    */config_adapter.py
    */reason_code_mapper.py

# 设置阶段性覆盖率门槛
[report]
fail_under = 30  # 第一轮补齐：30%，待集成后提升到70%
```

```bash
# run_regression_tests.sh - 使用配置文件门槛，不硬编码
pytest tests/ --cov=plugins.busbar_inspection --cov-report=term-missing
# 移除 --cov-fail-under=70，改用 .coveragerc 中的 fail_under
```

**审查规则**:
1. 新增模块分两阶段：先实现+测试（可排除覆盖率统计），后集成+移除排除
2. 覆盖率门槛应在 .coveragerc 中配置，不在脚本中硬编码
3. 覆盖率配置必须注释说明阶段性目标和提升计划
4. 模块集成后必须移除 omit 排除项并提升 fail_under 门槛

### 6.4 测试脚本参数覆盖配置文件

**Root Cause**: 命令行参数优先级高于配置文件，导致配置文件设置被忽略

**触发条件**:
- .coveragerc 设置 `fail_under = 30`
- 脚本使用 `--cov-fail-under=70`
- 命令行参数覆盖配置文件

**错误症状**:
```bash
# .coveragerc 中设置了 fail_under = 30
# 但脚本仍然使用 70% 门槛
pytest --cov-fail-under=70  # 命令行参数优先级更高
```

**正确范式**:
```bash
# 方案1：移除命令行参数，使用配置文件
pytest tests/ --cov=plugins.busbar_inspection --cov-report=term-missing

# 方案2：从配置文件读取并传递
COVERAGE_THRESHOLD=$(grep fail_under .coveragerc | awk '{print $3}')
pytest --cov-fail-under=$COVERAGE_THRESHOLD
```

**审查规则**:
1. 覆盖率门槛优先在 .coveragerc 中配置，便于版本控制和文档化
2. 脚本中的 --cov-fail-under 参数应移除或从配置文件读取
3. 配置文件修改后必须验证脚本是否正确读取

## 7. 第一轮优先整改模块建议（本项目当前最适合）

**模块名：`config_reason_contract`（配置映射 + 原因码统一层）**

选择理由（可验证）：

1. 当前配置为嵌套 YAML，算法初始化读取顶层键，存在高概率静默默认值风险。
2. 当前质量门禁返回 `1001~1005`，插件文案字典使用 `101~105`，存在语义断裂。
3. 该模块不改动检测算法本体，只修契约一致性，风险最低、收益最高。

第一轮完成标准：

1. 新增配置映射适配器并通过 `test_config_contract.py`。
2. 新增原因码映射并通过 `test_reason_code_contract.py`。
3. `run_targeted_tests.sh config` 返回 0。
