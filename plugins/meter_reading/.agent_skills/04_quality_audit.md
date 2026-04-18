# 04_quality_audit

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`。
2. 禁止在生产代码新增 `print()`。
3. 禁止把业务阈值重新写回推理主路径。
4. 禁止输出越界 `confidence` 或缺失必填 metadata。
5. 禁止未执行命令却宣称通过。

## 2. 本项目差异审计项（meter_reading）

### P0：阻断级

1. `ReadingStatus` 是否仍严格三态。
2. `meter_type / reading_status / pipeline_stage / fallback_level / timestamp_ms` 是否仍为所有结果必填。
3. 模拟表是否仍保留 `HRNet -> HoughCircle -> HoughLine` 降级链。
4. 缺失量程、OCR 非法串、HSV 不可分离是否仍进入 `NEED_MANUAL_REVIEW`，而不是伪造成功结果。
5. `detector_enhanced.py` 是否仍不依赖 SDK / `standalone/`。
6. 回归结论是否区分“代码级回归已跑”和“数据集级回归未就位”。

### P1：高风险

1. `reload_config()` 非法值是否保持旧配置。
2. `healthcheck()` 是否仍输出关键统计字段。
3. `postprocess()` 的失败/复核告警是否仍与当前契约一致。
4. `standalone/` 相关改动是否给出 smoke 说明。

## 3. 可执行审计命令

```bash
# A. 架构与反模式
rg -n "darkbreaker_sdk|standalone" detector_enhanced.py
rg -n "\bprint\(" plugin.py detector_enhanced.py
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py detector_enhanced.py
rg -n "LOW_CONFIDENCE" plugin.py detector_enhanced.py

# B. 合约与测试
./scripts/run_quality_gate.sh
python3 -m pytest tests/test_output_structure.py tests/test_confidence.py -q

# C. 空回归目录检查
find tests/regression -maxdepth 1 -name "test_*.py" | sort
find tests/fixtures -maxdepth 2 -type f | sort
```

## 4. 审计输出模板（必须使用）

```text
[AUDIT_RESULT] PASS|FAIL

[BLOCKERS]
<阻断项列表，空则写 NONE>

[HIGH_RISK]
<高风险项列表，空则写 NONE>

[ACTION_ITEMS]
<必须执行的修复动作，按优先级排序>

[EVIDENCE]
- quality gate 关键输出
- regression 是否真实执行 / 是否 skip
- 关键 rg / pytest 证据
```

## 5. 可回灌到通用模板的审查规则

1. 空 `tests/regression/` 或空 `tests/fixtures/` 必须显式 skip。
2. 状态枚举漂移必须有测试锁定，不能靠人工记忆。
3. metadata 必填字段必须有独立测试文件锁定。
4. “无量程/无合法 OCR/无可分离 HSV” 都属于典型的幻象默认值风险，适合沉淀为跨插件审查项。
