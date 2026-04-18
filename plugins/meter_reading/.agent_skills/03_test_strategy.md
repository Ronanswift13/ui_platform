# 03_test_strategy

## 1. 固定母版规则（测试治理）

1. 每条硬约束至少对应一个自动化测试锚点。
2. 每次缺陷修复必须新增防回归测试。
3. targeted、regression、quality gate 入口必须职责分离。
4. 空测试目录必须显式 skip，不允许假绿色。

## 2. 当前真实测试资产（meter_reading）

| targeted 模块 | 测试文件 | 关注点 |
|---|---|---|
| `analog` | `tests/test_analog_meter.py` | 指针角度、量程映射、降级链、模拟表 metadata |
| `digital` | `tests/test_digital_ocr.py` | OCR 清洗、非法串拒绝、数字表 metadata |
| `led` | `tests/test_led_indicator.py` | HSV 分类、可分离性、颜色编码、LED metadata |
| `validation` | `tests/test_input_validation.py` + `tests/test_confidence.py` | 输入兜底、置信度清洗、三态状态集 |
| `plugin` | `tests/test_plugin_integration.py` | `init/infer/postprocess/healthcheck`、多 ROI、上下文透传 |
| `contract` | `tests/test_output_structure.py` | 输出结构、metadata 必填字段、告警对象约束 |
| `replay` | `tests/test_replay.py` | expected_results schema、最小 fixture 目录、LED mock replay、真实样本缺失显式 skip |
| `all` | `tests/` | targeted + replay 用例总入口 |

## 3. 回归测试现状与规则

1. `tests/regression/` 当前为空。
2. `tests/fixtures/` 已建立最小六类目录，只有 `led_indicator` 有 mock image spec。
3. `tests/replay/expected_results.json` 已固定 planned slots 和输出 metadata schema。
4. 因此当前可执行的是“代码级回归 + mock replay plumbing”，不能宣称已经完成“标定样本回归”。
5. 脚本必须输出类似以下事实之一：
   - `[SKIP] tests/regression/ has no test_*.py`
   - `[SKIP] no present_labeled real-image replay samples`

## 4. meter_reading 必测重点

### P0：任何实现改动都优先关注

1. 模拟表 `fallback_level`、角度越界、量程缺失、metadata 完整性。
2. 数字表 OCR 清洗规则与 `ocr_text_raw` 保留。
3. LED 的 `off/red/green/yellow` 编码与 `hsv_not_separable` 复核路径。
4. 插件 `postprocess()` 对失败和人工复核的告警行为。
5. `ReadingStatus` 三态不漂移，`need_manual_review` 语义不漂移。

### P1：触及相关模块时必须补

1. `reload_config()` 的保旧配置行为。
2. `processing_time_ms` / `latency_violation` 的边界行为。
3. `standalone/` 路由、模拟场景和视频流 smoke。
4. `tests/regression/` 与 `tests/fixtures/` 的首批数据集回归入口。

## 5. 脚本职责契约

### `scripts/run_targeted_tests.sh`

1. 必须支持模块化执行。
2. 模块无测试文件时必须非零退出，阻断假绿色。
3. 只运行快速用例，不承诺数据集回归。

### `scripts/run_regression_tests.sh`

1. 必须先跑 `run_targeted_tests.sh all`。
2. 必须执行全量 `pytest`。
3. 若 `tests/regression/` 或 `tests/fixtures/` 为空，必须显式报告 skip。
4. 工具存在时执行静态/安全检查。

### `scripts/run_quality_gate.sh`

1. 必须先做快速架构/反模式检查。
2. 必须串联 regression gate。
3. 审计输出里必须保留 regression 是否真实执行的数据证据。

## 6. 阻断条件

任一命中即阻断：

1. `./scripts/run_targeted_tests.sh all` 非零。
2. `./scripts/run_regression_tests.sh` 非零。
3. 改动了 contract / metadata / 状态机，却没有更新对应测试。
4. 把空的 regression/fixtures 目录描述成“已完成回归”。
