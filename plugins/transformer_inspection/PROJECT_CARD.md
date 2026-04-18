# PROJECT_CARD

## Identity

- Plugin: `transformer_inspection`
- Type: `plugin-enhanced-detector`
- Main runtime: `plugin.py`
- Detector backend: `detector_enhanced.py`
- Auxiliary modules: `defect_detector.py`, `thermal_analyzer.py` (not in main runtime path)

## Current Goal

把 `transformer_inspection` 维护到“高频开发可用、低 prompt 成本可调用”的 HF 状态，重点是 skills / commands / scripts / tests 闭环，不重写主变巡视算法。

## Scope That Is Verified Today

- 插件可通过 `Plugin.create_standalone()` 初始化。
- `infer()` 能稳定返回 SDK `RecognitionResult` 列表。
- 当前已验证的状态链路只有：
  - 呼吸器硅胶
  - 油位计读数
  - 可选热像阈值告警
- 四个 HF 本地脚本可作为 implement / repair / audit 的统一入口。

## Required Local Commands

- `./scripts/run_targeted_tests.sh <module>`
- `./scripts/run_regression_tests.sh`
- `./scripts/run_quality_gate.sh`
- `./scripts/collect_root_cause.sh`

## Non-Goals For This Round

- 不改其它 `plugins/*`
- 不改 `ui/`、`platform_core/`、`darkbreaker_sdk/`
- 不把 `defect_detector.py` 或 `thermal_analyzer.py` 强行接入主链
- 不把阀门状态历史口径扩写成已验证能力

## Known Operational Limits

- 没有真实图像回放数据集。
- benchmark 仍是手动入口，不是默认 HF 门禁。
- manifest 的 capability 口径比当前运行时已验证能力更宽，引用时必须以 README 和 skills 为准。

## First Recommended Next Step After This HF Upgrade

补一个受控的 `tests/fixtures/` 小型真实样本集，把当前 regression 从”代码与 smoke 回归”推进到”最小视觉回放回归”。

## Phase 2 能力三分表 (2026-04-16 审计)

| 能力 | 分类 | 证据 |
|------|------|------|
| state_recognition | **verified** | 契约测试全通过；standalone 可运行；呼吸器硅胶+油位计链路已验证 |
| defect_detection | experimental | 代码存在(defect_detector.py)但未接入主链路；测试仅 smoke 级别；无 fixture 图像 |
| thermal_analysis | experimental | thermal_analyzer.py 存在但非主运行时路径；无热成像样本回归 |

**runtime_mode_support**: real_dl=❌ traditional_fallback=✅ simulation=✅
**测试状态**: 全量通过 (0 failures)，无 fixtures/regression 目录
**关键差距**: manifest 宣称 3 能力，仅 1 个 verified
