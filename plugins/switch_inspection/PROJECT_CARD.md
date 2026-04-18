# PROJECT_CARD

## Identity

- Plugin: `switch_inspection`
- Type: `plugin-enhanced-detector`
- Main runtime: `plugin.py`
- Detector backend: `detector_enhanced.py`
- Auxiliary module: `switch_consistency.py`

## Current Goal

把 `switch_inspection` 作为 HF 快速晋升样板插件维护到“高频开发可用、低 prompt 成本可调用”的状态，重点是 skills / commands / scripts / tests 闭环，不重写识别算法。

## Scope That Is Verified Today

- 插件可通过 `Plugin.create_standalone()` 初始化。
- `infer(..., context=None)` 在 standalone/demo 下可安全返回结果列表。
- detector 适配层提供：
  - `recognize_indicator_state()`
  - `recognize_linkage_state()`
  - `validate_interlock()`
  - `read_gauge()`
- 一致性校核模块 `switch_consistency.py` 有独立测试，不自动接入主插件链路。
- HF 治理入口已落地到本插件本地目录，不依赖 root 级治理文件改动。

## Required Local Commands

- `./scripts/run_targeted_tests.sh <module>`
- `./scripts/run_regression_tests.sh`
- `./scripts/run_quality_gate.sh`
- `./scripts/collect_root_cause.sh`

## Non-Goals For This Round

- 不改其它 `plugins/*`
- 不改 `ui/`、`platform_core/`、`darkbreaker_sdk/`
- 不重写 `detector_enhanced.py` 的核心识别算法
- 不把 `switch_consistency.py` 强行接入平台或跨插件联动

## Known Operational Limits

- 没有真实图像回放数据集。
- 可选 YOLOv8-ViT / OCR / fusion engine 默认不保证存在。
- SF6 表计读数默认关闭。

## First Recommended Next Step After This HF Upgrade

补一个受控的 `tests/fixtures/` 小型图像样本集，把当前 regression 从”代码回归”推进到”最小视觉回放回归”。

## Phase 2 能力三分表 (2026-04-16 审计)

| 能力 | 分类 | 证据 |
|------|------|------|
| state_recognition | **verified** | 契约测试全通过；recognize_indicator_state/recognize_linkage_state 可调用 |
| image_quality | experimental | 配置项和契约测试存在，但无真实清晰度评价样本回归 |
| five_prevention_logic_validation | blocked | validate_interlock() 存在但未接入主插件链路；switch_consistency.py 独立 |

**runtime_mode_support**: real_dl=❌ traditional_fallback=✅ simulation=✅
**测试状态**: 全量通过 (0 failures)，无 fixtures/regression 目录
**关键差距**: 五防逻辑校验为 description 宣称但实际 blocked；SF6 读数默认关闭
