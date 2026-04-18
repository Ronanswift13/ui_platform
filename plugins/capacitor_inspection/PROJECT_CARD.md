# PROJECT_CARD

## Identity

- Plugin: `capacitor_inspection`
- Type: `plugin-enhanced-detector`
- Main runtime: `plugin.py`
- Detector backend: `detector_enhanced.py`

## Current Goal

把 `capacitor_inspection` 提升到“准 HF”状态，重点是 skills / commands / scripts / tests / docs 闭环，不重写电容器缺陷和入侵算法。

## Scope That Is Verified Today

- 插件可通过 `Plugin.create_standalone()` 初始化。
- `infer()` 能稳定返回 SDK `RecognitionResult` 列表。
- 结构缺陷与入侵链路的插件合同已由本地单元/契约测试覆盖。
- 四个治理脚本已落地到插件目录内。

## Non-Goals For This Round

- 不改其它 `plugins/*`
- 不改 `ui/`、`platform_core/`、`darkbreaker_sdk/`
- 不重写 `detector_enhanced.py` 的核心识别算法
- 不把没有 replay 事实源的能力包装成稳态 HF

## Known Operational Limits

- 没有真实图像 replay 数据集。
- intrusion 正向识别仍缺少真实样本回放证据。
- 传统 CV 降级链里仍有硬编码阈值技术债。

## First Recommended Next Step After This Upgrade

补一个最小真实的 intrusion / capacitor defect 样本夹具集，把当前”合同测试通过”推进到”最小视觉回放回归”。

## Phase 2 能力三分表 (2026-04-16 审计)

| 能力 | 分类 | 证据 |
|------|------|------|
| defect_detection | experimental | 代码存在(detector_enhanced.py)，但测试因 import 错误全部无法收集 |
| intrusion_detection | experimental | 配置和代码路径存在，但无真实样本回放证据 |
| real_dl_onnx_inference | **blocked** | YOLOv8-ViT/YOLOv8-OBB 均为可选导入，当前环境不可用 |

**runtime_mode_support**: real_dl=❌ traditional_fallback=✅ simulation=✅
**测试状态**: ⚠️ 全部测试因 `ModuleNotFoundError: detector_enhanced` 无法收集
**关键差距**: 测试基础设施损坏是当前最大阻塞项；无 verified 能力
