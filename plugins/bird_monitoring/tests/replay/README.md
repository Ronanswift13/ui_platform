# bird_monitoring Replay Baseline

本目录固定 `bird_monitoring` 的最小 replay 样本槽位。

## 目标样本槽位

- `bird_no_bird_001`
- `bird_single_safe_001`
- `bird_unknown_review_001`
- `bird_complex_background_001`
- `bird_quality_dark_001`

## 现状说明

- 当前仓库内没有 bird 插件可直接复用的图像级样本。
- 现有 pytest 主要覆盖插件合同、风险评估和质量评估逻辑。
- 因此当前 replay 只先固定槽位、命名和 expected baseline 模板，真实图片需后续人工采集或脱敏标注。
- 除 `no_bird` 与 `quality_failed` 外，任何“检测到鸟”的 expected label 都必须依赖 `real_dl` 或显式隔离的 `standalone_simulation` demo，绝不得由当前 `simulation` 主链生成。
