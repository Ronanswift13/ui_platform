# capacitor_inspection Replay Baseline

本目录固定 `capacitor_inspection` 的最小 replay 样本槽位和 expected 结果模板。

## 目标样本槽位

- `cap_normal_structure_001`
- `cap_structural_tilt_001`
- `cap_intrusion_person_001`
- `cap_boundary_partial_occlusion_001`
- `cap_quality_dark_blur_001`

## 当前可复用来源

- 候选真实资产目录：
  - `training/data/processed/HV_220kV/capacitor/images`
  - `training/data/processed/HV_220kV/capacitor/classes.txt`

这些训练资产可以作为人工筛选来源，但当前不能直接当作插件级 replay baseline，因为：

- 插件目录下没有绑定到场景类型的基线映射
- 对应 `labels/` 目录为空，无法直接确认 anomaly / intrusion / boundary 分类
- 还没有 expected results 与插件输出口径对齐

以 [expected_results.json](/Users/ronan/Desktop/DarkBreaker/plugins/capacitor_inspection/tests/replay/expected_results.json) 作为后续采集、筛选和 pytest 回归接线的单一事实源。
