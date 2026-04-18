# capacitor_inspection Fixtures

本目录用于存放 `capacitor_inspection` 的最小视觉回放样本资产。

## 目录分类

- `normal/`: 结构完整、无遮挡、无入侵的基线样本
- `anomaly/`: 倾斜/倒塌/缺件/入侵等典型异常样本
- `boundary/`: 轻微倾斜、局部遮挡、边界误检风险样本
- `quality_fail/`: 过暗、模糊、强压缩等质量失败样本

## 命名建议

```text
cap_<scenario>_<index>.<ext>
示例:
cap_normal_structure_001.jpg
cap_intrusion_person_001.jpg
cap_quality_dark_001.jpg
```

## 当前状态

- 插件目录内尚无图像级 fixture。
- 仓库训练目录存在 `training/data/processed/*/capacitor/images` 候选真实资产，但当前缺少插件级 replay 筛选与 expected baseline 映射。
- 在真实样本落地前，不要把本目录中的占位结构当作“已验证视觉回放集”。
