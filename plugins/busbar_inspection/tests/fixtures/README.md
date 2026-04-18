# Fixtures（busbar_inspection）

存放可复现、去敏后的测试样本。

## 目录建议
- `quality/`：模糊、过曝、低对比、遮挡样本。
- `defects/`：销钉缺失、裂纹、异物样本。
- `roi_cases/`：边界 ROI、非法 ROI、多 ROI 并发样本。

## 规则
1. 不放真实生产敏感图片。
2. 文件命名带场景与期望：`blur_reason103_01.jpg`。
3. 与回归测试引用路径保持一致。
