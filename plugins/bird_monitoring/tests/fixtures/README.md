# bird_monitoring Fixtures

本目录用于存放 `bird_monitoring` 的最小视觉事实源。

## 目录分类

- `normal/`: 无鸟或单鸟安全场景
- `anomaly/`: 靠近导线、风险较高的鸟类场景
- `boundary/`: 复杂背景、unknown_bird、review_required 场景
- `quality_fail/`: 低清晰度、过暗、雨雾遮挡样本

## 命名建议

```text
bird_<scenario>_<index>.<ext>
示例:
bird_no_bird_001.jpg
bird_single_safe_001.jpg
bird_unknown_review_001.jpg
```

## 当前状态

- 插件目录内没有真实图像 fixture。
- 仓库内未发现 `bird_monitoring` 可直接复用的图片或 replay 资产。
- 当前 tests 主要使用 numpy 合成帧验证合同，不构成视觉回放事实源。
