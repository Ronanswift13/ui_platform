# fire_detection Fixtures

本目录用于存放 `fire_detection` 的最小事实源。

## 目录分类

- `normal/`: 无火无烟场景
- `anomaly/`: 火焰/烟雾等异常场景
- `boundary/`: 易混淆但不应直接误报的边界场景
- `quality_fail/`: 遮挡、低光、极端压缩等质量失败场景
- `drill/`: 演练/模拟专用配置或元数据资产

## 命名建议

```text
fire_<scenario>_<index>.<ext>
示例:
fire_visible_flame_001.png
fire_visible_smoke_001.png
fire_drill_electrical_001.json
```

## 当前状态

- 插件目录内没有真实火焰/烟雾图片。
- 仓库内未发现可直接复用的 fire/smoke 图像资产。
- 当前可立即复用的仅有 pytest 里的 mock frame 生成器和 drill API 行为，不构成真实视觉回放。
