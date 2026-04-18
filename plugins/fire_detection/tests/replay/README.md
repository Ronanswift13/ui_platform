# fire_detection Replay Baseline

本目录固定 `fire_detection` 的最小 replay 样本槽位。

## 目标样本槽位

- `fire_visible_flame_001`
- `fire_visible_smoke_001`
- `fire_drill_electrical_001`
- `fire_clear_scene_001`
- `fire_lowlight_occlusion_001`

## 现状说明

- 当前仓库内无真实火焰/烟雾图像可直接复用。
- 现有 tests 已有 `blank_frame` / `fire_frame` / `smoke_frame` / `thermal_hotspot_frame` 等 mock 生成器，可先作为 mock/simulation 级回归输入。
- 演练路径当前更适合通过配置/元数据固定，而不是要求真实图像。
