# meter_reading Fixtures

本目录用于存放 `meter_reading` 的最小回放样本资产。

## 最小试点目录

- `analog_normal/`: 清晰正视角模拟表，目标是稳定给出读数。
- `analog_boundary/`: 量程边界、指针角度边界或轻微倾斜模拟表。
- `analog_quality_fail/`: 模糊、过暗、低分辨率等模拟表质量失败样本。
- `digital_display/`: 数字表或七段码显示样本。
- `led_indicator/`: 红/绿/黄/灭灯指示灯样本。
- `glare_or_tilt_review_required/`: 强眩光或明显倾斜，目标是稳定进入人工复核。

历史粗分类目录 `normal/`、`anomaly/`、`boundary/`、`quality_fail/` 暂保留，后续可作为迁移入口；新的 replay baseline 以本节六个目录为准。

## 命名建议

```text
meter_<scenario>_<index>.<ext>
示例:
meter_analog_normal_001.jpg
meter_analog_boundary_001.jpg
meter_analog_quality_fail_001.jpg
meter_digital_display_001.jpg
meter_led_indicator_green_001.fixture.json
meter_glare_or_tilt_review_required_001.jpg
```

## 当前状态

- `led_indicator/` 当前包含一个 mock image spec，用于跑通最小 replay 测试骨架。
- analog / digital / glare_or_tilt 仅固定槽位和 expected_results 字段，尚无真实标注图像。
- `tests/test_matrix_template.md` 已给出命名规范和测试矩阵，可直接作为采集清单骨架。
- 仓库训练目录存在 `training/data/processed/*/meter/images` 候选资产，但 labels 为空，当前只能视为候选来源，不能直接当作 replay baseline。
