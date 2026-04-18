# meter_reading Replay Baseline

本目录固定 `meter_reading` 的最小可回归样本槽位。

## 目标样本槽位

- `meter_analog_normal_001`
- `meter_analog_boundary_001`
- `meter_analog_quality_fail_001`
- `meter_digital_display_001`
- `meter_led_indicator_green_001`
- `meter_glare_or_tilt_review_required_001`

`meter_led_indicator_green_001` 当前是 mock image spec，用来验证 replay plumbing。其余槽位为 planned，占位不等于已验收。

## expected_results 字段结构

顶层必填:

- `schema_version`
- `schema_ref`
- `plugin_id`
- `asset_root`
- `collection_status`
- `expected_results_schema`
- `samples`
- `candidate_sources`

每条 sample 必填:

- `sample_id`
- `source_type`
- `asset_kind`
- `plugin_id`
- `fixture_group`
- `scenario_type`
- `meter_type`
- `expected_runtime_mode`
- `expected_review_status`
- `expected_primary_labels`
- `expected_alarm_level`
- `collection_status`
- `expected_output_contract`
- `notes`

输出 metadata 必查:

- `meter_type`
- `reading_status`
- `pipeline_stage`
- `fallback_level`
- `timestamp_ms`
- `runtime_mode`
- `review_status`
- `failure_reason`

## 当前可复用来源

- 现有规划线索：
  - `plugins/meter_reading/tests/test_matrix_template.md`
- 仓库候选训练资产：
  - `training/data/processed/HV_220kV/meter/images`
  - `training/data/processed/HV_220kV/meter/classes.txt`
  - `training/data/processed/EHV_500kV/meter/images/test/samples.png`

这些训练资产当前只适合作为人工筛选和脱敏整理来源，不能直接视为插件级 expected baseline。

以 [expected_results.json](/Users/ronan/Desktop/DarkBreaker/plugins/meter_reading/tests/replay/expected_results.json) 为后续 pytest replay 的唯一基线入口。
