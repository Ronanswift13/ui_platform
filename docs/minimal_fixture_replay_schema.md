# Minimal Fixture / Replay Schema

本文件定义 DarkBreaker 插件最小 fixture / replay 事实源的统一结构。

## 目录约定

每个插件至少具备以下目录：

```text
plugins/<plugin_id>/tests/
├── fixtures/
│   ├── normal/
│   ├── anomaly/
│   ├── boundary/
│   └── quality_fail/
└── replay/
    ├── README.md
    └── expected_results.json
```

如插件存在纯演练/纯配置场景，可额外增加 `fixtures/drill/`。

## expected_results.json 顶层字段

- `schema_version`: 当前 schema 版本，建议从 `1.0` 开始
- `schema_ref`: 指向本说明文件
- `plugin_id`: 插件 ID
- `asset_root`: 样本资产根目录，通常是 `plugins/<plugin_id>/tests/fixtures`
- `collection_status`: 当前事实源状态，推荐值：
  - `planned_slots_only`
  - `partial_assets_available`
  - `ready_for_replay`
- `samples`: 样本槽位列表
- `candidate_sources`: 仓库内可复用但尚未完成插件级基线映射的候选来源

## 每条 sample 必填字段

- `sample_id`
- `source_type`
  - `real`
  - `synthetic`
  - `mock`
  - `placeholder`
- `asset_kind`
  - `image`
  - `mock_image_spec`
  - `sample_spec`
- `plugin_id`
- `fixture_group`
- `scenario_type`
  - 推荐值：`normal` / `anomaly` / `boundary` / `quality_fail`
  - 如插件存在演练模式，可用 `drill_simulation`
- `expected_runtime_mode`
  - 推荐值：`real_dl` / `traditional_fallback` / `simulation` / `simulation_only` / `blocked` / `not_applicable`
- `expected_review_status`
  - 推荐值：`clear` / `manual_review_required` / `failed` / `simulation_only` / `blocked` / `not_applicable`
- `expected_primary_labels`
- `expected_alarm_level`
  - 推荐值：`none` / `info` / `warning` / `alarm` / `error` / `critical` / `simulation`
- `expected_output_contract`
- `notes`

## 建议附加字段

- `asset_relpath`: 计划或现有样本相对路径
- `collection_status`: `planned` / `present_unlabeled` / `present_labeled`
- `candidate_source_refs`: 关联候选来源 ID 列表
- `expected_numeric_targets`: 量化目标，如读数值、角度、bbox 数量
- `expected_reason_codes`: 预期原因码或复核原因
- `acceptance_blockers`: 阻断该样本进入 verified 的缺口列表

## candidate_sources 建议字段

- `candidate_id`
- `source_type`
- `asset_relpath`
- `label_status`
  - `untriaged`
  - `partial`
  - `verified_for_plugin`
- `notes`

## 使用边界

- `source_type=placeholder` 只能表示“已固定槽位和命名规范”，不能表示样本已存在。
- `source_type=real` 仅用于仓库内确有真实资产、且路径可追溯的情况。
- 仓库内训练资产若尚未完成插件级场景筛选或 expected baseline 映射，只能进入 `candidate_sources`，不能直接当作已验证 replay 基线。
- `expected_results.json` 是“插件级回归事实源”，不是模型能力宣传文案。
