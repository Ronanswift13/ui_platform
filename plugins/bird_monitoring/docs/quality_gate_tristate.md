# 输入质量门三态设计记录

> 最后更新：2026-04-16

## 1. 状态定义

| status | is_valid | 推理行为 | 典型触发 |
|--------|----------|----------|---------|
| `pass` | true | 正常推理 | 分辨率、亮度、清晰度均达标 |
| `soft_fail` | true | 继续推理但强制 `review_required` | 分数低于 soft 阈值但未触发 hard_fail |
| `hard_fail` | false | 阻断推理，输出 `quality_failed`（blocked） | None / empty / 极小 / 过暗 / 过曝 / 过模糊 |

## 2. 阈值

在 `configs/default.yaml::quality`：
- `min_dimension`（hard）：像素最小边
- `clarity_threshold`（hard）：Laplacian 方差下限
- `soft_overall_threshold`：整体分下限
- `soft_clarity_threshold`：清晰度分下限
- `soft_brightness_threshold`：亮度分下限

## 3. `review_required` vs `blocked`

- `soft_fail → review_required`：结果能输出，只是打上复核标签，告警等级 WARNING。
- `hard_fail → blocked`：输出 `quality_failed`，告警等级 WARNING；需人工复查图像源。
- `blocked` 与 `review_required` 的差异在于：前者代表**不得基于此帧做任何判断**；
  后者代表**可以输出假设但必须人工复核**。

## 4. 测试矩阵

- `tests/test_quality_tristate.py`
  - pass: 正常帧
  - hard_fail: None / tiny / dark
  - soft_fail: 抬高 soft 阈值触发；验证推理继续并 review_required
  - quality_failed 告警级别为 WARNING
  - soft_fail 不得产出 quality_failed

## 5. 扩展点

- 后续可将 `soft_fail` 再细分为 `soft_brightness_only / soft_clarity_only` 以便
  UI 给出更具体的修图建议。当前保持单一 `soft_fail` 以控制契约复杂度。
