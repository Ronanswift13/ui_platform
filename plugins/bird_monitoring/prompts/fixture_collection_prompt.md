# Prompt: 真实 fixture 采集与 replay 接入

## 使用场景

当有人开始采集 `bird_monitoring` 的真实图像样本时，用本 prompt 指导 agent 把
原始素材转为可回归的 fixture。

## 任务目标

1. 按 `tests/replay/expected_results.json` 的 `sample_id` 槽位收集真实图像：
   - `bird_no_bird_001`（正常无鸟）
   - `bird_single_safe_001`（单鸟、安全距离）
   - `bird_unknown_review_001`（物种不明 → review）
   - `bird_complex_background_001`（复杂背景）
   - `bird_quality_dark_001`（过暗）
2. 文件以 `.jpg` 落入对应子目录（`tests/fixtures/{normal,anomaly,boundary,quality_fail}/`）。
3. 更新 `expected_results.json` 的 `collection_status` 从 `planned*` 改为 `collected`。
4. 在 `tests/test_replay_baseline.py` 中为真实样本增加 `pytest.mark.skipif`
   保护：当 `.jpg` 缺失时跳过，不污染 CI。
5. 对 `bird_safe / unknown_bird / review_required` 的样本，只有 `real_dl` 模式
   通过 preflight 后才允许进入 replay；simulation 下仍需 skip。

## 强制约束

- 合成图像不得冒充真实 fixture。`tests/regression/build_synthetic_fixtures.py`
  生成的 `.npy` 只覆盖 `no_bird_clear` 与 `quality_dark` 两个 simulation 可复现
  的 label。
- 真实图像必须脱敏（站名、车牌等）。
- 不得向 `evidence/` 写入任何 fixture。

## 验收清单

- [ ] 新样本 `.jpg` 存在且可被 cv2 读取
- [ ] `expected_results.json` 的 `collection_status` 与 `expected_runtime_mode`
      符合实际可复现条件
- [ ] `pytest` 对缺失真实样本 skip，不 failure
- [ ] 新样本不含敏感信息
