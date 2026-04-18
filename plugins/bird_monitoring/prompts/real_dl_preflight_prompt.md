# Prompt: real_dl preflight 升级

## 使用场景

当有人交付 `bird_monitoring` 的真实 YOLOv8 ONNX 模型，并希望把 runtime_mode
从 `simulation` 切到 `real_dl` 时，用本 prompt 指导 agent 验收。

## 任务目标

在 `plugins/bird_monitoring` 范围内：
1. 把新 ONNX 文件放到 `configs/default.yaml::model.path` 指向的相对位置。
2. 确认 `BirdDetector._preflight_onnx()` 通过（`report.passed=true`，`issues=[]`）。
3. 运行 `python -m pytest plugins/bird_monitoring/tests/ -q` 全绿。
4. 至少新增一条 replay 样本（真实小图）覆盖 `no_bird` 或 `bird_safe`，并更新
   `tests/replay/expected_results.json` 的 `collection_status` 与 `expected_runtime_mode`。

## 强制约束

- 不修改其它插件。
- 不修改 `platform_core/`, `darkbreaker_sdk/`。
- 不得在 simulation 模式下伪造 bird bbox。
- 不得在 detector 层接入硬件驱鸟；`deterrent_suggestion` 仍只是输出建议。
- preflight 失败必须回落 simulation 并在 `healthcheck.details.preflight` 暴露原因。

## 验收清单

- [ ] `healthcheck.details.preflight.passed == true`
- [ ] `healthcheck.details.runtime_mode == "real_dl"`
- [ ] `healthcheck.details.real_model_loaded == true`
- [ ] replay 测试至少绿 1 例真实样本
- [ ] 跨插件 pytest 无退化

## 禁止事项

- ❌ 绕过 preflight 直接开启 real_dl
- ❌ 把 experimental/enhanced_detector.py 接入生产主链
- ❌ 删减 `test_real_dl_preflight.py` 的失败分支
