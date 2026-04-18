# 07_learning_log

用于记录本插件重要故障、根因与预防动作。每次 `/repair` 或重大质量问题修复后必须追加。

## Entry Template
- Date:
- Context:
- Symptom:
- Root cause:
- Fix:
- Prevention:
- Follow-up:

## Entries

- Date: 2026-04-17
- Context: implement / 增强治理深度，补 detector 边界测试、工单阈值测试和现场式设备数据回放。
- Symptom: `DeviceHealthCalculator.calculate()` 中 `cpu_usage > 90` / `> 70` 使用硬编码阈值，L0 测试未覆盖健康指数和工单触发边界；回放测试只有 standalone smoke，不能证明更接近现场的遥测序列路径。
- Root cause: 早期最小治理只覆盖 standalone/infer，未把 detector 纯计算层和 `detect()` 工单触发边界固化为自动测试；CPU 使用率阈值也未接入 `configs/default.yaml`。
- Fix: 新增 `tests/test_detector.py`、`tests/test_device_replay.py`、`tests/fixtures/field_like_device_replay.json` 和 `scripts/run_targeted_tests.sh`；将 CPU 使用率 warning/alarm 阈值改为从 `thresholds.cpu_usage_warning_percent` / `cpu_usage_alarm_percent` 读取；吞异常路径改为 debug 日志。
- Prevention: 后续修改健康权重、阈值或工单优先级前，先跑 `scripts/run_targeted_tests.sh` 与 `scripts/run_sanity_checks.sh`；真实设备数据接入前必须保持 fixture 来源说明，不能把现场式夹具伪装成生产传感器闭环。
- Follow-up: 若获得授权的真实设备遥测样本，可替换或追加到 `tests/fixtures/`，并继续保持脱敏、无凭证、无外部服务依赖。
