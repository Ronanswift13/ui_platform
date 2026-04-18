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
- Context: B 类时序传感插件第一批质量治理回灌，聚焦算法/质量门测试、warning/deprecated 清理和稳定音频回放。
- Symptom: skill 仍记录“仅 standalone 测试、无质量门禁”，但当前代码已存在 detector/analyzer/config/process/standalone/WAV replay 测试和 sanity/targeted/quality gate 脚本。
- Root cause: 质量治理代码先落地，`.agent_skills` 未同步升级，导致后续 agent 可能按旧的最小治理路线误判。
- Fix: 将 `00/03/08` 更新为标准治理基线，明确 `tests/test_real_audio_replay.py` 是稳定 WAV 容器回放、`run_quality_gate.sh` 负责反模式和 deprecated API 扫描。
- Prevention: 后续只要新增测试、脚本或真实样本回放，必须同步更新 `03_test_strategy.md` 与 `08_task_routing.md`。
- Follow-up: 若获得脱敏现场录音样本，追加到 replay 测试并记录采样率、通道数、来源边界；不得把当前合成 WAV fixture 伪装成真实传感器闭环。
