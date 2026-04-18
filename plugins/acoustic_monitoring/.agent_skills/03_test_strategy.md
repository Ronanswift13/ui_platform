# 03_test_strategy

## 1. 固定母版规则

1. 每个硬约束至少对应 1 个自动化测试。
2. 每个 bug 修复必须新增防回归测试。
3. 测试分层执行：L0 快速、L1 集成。
4. 测试脚本必须返回明确退出码。

## 2. 当前测试现状

| 文件 | 层级 | 内容 |
|------|------|------|
| `tests/test_config_contract.py` | L0 | manifest sensor/time-series 声明、YAML 配置段、默认阈值字段 |
| `tests/test_process_contract.py` | L1 | 单条声学样本、空样本、阈值改动、metadata/RecognitionResult 虚拟 ROI |
| `tests/test_detector.py` | L0 | feature extractor、规则分数值域、配置阈值生效、模型 registry 路径 |
| `tests/test_analyzer.py` | L0 | 频谱/谐波/诊断章节与低采样率可用频带标识 |
| `tests/test_real_audio_replay.py` | L1 | 临时 WAV 容器写入/读取/回放，验证标准输出稳定 |
| `tests/test_standalone.py` | L1 | `create_standalone` + `process` + `healthcheck` + smoke route |

当前缺口仅限于真实现场录音样本和覆盖率门槛；现有 WAV replay 是稳定的合成容器回放，不应伪装成生产采集闭环。

## 3. 分层定义

- **L0 Targeted**：纯逻辑测试，不依赖 SDK，目标 < 2 分钟。
  - detector 特征提取输出形状
  - 各检测器阈值边界（给定已知波形，验证检测结果）
  - AcousticConfig 从 YAML 加载
- **L1 Integration**：`Plugin.create_standalone()` 全链路。
  - process() 正常音频 / 无音频 / 异常音频
  - healthcheck / shutdown / reinit

## 4. 最小测试入口

```bash
# 运行全部测试
python -m pytest plugins/acoustic_monitoring/tests/ -q

# 合同/smoke
cd plugins/acoustic_monitoring && ./scripts/run_sanity_checks.sh

# detector/analyzer/WAV replay
cd plugins/acoustic_monitoring && ./scripts/run_targeted_tests.sh

# 质量门禁
cd plugins/acoustic_monitoring && ./scripts/run_quality_gate.sh
```

## 5. 测试命名约定

```
test_{module}_{scenario}_{expected_behavior}
```

示例：`test_detector_normal_audio_no_anomaly`、`test_plugin_missing_audio_uses_mock`

## 6. 后续增强项

1. 引入脱敏真实现场 WAV/PCM 样本，并记录来源与采样参数。
2. 若启用真实深度学习模型，新增模型资产 preflight 与 fallback 回归。
3. 需要覆盖率治理时再建立 `.coveragerc`，不要把当前 quality gate 误报为覆盖率门禁。
