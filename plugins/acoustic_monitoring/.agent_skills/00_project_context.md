# 00_project_context

## 0. 治理等级

**标准治理基线** — 当前已补齐 detector/analyzer/config/process/standalone/真实 WAV 容器回放测试，并提供 sanity、targeted、quality gate 三类脚本入口。后续若引入真实模型或硬件采集链路，再升级为高频开发治理。

## 1. 固定母版规则（跨插件统一）

1. **接口契约冻结**：插件必须实现 `init/process/healthcheck`，签名与 SDK 一致。
2. **配置优先**：阈值必须来自 `configs/default.yaml`，不允许在推理主路径硬编码。
3. **降级可观测**：降级路径必须输出 `failure_reason` + `metadata`。
4. **输出可校验**：`anomaly_score`、`confidence` 必须满足值域 `[0, 1]`。
5. **测试分层**：至少包含 L0（单测）、L1（集成）分层执行入口。

## 2. 本项目差异规则（acoustic_monitoring 专属）

1. **业务目标**：变电站声学异常检测——局部放电、电晕放电、轴承故障、变压器嗡鸣、机械故障。
2. **输入为音频**：`np.ndarray` 波形 + `sample_rate`，非图像。无 ROI 概念。
3. **多异常类型并存**：`AcousticAnomalyType` 定义 8 种类型，severity 由类型决定。
4. **analyzer 为可选深度分析**：`AcousticAnalyzer` 提供频谱/谐波/时频详情，仅在需要诊断报告时调用。
5. **模拟音频兜底**：`process()` 在无 `audio` 输入时自动生成 mock 音频，保证 standalone 可演示。

## 3. 当前目录与职责边界

```
acoustic_monitoring/
├── plugin.py                   # SDK 适配层（配置加载、process 编排、告警生成、mock 音频）
├── detector.py                 # 算法层（特征提取 + 信号处理检测 + 深度学习推理）
├── analyzer.py                 # 分析层（频谱/谐波/时频详细分析，诊断建议）
├── configs/default.yaml        # 运行参数唯一来源
├── manifest.json               # 插件注册信息
├── standalone/                 # 独立运行（WebSocket 仪表盘 + AudioSessionManager）
│   ├── app.py
│   ├── audio_manager.py
│   └── templates/
├── tests/
│   ├── test_config_contract.py # manifest/YAML/配置契约
│   ├── test_process_contract.py# 统一声学输入输出壳
│   ├── test_detector.py        # L0 detector 阈值和模型路径边界
│   ├── test_analyzer.py        # 频谱/谐波/诊断输出结构
│   ├── test_real_audio_replay.py # WAV 容器回放稳定性
│   └── test_standalone.py      # L1 standalone smoke
├── scripts/
│   ├── run_sanity_checks.sh    # contract + standalone 最小门禁
│   ├── run_targeted_tests.sh   # detector/analyzer/WAV 回放
│   └── run_quality_gate.sh     # 反模式扫描 + 编译 + 全量 pytest
├── pytest.ini                  # 插件本地 pytest 配置
├── demo/run_demo.py            # 演示脚本
└── .agent_skills/              # AI 代理规则（本目录）
```

## 4. 模块职责边界

| 模块 | 职责 | 不应包含 |
|------|------|----------|
| `plugin.py` | SDK 适配、配置解析、告警生成、process 编排 | 信号处理算法细节 |
| `detector.py` | 特征提取、传统信号处理检测、模型推理 | SDK schema、告警逻辑 |
| `analyzer.py` | 深度频谱分析、诊断建议生成 | 检测决策、告警触发 |
| `standalone/audio_manager.py` | 会话管理、WebSocket 广播 | 算法逻辑 |

## 5. AI 自动闭环 vs 人工确认

### 可自动闭环
- `.agent_skills/` 规则维护
- `tests/` 测试补齐与执行
- 配置键一致性检查
- 质量审计扫描

### 需人工确认
- 异常类型清单变更（增删类型）
- severity 映射策略调整
- 模型文件路径变更（manifest.json）
- 采样率 / 超声参数等硬件相关配置

## 6. 可执行校验命令

```bash
# 配置可解析
python -c "import yaml; yaml.safe_load(open('plugins/acoustic_monitoring/configs/default.yaml'))"

# 插件可导入
python -c "from plugins.acoustic_monitoring.plugin import Plugin; print(Plugin.__name__)"

# 合同与 smoke
cd plugins/acoustic_monitoring && ./scripts/run_sanity_checks.sh

# 算法与真实音频容器回放
cd plugins/acoustic_monitoring && ./scripts/run_targeted_tests.sh

# 质量门禁
cd plugins/acoustic_monitoring && ./scripts/run_quality_gate.sh

# 全量插件测试
python -m pytest plugins/acoustic_monitoring/tests/ -q

# 性能基准
python -m plugins.acoustic_monitoring.scripts.benchmark
```
