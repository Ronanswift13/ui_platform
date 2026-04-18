# 04_quality_audit

> 最后更新：2026-04-16（主链收敛后）

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`（吞异常）。
2. 禁止在生产代码新增 `print()`。
3. 禁止硬编码业务阈值（须来自 config）。
4. 禁止输出越界 `confidence` 或非法 `bbox`。
5. 禁止提交未覆盖的新逻辑分支。
6. 禁止伪造检测数据（simulation 必须返回空，不得造合成鸟）。
7. 禁止对接驱离硬件（本插件仅输出建议 JSON）。

## 2. 本项目审计项状态

| 审计项 | 状态 | 证据 |
|---|---|---|
| 阈值双源 | ✅ 已清零 | 全部从 `config.risk_assessment.*_weight` 读取 |
| print 污染 | ✅ 已清零 | 17 处替换为 logger |
| 伪数据回退 | ✅ 已清零 | 未命中 → `unknown_bird`（非 sparrow） |
| runtime truth 可观测 | ✅ 已达成 | runtime_mode + 模型路径/存在性/session/real_model 字段 |
| 未初始化保护 | ✅ 已达成 | 返回 `error/9000` 结果，不 crash |
| 质量门 | ✅ 已达成 | `quality_failed` label + 9 测试覆盖 |
| 训练占位 | ✅ 已达成 | 正常、质量失败、错误结果附 training_placeholders |
| 主链唯一性 | ✅ 已达成 | plugin 固定加载 `BirdDetector` |
| 重复/实验检测器 | ⏳ 惰性债 | advanced 已迁入 `experimental/`；`BirdDetectorEnhanced` 仍在 detector.py |
| BIRD_DATABASE 外部化 | ⏳ 未完成 | 仍硬编码（需人工确认） |
| 驱离硬件耦合 | ✅ 已阻断 | 仅输出 JSON 建议，触发入口返回 False |
| 精度 regression | ⏳ blocked | 无 fixture 图片 |

## 3. 反模式清单（更新）

| 反模式 | 检测方法 | 严重度 | 状态 |
|--------|----------|--------|------|
| 硬编码阈值 | `rg 'RISK_THRESHOLDS\|_THRESHOLD\b' plugin.py detector.py` | 高 | 主链路已清零 |
| 吞异常 | `rg 'except.*pass' *.py` | 阻断 | 无 |
| print 调试 | `rg '\bprint\(' plugin.py detector.py` | 阻断 | 无（main.py 的 CLI 输出保留） |
| 重复/实验类型定义 | `rg 'class ThreatLevel\|class BirdDetection' *.py experimental detector.py` | 高 | legacy/advanced 未进入主链 |
| 默认 sparrow 回退 | `rg '"sparrow"' plugin.py` | 高 | 仅在 SPECIES_LABEL_MAP 内 |
| 驱离硬件调用 | `rg 'requests\.post\|urllib\.request\|urlopen\|serial\.\|RPi\.GPIO\|paho' plugin.py detector.py` | 阻断 | 无 |
| 随机检测进生产 | `rg 'np\.random|random\.' plugin.py detector.py` | 阻断 | 无；experimental/standalone demo 可命中但不得生产接入 |
| 伪造检测 | `rg '_simulate_detection' detector.py` + 人工检查是否返回非空 | 阻断 | 返回 [] |
| 未使用配置项 | 对比 YAML 键与代码读取 | 中 | quality 段新加全部读取 |
| 模型路径硬编码 | `rg '\.onnx\|\.pt\b' detector.py` | 中 | 走 `config.model.path` |

## 4. 审计命令（一条龙）

```bash
# 推荐入口（覆盖反模式 + pytest + 覆盖率，rg/grep 自动回落）
plugins/bird_monitoring/scripts/run_quality_gate.sh

# 仅跳过覆盖率门
plugins/bird_monitoring/scripts/run_quality_gate.sh --no-coverage
```

退出码语义：

| 退出码 | 含义 |
|---|---|
| 0 | 全绿 |
| 1 | 反模式命中 |
| 2 | pytest 失败 |
| 3 | 覆盖率不达标（`.coveragerc::fail_under`） |
| 4 | 环境缺失（pytest 不可用或 grep/rg 同时缺失） |

底层等效命令（仅作为 troubleshooting 参考）：

```bash
cd /Users/ronan/Desktop/DarkBreaker/plugins/bird_monitoring

rg 'except.*pass' *.py
rg '\bprint\(' plugin.py detector.py
rg 'class ThreatLevel|class BirdDetection|class BirdSpecies' *.py
rg 'RISK_THRESHOLDS|_THRESHOLD\b' plugin.py
rg 'requests\.post|urllib\.request|urlopen|serial\.|RPi\.GPIO|paho' plugin.py detector.py
rg 'np\.random|random\.' plugin.py detector.py
rg 'return[[:space:]].*"sparrow"' plugin.py

python3 -c "from plugins.bird_monitoring.plugin import BirdMonitoringPlugin as P; p = P.create_standalone(); print(p.healthcheck().details['runtime_mode'])"
python3 -m pytest plugins/bird_monitoring/tests/ -q
```

## 5. 零容忍准出检查清单

提交前必须全部通过：

- [ ] `rg '\bprint\(' plugin.py detector.py` → 无命中
- [ ] `rg 'except.*pass' *.py` → 无命中
- [ ] `rg 'requests\.post\|urllib\.request\|urlopen\|serial\.\|RPi\.GPIO' plugin.py detector.py` → 无命中
- [ ] `rg 'np\.random\|random\.' plugin.py detector.py` → 无命中
- [ ] `pytest plugins/bird_monitoring/tests/ -q` → 全绿
- [ ] 跨插件 pytest（至少 6 个插件）→ 无退化
- [ ] 新增 label 已更新 `test_standalone.py::test_infer_returns_results` 白名单
- [ ] manifest 三分表（verified / experimental / blocked）与代码事实一致
- [ ] 07_learning_log.md 已追加本次变更 Entry
