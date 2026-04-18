# 00_project_context

## 1. 输入完备性审计（当前真实状态）

| 输入项 | 状态 | 来源 | 处置规则 |
|---|---|---|---|
| `plugin.py` | 可用 | 当前插件目录 | 生命周期、平台契约、阈值判断、历史缓冲、告警输出的单一事实源 |
| `manifest.json` | 可用 | 当前插件目录 | 插件注册元数据、平台声明的 input/output schema、模型声明来源 |
| `predictor.py` | 可用 | 当前插件目录 | 趋势预测与泄漏检测后端，含深度学习优先和传统方法回退 |
| `analyzer.py` | 已接入主链路 | 当前插件目录 | `process()` 在预测后调用 `_analyze_trends()`，输出 `trend_analysis` 并纳入 `trend_diagnosis` |
| `run_standalone.py` / `standalone/app.py` | 可用 | 当前插件目录 | 独立服务入口，默认端口 `8094` |
| `demo/run_demo.py` | 可用 | 当前插件目录 | 当前最接近“人工回放”的入口，已本地验证可执行 |
| `configs/default.yaml` | 可用 | 当前插件目录 | sampling/window/thresholds/runtime/model/alarm_rules/upgrade_placeholders 的权威默认配置 |
| `tests/` | 可用 | 当前插件目录 | config/process/standalone/trend 合同测试已存在 |
| `scripts/run_sanity_checks.sh` | 可用 | 当前插件目录 | 标准治理基线的最小质量门，覆盖配置、主链路、趋势、standalone smoke |
| `data/results.db` | 可用但未见主链路写入 | 当前插件目录 | 可视为历史或预留产物，不得宣称当前 `process()` 已持久化结果 |

## 2. 治理等级

**标准治理基线**。

理由：

1. 有 `standalone/` 和 `demo/`，说明运行入口比纯骨架更完整。
2. `configs/default.yaml` 已补齐并被 `_parse_config()` 消费。
3. `tests/` 已覆盖默认配置、process 合同、standalone smoke、24 样本趋势预测主链路。
4. `scripts/run_sanity_checks.sh` 已成为当前最小质量门。

## 3. 固定母版规则（跨插件统一）

1. 事实源优先级固定：`plugin.py` > `predictor.py` > `manifest.json` > `requirements.txt` > 历史文档。
2. 不因目录名像“detection”就强行套 `detector.py` 模板。
3. 当前只有 `run_sanity_checks.sh`，不得伪造 targeted/regression/coverage 门禁。
4. 服务型插件要区分“平台契约已声明”和“本地主链路已验证”。
5. 深度学习能力、数据库能力、DGA 能力只有在当前调用链实际接入时才能算现状。

## 4. 当前插件真实结构

```text
gas_detection/
├── plugin.py
├── predictor.py
├── analyzer.py
├── manifest.json
├── requirements.txt
├── __main__.py
├── run_standalone.py
├── demo/
│   └── run_demo.py
├── standalone/
│   ├── app.py
│   └── templates/
├── data/
│   └── results.db
├── tests/
│   ├── test_config_contract.py
│   ├── test_process_contract.py
│   ├── test_standalone.py
│   └── test_trend_contract.py
├── scripts/
│   └── run_sanity_checks.sh
└── .agent_skills/             # 本轮补齐 00~08
```

## 5. 当前已核验事实

以下事实已在本地轻量验证：

1. `from plugins.gas_detection.plugin import GasDetectionPlugin` 可导入。
2. `GasDetectionPlugin().init()` 返回 `True`。
3. `process({"device_id": "...", "gas_readings": {...}})` 在基础场景下返回 `success=True`。
4. `process({})` 返回 `{"success": False, "error": "缺少气体读数数据"}`。
5. `GasDetectionPlugin.create_standalone()` 可执行。
6. `python3 -m plugins.gas_detection.demo.run_demo` 可执行。
7. 当同一设备累计 24 条样本后，`process()` 会进入 predictor + analyzer 主链路，返回 `predictions.available=True`、`trend_analysis.available=True`，并把趋势纳入主输出合同。

## 6. AI 自动闭环 vs 人工确认

### 可由 AI 自动闭环

- 维护 `.agent_skills/00~08`
- 维护现有合同测试与最小 sanity 脚本
- 校验 import / init / sample process / demo 回放链路
- 识别并记录配置契约与实现不一致问题

### 必须人工确认

- 是否引入真实模型文件或模型注册中心
- 是否启用独立服务对外暴露（`0.0.0.0:8094`）
- 是否让插件真的写入 `data/results.db`
- 是否调整平台输入输出契约字段
- 是否把 DGA 详细分析正式纳入主输出合同（趋势预测与趋势分析已纳入当前合同）

## 7. 最小可执行校验命令

```bash
# demo 回放
python3 -m plugins.gas_detection.demo.run_demo

# 最小质量门
cd plugins/gas_detection && ./scripts/run_sanity_checks.sh

# 插件测试
python3 -m pytest plugins/gas_detection/tests -q

# 基础 import + init + process
python3 - <<'PY'
import sys
from pathlib import Path
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.gas_detection.plugin import GasDetectionPlugin
p = GasDetectionPlugin()
print(p.init())
print(p.process({"device_id": "demo", "gas_readings": {"SF6": 200, "H2": 30}}))
PY

# 独立入口
python3 -m plugins.gas_detection
```
