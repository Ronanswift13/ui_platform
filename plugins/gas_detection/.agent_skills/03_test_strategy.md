# 03_test_strategy

## 1. 固定母版规则

1. 每个硬约束至少对应 1 个自动化测试或 1 个明确人工回放入口。
2. 每个 bug 修复必须新增防回归测试。
3. 不得把 demo 或 standalone 手工运行冒充为完整回归；当前正式入口是 pytest 与 `run_sanity_checks.sh`。
4. 测试结论必须区分“已本地执行”和“仅建议补齐”。

## 2. 当前测试现状

| 项目 | 状态 | 说明 |
|---|---|---|
| `tests/test_config_contract.py` | 已存在 | 默认 YAML、manifest 输出 schema、关键配置字段 |
| `tests/test_process_contract.py` | 已存在 | 最小输入、空输入、阈值修改、metadata、趋势降级 |
| `tests/test_trend_contract.py` | 已存在 | 24 样本 predictor + analyzer 主链路、spy 对象调用关系 |
| `tests/test_standalone.py` | 已存在 | standalone health/status/smoke |
| `scripts/run_sanity_checks.sh` | 已存在 | 配置 + pytest + 主链路 smoke |
| `demo/run_demo.py` | 已验证可执行 | 当前最佳人工回放入口 |
| `python3 -m plugins.gas_detection` | 可作为人工服务入口 | 需要人工观察 Web/API 层 |

**当前已核验但不等于测试覆盖的事实：**

1. import 可用
2. `init()` 可用
3. 基础 `process(sample)` 可用
4. 24 样本趋势预测路径进入主输出合同，`predictions` 与 `trend_analysis` 均可被测试断言

## 3. 分层建议

### L0 Targeted

纯逻辑测试，不依赖 Web 服务。

优先覆盖：

1. `_evaluate_gas_status()` 分级
2. `_update_history()` / `_get_history()` 缓冲行为
3. `_detect_leakage()` 传统回退路径
4. `create_standalone()` 消费 `configs/default.yaml` 后的默认行为
5. 24 样本进入预测路径后的 predictor/analyzer 调用关系和输出结构

### L1 Integration

插件级集成。

优先覆盖：

1. `init()` -> `process(sample)` -> `healthcheck()` -> `shutdown()`
2. `process({})` 返回缺少气体读数错误
3. `demo/run_demo.py` 最小回放
4. 24 样本趋势预测链路

### L2 Manual Service

仅在需要验证 standalone 页面时执行：

1. `python3 -m plugins.gas_detection`
2. 打开 `http://localhost:8094`
3. 检查服务启动与模板加载

## 4. 最小验证命令

```bash
# 人工回放
python3 -m plugins.gas_detection.demo.run_demo

# 最小质量门
cd plugins/gas_detection && ./scripts/run_sanity_checks.sh

# 自动化测试
python3 -m pytest plugins/gas_detection/tests -q

# 基础行为
python3 - <<'PY'
import sys
from pathlib import Path
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.gas_detection.plugin import GasDetectionPlugin
p = GasDetectionPlugin()
assert p.init() is True
assert p.process({})["success"] is False
ok = p.process({"device_id": "d1", "gas_readings": {"SF6": 100, "H2": 20}})
assert ok["success"] is True
PY
```

## 5. 已覆盖测试文件

1. `tests/test_process_contract.py`
   - `init()`、缺输入报错、基础成功路径、阈值改动、metadata 和趋势降级
2. `tests/test_trend_contract.py`
   - 24 样本 predictor/analyzer 主链路
   - spy predictor/analyzer 调用关系
3. `tests/test_config_contract.py`
   - YAML、manifest、输出 schema 与趋势字段
4. `tests/test_standalone.py`
   - health/status/smoke

## 6. 当前不应伪造的能力

1. 不存在现成 `run_targeted_tests.sh`
2. 不存在现成 `run_regression_tests.sh`
3. 不存在现成覆盖率门禁
4. DGA 详细分析尚未进入主输出合同

## 7. 后续最值得补的脚本

当前已补 `scripts/run_sanity_checks.sh`。后续若要扩展，应按顺序新增：

1. `scripts/run_targeted_tests.sh`：只跑 process/trend/config 快速合同。
2. `scripts/run_quality_gate.sh`：反模式扫描 + py_compile + pytest。
3. 覆盖率门槛：仅在质量门稳定后引入。
