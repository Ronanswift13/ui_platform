# 04_quality_audit

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`
2. 禁止在生产主链路新增 `print()`
3. 禁止无文档硬编码平台契约字段
4. 禁止把未验证能力写成“已支持”
5. 禁止修复行为问题却不补最小回归验证

## 2. 本项目高优先级审计项

1. **配置契约断裂**
   - `manifest.json.default_config` 声明了 `history_length` / `prediction_horizon`
   - 当前 `GasDetectionConfig`、`configs/default.yaml` 与 `_parse_config()` 已承接
   - 审计重点改为防止未来新增配置只写声明、不写消费逻辑
2. **模型注册键名错位**
   - 当前已保留 `sf6_forecast` / `multi_gas_forecast` / `equipment_health_trend` 与 `lstm` / `transformer` 兼容键
   - 后续新增模型键必须同步 manifest、YAML、dataclass、predictor
3. **阈值对象访问方式存在潜在 bug**
   - 按静态审查，`predictor.py` 中 `self.config.thresholds.get(gas, {}).get(...)` 假定阈值是 dict
   - 当前真实类型是 `GasThreshold` dataclass
   - 当前 24 样本趋势测试可覆盖主路径；新增阈值对象访问方式时必须补回归
4. **分析器主链路边界**
   - `GasDataAnalyzer.analyze_trends` 已接入主链路并输出 `trend_analysis`
   - DGA / 相关性详细分析仍不得写成当前主输出合同
5. **持久化能力未落地**
   - `data/results.db` 存在表结构
   - 当前主代码未见写入路径
6. **版本漂移**
   - `manifest.json.version = 1.0.0`
   - `plugin.py.PLUGIN_VERSION = 3.0.0`
7. **依赖声明漂移**
   - `manifest.json.dependencies` 含 `onnxruntime`
   - `requirements.txt` 未显式列出

## 3. 反模式清单

| 反模式 | 检测方法 | 严重度 |
|---|---|---|
| 配置字段只在 manifest 声明、不在代码消费 | `rg 'history_length|prediction_horizon'` | 阻断 |
| model id 键名不一致 | `rg 'lstm|transformer|sf6_forecast|multi_gas_forecast'` | 高 |
| 将 dataclass 当 dict 用 | `rg 'thresholds.get\\(gas, \\{\\}\\)\\.get' predictor.py` | 高 |
| 假持久化 | `rg 'results.db|detection_results|sqlite'` | 中 |
| 版本不一致 | 对比 `manifest.json` 与 `plugin.py` | 中 |
| 主链路 `print()` | `rg '\\bprint\\(' *.py --glob '!demo/*'` | 中 |

## 4. 审计命令

```bash
# 配置与模型键名检查
rg 'history_length|prediction_horizon|lstm|transformer|sf6_forecast|multi_gas_forecast' plugins/gas_detection/plugin.py plugins/gas_detection/predictor.py plugins/gas_detection/manifest.json

# 潜在阈值访问 bug
rg 'thresholds.get\\(gas, \\{\\}\\)\\.get' plugins/gas_detection/predictor.py

# 数据库/分析器接入情况
rg 'results.db|detection_results|sqlite|_analyzer|analyze_trends' plugins/gas_detection -S

# 最小质量门
cd plugins/gas_detection && ./scripts/run_sanity_checks.sh

# demo 回放
python3 -m plugins.gas_detection.demo.run_demo
```

## 5. 当前阻断级问题

当前没有已知阻断级问题。若 24 样本趋势预测、`trend_analysis` 或 standalone smoke 失败，应立即视为阻断。

## 6. 当前建议级问题

1. `manifest.json` 与实现版本不一致，容易误导平台侧排障。
2. `results.db` 会给人“持久化已接好”的错觉，需在文档和代码里继续明确边界。
3. DGA / 相关性详细分析仍是第二阶段入口，不是当前主输出合同。
