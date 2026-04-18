# 04_quality_audit

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`
2. 禁止在生产主链路新增 `print()`
3. 禁止把并行实现写成当前现状
4. 禁止修改合同却不补回归验证
5. 禁止把 demo 成功当成融合正确性证明

## 2. 本项目高优先级审计项

1. **增强引擎默认启用但对常见输入不稳**
   - demo 和常见 `{status, confidence}` 输入会触发增强引擎失败
   - 当前靠回退到基础融合保底
2. **manifest 输入 schema 漂移**
   - manifest 声明 `pre_processed`、`modality_results`
   - 当前 `process()` 不消费
3. **配置漂移**
   - manifest 声明 `max_history_length`、`modality_dims`
   - `_parse_config()` 当前不消费
4. **入口文案漂移**
   - `__main__.py` usage 提到 `run_standalone.py`
   - 当前文件不存在
5. **依赖漂移**
   - manifest 声明 `onnxruntime`
   - `requirements.txt` 未显式列出
6. **版本漂移**
   - `manifest.json.version = 1.0.0`
   - `plugin.py.PLUGIN_VERSION = 1.0.1`
   - `__init__.__version__ = 1.0.0`
7. **并行实现误用风险**
   - `plugin_v4_bayesian.py` 和 `fusion_engine.py` 容易被误认为现行主实现

## 3. 反模式清单

| 反模式 | 检测方法 | 严重度 |
|---|---|---|
| 增强引擎对 status 字符串崩溃 | 复现 demo/最小脚本 | 高 |
| `_parse_config()` 不消费 manifest 关键字段 | `rg 'max_history_length|modality_dims|_parse_config'` | 高 |
| 声明字段不被 `process()` 使用 | `rg 'pre_processed|modality_results'` | 中 |
| 缺失入口文案 | `test -f run_standalone.py` | 中 |
| 依赖漂移 | 对比 manifest 与 requirements | 中 |
| 版本漂移 | 对比 manifest / plugin / __init__ | 中 |

## 4. 审计命令

```bash
# 配置与输入漂移
rg 'pre_processed|modality_results|max_history_length|modality_dims|_parse_config' plugins/multimodal_fusion

# 缺失入口
test -f plugins/multimodal_fusion/run_standalone.py && echo EXISTS || echo MISSING

# demo 回放
python3 -m plugins.multimodal_fusion.demo.run_demo

# 基础 process
python3 - <<'PY'
import sys
from pathlib import Path
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin
p = MultimodalFusionPlugin(); p.init()
print(p.process({
    "device_id": "audit",
    "modalities": {
        "visual": {"status": "normal", "confidence": 0.9},
        "gas": {"status": "warning", "confidence": 0.8},
    }
}))
PY
```

## 5. 当前阻断/高风险问题

1. 增强引擎对常见输入格式不稳，当前主链路依赖回退兜底。
2. manifest 默认配置与 `_parse_config()` 存在实质漂移。

## 6. 当前建议级问题

1. `run_standalone.py` 缺失但文案仍引用。
2. `requirements.txt` 与 manifest 依赖声明不一致。
3. 并行实现文件需要继续在文档中明确降级为“候选路径”。
