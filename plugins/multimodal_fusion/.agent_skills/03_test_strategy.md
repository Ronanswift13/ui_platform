# 03_test_strategy

## 1. 固定母版规则

1. 每个硬约束至少对应 1 个自动化测试或 1 个明确人工回放入口。
2. 每个 bug 修复必须新增防回归测试。
3. 当前 `tests/` 为空时，不得把 demo 或 standalone 当成“已覆盖自动化测试”。
4. 测试结论必须区分“已本地执行”和“建议补齐”。

## 2. 当前测试现状

| 项目 | 状态 | 说明 |
|---|---|---|
| `tests/` | 空目录 | 当前没有 pytest 用例 |
| `scripts/` | 空目录 | 当前没有任何测试脚本 |
| `demo/run_demo.py` | 已验证可执行 | 当前最佳人工回放入口 |
| `python3 -m plugins.multimodal_fusion` | 可作为人工服务入口 | 会启动 standalone 服务 |

已核验但不等于测试覆盖：

1. `init()` 可用
2. 基础 `process()` 可用
3. demo 可跑
4. 增强引擎通常会失败并回退

## 3. 分层建议

### L0 Targeted

优先覆盖：

1. `_parse_config()` 对 `max_history_length` / `modality_dims` 的当前行为
2. `_process_modalities()` 的“注册插件/未注册插件/插件异常”三条路径
3. `_process_with_enhanced_engine()` 的回退行为
4. `_calculate_contributions()` 与状态聚合
5. `switch_fusion_strategy()` / `get_best_strategy()` 基础行为

### L1 Integration

优先覆盖：

1. `init()` -> `process(sample)` -> `shutdown()`
2. `demo/run_demo.py`
3. `create_standalone()` 默认配置回退
4. 已注册模态插件的最小集成

### L2 Manual Service

仅在需要验证 standalone 页面时执行：

1. `python3 -m plugins.multimodal_fusion`
2. 打开 `http://localhost:8096`
3. 检查服务启动与模板加载

## 4. 最小验证命令

```bash
# 人工回放
python3 -m plugins.multimodal_fusion.demo.run_demo

# 基础行为
python3 - <<'PY'
import sys
from pathlib import Path
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin
p = MultimodalFusionPlugin()
assert p.init() is True
assert p.process({})["success"] is False
ok = p.process({
    "device_id": "d1",
    "modalities": {
        "visual": {"status": "warning", "confidence": 0.7},
        "gas": {"overall_status": "alarm", "confidence": 0.9},
    }
})
assert ok["success"] is True
PY
```

## 5. 待补齐测试文件（优先级）

1. `tests/test_plugin_process.py`
   - 未初始化
   - 缺模态
   - 基础融合成功
   - shutdown/healthcheck
2. `tests/test_config_contract.py`
   - manifest 默认配置与 `_parse_config()` 漂移
3. `tests/test_enhanced_engine_fallback.py`
   - 常见 dict/status 输入触发回退
   - 数值 features 输入下的行为

## 6. 当前不应伪造的能力

1. 不存在现成 `run_sanity_checks.sh`
2. 不存在 regression 脚本
3. 不存在覆盖率门禁
4. 不存在已落地的外部模态插件集成测试

## 7. 后续最值得补的第一个脚本

建议先补：

`scripts/run_sanity_checks.sh`

理由：

1. 当前最缺的不是更复杂的测试矩阵，而是一个统一、可复用的最小事实校验入口。
2. 它可以串起 `init()`、基础 `process()`、demo 回放、增强引擎回退检查。
3. 它还能把 `run_standalone.py` 缺失、配置漂移这些问题固定成显式提醒。
