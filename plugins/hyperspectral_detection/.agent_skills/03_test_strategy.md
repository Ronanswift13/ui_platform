# 03_test_strategy

## 1. 固定母版规则

1. 每个硬约束至少对应 1 个自动化测试或 1 个明确人工回放入口。
2. 每个 bug 修复必须新增防回归测试。
3. 在 `tests/` 为空时，不得把 demo 或 standalone 称为“已覆盖自动化测试”。
4. 测试结论必须区分“已本地执行”和“建议补齐”。

## 2. 当前测试现状

| 项目 | 状态 | 说明 |
|---|---|---|
| `tests/` | 空目录 | 当前没有 pytest 用例 |
| `scripts/` | 空目录 | 当前没有任何测试脚本 |
| `demo/run_demo.py` | 已验证可执行 | 当前最佳人工回放入口 |
| `python3 -m plugins.hyperspectral_detection` | 可作为人工服务入口 | 会启动 standalone 服务 |

**当前已核验但不等于测试覆盖的事实：**

1. import 可用
2. `init()` 可用
3. `process({})` 可用
4. `create_standalone()` 可用
5. `analysis_type` 当前不改变输出

## 3. 分层建议

### L0 Targeted

纯逻辑测试，不依赖 Web 服务。

优先覆盖：

1. `_parse_config()` 对 manifest 键名漂移的当前行为
2. `process()` 缺图像时的模拟回退
3. `healthcheck()` / `shutdown()`
4. 光谱维度推断
5. `analysis_type` 当前无分支差异的事实

### L1 Integration

插件级集成。

优先覆盖：

1. `init()` -> `process(sample)` -> `healthcheck()` -> `shutdown()`
2. `create_standalone()` 默认配置回退
3. `demo/run_demo.py` 回放

### L2 Manual Service

仅在需要验证 standalone 页面时执行：

1. `python3 -m plugins.hyperspectral_detection`
2. 打开 `http://localhost:8095`
3. 检查模板加载与服务启动

## 4. 最小验证命令

```bash
# 人工回放
python3 -m plugins.hyperspectral_detection.demo.run_demo

# 基础行为
python3 - <<'PY'
import sys
from pathlib import Path
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.hyperspectral_detection.plugin import HyperspectralDetectionPlugin
p = HyperspectralDetectionPlugin()
assert p.init() is True
r = p.process({"device_id": "d1"})
assert r["success"] is True
assert "spectrum_analysis" in r
PY
```

## 5. 待补齐测试文件（优先级）

1. `tests/test_plugin_process.py`
   - `init()`
   - `process()` 模拟回退
   - `shutdown()` / `healthcheck()`
2. `tests/test_config_contract.py`
   - manifest 键名 vs `_parse_config()` 的当前错位
3. `tests/test_spectrum_shape_handling.py`
   - `(bands, h, w)` / `(h, w, bands)` / 2D 输入的光谱长度行为

## 6. 当前不应伪造的能力

1. 不存在现成 `run_sanity_checks.sh`
2. 不存在现成 regression 脚本
3. 不存在覆盖率门禁
4. 不存在真实模型 contract test

## 7. 后续最值得补的第一个脚本

建议先补：

`scripts/run_sanity_checks.sh`

理由：

1. 当前 `tests/` 为空，最先需要的是一个稳定的最小事实校验入口。
2. 它可以覆盖 import / init / simulated process / demo 回放。
3. 它还能把“配置键名漂移”和“光谱维度长度异常”固定成显式检查点。
