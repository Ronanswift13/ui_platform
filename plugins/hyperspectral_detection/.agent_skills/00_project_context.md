# 00_project_context

## 1. 输入完备性审计（当前真实状态）

| 输入项 | 状态 | 来源 | 处置规则 |
|---|---|---|---|
| `plugin.py` | 可用 | 当前插件目录 | 当前唯一核心实现文件；生命周期、平台契约、分析逻辑都以它为准 |
| `manifest.json` | 可用但偏声明层 | 当前插件目录 | 提供元数据和依赖声明，但没有 `input_schema` / `output_schema` |
| `requirements.txt` | 可用 | 当前插件目录 | 当前真实 Python 依赖入口 |
| `standalone/app.py` | 可用 | 当前插件目录 | 独立服务入口，默认端口 `8095` |
| `demo/run_demo.py` | 可用 | 当前插件目录 | 当前最佳人工回放入口，已本地验证可执行 |
| `configs/default.yaml` | 不存在 | 当前插件目录 | 不得假设已有配置文件；当前 `create_standalone()` 仍会回退默认配置 |
| `run_standalone.py` | 不存在 | 当前插件目录 | `__main__.py` 的 usage 文案提到它，但当前文件缺失 |
| `tests/` | 目录存在但为空 | 当前插件目录 | 不得宣称已有自动化测试 |
| `scripts/` | 目录存在但为空 | 当前插件目录 | 不得宣称已有 sanity / regression / quality gate 脚本 |
| `.agent_skills/` | 缺失 | 当前插件目录 | 本轮补齐 00~08 |

## 2. 治理等级

**最小治理**。

理由：

1. 有 `standalone/` 和 `demo/`，说明已经有可运行壳。
2. 但 `tests/`、`scripts/` 均为空，缺少最小自动校验闭环。
3. 配置文件缺失，manifest 与实现之间存在明显漂移。
4. 高光谱分析链路目前以模拟和占位结果为主，不能写成“已完成的生产算法管线”。

## 3. 固定母版规则（跨插件统一）

1. 事实源优先级固定：`plugin.py` > `manifest.json` > `requirements.txt` > 历史文档。
2. 不因插件名叫 detection 就强行套 `detector.py` 模板。
3. 没有 tests / scripts / configs 时必须如实说明。
4. 区分“当前代码真实行为”和“文案/manifest 声称支持的能力”。
5. 只有已接入主调用链的能力，才能写成当前合同。

## 4. 当前插件真实结构

```text
hyperspectral_detection/
├── plugin.py
├── manifest.json
├── requirements.txt
├── __main__.py
├── demo/
│   └── run_demo.py
├── standalone/
│   ├── app.py
│   └── templates/
├── tests/                     # 空目录
├── scripts/                   # 空目录
└── .agent_skills/             # 本轮补齐 00~08
```

## 5. 当前已核验事实

以下事实已本地轻量验证：

1. `from plugins.hyperspectral_detection.plugin import HyperspectralDetectionPlugin` 可导入。
2. `HyperspectralDetectionPlugin().init()` 返回 `True`。
3. `process({})` 当前会成功，并自动生成模拟高光谱立方体，而不是报缺少图像错误。
4. `process({"device_id": "d1"})` 返回 `success=True`、`overall_status="normal"`、`num_bands=224`。
5. `HyperspectralDetectionPlugin.create_standalone()` 可执行，并回到默认 `HyperspectralConfig`。
6. `python3 -m plugins.hyperspectral_detection.demo.run_demo` 可执行。
7. `analysis_type` 当前不影响输出结构。
8. 光谱均值计算对某些 3D 输入形状存在维度误判风险。

## 6. AI 自动闭环 vs 人工确认

### 可由 AI 自动闭环

- 维护 `.agent_skills/00~08`
- 新增最小测试骨架
- 新增最小 sanity 脚本
- 校验 import / init / simulated process / demo 回放
- 校验 manifest / requirements / docstring 漂移

### 必须人工确认

- 是否引入真实高光谱模型或 PCA 管线
- 是否公开暴露 standalone 服务
- 是否调整平台输入输出契约
- 是否新增真实文件/设备输入链路
- 是否把当前占位检测结果替换为生产算法输出

## 7. 最小可执行校验命令

```bash
# demo 回放
python3 -m plugins.hyperspectral_detection.demo.run_demo

# 基础 import + init + process
python3 - <<'PY'
import sys
from pathlib import Path
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.hyperspectral_detection.plugin import HyperspectralDetectionPlugin
p = HyperspectralDetectionPlugin()
assert p.init() is True
print(p.process({"device_id": "demo"}))
PY

# 独立服务入口
python3 -m plugins.hyperspectral_detection
```
