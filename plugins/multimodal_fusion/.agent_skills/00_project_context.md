# 00_project_context

## 1. 输入完备性审计（当前真实状态）

| 输入项 | 状态 | 来源 | 处置规则 |
|---|---|---|---|
| `plugin.py` | 可用 | 当前插件目录 | 当前 manifest 指向的主实现；平台契约与融合主链路以此为准 |
| `fusion_engine_enhanced.py` | 可用但主链路不稳定 | 当前插件目录 | 当前 `init()` 会优先启用，但常见 dict/status 输入下容易失败并回退 |
| `fusion_engine.py` | 可用但未直接挂到当前主插件 | 当前插件目录 | 视为基础引擎参考实现，不写成当前主入口 |
| `plugin_v4_bayesian.py` | 可用但未接入当前 manifest | 当前插件目录 | 视为并行/候选实现，不得当成当前现状 |
| `manifest.json` | 可用 | 当前插件目录 | 插件元数据、声明式输入输出 schema、模态依赖来源 |
| `requirements.txt` | 可用但依赖声明偏少 | 当前插件目录 | 真实 Python 安装入口 |
| `standalone/app.py` | 可用 | 当前插件目录 | 独立服务入口，默认端口 `8096` |
| `demo/run_demo.py` | 可用 | 当前插件目录 | 当前最佳人工回放入口，已本地验证可执行 |
| `configs/default.yaml` | 不存在 | 当前插件目录 | 不得假设已有 YAML 配置；`create_standalone()` 依赖 SDK 缺省兜底 |
| `run_standalone.py` | 不存在 | 当前插件目录 | `__main__.py` usage 提到它，但文件缺失 |
| `tests/` | 目录存在但为空 | 当前插件目录 | 不得宣称已有自动化测试 |
| `scripts/` | 目录存在但为空 | 当前插件目录 | 不得宣称已有脚本化门禁 |

## 2. 治理等级

**最小治理**。

理由：

1. 有 `standalone/` 和 `demo/`，具备基本运行壳。
2. 但 `tests/`、`scripts/` 为空，缺少最小自动化闭环。
3. 存在多套融合实现并存，且当前主链路与增强链路之间有明显漂移。
4. manifest 声明字段与当前 `process()` 真正消费的字段并不完全一致。

## 3. 固定母版规则（跨插件统一）

1. 事实源优先级固定：`plugin.py` > `demo/` > `manifest.json` > `requirements.txt` > 并行实现文件。
2. 这是“服务集成/数据融合插件”，不是 detector 模板。
3. 并行实现或未来版文件不能当成当前 manifest 主链路。
4. 没有 tests / scripts / configs 时必须如实说明。
5. 输入 schema 声明与代码真实消费字段必须分开写。

## 4. 当前插件真实结构

```text
multimodal_fusion/
├── plugin.py
├── fusion_engine.py
├── fusion_engine_enhanced.py
├── plugin_v4_bayesian.py
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

1. `MultimodalFusionPlugin().init()` 返回 `True`。
2. `process({})` 返回 `{"success": False, "error": "缺少模态数据"}`。
3. `process({"device_id": "...", "modalities": {...}})` 基础链路可返回成功。
4. `python3 -m plugins.multimodal_fusion.demo.run_demo` 可执行。
5. `create_standalone()` 可执行，并回到默认 `MultimodalConfig`。
6. 当前增强引擎通常会先尝试启用，但 demo 这类常见 dict/status 输入下会报错并自动回退到基础融合。
7. `max_history_length` 在 dict 配置注入时当前不会被 `_parse_config()` 正确消费。

## 6. AI 自动闭环 vs 人工确认

### 可由 AI 自动闭环

- 维护 `.agent_skills/00~08`
- 新增最小测试骨架
- 新增最小 sanity 脚本
- 校验 `init()` / `process()` / demo 回放 / 增强引擎回退
- 标注 manifest/实现漂移

### 必须人工确认

- 是否切换到 `plugin_v4_bayesian.py` 或其他并行实现
- 是否真正接入外部模态插件和模型注册表
- 是否修改平台输入输出契约
- 是否公开 standalone 服务
- 是否把当前增强引擎失败视为 bug 还是暂时接受的回退行为

## 7. 最小可执行校验命令

```bash
# demo 回放
python3 -m plugins.multimodal_fusion.demo.run_demo

# 基础 process
python3 - <<'PY'
import sys
from pathlib import Path
root = Path('/Users/ronan/Desktop/DarkBreaker')
sys.path.insert(0, str(root))
from plugins.multimodal_fusion.plugin import MultimodalFusionPlugin
p = MultimodalFusionPlugin()
assert p.init() is True
print(p.process({
    "device_id": "demo",
    "modalities": {
        "visual": {"status": "warning", "confidence": 0.7},
        "gas": {"overall_status": "alarm", "confidence": 0.9},
    }
}))
PY
```
