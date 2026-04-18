# 07 Learning Log

Use this file to record bugs, root causes, and preventive actions that matter for future `/repair` and `/audit` work.

## Entry Template

- Date:
- Context:
- Symptom:
- Root cause:
- Fix:
- Prevention:
- Follow-up:

## Entries

### 2026-04-08 - Consistency helper introduced

- Context: local plugin scope
- Symptom: runtime switch plugin only covered video state output
- Root cause: no packaged helper existed for video/sensor/action comparison
- Fix: `switch_consistency.py` and `tests/test_switch_consistency.py` were added
- Prevention: keep consistency logic tested independently until runtime integration is explicitly approved
- Follow-up: do not document it as runtime-integrated unless `plugin.py` calls it

### 2026-04-09 - HF governance uplift

- Context: promote `switch_inspection` from STD to HF sample plugin
- Symptom: scripts were missing, commands referenced nonexistent assets, plugin/detector contracts drifted, demo and standalone smoke broke on `context=None`
- Root cause: governance assets were generated before local scripts/tests/contracts were finalized
- Fix: added four governance scripts, added contract/governance tests, normalized command files, added `PROJECT_CARD.md`, and repaired plugin/detector adapter boundaries
- Prevention: every governance change must update `08_task_routing.md` and be covered by `tests/test_quality_gate_contract.py`
- Follow-up: add a small replay fixture set so regression is not only code-level

### 2026-04-17 - Module-level import triggers torch / OMP heavy init

- Context: clean-start 场景（multiprocessing spawn / gunicorn preload）下 `create_standalone()` 失败，从 busbar/capacitor 同根因推广修复
- Symptom: `import plugins.switch_inspection` 即触发 torch 加载，因 `plugin.py` 模块级 `from plugins._model_resolution import resolve_plugin_model_config` 触发 `_model_resolution → training → torch` 链
- Root cause: 设备类插件共用 `_model_resolution` 解析器，该模块在 import 阶段通过 `training/__init__.py → train_main.py` 拉入 torch；`__init__.py` eager import 进一步放大
- Fix: (1) `plugin.py` 模块级 import 替换为 `_get_model_resolver()` 延迟函数，`init()` 时才调用；(2) `__init__.py` 改为 PEP 562 `__getattr__` 延迟加载；(3) 新增 `test_import_weight.py` 4 个回归测试
- Prevention: 设备类插件 `_model_resolution` 必须 init-time lazy import；`__init__.py` 禁止 eager `from .plugin import`
- Follow-up: 与 busbar/capacitor/transformer 共享同一修复模式，可纳入插件模板 checklist

### 2026-04-09 - Standalone simulation panel was empty

- Context: repair standalone dashboard for local demo / simulation review
- Symptom: selecting "模拟数据" still left a black canvas with no plugin-local debug or monitoring controls
- Root cause: the page only rendered a passive canvas and result placeholders; simulation behavior depended on no plugin-owned scene engine or mode router
- Fix: added a plugin-local browser-side simulation/monitoring layer in `standalone/templates/switch_inspection.html`, plus template contract tests for controls and runtime isolation text
- Prevention: standalone pages that advertise simulated data must include plugin-owned controls and an explicit isolation note separating demo behavior from runtime algorithm delivery
- Follow-up: if future work adds real replay fixtures, wire them into the standalone page as a separate data source instead of reusing the browser simulation path
