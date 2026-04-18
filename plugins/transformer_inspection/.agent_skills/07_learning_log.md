# 07 Learning Log

Used to record confirmed issues, root causes, and prevention notes for this plugin.

## Entry Template

- Date:
- Context:
- Symptom:
- Root cause:
- Fix:
- Prevention:
- Follow-up:

## Entries

- Date: 2026-04-08
- Context: initial agent-skills bootstrap for `transformer_inspection`
- Symptom: `python3 -m pytest plugins/transformer_inspection/tests -q` failed with SDK drift and contract mismatches
- Root cause: old tests still depended on legacy `StandalonePluginRunner(plugin_dir=...)` and ROIs without `roi_type`
- Fix: rebuild tests around current SDK contracts and local standalone factory
- Prevention: run plugin-local pytest whenever SDK contract changes
- Follow-up: keep targeted test modules aligned with the HF scripts

---

- Date: 2026-04-17
- Context: clean-start 场景（multiprocessing spawn / gunicorn preload）下 `create_standalone()` 失败，从 busbar/capacitor 同根因推广修复
- Symptom: `import plugins.transformer_inspection` 即触发 torch 加载，因 `plugin.py` 模块级 `from plugins._model_resolution import resolve_plugin_model_config` 触发 `_model_resolution → training → torch` 链
- Root cause: 设备类插件共用 `_model_resolution` 解析器，模块级 import 时触发 torch 初始化；`__init__.py` eager import 进一步放大
- Fix: (1) `plugin.py` 模块级 import 替换为 `_get_model_resolver()` 延迟函数，`init()` 时才调用；(2) `__init__.py` 改为 PEP 562 `__getattr__` 延迟加载；(3) 新增 `test_import_weight.py` 4 个回归测试
- Prevention: 设备类插件 `_model_resolution` 必须 init-time lazy import；`__init__.py` 禁止 eager `from .plugin import`
- Follow-up: 与 busbar/capacitor/switch 共享同一修复模式，可纳入插件模板 checklist

---

- Date: 2026-04-09
- Context: HF governance upgrade
- Symptom: `plugin.py` and `detector_enhanced.py` had drift on defect return shape, nested inference config, and thermal/state adaptation
- Root cause: plugin assumed dict-style detector outputs and a generic `recognize_state()` path that the detector did not actually expose
- Fix: add plugin-side adapters for defect/silica/oil-level/thermal outputs, align detector config reads to `inference.*`, and replace silent/stale logging patterns
- Prevention: treat `.agent_skills/02_algorithm_contract.md` plus targeted tests as the source of truth before editing runtime code
- Follow-up: add a small real image fixture set so regression can validate more than synthetic inputs
