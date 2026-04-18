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
- Context: initial agent-skills bootstrap for `capacitor_inspection`
- Symptom: plugin and detector read confidence thresholds from different config paths
- Root cause: detector used top-level keys while YAML stored thresholds under `inference.*`
- Fix: align detector reads to nested config
- Prevention: keep `.agent_skills/02_algorithm_contract.md` and config contract tests in sync with YAML shape
- Follow-up: continue cleaning remaining hardcoded traditional-CV thresholds

---

- Date: 2026-04-17
- Context: clean-start 场景（multiprocessing spawn / gunicorn preload）下 `create_standalone()` 失败
- Symptom: `import plugins.capacitor_inspection` 即触发 torch 加载，导致 fork-safe / spawn 场景下 OMP 与 CUDA 初始化冲突或超时
- Root cause: `plugin.py` 模块级 `from plugins._model_resolution import resolve_plugin_model_config` 触发 `_model_resolution → training/__init__.py → train_main.py → import torch` 链；`__init__.py` 直接 `from .plugin import CapacitorInspectionPlugin` 进一步放大
- Fix: (1) 将模块级 import 替换为 `_get_model_resolver()` 延迟加载函数，仅在 `init()` 调用时执行 import；(2) 将 `__init__.py` 改为 PEP 562 `__getattr__` 延迟加载；(3) fallback 确保 standalone 安全
- Prevention: 设备类插件的 `_model_resolution` / `training` 依赖必须在 `init()` 内延迟加载；`__init__.py` 使用 PEP 562；新增 `test_import_weight.py` 回归测试锁定
- Follow-up: 此模式已推广至 busbar / switch / transformer，可作为后续新设备插件模板

---

- Date: 2026-04-09
- Context: quasi-HF governance upgrade
- Symptom: plugin routing and result adaptation were mismatched with current SDK and detector dataclasses
- Root cause: plugin still assumed old ROI routing and dict-shaped detector outputs, and the demo used pre-SDK ROI construction
- Fix: add plugin-side routing/result adapters, refresh demo/tests/scripts/commands, and remove stale command template assumptions
- Prevention: treat targeted tests plus `08_task_routing.md` as the default write path before editing runtime code
- Follow-up: add a small replay fixture set for intrusion and structural positives
