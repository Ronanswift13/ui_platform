# 07_learning_log

用于记录当前插件已核验的经验、暴露出的契约问题，以及后续修复时必须记住的教训。

## Entry Template

- Date:
- Context:
- Symptom:
- Root cause:
- Fix:
- Prevention:
- Follow-up:

## Entries

- Date: 2026-04-09
- Context: standalone 配置回退核验
- Symptom: 插件目录没有 `configs/default.yaml`，但 `create_standalone()` 仍然成功
- Root cause: SDK 配置加载缺文件返回空字典，实例随后回到 `_default_config()`
- Fix: 在 skill 中明确记录这是“默认配置兜底”，不是“已有配置文件”
- Prevention: 未来若需要可调 YAML，先补真实 `configs/default.yaml`
- Follow-up: 新增 YAML 后必须重新验证 `init()` 语义是否正确

---

- Date: 2026-04-09
- Context: 初始化合同核验
- Symptom: `init({'voxel_size': 0.2})` 后 `dl_enabled=True`，但 `point_processor.voxel_size` 仍为 0.1
- Root cause: `init()` 当前只接受 `model_registry`，非空 dict 被误当成 registry
- Fix: 后续应拆分 `init(config)` 与 `set_model_registry()`，或显式区分参数类型
- Prevention: 配置注入和依赖注入不能共用一个模糊参数位
- Follow-up: 修复后补 `tests/test_plugin_contract.py`

---

- Date: 2026-04-09
- Context: 生命周期一致性核验
- Symptom: 未初始化时 `process_point_cloud()` 仍能运行，`shutdown()` 后 `healthcheck()` 仍是 `OK`
- Root cause: 当前主处理链和健康检查都没有真正绑定 `_is_initialized`
- Fix: 在 skill 中把它标为高风险契约问题，而非正式设计
- Prevention: 生命周期状态必须由测试锁住
- Follow-up: 先补测试，再决定是允许无状态处理还是要求强制 init

---

- Date: 2026-04-09
- Context: 多实现并存核验
- Symptom: `__init__.py` 导出 `SemanticSLAMPlugin` 且版本为 `2.0.0`，但 manifest 主类仍是 `SLAMMappingPlugin` `1.0.0`
- Root cause: 目录内同时存在主实现与候选语义实现，且元数据未统一
- Fix: 当前 skill 体系中把语义实现降级为“并行实现/演进方向”
- Prevention: 只有 manifest 指向的实现，才能写成当前现状
- Follow-up: 若未来切换 manifest，再同步更新 `00/02/04/08`
