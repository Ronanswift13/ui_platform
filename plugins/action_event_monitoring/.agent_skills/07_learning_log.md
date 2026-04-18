# 07_learning_log

用于记录当前插件真实落地过的经验，以及仍可指导未来演进的历史教训。

## Entry Template

- Date:
- Context:
- Symptom:
- Root cause:
- Fix:
- Prevention:
- Follow-up:

## Entries

- Date: 2026-04-07
- Context: 历史上游设计希望把动作链分析结果做成更可操作的候选事件闭环
- Symptom: 旧文档大量提到 CandidateEvent、人工复核 API、设备查询 API，但当前本地插件目录并没有这些实现
- Root cause: 经验日志记录了上游设计方向，却被误当成当前本地事实
- Fix: 在当前 skill 体系中把这些内容降级为“历史经验/未来路线”，不再作为现有合同
- Prevention: 经验日志只能记录“已发生教训”或“明确标注的未来路线”，不能替代当前代码事实源
- Follow-up: 如果未来真的补本地 API/复核流转，再回写到 `02/04/08`

---

- Date: 2026-04-07
- Context: 历史根因分析经验
- Symptom: 证据不足时若仍给出高置信度结论，会误导值班人员
- Root cause: 动作链和根因分析天然依赖外部证据，缺少录波、巡视、二次设备健康监测时，结论充分性不足
- Fix: 保留“证据不足必须降级”的经验，作为当前 `analysis_result` 审查重点
- Prevention: 任何自动分析输出都必须关注 `confidence`、`evidence_sufficiency`、`manual_review_items`
- Follow-up: 若后续新增本地候选事件模型，应把这些字段作为必填项而不是可选项

---

- Date: 2026-04-08
- Context: 本地治理整改
- Symptom: 当时插件目录只有 `04` 和 `07` 两个 skill，且内容呈现“伪完整”状态，容易误导 AI 认为已有 tests / API / review 闭环
- Root cause: 文档先于治理骨架落地，且没有把“当前本地实现”和“历史设计愿景”分开
- Fix: 以 `plugin.py + manifest.json + configs/` 为事实源，补齐最小可信的 `00~08`；该历史阶段已被 2026-04-17 的 tests/standalone/entrypoints/global mapping 升级覆盖
- Prevention: 当前已进入标准治理基线，仍需避免把未落地的 API/UI/复核闭环写成现状
- Follow-up: 已补 `tests/` 最小骨架；下一步是 targeted/quality gate 或 UI/cockpit 授权接线

---

- Date: 2026-04-17
- Context: 平台统一化缺口收口，仅允许修改 `plugins/action_event_monitoring`
- Symptom: 本插件已有局部 `tests/` 与 `standalone/`，但缺少 `requirements.txt`、`demo/run_demo.py`、`__main__.py`、`run_standalone.py` 和 `Plugin = ActionEventMonitoringPlugin` 别名；历史 skill 仍记录“无 tests/standalone”，与当前事实源不一致
- Root cause: 早期最小治理只保证 import/init/process；后续补了局部 standalone/tests，但没有同步补齐顶层命令入口和本地 manifest standalone 元数据
- Fix: 新增本地 requirements、demo、模块入口、standalone 运行入口、Plugin 别名、entrypoint 合同测试，并把 sanity 脚本扩展为同时校验本地入口和 pytest；随后补入全局 integration standalone 清单与 `platform_core/plugin_manager/installer.py` 分类/端口映射，统一端口为 `8097`
- Prevention: 后续若要让 dashboard/cockpit 真正出现统一入口，必须在获得 UI 目录授权后修改 UI 清单；在此之前只能声明“后端安装器与 integration 清单已接入”，不能伪造 UI 接线完成
- Follow-up: UI/dashboard/cockpit 统一入口仍需获得前端目录授权后另行接线；当前只能声明后端安装器、integration 清单和本地 standalone 已接入

## 仍然有效的历史经验

1. 证据不足时必须降级，而不是高置信度输出。
2. 误动、顺序异常、缺少断路器响应等场景天然需要人工复核。
3. 缺失历史库、录波、巡视、二次设备健康监测时，应给出明确缺口说明。

## 当前本地尚未落地的能力（仅作路线，不作现状）

1. CandidateEvent 本地模型与接口
2. 人工复核状态流转接口
3. 设备+时间窗统一查询接口
4. 完整 regression/coverage 质量门
