# 06 受控重构策略

## 1. When to use
- 重命名或移动 `plugin.py`、`protocols.py`、`core/*`、`standalone/*`
- 修改 config schema、zone schema、scenario schema
- 修改 `BaseAdapter`、`Plugin` 入口、standalone route surface
- 单次改动跨越 5 个以上文件或同时触达主插件链路与 V3 演练链路

## 2. Inputs
- 重构目标说明
- 受影响文件清单
- 当前测试与覆盖率基线
- `manifest.json`、`configs/default.yaml`、`.coveragerc`

## 3. Outputs
- 分批次的改动计划
- 每批对应的 targeted / regression 验证命令
- 需要同步更新的 docs / skills / tests 列表

## 4. Hard Constraints
- 保持 `manifest.json -> plugin.py -> IndoorFencePlugin` 入口契约
- 保持 `create_standalone()` 与 `Plugin = IndoorFencePlugin`
- 配置字段重构时必须同步更新 `configs/default.yaml`、`core/config_manager.py`、相关测试；对外字段变化再同步 `manifest.json`
- route / UI 相关重构时必须同步更新模板、前端脚本和 API 测试
- 不要把 legacy 清理和新功能开发混在同一次无边界扩散里
- 每一批完成后都要跑最近的 `run_targeted_tests.sh <module>`
- 全部完成后必须跑 `./scripts/run_regression_tests.sh`
- 若重构提炼出新风险模式，必须回写 `04_quality_audit.md` 和 `07_learning_log.md`

## 5. Algorithm / Logic Contract

### 推荐扩散顺序

```text
1. 契约层
   protocols.py / manifest.json / configs/default.yaml / standalone/configs/zone.yaml
2. 算法或适配层
   core/* / adapters/* / detection/*
3. standalone 支撑层
   standalone/video_stream.py / realtime_pipeline.py / training*.py / simulator*.py
4. 入口层
   plugin.py / run_standalone.py / standalone/app.py
5. 测试与文档
   tests/* / .agent_skills/* / CLAUDE.md / PROJECT_CARD.md
```

### indoor_fence 特殊规则
- 若任务只想增强新版检测能力，优先重构 `detection/*`，不要继续加重 `detector.py` 的历史包袱
- 若任务触及主插件链路与 V3 演练链路的同名概念，必须说明“哪个是当前交付主路径，哪个是演练路径”
- 若任务改了 `configs/scenarios/*.json` 或 `zone.yaml` 结构，必须连带更新 scenario / zone 相关测试

## 6. Validation Rules

```bash
# 先跑最近模块
./scripts/run_targeted_tests.sh <module>

# 全量回归
./scripts/run_regression_tests.sh

# 交付前闸门
./scripts/run_quality_gate.sh

# 影响面扫描
rg -n "旧接口名|旧字段名" . --glob '*.py' --glob '*.md' --glob '*.yaml' --glob '*.json'
```

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| 只改代码不改配置/schema/test | runtime 更新失败 | 按契约层优先顺序传播 |
| route 重构未联动模板/前端 | UI 断裂 | 同步更新 API + template + JS + tests |
| 主链路与 V3 链路混改 | 责任边界模糊 | 在计划里先标明落点 |
| legacy 清理与新功能混合 | diff 失控、回归难定位 | 拆成独立批次 |
| 只跑 targeted 不跑 regression | 跨模块回归漏检 | 完成后统一跑 regression |

## 8. Required Tests
- `./scripts/run_targeted_tests.sh <module>`
- `./scripts/run_regression_tests.sh`
- `./scripts/run_quality_gate.sh`

