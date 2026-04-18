# 06_refactor_policy

## 1. 当前重构原则

1. 先维护标准治理基线，再做结构美化。
2. 先锁住 `process()` 契约，再调整内部实现。
3. 配置契约已修复后，继续保持 YAML、dataclass、manifest、测试同步。
4. 历史样本 24 条后的预测链路，是当前必须持续回归锁住的行为。

## 2. 允许的低风险重构

1. 补齐 `.agent_skills/00~08`
2. 维护 `scripts/run_sanity_checks.sh`
3. 维护 process/config/trend/standalone 测试
4. 扩展 `configs/default.yaml` 时同步 `_parse_config()`
5. 扩展 `_parse_config()` 时同步 manifest / YAML 中的关键字段
6. 同步 `manifest.json.version` 与实现版本

## 3. 中风险重构（建议先补测试）

1. 扩展 `predictor.py` 模型路径时保持传统预测 fallback。
2. 继续对齐 `model_ids` 命名，避免打断 model registry 路径。
3. 扩展 `analyzer.py` 结果时只能新增字段或兼容字段，不破坏当前 `trend_analysis` 合同。
4. 引入真实数据库写入前先抽象独立存储层

## 4. 高风险重构（需人工确认）

1. 修改 `process()` 输入输出 schema
2. 把当前插件改造成另一种平台接口形态
3. 默认强制依赖模型文件或外部服务
4. 修改 standalone 对外暴露方式或端口
5. 让插件主动写入数据库或跨插件联动

## 5. 当前强制流程

1. 先读 `00/01/02/03`
2. 明确此次改动属于：
   - 文档治理
   - 测试补齐
   - 配置修复
   - 运行时缺陷修复
3. 若改动 `plugin.py` / `predictor.py` / `manifest.json`
   - 至少跑 `cd plugins/gas_detection && ./scripts/run_sanity_checks.sh`
   - 必要时再跑 demo 回放
4. 若修复预测路径
   - 必须新增“24 样本历史”回归用例或脚本检查
