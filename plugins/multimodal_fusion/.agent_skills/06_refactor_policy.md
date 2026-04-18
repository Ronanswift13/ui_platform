# 06_refactor_policy

## 1. 当前重构原则

1. 先锁住融合合同和回退行为，再做引擎升级。
2. 先补最小测试与脚本，再考虑切换主实现。
3. 当前以 `plugin.py` 主链路为准，不跨越到 `plugin_v4_bayesian.py`。
4. 任何涉及增强引擎的改动都必须明确“失败是否允许回退”。

## 2. 允许的低风险重构

1. 补齐 `.agent_skills/00~08`
2. 新增 `scripts/run_sanity_checks.sh`
3. 新增 `tests/test_plugin_process.py`
4. 同步 `requirements.txt` 与 manifest 依赖
5. 修正文案里缺失的 `run_standalone.py` 引用

## 3. 中风险重构（建议先补测试）

1. 修复增强引擎对常见 dict/status 输入的失败
2. 对齐 `_parse_config()` 与 manifest 默认配置
3. 让 `pre_processed` / `modality_results` 真正参与处理，或从 manifest 删除
4. 增加外部模态插件注册/发现层

## 4. 高风险重构（需人工确认）

1. 切换 manifest 指向 `plugin_v4_bayesian.py`
2. 修改 `process()` 输入输出合同
3. 引入真实模型/GPU/远程推理
4. 对下游决策总线或闭环控制做自动动作
5. 改变 standalone 服务暴露方式

## 5. 当前强制流程

1. 先读 `00/01/02/03`
2. 明确改动属于：
   - 文档治理
   - 测试/脚本治理
   - 合同修复
   - 引擎/算法修复
3. 若改动 `plugin.py` / `fusion_engine_enhanced.py` / `manifest.json`
   - 至少跑 demo 回放
   - 至少跑一次基础 `process()` 校验
   - 至少确认增强引擎回退行为
