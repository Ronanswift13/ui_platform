# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. 当前 active plugin 以 `manifest.json -> plugin.py::MultimodalFusionPlugin` 为准。
2. `fusion_engine_enhanced.py`、`fusion_engine.py`、`plugin_v4_bayesian.py` 都是实现资源，但不等于“当前稳定合同”。
3. 服务编排层与并行算法层要分开描述。
4. 不得把 manifest 中声明过的能力自动当成已稳定落地能力。

## 2. 当前架构事实（multimodal_fusion）

1. `plugin.py` 负责：
   - 生命周期管理
   - 模态插件注册
   - 模态结果汇总
   - 策略切换
   - 增强融合引擎优先尝试
   - 基础融合回退
   - 诊断报告、建议、告警输出
2. `fusion_engine_enhanced.py` 提供：
   - cross-attention 融合
   - 贝叶斯决策网络
   - 策略管理
3. `plugin_v4_bayesian.py` 是另一套贝叶斯融合插件原型，不是当前 manifest 主类。
4. 当前主接口是 `process(inputs)`，不是 `detect()`。

## 3. 当前模块职责边界

1. `plugin.py`
   - 可以做：模态编排、回退、状态聚合、告警输出
   - 不应继续堆：Web 逻辑、数据库持久化、远程模型下载
2. `fusion_engine_enhanced.py`
   - 可以做：增强融合算法
   - 不应在未稳定输入合同前被写成“可靠默认链路”
3. `plugin_v4_bayesian.py`
   - 可以做：未来演进参考
   - 不应在未切 manifest 前被写成当前能力
4. `standalone/*`
   - 可以做：本地服务运行和模板渲染
   - 不应定义新的业务合同

## 4. 当前架构红线

1. 不得把当前插件写成单模态 detector。
2. 不得把 `pre_processed` / `modality_results` 写成当前 `process()` 已消费字段。
3. 不得把增强引擎写成当前对常见字典输入稳定可用的默认链路。
4. 不得把 `plugin_v4_bayesian.py` 写成现行实现。
5. 不得把 `run_standalone.py` 写成当前可执行入口；它不存在。

## 5. 当前可接受的最小演进方向

1. 新增 `scripts/run_sanity_checks.sh`
2. 补最小 `tests/test_plugin_process.py`
3. 对齐 `_parse_config()` 与 manifest 默认配置
4. 修复增强引擎对常见 dict 输入的脆弱性
5. 再决定是否切换到贝叶斯主链路或真正接外部模态插件

## 6. 高风险改动（需人工确认）

1. 切换 manifest 指向其他实现文件
2. 修改 `process()` 输入输出合同
3. 引入真实外部模型/远程依赖
4. 更改 standalone 对外暴露方式
5. 自动控制下游闭环或决策总线
