# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. 先认清这是“服务集成/时序分析编排插件”，不是视觉检测器。
2. 平台契约与运行入口以 `plugin.py` 为准，`manifest.json` 是声明层，不等于代码已完整实现。
3. 缺失的配置、测试、脚本只能如实暴露，不能由 skill 补脑。
4. 允许补治理骨架，但不允许为了迎合模板把结构改成虚假的 `detector.py + tests + scripts` 成熟形态。

## 2. 当前架构事实（gas_detection）

1. `plugin.py` 是编排层：
   - 生命周期：`init()` / `shutdown()` / `cleanup()`
   - 平台处理：`process(inputs)`
   - 状态输出：`get_plugin_status()` / `healthcheck()` / `plugin_info`
   - 内存态：每个 `device_id` 一组历史缓冲和告警状态
2. `predictor.py` 是可选增强层：
   - 优先尝试 `ai_models.deep_learning.gl_translstm`
   - 其次尝试 `model_registry`
   - 最后回退传统线性外推
3. `analyzer.py` 是离线分析工具层：
   - 提供趋势、异常、DGA、相关性分析
   - 当前并未接入 `process()` 主返回
4. `standalone/` 与 `run_standalone.py` 只是服务壳，不承载业务判断。

## 3. 当前模块职责边界

1. `plugin.py`
   - 可以做：输入解包、历史更新、阈值分级、调用预测/泄漏检测、生成告警和建议
   - 不应继续做：复杂 DGA 推理实现、数据库 ORM、Web 控制器逻辑
2. `predictor.py`
   - 可以做：预测、泄漏检测、模型初始化、模型清理
   - 不应做：平台 schema 拼装、HTTP 服务逻辑、SDK 生命周期管理
3. `analyzer.py`
   - 可以做：纯分析函数
   - 不应假装已经是主链路输出
4. `standalone/*`
   - 可以做：运行容器和模板渲染
   - 不应持有新的业务阈值源

## 4. 当前架构红线

1. 不得把本插件描述成已有 `detect()` 主接口的检测器插件；当前主接口是 `process()`.
2. 不得声称 `manifest.json.default_config` 会自动完整生效；当前 `_parse_config()` 实际只处理 `thresholds`.
3. 不得声称 `data/results.db` 是当前处理链路的一部分；目前未见写入调用。
4. 不得把 `analyzer.py` 的能力写成当前 `process()` 已输出字段。
5. 不得改动 `process()` 返回字段而不同时审视平台消费方。

## 5. 当前可接受的最小演进方向

1. 新增 `configs/default.yaml`，把真实可调参数从代码默认值迁出。
2. 扩展 `_parse_config()`，真正解析 `history_length`、`prediction_horizon`、`leak_detection_window` 等字段。
3. 为 `process()` 基础链路补最小测试。
4. 新增最小 sanity 脚本，锁住 import / init / sample process / demo 回放。
5. 如果要暴露更多趋势/DGA 字段，先把 `analyzer.py` 以显式可观测方式接入主链路。

## 6. 高风险改动（需人工确认）

1. 调整 `process()` 输入或输出 schema。
2. 变更 `manifest.json` 的 `id` / `entrypoint` / `plugin_class`.
3. 引入真实数据库写入、网络请求、远端模型下载。
4. 修改独立服务监听地址、端口或公开方式。
5. 将当前“可降级运行”改为“强依赖模型或配置文件”。
