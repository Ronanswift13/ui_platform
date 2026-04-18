# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. 当前插件是“单文件算法插件 + `process()` 平台适配壳”，不是 detector 双层架构。
2. 没有 `input_schema` / `output_schema` 时，输入输出合同必须直接以 `plugin.py::process()` 为准。
3. 占位算法、模拟输入、文案承诺必须与真实能力分开写。
4. 缺失的配置文件、脚本、测试不能靠 skill 文案补成既有能力。

## 2. 当前架构事实（hyperspectral_detection）

1. `plugin.py` 同时承担：
   - 生命周期管理
   - 配置解析
   - 高光谱均值光谱分析
   - 缺陷检测占位输出
   - 材料识别占位输出
   - 建议生成
2. `standalone/app.py` 只负责运行 `StandalonePluginRunner`。
3. `demo/run_demo.py` 是当前最接近真实回放的入口。
4. 当前没有：
   - `detector.py`
   - `configs/default.yaml`
   - `run_standalone.py`
   - 自动化测试
   - 已落地的模型推理层

## 3. 当前模块职责边界

1. `plugin.py`
   - 可以做：输入解包、默认模拟、均值光谱摘要、占位结果拼装
   - 不应继续堆：Web 逻辑、数据库逻辑、跨插件联动、下载模型逻辑
2. `standalone/*`
   - 可以做：本地服务和模板展示
   - 不应定义新的业务阈值或输入合同
3. `demo/*`
   - 可以做：人工验证与示例
   - 不应被写成自动测试替代品

## 4. 当前架构红线

1. 不得把它描述成已有 PCA + 深度学习 + 缺陷分割完整管线；当前实现远未达到该程度。
2. 不得把 `analysis_type` 写成当前可切换处理分支；目前它未参与控制流。
3. 不得把 `_model_registry` 写成当前实际生效依赖；目前仅存储，不参与推理。
4. 不得把 `run_standalone.py` 写成当前可执行入口；它当前不存在。
5. 不得把 manifest 中的依赖与默认配置写成“已全部接入实现”。

## 5. 当前可接受的最小演进方向

1. 补 `configs/default.yaml`
2. 对齐 `_parse_config()` 与 `manifest.json.default_config`
3. 修正光谱维度推断
4. 补最小测试和 sanity 脚本
5. 再考虑接入真实 PCA / model registry / 缺陷定位逻辑

## 6. 高风险改动（需人工确认）

1. 修改 `process()` 输入输出合同
2. 引入真实模型文件或远端模型依赖
3. 调整 standalone 服务暴露方式
4. 让插件依赖真实相机/文件 I/O 或数据库
5. 新增外部网络访问
