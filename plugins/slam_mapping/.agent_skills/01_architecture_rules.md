# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. 当前 active plugin 以 `manifest.json -> plugin.py::SLAMMappingPlugin` 为准。
2. `semantic_slam_plugin.py` 是并行实现资源，不是当前默认主链路。
3. SDK 兼容接口和业务入口必须分层描述。
4. 不得把 demo / smoke test 成功包装成“完整建图能力已验证”。

## 2. 当前架构事实（slam_mapping）

1. `plugin.py` 同时承担：
   - 点云预处理
   - 地面分割
   - ICP 配准
   - 占据栅格地图更新
   - 物体聚类与特征提取
   - 路径规划
   - 沉降监测
   - 设备注册与查询
2. `process_point_cloud()` 是当前核心业务入口。
3. `infer()` / `postprocess()` 只是 SDK 兼容空壳。
4. `healthcheck()` 与 `shutdown()` 当前没有真实绑定初始化状态。
5. `semantic_slam_plugin.py` 提供语义 SLAM / 变化检测 / 语义地图的另一套实现，但 manifest 未指向它。

## 3. 当前模块职责边界

1. `plugin.py`
   - 可以做：点云处理、地图状态更新、设备位置查询、巡检路径规划
   - 不应继续堆：Web 控制器、数据库 ORM、跨插件联动细节、远程模型下载
2. `semantic_slam_plugin.py`
   - 可以做：未来语义 SLAM 演进或并行方案
   - 不应在未切换 manifest 前被写成现状
3. `standalone/*`
   - 可以做：本地服务运行和模板展示
   - 不应定义新的算法合同
4. `tests/*`
   - 当前只证明最小导入和 smoke 路径
   - 不证明建图精度、配准精度、路径规划正确性

## 4. 当前架构红线

1. 不得把当前插件描述成已有标准 `process()` 平台入口；当前没有该方法。
2. 不得把 `init(config_dict)` 写成真实配置注入路径；当前会误判为 model registry。
3. 不得把 `healthcheck()` 写成真实初始化状态源；当前始终返回 `OK`。
4. 不得把 `semantic_slam_plugin.py` 当成当前 manifest 行为。
5. 不得把 `data/results.db` 当成当前主链路持久化存储。

## 5. 当前可接受的最小演进方向

1. 新增 `scripts/run_sanity_checks.sh`
2. 修复 `init()` 的配置/registry 语义
3. 修复 `shutdown()` / `healthcheck()` / `_is_initialized` 一致性
4. 为 `process_point_cloud()` 增加初始化保护或明确其无状态合同
5. 在补足测试前，不切换到语义 SLAM 实现

## 6. 高风险改动（需人工确认）

1. 切换 manifest 指向 `semantic_slam_plugin.py`
2. 修改 `process_point_cloud()` 返回结构
3. 新增标准 `process()` 并改变平台接入方式
4. 引入真实模型、GPU 推理或外部地图服务
5. 启用导出到任意外部文件路径
