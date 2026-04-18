# 07 经验回灌知识库

## 1. When to use
在以下情况时必须更新本文件：
- 修复一个非平凡 bug 后
- 发现一个隐藏的边界条件后
- 降级路径在生产或测试中被触发后
- 性能优化取得显著效果后
- 重构引入了意外回归后

## 2. Inputs
- 问题症状描述
- 根因分析结果（参考 `prompts/root_cause_prompt.md` 模板）
- 修复 diff 或 commit hash

## 3. Outputs
- 新增经验条目（追加到本文件 Known Issues 段落）
- 可选: 新增到 `04_quality_audit.md` 的反模式清单
- 可选: 新增到 `06_refactor_policy.md` 的扩散清单

## 4. Hard Constraints
1. 每条经验必须包含 6 个字段: 日期、症状、根因、类别、修复、预防
2. 根因类别必须属于以下枚举值之一:
   - `数据质量` - 传感器噪声、异常值、时间戳问题
   - `算法发散` - EKF 发散、滤波器不稳定
   - `资源泄漏` - 文件/连接/内存未释放
   - `配置错误` - 参数缺失、类型错误、路径错误
   - `依赖缺失` - 模块/包/模型文件不存在
   - `并发问题` - 竞态条件、死锁、数据竞争
3. 修复必须关联具体 commit hash 或 test case
4. 预防措施必须可自动化验证（测试用例或 lint 规则）
5. 不得记录仅针对单次调试的临时信息

## 5. Algorithm / Logic Contract

### 回灌流程

```
1. 填写经验条目 ─→ 使用下方模板
2. 添加防回归测试 ─→ tests/ 中新增或补充
3. 评估通用性:
   ├─ 通用模式 ─→ 更新 04_quality_audit.md 反模式清单
   ├─ 跨模块影响 ─→ 更新 06_refactor_policy.md 扩散清单
   └─ 仅限本模块 ─→ 只记录在本文件
4. 下次修复前先检索本文件 ─→ 避免重复犯错
```

### 检索方法
```bash
# 按类别检索
grep "类别.*算法发散" .agent_skills/07_learning_log.md

# 按文件检索
grep "相关文件.*fusion" .agent_skills/07_learning_log.md
```

## 6. Validation Rules
- 每条经验的 6 个字段不得为空
- 类别字段必须使用上述 6 个标准枚举值
- 预防字段必须引用具体的测试文件或 lint 命令
- 新增条目后运行 `pytest tests/ -q` 确认引用的测试存在且通过

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| 经验未记录 | 重复犯错 | 修复 PR 中强制包含回灌条目 |
| 预防措施无测试 | 无法自动验证 | 阻塞合并 |
| 类别标注错误 | 检索失效 | 使用枚举值限制 |
| 条目过于碎片化 | 无法复用 | 要求写通用化结论 |

## 8. Required Tests
- 无直接测试要求（本文件是知识库）
- 间接要求: 每条经验引用的测试必须存在且通过

---

## Known Issues (已知问题回灌)

### 条目模板

复制以下模板追加新条目：

```
### [简短标题]

| 字段 | 内容 |
|------|------|
| 日期 | YYYY-MM-DD |
| 症状 | [现象描述] |
| 根因 | [根本原因] |
| 类别 | 数据质量 / 算法发散 / 资源泄漏 / 配置错误 / 依赖缺失 / 并发问题 |
| 修复 | [修复方案 + commit hash 或 PR] |
| 预防 | [测试用例路径 + lint 规则] |
| 相关文件 | [涉及的源码文件] |
```

---

### 硬件设备缺失 Silent Fail

| 字段 | 内容 |
|------|------|
| 日期 | 2026-03-07 |
| 症状 | 当摄像头或 LiDAR 硬件不可用时，系统日志不明确，用户无法判断是否进入降级模式 |
| 根因 | 1) 硬件连接失败时使用普通 `logger.error()` 而非 `FALLBACK` 格式<br>2) 模拟模式下 `_connected=False` 导致 `get_person_detections()` 返回空列表<br>3) 缺少硬件降级场景的测试覆盖 |
| 类别 | 依赖缺失 + 配置错误 |
| 修复 | 1) 统一硬件降级日志格式，使用 `FALLBACK` 前缀和结构化 `extra` 字段<br>2) 修改 `get_person_detections()` 检查逻辑为 `is_connected or is_simulated`<br>3) 新增 8 个测试用例覆盖摄像头和 LiDAR 降级场景<br>4) 确保模拟模式下所有必要组件 (tracker) 被正确初始化 |
| 预防 | `tests/test_camera_adapter.py::test_camera_hardware_unavailable_fallback`<br>`tests/test_camera_adapter.py::test_camera_simulation_detections`<br>`tests/test_lidar_adapter.py::test_lidar_serial_unavailable_fallback`<br>`tests/test_lidar_adapter.py::test_lidar_simulation_mode_works`<br>质量门禁: `grep -r "FALLBACK" adapters/` 验证降级日志 |
| 相关文件 | `adapters/camera_adapter.py`<br>`adapters/lidar_adapter.py`<br>`tests/test_camera_adapter.py`<br>`tests/test_lidar_adapter.py`<br>`.agent_skills/04_quality_audit.md` (新增硬件降级规则) |

**通用化结论**:
- 所有硬件适配器必须支持模拟模式降级
- 降级日志必须使用统一格式: `FALLBACK: {描述}` + 结构化 `extra` 字段
- 模拟模式下必须初始化所有必要组件，确保功能可用
- 状态检查逻辑必须考虑模拟模式: `is_connected or is_simulated`
- 每个硬件降级场景必须有测试覆盖，包括：
  - 硬件不可用时自动降级
  - 模拟模式能生成有效数据
  - 禁用降级时正确失败


| 字段 | 内容 |
|------|------|
| 日期 | 2026-03-07 |
| 症状 | 当 `models/indoor/person_yolov8n.onnx` 不存在时，YOLODetector 静默降级到模拟模式，无日志记录，导致系统行为不可预期 |
| 根因 | `YOLODetector.__init__()` 中使用 bare `except Exception` 捕获所有异常并设置 `_simulation_mode = True`，但未记录降级原因和日志 |
| 类别 | 依赖缺失 |
| 修复 | 1) 添加 `os.path.exists()` 检查，区分文件不存在、加载失败、依赖缺失三种场景<br>2) 每种降级场景使用 `logger.error()` 记录，包含 `extra` 字段<br>3) 添加 `is_simulation_mode()` 和 `get_fallback_reason()` 方法供外部查询<br>4) 新增 3 个测试用例覆盖降级路径 |
| 预防 | `tests/test_detection.py::test_yolo_detector_model_missing_fallback`<br>`tests/test_detection.py::test_yolo_detector_model_exists_recovery`<br>`tests/test_detection.py::test_yolo_detector_no_path_provided`<br>质量门禁: `grep -r "except.*pass" detection/` 扫描 silent fail |
| 相关文件 | `detection/yolo_detector.py`<br>`tests/test_detection.py`<br>`.agent_skills/04_quality_audit.md` (新增规则 9) |

**通用化结论**:
- 所有外部资源加载 (模型、配置、硬件) 必须区分"未提供"、"不存在"、"加载失败"三种场景
- 降级事件必须使用结构化日志 (`extra` 字段) 便于追踪
- 降级状态必须可通过 API 查询，不能仅依赖日志
- 每个降级路径必须有对应测试用例

---

### Web UI 视频流管道缺失

| 字段 | 内容 |
|------|------|
| 日期 | 2026-03-08 |
| 症状 | Web UI 视频区域始终显示 "Awaiting video feed..."，帧率/帧数/检测数均为 0，模拟模式和 RTSP 摄像头均无画面 |
| 根因 | 1) 无后台推理循环 — `infer()` 仅通过 `/api/detect`（文件上传）调用<br>2) 无视频帧 HTTP 端点 — `CameraAdapter.get_frame()` 存在但无路由服务<br>3) HTML 只有 `<div>` 占位符，无 `<img>`/`<canvas>` 渲染元素<br>4) 模拟帧返回纯黑 `np.zeros` 数组<br>5) 数据源切换仅修改 CSS class，未通知后端 |
| 类别 | 配置错误 |
| 修复 | 1) 新增 `standalone/video_stream.py` — 后台推理线程 + MJPEG 生成器 + 帧标注<br>2) 在 `plugin.py` 注册 5 个视频流路由 (stream/start/stop/snapshot/stats)<br>3) 将 HTML `<div>` 替换为 `<img>` + JS 流控制和数据源联动<br>4) 改进 `camera_adapter.py` 模拟帧渲染 (网格/检测框/水印) |
| 预防 | `tests/test_video_stream.py::test_video_stream_service_start_stop`<br>`tests/test_video_stream.py::test_simulation_frame_not_black`<br>质量门禁: 检查 MJPEG 端点注册 |
| 相关文件 | `standalone/video_stream.py`<br>`plugin.py` (get_standalone_routes)<br>`adapters/camera_adapter.py` (get_frame)<br>`standalone/templates/indoor_fence.html`<br>`.agent_skills/04_quality_audit.md` (新增管道完整性规则) |

**通用化结论**:
- Web UI 视频展示必须实现端到端管道: 帧源→推理→标注→编码→传输→渲染
- 后台推理必须在独立线程中运行，并提供 start/stop 生命周期接口
- 模拟模式的帧必须包含可视化信息 (网格、检测框、水印)，不能返回纯黑帧
- 前端数据源切换必须双向联动后端配置，而非仅修改 CSS
- MJPEG 是最简实现方案: `<img src="/api/stream">` + `StreamingResponse(multipart/x-mixed-replace)`

---

### Jinja2 模板块名称不匹配导致前端完全失效

| 字段 | 内容 |
|------|------|
| 日期 | 2026-03-08 |
| 症状 | 实现完整的视频流管道 (VideoStreamService + MJPEG 端点 + 前端 JS) 后，Web UI 选择"模拟数据"仍然全黑无画面。后端 API (curl 测试) 全部正常返回数据，`/api/indoor-fence/snapshot` 返回 76KB JPEG 图片。但页面上 `videoStream` 元素和 `startVideoStream` 函数均不存在 |
| 根因 | 插件模板 `indoor_fence.html` 使用了 `{% block plugin_content %}` 和 `{% block plugin_scripts %}`，但基础模板 `base_standalone.html` 定义的是 `{% block content %}` 和 `{% block extra_scripts %}`。Jinja2 对子模板中未定义的块名**静默忽略** — 内容被直接丢弃，不会抛出任何异常。页面渲染了基础模板的侧边栏和状态栏（造成"部分正常"的假象），但插件的视频面板和 JS 代码完全缺失 |
| 类别 | 配置错误 |
| 修复 | 将 `{% block plugin_content %}` → `{% block content %}`，`{% block plugin_scripts %}` → `{% block extra_scripts %}`。修复后 `curl http://localhost:8081/ \| grep -c videoStream` 从 0 变为 3 |
| 预防 | 创建子模板前必须 `grep '{%.*block.*%}' base_standalone.html` 查阅可用块<br>渲染验证: `curl -s http://localhost:8081/ \| grep -c '关键元素ID'` 必须 > 0<br>质量门禁: `.agent_skills/04_quality_audit.md` 新增"Jinja2 模板块名称一致性规则" |
| 相关文件 | `standalone/templates/indoor_fence.html`<br>`darkbreaker_sdk/standalone/templates/base_standalone.html`<br>`.agent_skills/04_quality_audit.md` (新增 Jinja2 块名规则) |

**通用化结论**:
- Jinja2 模板继承时，子模板的块名必须与父模板完全一致，否则内容被静默丢弃
- **高危诊断特征**: 后端 API 完全正常 + 页面 HTTP 200 + 前端功能缺失 = 模板渲染层断裂
- 创建子模板前必须先查阅父模板的块定义 (`grep block`)
- Jinja2 不报错的设计是为了灵活性，但对调试极其不友好 — 建议开发模式使用 `StrictUndefined`
- 修复后必须验证渲染完整性: 检查关键 HTML 元素和 JS 函数存在于最终页面中

---

### standalone 启动依赖漂移导致跨插件启动失败

| 字段 | 内容 |
|------|------|
| 日期 | 2026-03-26 |
| 症状 | `python run_standalone.py` 或 VS Code 直接运行插件入口时，部分插件报 `ModuleNotFoundError: No module named 'numpy'`，而另一些插件可启动，导致“同仓不同命”的独立运行体验。 |
| 根因 | 大多数插件的 `run_standalone.py` 只做了 `sys.path` 注入，没有统一处理解释器选择；当入口脚本被系统 Python 启动时，会绕过项目根目录 `venv`，从而在导入 `darkbreaker_sdk` 或插件依赖时触发缺包。此前修复只落在个别插件，未形成全仓统一策略。 |
| 类别 | 依赖缺失 |
| 修复 | 统一为全部 `plugins/*/run_standalone.py` 增加“虚拟环境自动切换”逻辑：若当前不在 venv 中，则优先 `execv` 到项目根目录 `venv/bin/python`，其次兼容插件目录 `.venv/bin/python`；若两者都不存在，则打印标准化创建命令并退出。随后用系统 `python3` 对全量 standalone 入口做批量冒烟验证，14 个插件均成功进入 Uvicorn 启动链路。 |
| 预防 | 回归验证脚本：批量执行 `python3 plugins/*/run_standalone.py`，断言日志中不再出现 `ModuleNotFoundError`，并至少出现 `Application startup complete` / `Uvicorn running` / `address already in use` 之一；质量门禁：扫描 `plugins/*/run_standalone.py` 必须包含 `PROJECT_VENV_PYTHON` 与 `os.execv`。 |
| 相关文件 | `plugins/*/run_standalone.py`<br>`requirements.txt`<br>`darkbreaker_sdk/interfaces/base_plugin.py` |

**跨插件可复用经验总结**:
- standalone 启动问题先查“解释器漂移”，再查业务代码；先看 `sys.executable`，不要先怀疑算法模块。
- 对同一仓库多插件项目，应优先统一到“项目根 venv”，避免每个插件各自维护环境导致依赖分叉。
- `run_standalone.py` 必须在**导入任何项目模块之前**完成 venv 切换，否则 guard 形同虚设。
- 对启动链路做验证时，不要求所有硬件都在线；只要能进入 Uvicorn 启动阶段，硬件缺失应走模拟/降级路径而不是 import fail。
- 端口占用属于运行态问题，不应与 `numpy`/依赖缺失混为一谈；两者要分层诊断。

---

### 多目标跟踪贪婪分配导致交叉匹配 + 融合管道断裂

| 字段 | 内容 |
|------|------|
| 日期 | 2026-03-27 |
| 症状 | 1) 多人场景下目标 ID 频繁跳变，轨迹交叉错乱<br>2) 仿真界面无法显示实时轨迹和距离，连接外设后仍无法正常跟踪<br>3) UWB/IMU 传感器数据未参与定位融合<br>4) 仿真仅使用 Camera 单传感器数据 |
| 根因 | 1) `RealtimeMultiPersonTracker._associate()` 使用贪婪最近邻匹配，O(n²) 扫描取局部最优，多目标场景下产生交叉分配<br>2) 代价矩阵仅使用 2D 距离 (dx²+dy²)，忽略 UWB 提供的 z 维度<br>3) `SensorFusionV3` 未集成到 `RealtimeMultiPersonPipeline`，管道直接从 Simulator 提取原始 Camera 检测跳过融合<br>4) `SensorFusionV3._associate()` 同样使用贪婪匹配<br>5) IMU 数据在 `_extract_observations()` 中被 `pass` 跳过<br>6) `SimulatorConfig` 默认仅启用 `[SensorType.CAMERA]`，仿真模式下 UWB/LiDAR/IMU 数据不生成 |
| 类别 | 算法发散 |
| 修复 | 1) 替换 `RealtimeMultiPersonTracker._associate()` 为 Hungarian 最优分配 (`core/tracking/hungarian.py`)<br>2) 代价矩阵升级为 3D 欧几里得距离 (dx²+dy²+dz²)<br>3) 替换 `SensorFusionV3._associate()` 为 Hungarian 最优分配<br>4) 在 `SensorFusionV3.update()` 中集成 IMU 速度更新 (`ekf.update_imu()`)<br>5) 在 `RealtimeMultiPersonPipeline` 中插入 `SensorFusionV3` 融合阶段<br>6) 新增 `_get_sensor_data()` 方法将多传感器原始数据送入融合引擎<br>7) 更新 `SimulatorConfig` 默认启用 Camera+LiDAR+UWB+IMU<br>8) 新增 14 个测试覆盖 Hungarian 分配、3D 跟踪、IMU 融合、多传感器管道 |
| 预防 | `tests/test_realtime_tracking.py::TestHungarianAssociation` (6 tests)<br>`tests/test_realtime_tracking.py::TestMultiSensorFusionPipeline` (3 tests)<br>`tests/test_realtime_tracking.py::TestSensorFusionV3Hungarian` (3 tests)<br>`tests/test_fusion_v3.py::test_fusion_multi_target_hungarian`<br>`tests/test_fusion_v3.py::test_fusion_imu_integration`<br>`tests/test_ekf.py::test_ekf_update_imu` |
| 相关文件 | `core/tracking/realtime_tracker.py`<br>`core/fusion/sensor_fusion_v3.py`<br>`standalone/realtime_pipeline.py`<br>`core/tracking/hungarian.py` |

**通用化结论**:
- 多目标跟踪必须使用 Hungarian 最优分配而非贪婪匹配，特别是目标数 > 3 时贪婪会产生明显的交叉分配
- 有 3D 传感器 (UWB) 时，代价矩阵必须使用 3D 距离，否则同 (x,y) 不同 z 的目标会被错误合并
- 融合引擎必须完整嵌入管道主循环，不能跳过直接使用原始检测
- 仿真模式必须模拟全部传感器类型，否则融合算法在仿真中永远不被测试到
- IMU 数据虽然不提供位置，但通过 EKF 速度更新可以显著改善运动状态估计
