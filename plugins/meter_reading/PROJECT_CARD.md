# PROJECT_CARD

## 1. 项目名称
变电站插件 — 表计读数算法迭代改进项目 (Meter Reading Plugin Enhancement)

## 2. 项目类型
- **plugin_update**

## 3. 输入源
| 数据源 | 频率 | 时间戳 | 单位 | 协议 | 噪声特点 |
|--------|------|--------|------|------|----------|
| 巡检摄像头帧 (BGR np.ndarray) | 按任务触发，单帧推理 | Unix timestamp (ms) | 像素 | DarkBreaker SDK BasePlugin.infer() | 光照不均、反光眩光、倾斜畸变、低分辨率 |
| ROI 区域 (归一化坐标) | 与帧同步 | — | 归一化 [0,1] | SDK ROI schema (x, y, w, h) | 框偏移、遮挡、多表重叠 |
| 设备上下文 (task/site/device/component ID) | 每次调用携带 | — | 字符串 | PluginContext | 无 |

**支持的表计类型 (9种)**:
pressure_gauge, temperature_gauge, oil_level_gauge, sf6_density_gauge,
digital_display, led_indicator, ammeter, voltmeter, seven_segment

## 4. 输出目标
| 输出项 | 格式 | 刷新频率 | 精度要求 | 延迟要求 |
|--------|------|----------|----------|----------|
| RecognitionResult | SDK schema (value: float, confidence: float, label: str, metadata: dict) | 每帧一组结果 | 模拟表: 满量程 ±2%; 数字表: 完全匹配 | ≤500ms/帧 (含预处理+推理+后处理) |
| Alarm | SDK schema (level: INFO/WARNING/CRITICAL) | 由 postprocess 触发 | — | 与推理同步 |
| HealthStatus | SDK HealthStatus enum | 周期性心跳 | — | 实时 |

**Standalone Web 端口**: 8091，提供仪表盘(成功率/推理次数/待人工复核数)

## 5. 关键约束
### 物理约束
- 表盘视角倾斜 ≤45° (perspective_correction.max_rotation)
- 模拟表指针角度范围 [-135°, 135°]
- LED 指示灯需 HSV 色彩空间可分离

### 工程约束
- 依赖 darkbreaker-sdk >= 1.0.0，遵循 BasePlugin 接口
- 模型推理仅支持 CPU/CUDA (ONNX Runtime)
- 单帧推理时间 ≤500ms (performance.max_reading_time_ms)
- 成功率目标 ≥95% (performance.success_rate)

### 资源约束
- 无 GPU 时需纯 OpenCV fallback 可用
- 模型文件不超过 50MB (部署约束)
- 内存占用 ≤512MB

### 安全约束
- 不访问外部网络
- 不持久化原始图像 (隐私合规)
- 配置变更需热加载，不可中断推理

## 6. 验收标准
| 编号 | 验收项 | 目标 | 测试方法 |
|------|--------|------|----------|
| AC-1 | 模拟表读数精度 | 满量程误差 ≤2% | 标定图片集回归测试 |
| AC-2 | 数字表 OCR 准确率 | ≥98% 字符级准确 | 字符级 benchmark |
| AC-3 | LED 状态识别 | ≥99% 准确率 | 红/绿/黄/灭四类标注集 |
| AC-4 | 单帧推理延迟 | ≤500ms (CPU) | 100 帧 P95 统计 |
| AC-5 | 整体成功率 | ≥95% (非 NEED_MANUAL_REVIEW) | 全量测试集 |
| AC-6 | 透视校正有效性 | 倾斜 ≤45° 时精度不劣化 | 合成倾斜图片测试 |
| AC-7 | Fallback 链路完整 | HRNet → HoughCircle → HoughLine 逐级降级可用 | 模拟模型缺失场景 |
| AC-8 | Healthcheck | init/infer/postprocess 全链路无异常 | 冒烟测试 |
| AC-9 | 低置信度复核 | confidence < 0.5 正确标记 NEED_MANUAL_REVIEW | 边界用例集 |

## 7. 禁止事项
- 禁止引入 PyTorch/TensorFlow 等重依赖 (仅允许 ONNX Runtime)
- 禁止修改 darkbreaker_sdk 接口层
- 禁止改动 standalone/templates/ 之外的 UI 文件
- 禁止访问线上接口或外部 API
- 禁止在代码中留 TODO/FIXME/HACK 未解决标记
- 禁止在推理主路径新增未声明的量程硬编码；当前基线量程由 `detector_enhanced.py::METER_RANGES` 注册表维护，变更时需同步契约与测试
- 禁止删除已有 fallback 链路
- 禁止降低现有测试覆盖率

## 8. 已知参考物
| 类型 | 路径/名称 | 说明 |
|------|-----------|------|
| 核心算法 | `detector_enhanced.py` (~955行) | V3.0 增强检测器，含 HRNet keypoint + Hough fallback |
| 插件接口 | `plugin.py` (~404行) | BasePlugin 实现，含 init/infer/postprocess/healthcheck |
| 默认配置 | `configs/default.yaml` | 30+ 可调参数 |
| 插件清单 | `manifest.json` | 插件元数据 (id: meter_reading, version: 1.0.0) |
| Web UI | `standalone/templates/meter_reading.html` | 仪表盘模板 |
| Demo | `demo/run_demo.py` | 演示脚本 |
| 模型ID | hrnet_meter_keypoint / crnn_meter_ocr / meter_type_classifier | 模型注册表标识 |
| 预置量程 | METER_RANGES (detector_enhanced.py 内) | 压力/温度/油位/SF6 量程映射 |
| SDK 接口 | darkbreaker_sdk.interfaces.BasePlugin | 插件基类定义 |

## 9. 当前任务
### V3.1 迭代 (2026-03-19) - 已完成
- [x] **detector_enhanced.py 升级至 V3.1**, 严格遵循算法合约:
  - 移除旧的第四状态, 严格三态: SUCCESS / FAILED / NEED_MANUAL_REVIEW
  - 新增 CLAHE + 去眩光 + 对比度增强预处理流水线
  - 实现 HRNet -> HoughCircle -> HoughLine 三级降级链路 (fallback_level 追踪)
  - 统一置信度计算 `c = min(c_det, c_geom, c_parse)`, NaN/负数/超1校验
  - 指针角度越界 `[-135, 135]` 强制进入 NEED_MANUAL_REVIEW
  - 透视校正含视角超限检测 (`max_rotation=45`)
  - 量程缺失时强制进入 NEED_MANUAL_REVIEW, 不使用默认值
  - LED: HSV 可分离性校验 + 发光面积占比 + 低饱和度/颜色簇重叠检测
  - 数字表: 严格 OCR 文本清洗 (多小数点/中间负号/非法字符剥离)
  - 所有结果 metadata 输出符合合约要求
  - 配置热加载 `reload_config()` 支持, 校验失败保持旧配置
- [x] **plugin.py 升级**: MeterType 枚举传递, metadata 必填字段完整
- [x] **configs/default.yaml 修正**: angle_range [-135,135], 新增 led_detection/CLAHE 参数
- [x] **测试全量通过**: 98 个测试用例 (L0+L1), 覆盖全部合约场景
- [x] 所有阈值配置驱动, 无硬编码业务参数
- [x] 无 TODO/FIXME/HACK 残留

### 本轮不做
- 不训练或替换模型
- 不改动 Web UI 模板
- 不做线上部署
- 不做 L2 回归测试 (需标定图片集)

## Phase 2 能力三分表 (2026-04-16 本轮审计后)

| 能力 | 分类 | 证据 |
|------|------|------|
| output_structure_contract | **verified** | `test_output_structure.py` 覆盖 RecognitionResult / Alarm / metadata 必填字段，新增 runtime_mode / review_status / failure_reason 契约 |
| expected_results_schema | **verified** | `tests/replay/expected_results.json` schema 由 `test_replay.py` 校验 |
| mock_led_replay_runner | **verified-with-limit** | `meter_led_indicator_green_001.fixture.json` 可执行 mock replay，仅证明 replay plumbing 和 HSV mock 语义 |
| analog_meter_reading | experimental | 代码完整(HRNet + Hough fallback)，但 replay 中只有 planned analog 槽位，无真实标注图像和值 |
| digital_ocr_reading | experimental | OCR 清洗/解析有单元测试；真实数字表图片和人工字符标签缺失，传统 OCR 当前仍返回空串 |
| led_indicator_real_image_reading | experimental | HSV 逻辑和 mock replay 可跑；真实红/绿/黄/灭灯现场图像尚未进入 replay |
| perspective_correction | experimental | 透视校正代码完整，max_rotation=45°；glare/tilt 只有 planned 复核槽位 |
| fallback_chain | experimental | HRNet→HoughCircle→HoughLine 链路代码存在，现有测试只证明可返回 fallback_level，不能证明真实逐级降级效果 |
| real_dl_keypoint_inference | **blocked** | HRNet 模型为可选依赖，当前不可用 |
| real_dl_ocr_inference | **blocked** | CRNN OCR 模型为可选依赖，当前不可用 |

**runtime_mode_support**: real_dl=❌ traditional_fallback=✅ simulation=✅
**测试状态**: ✅ `plugins/meter_reading/tests` 可收集并运行；本轮新增最小 replay 骨架后以 pytest 实测为准。
**关键差距**: 最大缺口已从 import/collect 转为真实 fixture/replay 样本不足；`tests/regression/` 仍为空。

## 最小试点回放基线 (2026-04-16)

| 槽位 | 状态 | 用途 |
|------|------|------|
| `analog_normal` | planned | 清晰正视角模拟表，需补真实图像、人工读数、量程和单位 |
| `analog_boundary` | planned | 指针角度/量程边界样本，需补边界标签和容差 |
| `analog_quality_fail` | planned | 模糊/过暗/遮挡样本，验证不伪造读数 |
| `digital_display` | planned | 数字表/七段码样本，需补人工字符和数值标签 |
| `led_indicator` | present mock | 当前仅有绿色纯色 mock fixture，可跑 replay plumbing |
| `glare_or_tilt_review_required` | planned | 强眩光/倾斜样本，验证 manual_review_required 语义 |

`expected_results.json` 的输出 metadata 必查字段:

- `meter_type`
- `reading_status`
- `pipeline_stage`
- `fallback_level`
- `timestamp_ms`
- `runtime_mode`
- `review_status`
- `failure_reason`
