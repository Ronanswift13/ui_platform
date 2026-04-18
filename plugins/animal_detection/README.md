# Animal Detection Plugin

DarkBreaker 动物入侵检测插件 — 基于 YOLOv8 ONNX 推理，支持多目标跟踪、热成像验证和自动驱离。

## 目录结构

```
animal_detection/
├── core/                   # 核心算法模块
│   ├── detector.py         # YOLOv8 ONNX 检测器
│   ├── onnx_inference.py   # ONNX 推理引擎（COCO/自定义模型）
│   ├── tracker.py          # 多目标跟踪（IoU 匹配）
│   ├── thermal_validator.py# 热成像验证
│   ├── deterrent.py        # 驱离控制（音频/灯光）
│   ├── event_schema.py     # 统一事件契约
│   └── statistics.py       # 入侵统计分析
├── configs/                # 配置文件
│   └── default.yaml        # 运行参数与阈值
├── models/                 # ONNX 模型文件
├── standalone/             # 独立运行 Web UI
├── tests/                  # 测试套件
├── scripts/                # 构建/测试脚本
├── plugin.py               # DarkBreaker SDK 适配层
└── run_standalone.py       # 独立启动入口（含 venv guard）
```

## 快速启动

```bash
cd plugins/animal_detection
python3.13 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python run_standalone.py
```

启动后访问 `http://localhost:8082`，健康检查：`GET /api/health`。

## 支持的动物类型

| 类别 | 枚举值 | 风险等级 | 热验证要求 |
|------|--------|----------|-----------|
| 鼠 | mouse | HIGH | 必须 |
| 蛇 | snake | CRITICAL | 必须 |
| 猫 | cat | MEDIUM | 可选 |
| 狗 | dog | MEDIUM | 可选 |
| 鸟 | bird | LOW | 不需要 |
| 家禽 | poultry | LOW | 不需要 |
| 昆虫 | insect | LOW | 不需要 |
| 其他 | other | MEDIUM | 可选 |

## 处理流水线

```
摄像头帧 → detector.detect()
              ↓
         tracker.update()
              ↓
    thermal_validator.validate()  (可选)
              ↓
    event_schema.build_intrusion_event()
              ↓
    deterrent.evaluate_and_trigger()  (可选)
              ↓
         plugin.emit_event()
```

## 配置

所有参数在 `configs/default.yaml` 中定义：

- `model.path` — ONNX 模型路径
- `inference.confidence_threshold` — 检测置信度阈值（默认 0.5）
- `inference.nms_threshold` — NMS IoU 阈值（默认 0.45）
- `thermal.enabled` — 热成像验证开关（默认 false）
- `tracking.enabled` — 目标跟踪开关（默认 true）
- `deterrent.enabled` — 驱离功能开关（默认 false）

## 降级策略

| 故障 | 降级行为 |
|------|---------|
| ONNX 模型缺失 | 返回空检测，记录 ERROR |
| 热成像不可用 | 跳过热验证，仅视觉检测 |
| 跟踪器异常 | 重置跟踪状态，继续检测 |
| 驱离设备不可用 | 仅记录事件，不执行驱离 |

## 测试

```bash
# 模块针对性测试
./scripts/run_targeted_tests.sh detector
./scripts/run_targeted_tests.sh tracker
./scripts/run_targeted_tests.sh event
./scripts/run_targeted_tests.sh plugin

# 回归测试
./scripts/run_regression_tests.sh

# 质量闸门
./scripts/run_quality_gate.sh
```

## 性能目标

- 单帧检测 P95 ≤ 100ms（CPU, 640×480）
- ONNX 模型 ≤ 50MB
- 运行时内存 ≤ 512MB
