# Real-DL 升级决策记录

> 最后更新：2026-04-16

## 1. 背景

`bird_monitoring` 当前 `runtime_mode=simulation`，没有真实 ONNX 模型。
本决策记录捕获从 simulation → real_dl 的升级边界。

## 2. 决策

### 2.1 升级先决条件（必须同时满足）

1. 可用的 YOLOv8 风格 ONNX 文件（路径见 `configs/default.yaml::model.path`）。
2. `onnxruntime` 可安装（CPU 或 CUDA provider 任一）。
3. 模型输入 shape 与 `config.model.input_size` 对齐（默认 `[640, 640]`）。
4. 模型输出 channel 数与 `BirdDetector.CLASSES`（默认 10）对齐：YOLOv8 语义为 `4 + num_classes`。
5. 类别名称映射覆盖 `SPECIES_LABEL_MAP` 的所有键，未命中项必须归入 `unknown_bird`。

### 2.2 preflight 合同

`BirdDetector._preflight_onnx()` 负责校验 §2.1 的 3~5 项。失败时：
- **不抛异常**。
- `self.session` 置空、`real_model_loaded=False`。
- `model_load_error` 记录 `preflight_failed:<issues>`。
- `runtime_mode` 自动回落 `simulation`。

### 2.3 健康检查可见性

`plugin.healthcheck().details` 必须包含：
- `runtime_mode`
- `model_path_configured / model_path_resolved`
- `model_file_exists / real_model_loaded / onnx_session_ready`
- `model_load_error`
- `preflight`（含 `performed / passed / checks / issues`）

## 3. 已拒绝的方案

- ❌ 直接启用 `experimental/enhanced_detector.py` 作为生产检测器。原因：含随机
  行为、硬件驱鸟语义；且未经过 real_dl preflight。
- ❌ 在 simulation 模式合成鸟类 bbox 以"演示"升级效果。原因：破坏 runtime truth。
- ❌ 默认启用 CUDA provider。原因：部署环境不确定，statically 固定会在 CPU-only
  机器上抛异常。

## 4. 下一步待办（blocked）

- 采集最小真实 fixture 集（见 `docs/fixture_collection_plan.md` 计划槽位）。
- 建立 precision/recall regression baseline（需真实标注）。
- 相机标定 → 真实距离估计（当前距离为启发式）。
