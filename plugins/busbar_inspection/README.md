# 母线自主巡视插件（C组）`busbar_inspection`

> 适用平台：输变电“激光星芒破夜绘明”监测平台（插件化架构）  
> 责任范围：绝缘子串 / 金具 / 导线连接点 / 异物悬挂 等  
> 核心目标：在 **4K 大视场**下实现 **远距小目标缺陷**的稳定检测，并支持 **多 ROI 批处理（Batch > 1）**、**误报原因码**输出与 **建议变焦倍率**输出，便于平台执行二次抓拍与取证。  
> （与统筹规划对齐：远距小目标检测、多目标并发、环境干扰过滤原因码、建议变焦倍率等要求。）  

---

## 1. 功能概述

### 1.1 核心功能（必须交付可用）

1) **远距小目标缺陷检测（4K大视场）**  
- 典型缺陷：  
  - `pin_missing`：销钉/开口销缺失（常见小目标）  
  - `crack`：绝缘子/金具裂纹（细长、低对比）  
  - `loose_fitting`：金具松动/错位（结构变化）  
  - `broken_part`：部件破损/缺失（结构性缺陷）  
- 典型目标区域：绝缘子串端部销钉、连接点、耐张线夹、间隔棒、金具连接处。

2) **异物悬挂检测**  
- `foreign_object`：塑料袋、风筝线、鸟巢/树枝、漂浮物等。

3) **多 ROI 并发推理（Batch > 1）**  
- 同一帧内多个部件 ROI 同时推理：  
  - 支持“ROI裁剪后拼 Batch”或“Tile（切片）后拼 Batch”的方式；  
  - 目标：单帧多部件处理效率稳定、吞吐可控。

4) **环境干扰过滤 + 误报原因码（Reason Code）**  
- 对“逆光/遮挡/模糊/雨雾/反光”等造成的疑似误报，必须输出 `reason_code`，便于平台决定：调焦、变焦、换角度、二次抓拍或人工复核。

5) **建议变焦倍率输出（Zoom Suggestion）**  
- 对于小目标检测不稳定/目标过小，需要输出 `suggested_zoom`（倍率或相对倍率）与 `suggested_action`，驱动平台执行“二次变焦/二次抓拍”。

> 注意：本插件只“提出建议与证据”，PTZ/变焦/对焦动作由平台执行（插件不直接控制云台）。

### 1.2 可选增强（建议逐步上线）

- **超分/锐化增强**（SR / Unsharp Mask）用于微小裂纹、销钉纹理提升  
- **跨帧稳定性（Track & Verify）**：以 3~5 帧一致性减少鸟类/抖动误报  
- **自适应曝光/去雾评估**：更稳的雨雾干扰识别与策略建议  
- **场景级母线结构检测**：自动生成 ROI（减少人工 ROI 配置成本）

---

## 2. 输入与ROI（识别区域）类型说明

### 2.1 输入约定

- `frame`：单帧图像  
  - 类型：`np.ndarray`（H×W×3），默认 **BGR**（与 OpenCV 一致）  
  - 分辨率：建议 ≥ 1920×1080；目标场景按 4K（3840×2160）优化  
- `rois`：ROI 列表，每个 ROI 至少包含：
  - `roi_id`：唯一ID  
  - `roi_type`：ROI 类型（见下表）  
  - `bbox`：ROI 框（**建议使用归一化坐标** 0~1；平台也可传像素坐标，但需明确）  
- `context`：上下文（可选但推荐）
  - `task_id/site_id/device_id/component_id/timestamp`
  - 摄像头当前变焦倍率 `zoom_current`（若可取到）
  - 历史帧信息（若平台提供）

### 2.2 ROI 类型表（建议平台统一字典）

| ROI类型 | 说明 | 主要检测目标（label） | 备注 |
|---|---|---|---|
| `insulator_string` | 绝缘子串整体 | `crack`, `broken_part`, `foreign_object` | 可选：细分到端部区域 |
| `hardware_fitting` | 金具/线夹/连接金具 | `pin_missing`, `loose_fitting`, `broken_part` | **销钉缺失**重点ROI |
| `conductor_joint` | 导线连接点/接头 | `loose_fitting`, `foreign_object` | 温升由温度插件负责 |
| `spacer_damper` | 间隔棒/防振锤 | `loose_fitting`, `broken_part` | 小目标，需变焦建议 |
| `foreign_object_zone` | 异物高发区域（走线附近） | `foreign_object` | 允许更大ROI |
| `busbar_span` | 母线长跨（全局） | `foreign_object`（粗检） | 常用于先粗检再二次变焦 |

> ROI 设计建议：对“小目标（销钉/裂纹）”尽量给 **更精确的局部 ROI**，否则必须启用切片 + 变焦建议机制。

---

## 3. 输出格式（统一JSON字段 + 扩展字段）

平台对输出做强校验：字段缺失/类型错误/`bbox`非法/`confidence`不在[0,1] 一律视为失败，并写入 evidence/errors.log。

### 3.1 RecognitionResult（单ROI输出）

> **统一字段（平台强制）**：  
`task_id, site_id, device_id, component_id, timestamp, roi_id, bbox, label, value(可空), confidence, evidence, model_version, code_version, reason_code(可选), extra(可选)`

**建议扩展字段（强烈建议保持一致）**：

- `reason_code`：误报/失败/策略原因（整数，详见 7.3）  
- `extra.suggested_zoom`：建议变焦倍率（float）  
- `extra.suggested_action`：建议动作（字符串枚举）  
- `extra.quality`：质量评估（如 `clarity_score/overexposed/backlight`）  
- `extra.debug`：调试信息（tile数量、耗时等，生产可关闭）

示例：

```json
{
  "task_id": "task_20250410_001",
  "site_id": "site_110kV_01",
  "device_id": "cam_busbar_03",
  "component_id": "busbar",
  "timestamp": "2025-04-10T10:01:02Z",

  "roi_id": "roi_12",
  "bbox": {"x": 0.41, "y": 0.18, "w": 0.08, "h": 0.12},
  "label": "pin_missing",
  "value": null,
  "confidence": 0.86,

  "evidence": {
    "frame_raw": "evidence/.../frame_raw.jpg",
    "frame_annotated": "evidence/.../frame_annotated.jpg",
    "result_json": "evidence/.../result.json"
  },

  "model_version": "busbar-det@1.2.0",
  "code_version": "sha256:0c2b...d91a",
  "reason_code": null,
  "extra": {
    "suggested_zoom": 1.0,
    "suggested_action": "NONE",
    "quality": {
      "clarity_score": 0.72,
      "backlight": false,
      "occlusion": 0.08
    },
    "debug": {
      "tiles": 12,
      "batch_size": 8,
      "latency_ms": 243
    }
  }
}
```

### 3.2 Alarm（告警输出）

平台规则引擎（或本插件 postprocess）统一输出：

```json
{
  "alarm_id": "alarm_20250410_0001",
  "severity": "HIGH",
  "message": "母线金具连接处疑似销钉缺失",
  "related_roi_id": "roi_12",
  "timestamp": "2025-04-10T10:01:02Z",
  "evidence": {
    "frame_annotated": "evidence/.../frame_annotated.jpg",
    "result_json": "evidence/.../result.json"
  }
}
```

---

## 4. 算法检测说明（可直接落地实现）

> 母线场景的难点是：**目标小、背景复杂、光照强变化、拍摄距离远**。工程上要稳定，需要把算法拆成：  
**质量评估 → 远距小目标检测（切片/多尺度） → 误报过滤（原因码） → 变焦建议 → 规则告警**。

### 4.1 远距小目标检测：切片 + 多尺度（核心）

#### 4.1.1 切片（Tiling）策略

对每个 ROI（或整帧）按固定 `tile_size` 切片，重叠 `overlap`，避免小目标被切断：

- ROI 像素框：`(x0,y0,x1,y1)`
- 切片步长：`stride = tile_size - overlap`
- 生成 tile 列表：`[(tx0,ty0,tx1,ty1), ...]`

**合并策略**：tile 内推理得到的 bbox 转回 ROI 坐标系，再做全局 NMS（非极大值抑制）。

NMS 的 IoU 定义：

\[
IoU(B_1,B_2)=\frac{|B_1\cap B_2|}{|B_1\cup B_2|}
\]

当 `IoU > iou_thr` 时保留置信度更高者。

#### 4.1.2 多尺度推理（Multi-scale / TTA）

对远距小目标常用两种工程可落地做法：

- **尺度金字塔**：对 tile resize 到 `input_size` 的同时，额外对 tile 做一次 `scale_up`（如 1.2~1.5）再推理；  
- **小目标专用模型**：训练时提升小目标采样权重（mosaic/Copy-Paste），推理端只做切片即可。

> 推荐：**切片 + 单尺度**先跑稳；再上多尺度作为可选增强开关。

#### 4.1.3 模型建议（不限定框架）

- 检测模型：YOLOv8 / RT-DETR / Faster R-CNN 均可  
- 导出：ONNX（平台更易部署）  
- 推理：ONNXRuntime（CPU/GPU）或 TensorRT（GPU，建议）

### 4.2 多目标并发处理（Batch Size > 1）

核心思想：把多个 ROI 的 tile 拼成 batch，一次 forward 处理多个 tile。

**拼 Batch 的粒度**（两选一）：

1) **ROI Crop Batch**：每个 ROI 裁剪后 resize 到输入尺寸，batch 推理  
- 优点：实现简单  
- 缺点：远距小目标容易被缩小，需要更精细 ROI 或额外 scale-up

2) **Tile Batch（推荐）**：ROI 内先切片，然后把 tile 拼 batch  
- 优点：对小目标友好、精度更稳  
- 缺点：tile 数多，需要合理的 tile_size/stride 控制速度

---

### 4.3 环境干扰过滤：质量评估 + 原因码（强制）

母线误报大头来自：逆光、遮挡、模糊、雨雾、反光、运动干扰（鸟类/树叶）。

建议在 infer 前做 **质量评估（Quality Gate）**，infer 后做 **误报解释（Reason Code）**：

#### 4.3.1 模糊检测（Blur）

用 Laplacian 方差（工程稳、成本低）：

\[
B = Var(\nabla^2 I)
\]

若 `B < blur_thr` 则判为模糊：

- `reason_code = 103`（模糊）
- `suggested_action = "REFOCUS_OR_RECAPTURE"`

#### 4.3.2 逆光/过曝检测（Backlight / Overexposure）

- 计算亮度通道 `Y` 的直方图：  
  - 高亮像素比例 `p_high = mean(Y > y_high)`  
  - 动态范围 `dr = P95(Y) - P5(Y)`  
- 若 `p_high > overexp_ratio` 或 `dr < dr_min`：  
  - `reason_code = 101`（逆光/过曝）  
  - `suggested_action = "ADJUST_EXPOSURE_OR_CHANGE_VIEW"`

#### 4.3.3 遮挡估计（Occlusion）

简化可落地指标：边缘能量或纹理能量偏低：

- `edge_energy = mean(|Sobel(I)|)`  
- `edge_energy < edge_thr` 视为遮挡/不可见：
  - `reason_code = 102`（遮挡）  
  - `suggested_action = "CHANGE_VIEW_OR_RECAPTURE"`

#### 4.3.4 目标过小（需要变焦）

当检测框的像素尺寸小于阈值（例如 `< 18px`）：

- `reason_code = 201`（目标过小）  
- 输出 `suggested_zoom`（见 4.4）

---

### 4.4 建议变焦倍率（Zoom Suggestion）

目标：把“销钉/裂纹”在最终取证图中达到可读像素尺寸 `target_px`（经验值：`60~120px`）。

设检测框最大边像素为 `s_px`，建议倍率：

\[
z = clamp\left(\frac{target\_px}{\max(s\_{px}, \epsilon)}, z_{min}, z_{max}\right)
\]

- `z_min`：最小变焦倍率（通常 1.0）  
- `z_max`：最大变焦倍率（按相机能力）  
- 若平台可提供 `zoom_current`，可输出 **相对倍率**：`z_rel = z / zoom_current`

建议动作枚举（统一口径）：

- `NONE`：无需调整  
- `ZOOM_IN`：需要拉近  
- `REFOCUS`：需要调焦  
- `RECAPTURE`：需要二次抓拍  
- `CHANGE_VIEW`：建议换角度/换预置位（由任务引擎决定）

---

### 4.5 结果识别与告警策略（工程可用）

- 当 `label in {pin_missing, crack, broken_part}` 且 `confidence >= alarm_thr_high`：`HIGH`  
- 当 `foreign_object` 且 `confidence >= alarm_thr_mid`：`MEDIUM`  
- 当 `confidence < alarm_thr_low` 或触发质量门禁：输出 `reason_code`，进入“待复核/二次抓拍”链路

---

## 5. 工程实现（插件代码骨架，可直接复制到 `plugin.py`）

> 说明：以下为“可落地”的最小实现范式：  
- ONNX 模型推理（onnxruntime）  
- 切片 + Batch 推理  
- NMS 合并  
- 质量评估 + 原因码 + 变焦建议  
你只需要把 `ModelRunner` 中的 `postprocess()` 对齐你们的 ONNX 输出即可。

### 5.1 `manifest.json`（完整示例）

```json
{
  "name": "busbar_inspection",
  "version": "0.1.0",
  "entrypoint": "busbar_inspection.plugin:BusbarInspectionPlugin",
  "supported_tasks": ["busbar_inspection"],
  "model_version": "busbar-det@1.2.0",
  "author": "C组",
  "description": "母线自主巡视：远距小目标缺陷检测 + 多ROI批处理 + 原因码 + 变焦建议",
  "dependencies": [
    "numpy>=1.26",
    "opencv-python>=4.9",
    "onnxruntime-gpu>=1.18; platform_system!='Darwin'",
    "onnxruntime>=1.18; platform_system=='Darwin'"
  ]
}
```

### 5.2 `plugin.py`（核心骨架）

```python
# 说明：此代码块用于 README 交付口径，复制到 plugins/busbar_inspection/plugin.py 后即可作为实现底座。
# 你们需要补齐：模型输出映射、ROI聚合、NMS、label映射等。

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import hashlib

import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception:
    ort = None


@dataclass
class BBox:
    x: float  # normalized 0..1
    y: float
    w: float
    h: float


@dataclass
class ROI:
    roi_id: str
    roi_type: str
    bbox: BBox


def _norm_to_px(b: BBox, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = int(max(0, b.x * W))
    y0 = int(max(0, b.y * H))
    x1 = int(min(W, (b.x + b.w) * W))
    y1 = int(min(H, (b.y + b.h) * H))
    return x0, y0, x1, y1


def tile_boxes(x0: int, y0: int, x1: int, y1: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    stride = max(1, tile - overlap)
    boxes = []
    for ty in range(y0, y1, stride):
        for tx in range(x0, x1, stride):
            bx0, by0 = tx, ty
            bx1, by1 = min(tx + tile, x1), min(ty + tile, y1)
            boxes.append((bx0, by0, bx1, by1))
    return boxes


def clarity_score_laplacian(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    v = lap.var()
    return float(np.clip(v / 1500.0, 0.0, 1.0))


def is_overexposed(gray: np.ndarray, y_high: int = 245, ratio_thr: float = 0.25) -> bool:
    p_high = float((gray > y_high).mean())
    return p_high >= ratio_thr


def is_low_contrast(gray: np.ndarray, dr_min: int = 35) -> bool:
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    return (p95 - p5) < dr_min


class ModelRunner:
    def __init__(self, model_path: str, input_size: int, providers: Optional[List[str]] = None):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed.")
        self.model_path = model_path
        self.input_size = input_size
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def forward(self, batch_chw: np.ndarray) -> Any:
        return self.sess.run(None, {self.input_name: batch_chw})

    def postprocess(self, outputs: Any, conf_thr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TODO：根据模型输出格式改写，返回：
        #   boxes_xyxy: (M,4) in input_size coordinates
        #   scores: (M,)
        #   class_ids: (M,)
        det = outputs[0]
        det = det.reshape(-1, det.shape[-1])
        det = det[det[:, 4] >= conf_thr]
        boxes = det[:, 0:4].astype(np.float32)
        scores = det[:, 4].astype(np.float32)
        cls = det[:, 5].astype(np.int32)
        return boxes, scores, cls


class BusbarInspectionPlugin:
    def __init__(self):
        self.cfg = None
        self.model: Optional[ModelRunner] = None
        self.model_version = "busbar-det@unknown"
        self.code_version = self._calc_code_hash()

    def _calc_code_hash(self) -> str:
        h = hashlib.sha256()
        h.update(b"busbar_inspection_plugin_v0")
        return "sha256:" + h.hexdigest()[:12]

    def init(self, config: Dict[str, Any]) -> "BusbarInspectionPlugin":
        self.cfg = config
        self.model_version = config["model"]["model_version"]
        self.model = ModelRunner(
            model_path=config["model"]["model_path"],
            input_size=int(config["model"]["input_size"]),
            providers=config["runtime"].get("providers"),
        )
        return self

    def healthcheck(self) -> Dict[str, Any]:
        return {"ok": self.model is not None, "model_version": self.model_version, "code_version": self.code_version}

    def infer(self, frame: np.ndarray, rois: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        assert self.model is not None, "Call init(config) first."
        t0 = time.time()
        H, W = frame.shape[:2]
        cfg = self.cfg

        tile_size = int(cfg["tiling"]["tile_size"])
        overlap = int(cfg["tiling"]["overlap"])
        batch_size = int(cfg["runtime"]["batch_size"])
        conf_thr = float(cfg["thresholds"]["conf_thr"])

        # quality gate
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clarity = clarity_score_laplacian(gray)

        overexp = is_overexposed(gray, y_high=int(cfg["quality"]["y_high"]), ratio_thr=float(cfg["quality"]["overexp_ratio"]))
        low_contrast = is_low_contrast(gray, dr_min=int(cfg["quality"]["dr_min"]))

        # build ROI list
        roi_objs: List[ROI] = []
        for r in rois:
            bb = r["bbox"]
            roi_objs.append(ROI(
                roi_id=r["roi_id"],
                roi_type=r.get("roi_type", "unknown"),
                bbox=BBox(x=float(bb["x"]), y=float(bb["y"]), w=float(bb["w"]), h=float(bb["h"]))
            ))

        # collect tiles
        meta = []
        batch = []
        for roi in roi_objs:
            rx0, ry0, rx1, ry1 = _norm_to_px(roi.bbox, W, H)
            for (tx0, ty0, tx1, ty1) in tile_boxes(rx0, ry0, rx1, ry1, tile=tile_size, overlap=overlap):
                crop = frame[ty0:ty1, tx0:tx1]
                if crop.size == 0:
                    continue
                batch.append(self.model.preprocess(crop))
                meta.append((roi, (tx0, ty0, tx1, ty1)))

        # run inference in batches (TODO: map outputs back)
        for i in range(0, len(batch), batch_size):
            outs = self.model.forward(np.stack(batch[i:i+batch_size], axis=0))
            _boxes, _scores, _cls = self.model.postprocess(outs, conf_thr=conf_thr)
            # TODO: 将 _boxes 映射到 frame 坐标 + ROI聚合 + NMS + label映射

        # demo-safe return (replace with real detections)
        results = []
        for roi in roi_objs:
            reason = None
            action = "NONE"
            zoom = 1.0

            if clarity < float(cfg["quality"]["blur_thr"]):
                reason = 103
                action = "REFOCUS_OR_RECAPTURE"
            elif overexp or low_contrast:
                reason = 101
                action = "ADJUST_EXPOSURE_OR_CHANGE_VIEW"

            results.append({
                "task_id": context.get("task_id", ""),
                "site_id": context.get("site_id", ""),
                "device_id": context.get("device_id", ""),
                "component_id": context.get("component_id", "busbar"),
                "timestamp": context.get("timestamp", ""),
                "roi_id": roi.roi_id,
                "bbox": {"x": roi.bbox.x, "y": roi.bbox.y, "w": roi.bbox.w, "h": roi.bbox.h},
                "label": "ok",
                "value": None,
                "confidence": 1.0 if reason is None else 0.2,
                "evidence": {},
                "model_version": self.model_version,
                "code_version": self.code_version,
                "reason_code": reason,
                "extra": {
                    "suggested_zoom": zoom,
                    "suggested_action": action,
                    "quality": {"clarity_score": float(clarity)},
                    "debug": {"latency_ms": int((time.time() - t0) * 1000)}
                }
            })

        return results

    def postprocess(self, results: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        alarms = []
        thr_hi = float(rules.get("alarm_thr_high", 0.75))
        thr_mid = float(rules.get("alarm_thr_mid", 0.60))
        for r in results:
            label = r.get("label", "ok")
            conf = float(r.get("confidence", 0.0))
            if label in ("pin_missing", "crack", "broken_part") and conf >= thr_hi:
                alarms.append({"alarm_id": f"alarm_{int(time.time())}", "severity": "HIGH",
                               "message": f"检测到{label}", "related_roi_id": r.get("roi_id"),
                               "timestamp": r.get("timestamp"), "evidence": r.get("evidence", {})})
            elif label == "foreign_object" and conf >= thr_mid:
                alarms.append({"alarm_id": f"alarm_{int(time.time())}", "severity": "MEDIUM",
                               "message": "检测到异物悬挂", "related_roi_id": r.get("roi_id"),
                               "timestamp": r.get("timestamp"), "evidence": r.get("evidence", {})})
        return alarms
```

---

## 6. 配置文件（`configs/default.yaml` 完整示例）

> 该配置建议由平台统一下发；插件仅消费配置，不写死阈值。

```yaml
model:
  model_path: "models/busbar_det.onnx"
  model_version: "busbar-det@1.2.0"
  input_size: 640

runtime:
  providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
  batch_size: 8

tiling:
  tile_size: 1280
  overlap: 320
  roi_margin_px: 10

thresholds:
  conf_thr: 0.25
  nms_iou: 0.50

quality:
  blur_thr: 0.35         # clarity_score < 0.35 => 模糊
  y_high: 245            # 过曝阈值
  overexp_ratio: 0.25    # 高亮像素比例阈值
  dr_min: 35             # 动态范围过低视为低对比/逆光

zoom:
  target_px: 90          # 希望小目标最终达到的像素尺寸
  min_obj_px: 18         # <18px 直接建议变焦
  zmin: 1.0
  zmax: 12.0

labels:
  - ok
  - pin_missing
  - crack
  - loose_fitting
  - broken_part
  - foreign_object

rules:
  alarm_thr_high: 0.75
  alarm_thr_mid: 0.60
  low_conf_thr: 0.40
```

---

## 7. 性能指标与验收（工程可用口径）

### 7.1 性能指标（建议最低线）

1) **正确性（以验收数据集统计）**
- `pin_missing`：Recall ≥ 0.85，Precision ≥ 0.85（远距小目标核心KPI）
- `crack`：Recall ≥ 0.70，Precision ≥ 0.80（裂纹更难，先保证低误报）
- `foreign_object`：Recall ≥ 0.85，Precision ≥ 0.85

2) **速度（单帧4K，包含切片 + NMS）**
- GPU（如 3060/3080）：P95 ≤ 800ms（建议目标 300~500ms）  
- CPU（可用但不推荐）：P95 ≤ 5s（用于回放或兜底）

3) **稳定性**
- 连续运行 24h 不崩溃、不内存泄漏  
- 质量门禁触发时必须输出原因码与建议动作（不允许“沉默失败”）

### 7.2 交付数据与回放要求

- `examples/`：≥ 200 张 4K 图（覆盖：晴天/阴天/逆光/雨雾/遮挡/不同距离）  
- `expected/expected.json`：对每张样例给出期望输出（至少给出 `roi_id/label/confidence/reason_code`）  
- `tests/`：至少 1 个单测：跑一次 infer 并通过 schema 校验（平台黑盒测试需要）

### 7.3 失败原因码（建议统一）

| reason_code | 含义 | suggested_action（建议） |
|---:|---|---|
| 101 | 逆光/过曝/低对比 | `ADJUST_EXPOSURE_OR_CHANGE_VIEW` |
| 102 | 遮挡/不可见 | `CHANGE_VIEW_OR_RECAPTURE` |
| 103 | 模糊/失焦 | `REFOCUS_OR_RECAPTURE` |
| 104 | 雨雾/霾导致低能见度 | `RECAPTURE` |
| 105 | 运动干扰（鸟/树叶/抖动） | `RECAPTURE` |
| 201 | 目标过小，需要变焦 | `ZOOM_IN` |
| 202 | 检测不稳定（需二次抓拍确认） | `RECAPTURE` |
| 301 | 结果不可信（规则冲突/多解释） | `MANUAL_REVIEW` |

---

## 8. 局限性与风险控制（必须明确）

- **极端逆光/强反光**：即使模型识别到目标，也可能纹理不可见，必须通过 reason_code 引导二次抓拍  
- **裂纹类缺陷**：低对比、细长目标，强依赖清晰度与分辨率；建议优先做“变焦建议 + 复核链路”  
- **无遮挡但目标尺寸极小**：切片仍可能不足，必须输出 `suggested_zoom`  
- **鸟类/树叶误报**：建议在平台层做跨帧一致性复核（或插件内部做轻量缓存）

---

## 9. 未来改进路线（不影响当前交付）

- [ ] 引入轻量超分模型（提升裂纹/销钉纹理）
- [ ] 端到端“粗检→二次变焦→精检”闭环（插件给 zoom 建议，平台执行）
- [ ] 场景结构先验：母线几何拓扑约束降低误报
- [ ] 半监督/主动学习：把人工复核数据自动回流训练

---

## 10. 技术支持与版本策略

- `model_version`：模型训练产物版本（必须可追溯）  
- `code_version`：代码版本hash（建议 git commit；至少文件hash）  
- 出现线上问题时：必须能用同一 `examples/` 回放复现；否则视为不可维护交付。

如需平台侧对接：请按平台统一的 Plugin ABC 接口实现 `init/infer/postprocess/healthcheck`，并确保输出严格符合平台 JSON Schema。
