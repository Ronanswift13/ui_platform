# 开关间隔自主巡视插件（B组）`switch_inspection`

> **用途**：面向变电站“开关间隔（Bay）”的自主巡视算法插件，覆盖断路器、隔离开关、接地开关、机构箱、分合指示牌、避雷器，以及（可选）SF6 压强/密度表读数。  
> **交付目标**：工程可用、可回放复现、输出字段统一、可被平台任务引擎调度与UI呈现；支持“动态定位 + AI 调焦”的对接参数（目标框稳定阈值、清晰度评价函数）。  

---

## 1. 功能概述

开关间隔的核心是“状态对不对、逻辑对不对、图像清不清”。本插件按 **三条主线**落地：

### 1.1 核心功能（必须交付可用）
1) **分合位状态识别（State Recognition）**  
- 断路器分/合/中间态识别  
- 隔离开关分/合识别  
- 接地开关分/合识别  
- 证据融合：连杆角度 + 指示牌文字(OCR) + 指示牌颜色(红/绿) +（可选）触头位置  

2) **互锁/逻辑校验（Interlock / Five-Prevention）**  
- 结合隔离开关与接地开关互锁逻辑，输出“状态异常告警”  
- 规则可配置（平台侧可复用统一规则引擎；插件侧提供最小可用实现）

3) **清晰度评价（Clarity Score 0–1）**  
- 输出 `clarity_score ∈ [0,1]`，用于主控层触发自动调焦/二次抓拍  
- 配置参数包括：模糊阈值、稳定帧数、稳定 IoU 阈值

4) **低置信度回退（Fallback）**  
- 当 `confidence < 阈值` 或 `clarity_score < 阈值`：输出 `reason_code` 与 `suggested_action`  
- 由平台任务层执行：二次变焦/二次抓拍/重新对焦（本插件只提出建议与证据）

### 1.2 升级功能（逐步上线）
- **SF6 压强/密度表读数（指针式/数字式）**：输出 `value + unit + confidence`  
- **外观缺陷（机构箱破损/锈蚀/异物）**：可复用 A组缺陷检测逻辑或引入深度模型

---

## 2. 输入与ROI（识别区域）类型说明

### 2.1 输入约定
- `frame`：BGR 图像（OpenCV）或 RGB（需在配置里声明）；推荐统一为 **BGR**。  
- `rois`：平台 UI 编辑得到的 ROI 列表（每个ROI含 `id/type/bbox` 等）。  
- `context`：平台提供任务上下文（task_id/site_id/device_id/point_id/timestamp 等）。  

### 2.2 ROI 类型表（建议平台统一字典）

| roi.type | 含义 | 典型目标 | 输出 label/value |
|---|---|---|---|
| `breaker_indicator` | 断路器分合指示窗/牌 | “分/合/OPEN/CLOSE”、红/绿 | `breaker_state`=`open/closed/intermediate/unknown` |
| `isolator_indicator` | 隔离开关指示窗/牌 | 分/合 | `isolator_state` |
| `grounding_indicator` | 接地开关指示窗/牌 | 分/合 | `grounding_state` |
| `breaker_linkage` | 断路器连杆/机构 | 连杆角度/位置 | `breaker_angle_deg` + state |
| `isolator_linkage` | 隔离刀闸连杆/刀闸 | 刀闸角度 | `isolator_angle_deg` + state |
| `grounding_handle` | 接地柄/连杆 | 位置/角度 | `grounding_angle_deg` + state |
| `gauge_pressure` | SF6 压强表 | 指针/数字 | `sf6_pressure`（float） |
| `gauge_density` | 密度表/继电器 | 指针/数字 | `sf6_density`（float） |
| `mechanism_box` | 机构箱外观 | 门、面板、锈蚀/破损（可选） | `defect` 或 `door_state` |
| `arrester` | 避雷器外观 | 裂纹/污损（可选） | `defect` |
| `clarity_anchor` | 清晰度评估锚点 | 设备关键纹理区 | `clarity_score` |

> **工程建议**：每个点位至少配置 1 个 `clarity_anchor`，并为关键状态对象配置对应 `*_indicator` 与 `*_linkage`（多证据更稳）。  

---

## 3. 输出格式（统一JSON字段 + 扩展字段）

### 3.1 RecognitionResult（单ROI输出）
平台会对该结构做强校验（建议使用 Pydantic）。最小字段如下：

```json
{
  "task_id": "task_xxx",
  "site_id": "site_001",
  "device_id": "cam_01",
  "component_id": "bay_110kV_141",
  "timestamp": "2025-04-10T18:11:03Z",

  "roi_id": "roi_breaker_indicator_01",
  "bbox": {"x": 0.12, "y": 0.18, "width": 0.22, "height": 0.16},
  "label": "breaker_state",
  "value": "closed",
  "confidence": 0.92,

  "evidence": "evidence/site_001/point_01/run_xxx/frame_annotated.png",
  "model_version": "1.0.0",
  "code_version": "sha256:....",

  "reason_code": null,
  "extra": {
    "ocr_text": "合",
    "color_hint": "green",
    "angle_deg": 12.4,
    "clarity_score": 0.86,
    "suggested_action": null
  }
}
```

### 3.2 Alarm（告警输出）
```json
{
  "alarm_id": "alarm_xxx",
  "severity": "error",
  "message": "互锁逻辑异常：检测到接地开关合闸，但隔离开关未断开",
  "related_roi_id": "roi_grounding_indicator_01",
  "evidence": "evidence/.../result.json",
  "timestamp": "2025-04-10T18:11:03Z"
}
```

---

## 4. 算法检测说明（可直接落地实现）

本插件的“工程可用”基线实现依赖 OpenCV +（可选）OCR。深度模型（YOLO/Seg）可作为替换组件接入，但不得改变输出 Schema。

### 4.1 清晰度评价（Clarity Score）

推荐采用 **Laplacian 方差**或 **Tenengrad**。这里给出可落地的 Laplacian 方差实现：

- 定义模糊度：  
\[
B = Var(\nabla^2 I)
\]
- 将其映射到 \([0,1]\)：
\[
score = \sigma\Big(\frac{B - \mu}{\tau}\Big) = \frac{1}{1+e^{-(B-\mu)/\tau}}
\]

其中 \(\mu\) 是“可接受清晰度阈值”，\(\tau\) 控制过渡陡峭度（配置化）。

**参考实现：**
```python
import cv2, numpy as np

def clarity_score_laplacian(bgr: np.ndarray, mu: float = 120.0, tau: float = 30.0) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    v = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = 1.0 / (1.0 + np.exp(-(v - mu) / max(tau, 1e-6)))
    return float(np.clip(score, 0.0, 1.0))
```

> 平台对接参数：`min_clarity_score`、`stable_iou_threshold`、`stable_frames`（用于“动态定位+AI调焦”闭环）。  

---

### 4.2 指示牌：OCR + 颜色提示（最稳的“软证据”）

#### 4.2.1 OCR（可选但推荐）
- 词表：`["合","分","OPEN","CLOSE","ON","OFF"]`
- 输出：`ocr_text, ocr_confidence`

```python
def try_ocr_text(bgr: np.ndarray, engine: str = "easyocr"):
    try:
        import easyocr
        reader = easyocr.Reader(['ch_sim','en'], gpu=False)
        results = reader.readtext(bgr)
        if not results:
            return None, 0.0
        # 取置信度最高的一条
        (bbox, text, conf) = max(results, key=lambda x: x[2])
        return text.strip(), float(conf)
    except Exception:
        return None, 0.0
```

#### 4.2.2 颜色提示（红/绿占比）
- 在 HSV 空间统计颜色占比（对指示窗非常有效）

```python
def hsv_ratio(bgr: np.ndarray, lower, upper) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    return float(mask.mean() / 255.0)

def color_hint_red_green(bgr: np.ndarray):
    green = hsv_ratio(bgr, [35,40,40], [85,255,255])
    red1 = hsv_ratio(bgr, [0,60,60], [10,255,255])
    red2 = hsv_ratio(bgr, [170,60,60], [180,255,255])
    red = max(red1, red2)
    return {"red_ratio": red, "green_ratio": green}
```

---

### 4.3 连杆角度：几何证据（硬证据）

对连杆ROI：Canny → HoughLinesP → 取主方向线段角度：
\[
\theta = atan2(dy, dx) \cdot 180/\pi
\]

```python
def dominant_line_angle_deg(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=0.25*min(bgr.shape[:2]),
                            maxLineGap=10)
    if lines is None:
        return None
    # 选最长线段
    x1,y1,x2,y2 = max(lines[:,0,:], key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2)
    ang = np.degrees(np.arctan2((y2-y1), (x2-x1)))
    return float(ang)
```

---

### 4.4 状态融合判决（工程可用的“多证据融合”）

为每类开关定义 `open/closed` 的特征得分，线性融合后取最大类：

\[
S_{closed}=w_t s_{text}^{(closed)} + w_c s_{color}^{(closed)} + w_a s_{angle}^{(closed)}
\]
\[
S_{open}=w_t s_{text}^{(open)} + w_c s_{color}^{(open)} + w_a s_{angle}^{(open)}
\]

其中：
- `s_text`：OCR命中词表与置信度映射  
- `s_color`：红/绿占比映射  
- `s_angle`：与标定角 `θ_open/θ_closed` 的距离映射（需要点位标定或经验阈值）

建议将融合权重、阈值写入 `configs/default.yaml`。

---

### 4.5 互锁逻辑校验（最小可用实现）

最小规则（可配置）：
- R1：`isolator != open` 且 `grounding == closed` ⇒ **异常**（带电合接地刀风险）
- R2：`breaker == closed` 且 `isolator == open` ⇒ **异常**（可能的异常工况/识别错误需复核）

> 工程建议：互锁校验更稳健需要“时序状态”，可在插件内部缓存最近 N 帧结果，或由平台任务层做状态机。

---

### 4.6 SF6 压强/密度表读数（可选，建议B组最终交付）

#### 指针式（基线）
- 圆检测（HoughCircles）获取中心与半径  
- 指针线检测（HoughLinesP）获取方向角  
- 角度映射量程：
\[
value = v_{min} + \frac{\theta-\theta_{min}}{\theta_{max}-\theta_{min}} (v_{max}-v_{min})
\]

提供每个点位的 `theta_min/theta_max/value_min/value_max/unit` 标定参数。

#### 数字式（OCR）
- ROI 内 OCR 提取数字与单位，输出 `value/unit`，并给出 `reason_code`（如OCR失败）

---

## 5. 工程实现（插件代码骨架，可直接复制到 `plugin.py`）

> 以下代码以“规则/传统视觉”作为可用基线；如果后续替换为深度模型，只替换 `detect_*` 内部实现，不得改变输出字段。

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
import time, hashlib, json
import numpy as np
import cv2

# ---- 说明：以下类型在平台 platform_core 中应统一定义；此处仅示例 ----
@dataclass
class BBox:
    x: float; y: float; width: float; height: float
@dataclass
class ROI:
    id: str
    type: str
    bbox: BBox
    meta: dict[str, Any] | None = None
@dataclass
class PluginContext:
    task_id: str
    site_id: str
    device_id: str
    point_id: str
    timestamp: str
@dataclass
class RecognitionResult:
    task_id: str; site_id: str; device_id: str; component_id: str; timestamp: str
    roi_id: str; bbox: dict; label: str; value: Any | None; confidence: float
    evidence: str; model_version: str; code_version: str
    reason_code: Optional[int] = None
    extra: dict[str, Any] | None = None

def crop_roi(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = int(max(0, min(w-1, bbox.x * w)))
    y1 = int(max(0, min(h-1, bbox.y * h)))
    x2 = int(max(0, min(w, (bbox.x + bbox.width) * w)))
    y2 = int(max(0, min(h, (bbox.y + bbox.height) * h)))
    if x2 <= x1 or y2 <= y1:
        return frame[0:1,0:1].copy()
    return frame[y1:y2, x1:x2].copy()

def clarity_score_laplacian(bgr: np.ndarray, mu: float, tau: float) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    v = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = 1.0 / (1.0 + np.exp(-(v - mu) / max(tau, 1e-6)))
    return float(np.clip(score, 0.0, 1.0))

def hsv_ratio(bgr: np.ndarray, lower, upper) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    return float(mask.mean() / 255.0)

def dominant_line_angle_deg(bgr: np.ndarray) -> Optional[float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=int(0.25*min(bgr.shape[:2])),
                            maxLineGap=10)
    if lines is None:
        return None
    x1,y1,x2,y2 = max(lines[:,0,:], key=lambda l: (l[2]-l[0])**2 + (l[3]-l[1])**2)
    return float(np.degrees(np.arctan2((y2-y1), (x2-x1))))

class SwitchInspectionRuntime:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def recognize_indicator(self, roi_img: np.ndarray) -> dict[str, Any]:
        # 颜色提示
        green = hsv_ratio(roi_img, self.cfg["state_recognition"]["color_hint"]["hsv_green"][0],
                          self.cfg["state_recognition"]["color_hint"]["hsv_green"][1])
        red1 = hsv_ratio(roi_img, self.cfg["state_recognition"]["color_hint"]["hsv_red1"][0],
                         self.cfg["state_recognition"]["color_hint"]["hsv_red1"][1])
        red2 = hsv_ratio(roi_img, self.cfg["state_recognition"]["color_hint"]["hsv_red2"][0],
                         self.cfg["state_recognition"]["color_hint"]["hsv_red2"][1])
        red = max(red1, red2)

        # OCR（可选）
        ocr_text, ocr_conf = None, 0.0
        if self.cfg["state_recognition"]["ocr"]["enabled"]:
            try:
                import easyocr
                reader = easyocr.Reader(['ch_sim','en'], gpu=False)
                res = reader.readtext(roi_img)
                if res:
                    (_, text, conf) = max(res, key=lambda x: x[2])
                    ocr_text, ocr_conf = text.strip(), float(conf)
            except Exception:
                pass

        return {"red_ratio": red, "green_ratio": green, "ocr_text": ocr_text, "ocr_conf": ocr_conf}

    def decide_state(self, evidence: dict[str, Any], angle: Optional[float], kind: str) -> tuple[str, float, dict[str, Any], Optional[int]]:
        # kind: breaker/isolator/grounding
        w = self.cfg["state_recognition"]["fusion_weights"]
        # text score
        s_text_open = 0.0; s_text_closed = 0.0
        t = (evidence.get("ocr_text") or "").upper()
        c = evidence.get("ocr_conf", 0.0)
        if any(k.upper() in t for k in self.cfg["state_recognition"]["ocr"]["keywords_open"]):
            s_text_open = c
        if any(k.upper() in t for k in self.cfg["state_recognition"]["ocr"]["keywords_closed"]):
            s_text_closed = c

        # color score
        red = evidence.get("red_ratio", 0.0)
        green = evidence.get("green_ratio", 0.0)
        s_color_open = red
        s_color_closed = green

        # angle score
        s_angle_open = 0.0; s_angle_closed = 0.0
        reason = None
        if angle is not None:
            th_open = self.cfg["state_recognition"]["angle_reference"][kind]["open_deg"]
            th_closed = self.cfg["state_recognition"]["angle_reference"][kind]["closed_deg"]
            # 距离越近分数越高
            def score_to(ref):
                d = abs(angle - ref)
                return float(np.exp(- (d*d) / (2*(self.cfg["state_recognition"]["angle_sigma_deg"]**2))))
            s_angle_open = score_to(th_open)
            s_angle_closed = score_to(th_closed)
        else:
            reason = 1003  # angle not found

        S_open = w["text"]*s_text_open + w["color"]*s_color_open + w["angle"]*s_angle_open
        S_closed = w["text"]*s_text_closed + w["color"]*s_color_closed + w["angle"]*s_angle_closed

        if max(S_open, S_closed) < self.cfg["state_recognition"]["min_state_score"]:
            return "unknown", float(max(S_open, S_closed)), {"S_open": S_open, "S_closed": S_closed}, 1002 if evidence.get("ocr_text") is None else 1000

        if S_open > S_closed:
            return "open", float(S_open), {"S_open": S_open, "S_closed": S_closed}, reason
        else:
            return "closed", float(S_closed), {"S_open": S_open, "S_closed": S_closed}, reason
```

---

## 6. 配置文件（`configs/default.yaml` 完整示例）

> 注意：你压缩包里的 default.yaml/manifest.json 目前包含省略号 `...`，工程上必须替换为可解析的完整配置。下面给出可直接使用的完整版本。

```yaml
model:
  path: "models/switch_detector.onnx"
  device: "cpu"   # cpu/cuda

inference:
  confidence_threshold: 0.6
  nms_threshold: 0.4
  max_detections: 50

state_recognition:
  min_state_score: 0.55

  fusion_weights:
    text: 0.5
    color: 0.3
    angle: 0.2

  angle_sigma_deg: 18

  # 每类开关的标定参考角（建议按点位标定；没有标定时可先给经验值）
  angle_reference:
    breaker:
      open_deg: -60
      closed_deg: 30
    isolator:
      open_deg: -70
      closed_deg: 20
    grounding:
      open_deg: -80
      closed_deg: 10

  ocr:
    enabled: true
    engine: "easyocr"
    keywords_open: ["分", "OPEN", "OFF"]
    keywords_closed: ["合", "CLOSE", "ON"]

  color_hint:
    enabled: true
    hsv_green: [[35, 40, 40], [85, 255, 255]]
    hsv_red1:  [[0, 60, 60], [10, 255, 255]]
    hsv_red2:  [[170, 60, 60], [180, 255, 255]]

logic_validation:
  enabled: true
  rules:
    - name: "防止带电合接地刀"
      condition: "isolator_open_before_grounding"
      severity: "error"
    - name: "异常工况提示：合闸但刀闸断开"
      condition: "breaker_closed_while_isolator_open"
      severity: "warning"

image_quality:
  enabled: true
  method: "laplacian_var"
  mu: 120
  tau: 30
  min_clarity_score: 0.70

focus_link:
  stable_iou_threshold: 0.70
  stable_frames: 3

gauge_reading:
  enabled: false
  pointer:
    theta_min_deg: -120
    theta_max_deg: 120
    value_min: 0.0
    value_max: 1.0
    unit: "MPa"

device_types: ["breaker", "isolator", "grounding"]
```

---

## 7. 性能指标与验收（工程可用口径）

### 7.1 性能指标（建议最低线）
- 单帧单ROI CPU：**< 300 ms**（1080p ROI裁剪后）
- 清晰度评分：对“清晰/轻微模糊/严重模糊”具有单调性（可用三档测试集验证）
- 状态识别：同点位（固定ROI）**≥ 95%**（建议按点位分开评估）
- 逻辑校验误报：**< 2%**（需要时序/多证据完整输入）

### 7.2 交付数据与回放要求
- 回放 demo：≥ 20 段视频或 ≥ 200 张图片（与平台回放接口兼容）
- 每个 ROI 类型至少提供 30 个样本（含弱光/逆光/遮挡）

### 7.3 失败原因码（建议统一）
- `1001` 清晰度过低（clarity_score < min）
- `1002` OCR失败或无文本
- `1003` 未检测到有效角度/线段
- `1004` 表盘/指针检测失败
- `2001` 互锁逻辑异常（带电合接地刀风险）
- `2002` 异常工况提示（breaker closed & isolator open）
- `9000` 未知错误

---

## 8. 局限性与风险控制（必须明确）
1) **角度阈值依赖点位标定**：不同摄像机角度导致角度范围变化，务必按点位标定 `angle_reference`  
2) **强逆光/反光影响OCR与颜色提示**：建议在平台侧做曝光/增益策略与多帧融合  
3) **互锁逻辑严谨性依赖时序**：单帧仅做“安全提示”，建议平台任务层实现状态机与多帧一致性  
4) **表计读数受表盘反光与分辨率影响**：推荐高分辨率采集或二次变焦策略

---

## 9. 未来改进路线（不影响当前交付）
- 引入深度模型做“指示窗/连杆/表计”的检测与分割（提升鲁棒性）
- 多帧融合：状态稳定投票、时序一致性检查
- 结合站内遥测/遥信（如果可用）做交叉校验（降低误报）

---

## 10. 技术支持与版本策略
- 本插件版本：`1.0.0`（规则/传统视觉基线）  
- 未来升级：保持输出 Schema 不变，仅替换内部 `recognize_* / gauge_reading` 实现。  
