# 02_algorithm_contract

## 1. 固定母版规则（跨项目通用算法契约）

1. **输入先校验后推理**：非法输入直接失败，不进入模型调用。
2. **一条主链 + 显式降级链**：每级降级必须可观测、可测试。
3. **输出字段完整**：每个结果必须包含 `label/confidence/bbox/metadata`。
4. **阈值来源唯一**：阈值必须来源配置映射，不允许推理路径临时常量。
5. **异常可追踪**：异常必须记录并返回可解释失败信息，不允许静默吞错。

## 2. 本项目差异契约（busbar_inspection）

### 2.1 输入契约

- `frame`：`np.ndarray`，形状 `H x W x 3`，dtype `uint8`，BGR。
- `rois`：`list[ROI]`，每个 ROI 的 `bbox.x/y/width/height` 必须在 `[0,1]` 且 `width,height > 0`。
- `context`：必须包含 `task_id/site_id/device_id/component_id` 字段；字段允许空字符串，不允许缺字段名。

**非法输入处理**：
- ROI 越界或面积为 0：该 ROI 直接跳过并记 `metadata.error="invalid_roi"`。
- `frame` 维度不合法：整帧返回空结果并计错误计数。

### 2.2 输出契约

#### 2.2.1 检测结果

每个 `RecognitionResult` 必须满足：

1. `label` 取值集合：`pin_missing/crack/foreign_object/quality_failed`（本轮基线）。
2. `confidence` 必须在 `[0,1]`。
3. `bbox` 归一化后仍在 `[0,1]`，且宽高 `> 0`。
4. `metadata` 必须包含：
   - `quality.clarity_score`
   - `quality.is_overexposed`
   - `quality.is_low_contrast`
   - `quality.is_occluded`
   - `suggested_action`
5. 当 `label == "quality_failed"` 时，`failure_reason` 必须非空。

#### 2.2.2 告警结果

- `pin_missing` -> `AlarmLevel.ERROR`
- `crack` -> `AlarmLevel.WARNING`
- `foreign_object` -> `AlarmLevel.WARNING`
- `quality_failed` -> `AlarmLevel.WARNING`

### 2.3 主状态机（强制）

```text
S0 输入校验
  ├─ 非法输入 -> S_fail
  └─ 合法 -> S1

S1 ROI提取
  ├─ ROI无效 -> S_skip_roi
  └─ ROI有效 -> S2

S2 质量门禁
  ├─ fail -> S_quality_failed
  └─ pass -> S3

S3 检测主链
  ├─ YOLOv8-ViT可用且成功 -> S4
  ├─ 否则 model_registry 成功 -> S4
  └─ 否则传统CV -> S4

S4 环境噪声过滤 + 全局NMS
  └─ S5

S5 变焦建议
  └─ S6 输出组装
```

### 2.4 原因码统一契约

本项目外部可见原因码固定为以下集合：

| 外部原因码 | 语义 |
|---|---|
| 101 | 光照异常（过曝/欠曝/低对比） |
| 102 | 遮挡/不可见 |
| 103 | 模糊/失焦 |
| 104 | 雨雾/低能见度 |
| 105 | 运动干扰 |
| 201 | 目标过小，需要变焦 |
| 202 | 检测不稳定 |
| 301 | 环境干扰（鸟类/飞虫/水滴） |

内部子码到外部码映射（强制）：

| 内部码 | 外部码 |
|---|---|
| 1001 | 103 |
| 1002 | 101 |
| 1003 | 101 |
| 1004 | 102 |
| 1005 | 101 |
| 2001 | 201 |
| 2002 | 202 |
| 3001/3002/3003 | 301 |

### 2.5 变焦建议契约

- 目标尺寸：`s_px = max(w_px, h_px)`
- 推荐倍率：

\[
z = clamp(\frac{target\_px}{max(s\_px, \epsilon)}, z_{min}, z_{max})
\]

- `suggested_action` 取值：`NONE/ZOOM_IN/REFOCUS/RECAPTURE/CHANGE_VIEW`
- 当 `s_px < min_obj_px` 时必须输出 `ZOOM_IN` 或 `RECAPTURE`。

### 2.6 配置映射契约（必须落地测试）

当前 YAML 为嵌套结构；算法层必须通过“统一配置视图”读取参数：

| 统一键 | YAML 路径 | 默认值 |
|---|---|---|
| `confidence_threshold` | `thresholds.conf_thr` | 0.25 |
| `nms_threshold` | `thresholds.nms_iou` | 0.50 |
| `tile_size` | `tiling.tile_size` | 1280 |
| `tile_overlap` | `tiling.overlap` | 320 |
| `clarity_threshold` | `quality.blur_thr` | 0.35 |
| `min_object_size_px` | `zoom.min_obj_px` | 18 |
| `target_size_px` | `zoom.target_px` | 90 |
| `zoom_min` | `zoom.zmin` | 1.0 |
| `zoom_max` | `zoom.zmax` | 12.0 |

**禁止**：算法层直接读取不存在的顶层键并静默回落默认值。

## 3. 可执行验证规则

```bash
# 1) 快速契约测试（完成后）
python -m pytest tests/test_config_contract.py tests/test_reason_code_contract.py -q

# 2) 检查原因码字典一致性（文本扫描）
rg -n "REASON_CODES|REASON_DESCRIPTIONS|reason_code" plugin.py detector_enhanced.py

# 3) 检查输出字段（通过测试断言 metadata 必填字段）
python -m pytest tests/test_plugin_contract.py -q
```

## 4. AI 自动闭环 / 人工确认

### 可自动闭环

- 配置映射适配器实现与测试
- 原因码标准化实现与测试
- 输出字段完整性测试
- 降级链路状态机测试

### 必须人工确认

- 外部原因码字典是否与平台告警中心一致
- 缺陷标签全集是否扩展到 8 类（当前输出基线为 3 类缺陷 + 质量失败）
- 性能目标阈值（CPU 5s vs 更严格上线目标）

## 5. 必测清单（未覆盖即不合格）

1. `test_config_contract.py`：YAML 值变更能被算法层读取。
2. `test_reason_code_contract.py`：内部码到外部码映射正确。
3. `test_quality_gate_contract.py`：模糊/过曝/低对比/遮挡触发正确原因码。
4. `test_bbox_contract.py`：切片 remap 后 bbox 不越界。
5. `test_fallback_chain.py`：YOLO 不可用时回退到传统路径。
