# 02 算法契约

## 1. When to use
在以下场景查阅本文件：
- 修改检测/跟踪/验证算法
- 调整阈值参数
- 排查误检/漏检问题
- 新增动物类型或事件类型

## 2. Inputs

### 2.1 帧输入
- 摄像头帧: BGR uint8 ndarray, shape (H, W, 3), 最大 4096×4096
- 热成像帧: 灰度 uint8 ndarray, shape (H, W), 可选
- 配置参数: configs/default.yaml

### 2.2 配置输入
- 所有阈值/参数必须来自 `configs/default.yaml`，不允许在推理路径中硬编码 (QR-1)
- 配置加载优先级: 命令行参数 > 环境变量 > configs/default.yaml

## 3. Outputs

### 3.1 检测结果
```python
@dataclass
class AnimalDetectionResult:
    detection_id: str          # UUID[:12]
    animal_class: str          # AnimalClass 枚举值
    confidence: float          # ∈ [0, 1], 必须经 _sanitize_confidence() 清洗 (QR-11)
    bbox: Optional[BoundingBox]  # 归一化坐标 [0, 1]
    world_position: Optional[WorldPosition]
    track_id: Optional[str]
    is_thermal_validated: bool
    thermal_diff_c: Optional[float]
    stay_duration_s: float
```

### 3.2 事件
```python
@dataclass
class AnimalEvent:
    # 统一契约核心: timestamp / type / location / value / confidence
    # + evidence(证据链) + suggestion(处置建议) + trace_id
```

### 3.3 枚举状态集 (QR-8: 测试必须锁定成员数)

| 枚举 | 成员数 | 值域 |
|------|--------|------|
| AnimalClass | 8 | mouse, cat, snake, bird, dog, poultry, insect, other |
| EventType | 10 | intrusion_detected, intrusion_confirmed, intrusion_cleared, deterrent_triggered, deterrent_result, thermal_validated, thermal_rejected, tracking_update, stay_exceeded, statistics_report |
| RiskLevel | 4 | low, medium, high, critical |

## 4. Hard Constraints (绝对约束)

| # | 约束 | 检测方法 |
|---|------|----------|
| HC-1 | 接口签名严格遵循 BaseDetector Protocol | 类型检查 |
| HC-2 | 所有阈值参数必须从 configs/default.yaml 读取，不允许硬编码 (QR-1) | `grep -rn '0\.[0-9]' core/*.py` |
| HC-3 | 降级链路必须保留并可审计 (QR-3) | 降级测试 + metadata |
| HC-4 | 单帧检测延迟 P95 ≤ 100ms | 内置计时 benchmark |
| HC-5 | 不允许外部网络调用 | 代码审查 |
| HC-6 | 不允许持久化原始图像到磁盘（除证据链路外） | 代码审查 |
| HC-7 | 最小依赖: numpy, opencv, pydantic, pyyaml, onnxruntime | requirements.txt |
| HC-8 | GPU 为可选，CPU 必须可运行 | CI 测试 |
| HC-9 | ONNX 模型 ≤ 50MB, 运行时内存 ≤ 512MB | 模型文件检查 + psutil |
| HC-10 | confidence ∈ [0, 1], 必须经 _sanitize_confidence() 清洗 (QR-11) | 赋值入口审查 |
| HC-11 | 召回率 ≥ 85%, 精确率 ≥ 80%, 误报率 < 5% | 测试集评估 |
| HC-12 | 测试覆盖率: 整体 ≥ 70%, core/ ≥ 80% | pytest --cov |
| HC-13 | 算法层 (core/) 与 SDK 接口层 (plugin.py) 职责分离 | 架构审查 |
| HC-14 | 推理必须无状态（除跟踪器状态外） | 代码审查 |
| HC-15 | 枚举状态集严格等于合约定义，测试锁定成员数 (QR-8) | `len(Enum)` 断言 |

## 5. Algorithm / Logic Contract

### 5.1 YOLO 检测流程

```python
def detect(frame: np.ndarray) -> List[Detection]:
    # 1. 输入验证: shape 检查, 最大 4096x4096 (QR-13)
    # 2. 预处理: resize to 640x640, normalize to [0,1], HWC→CHW
    # 3. ONNX 推理 (CPU/CUDA)
    # 4. 后处理:
    #    a. 解析原始输出 (1, 4+C, N) → 坐标 + 类别分数
    #    b. 置信度过滤 (>= confidence_threshold)
    #    c. 归一化坐标到 [0,1]
    #    d. 越界 bbox 处理: 警告 + 置信度衰减 (QR-9)
    #    e. 按类别 NMS (IoU threshold)
    #    f. 截断到 max_detections
    # 5. 构建 AnimalDetectionResult, confidence 经 _sanitize_confidence() (QR-11)
    return detections
```

### 5.2 多尺度检测策略

```python
def multi_scale_detect(frame: np.ndarray) -> List[Detection]:
    # 1. 原图推理
    # 2. 4象限裁剪推理 (overlap 20%)
    # 3. 坐标映射回原图
    # 4. 全局 NMS 合并
    return detections
```

### 5.3 置信度阈值 (按动物类型)

| 动物类型 | 最低置信度 | 热验证要求 | 风险等级 |
|---------|-----------|-----------|---------|
| 鼠 (mouse) | 0.5 | 必须 | HIGH |
| 蛇 (snake) | 0.6 | 必须 | CRITICAL |
| 猫 (cat) | 0.5 | 可选 | MEDIUM |
| 鸟 (bird) | 0.4 | 不需要 | LOW |
| 狗 (dog) | 0.5 | 可选 | MEDIUM |
| 家禽 (poultry) | 0.5 | 不需要 | LOW |
| 昆虫 (insect) | 0.5 | 不需要 | LOW |
| 其他 (other) | 0.5 | 可选 | MEDIUM |

### 5.4 ByteTrack 跟踪流程

```python
def update(detections: List[Detection]) -> List[Track]:
    # 1. 预测: 卡尔曼滤波预测位置
    # 2. 匹配: IoU + 外观特征匹配
    # 3. 更新: 更新跟踪状态
    # 4. 管理: 创建/删除轨迹
    #    - max_age: 30 帧无匹配则删除
    #    - min_hits: 3 次匹配确认新轨迹
    return tracks
```

### 5.5 热验证流程

```python
def validate(bbox: BoundingBox, thermal_frame: np.ndarray) -> bool:
    # 1. 提取 ROI (bbox 区域)
    # 2. 计算目标区域平均温度
    # 3. 计算背景区域平均温度
    # 4. thermal_diff = abs(target_temp - background_temp)
    # 5. is_valid = thermal_diff >= heat_diff_threshold_c (默认 2.0°C)
    return is_valid
```

### 5.6 驻留时间计算

```python
def update_stay_duration(track: Track, current_time: float):
    if track.is_stationary:  # 速度 < 0.1 m/s
        track.stay_duration += current_time - track.last_update
    else:
        track.stay_duration = 0  # 移动则重置
```

### 5.7 统一置信度计算 (QR-5)

```python
# 所有链路的 confidence 必须经过统一语义处理:
# 1. 原始检测置信度 c_det: YOLO 模型输出 class_score
# 2. 空间一致性置信度 c_spatial: bbox 是否在合法区域内
# 3. 热验证置信度 c_thermal: 温差是否满足阈值
# 最终: confidence = _sanitize_confidence(min(c_det, c_spatial, c_thermal))
# 若热验证未启用, 则 c_thermal = 1.0 (不参与约束)
```

## 6. Validation Rules

```bash
# 检测延迟测试
python -c "from core.detector import YOLOv8Detector; d=YOLOv8Detector('models/animal_yolov8n.onnx'); d.load(); print(d.get_stats())"

# 模型完整性验证
python -c "import onnxruntime; onnxruntime.InferenceSession('models/animal_yolov8n.onnx')"

# 枚举锁定测试
python -c "from core.event_schema import AnimalClass, EventType, RiskLevel; assert len(AnimalClass)==8; assert len(EventType)==10; assert len(RiskLevel)==4; print('PASS')"
```

## 7. Failure Modes

| 故障 | 症状 | 处理 | 降级等级 |
|------|------|------|---------|
| 模型加载失败 | ImportError/FileNotFoundError | 返回空检测 + ERROR 日志 | L1 |
| 推理超时 | 延迟 > 1s | 跳过当前帧 + WARN 日志 | L1 |
| 内存不足 | OOM | 降低输入分辨率 / 关闭多尺度 | L2 |
| 跟踪器发散 | ID 频繁切换 | 重置跟踪状态 | L1 |
| 热成像不可用 | 无帧数据 | 跳过热验证，仅视觉检测 | L1 |
| 驱离设备不可用 | 通信失败 | 仅记录事件，不执行驱离 | L1 |
| 未知类别 ID | class_map 无匹配 | 映射为 OTHER + metadata 标记 (QR-10) | L0 |

## 8. Required Tests

### 必须存在的测试 (88 项)

**A. 输入验证 (8 项)**
- DT-01: 空帧输入 → 空检测列表
- DT-02: 噪声帧输入 → 空检测列表
- DT-03: 超大图像 (>4096) → 拒绝或降采样
- DT-04: 非 uint8 输入 → 类型错误
- DT-05: 单通道输入 → 类型错误
- DT-06: ROI 超出图像范围 → 裁剪到有效区域
- DT-07: ROI 面积为 0 → 空检测
- DT-08: 模型未加载时调用 detect → 空列表 + 警告

**B. 置信度与过滤 (8 项)**
- CF-01: 低置信度 (conf=0.3) → 被过滤
- CF-02: 边界置信度 (conf=0.5) → 保留
- CF-03: 高置信度 (conf=0.95) → 保留
- CF-04: NaN confidence → _sanitize_confidence → 0.0 (QR-11)
- CF-05: Inf confidence → _sanitize_confidence → 0.0 (QR-11)
- CF-06: 负数 confidence → _sanitize_confidence → 0.0 (QR-11)
- CF-07: 超1 confidence → _sanitize_confidence → 1.0 (QR-11)
- CF-08: bbox 越界 → 置信度衰减 + 警告 (QR-9)

**C. 热验证 (5 项)**
- TH-01: 无温差 (diff=0°C) → 验证失败
- TH-02: 临界温差 (diff=2.0°C) → 验证通过
- TH-03: 高温差 (diff=10°C) → 验证通过
- TH-04: 热成像帧缺失 → 跳过验证
- TH-05: 热验证关闭时 → c_thermal=1.0 不参与约束

**D. 跟踪 (5 项)**
- TR-01: 新目标首次出现 → 分配新 ID
- TR-02: 连续 max_age 帧无检测 → 删除轨迹
- TR-03: 目标消失后重现 → 新 ID
- TR-04: 两目标交叉 → 保持各自 ID
- TR-05: 跟踪器异常 → 重置状态 + 继续检测

**E. 事件生成 (6 项)**
- EV-01: 无检测 → INTRUSION_CLEARED 事件
- EV-02: 单检测 → INTRUSION_DETECTED 事件
- EV-03: 多类型检测 → 聚合事件 (最高风险)
- EV-04: 蛇检测 → risk=CRITICAL
- EV-05: 事件字段完整性 → 所有必填字段存在 (QR-12)
- EV-06: trace_id 唯一性 → 每次调用不同

**F. 枚举与 Schema (6 项)**
- EN-01: len(AnimalClass) == 8 (QR-8)
- EN-02: len(EventType) == 10 (QR-8)
- EN-03: len(RiskLevel) == 4 (QR-8)
- EN-04: AnimalDetectionResult.to_dict() 字段完整 (QR-12)
- EN-05: AnimalEvent.to_dict() 字段完整 (QR-12)
- EN-06: build_intrusion_event() 输出合规

**G. 降级链路 (4 项)**
- DG-01: 模型文件缺失 → load() 返回 False, detect() 返回空
- DG-02: onnxruntime 缺失 → load() 返回 False + 错误日志
- DG-03: 热成像不可用 → 仅视觉检测
- DG-04: 驱离设备不可用 → 仅记录事件

**H. 性能 (3 项)**
- PF-01: 单帧检测延迟 < 100ms (640x480 输入, CPU)
- PF-02: 多尺度检测延迟 < 500ms (640x480 输入, CPU)
- PF-03: NMS 处理 100 个候选框 < 10ms

**I. 配置 (4 项)**
- CG-01: 修改 YAML confidence_threshold → 检测行为变化 (QR-4)
- CG-02: 修改 YAML nms_threshold → NMS 行为变化 (QR-4)
- CG-03: 配置缺失时使用默认值 → 行为正确 (QR-4)
- CG-04: 配置值越界 → 拒绝或警告
