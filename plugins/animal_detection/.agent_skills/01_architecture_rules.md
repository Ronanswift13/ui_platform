# 01 架构规则

## 1. When to use
在以下场景查阅本文件：
- 添加新模块或修改现有模块
- 代码审查时检查架构合规性
- 重构代码时确认依赖方向

## 2. Inputs
- 源代码文件
- 模块导入语句

## 3. Outputs
- 架构合规性判定
- 违规修复建议

## 4. Hard Constraints

### 依赖方向规则

```
plugin.py
    ↓
core/ ←────────── standalone/
    ↓                  ↓
configs/          templates/
```

**禁止的依赖方向：**
- `core/` 不得 import `standalone/`
- `core/` 不得 import `demo/`
- `core/` 不得 import `tests/`

### 模块职责

| 模块 | 职责 | 可依赖 |
|------|------|--------|
| `core/detector.py` | YOLO 检测 | numpy, cv2, onnxruntime |
| `core/tracker.py` | 目标跟踪 | numpy, core/detector |
| `core/deterrent.py` | 驱离控制 | core/event_schema |
| `core/thermal_validator.py` | 热验证 | numpy, cv2 |
| `core/event_schema.py` | 事件定义 | dataclasses, enum |
| `core/statistics.py` | 统计分析 | core/event_schema |
| `standalone/app.py` | Web 服务 | flask, core/* |
| `plugin.py` | SDK 接口 | core/*, darkbreaker_sdk |

### 接口契约

**检测器接口**（隐式契约，未定义 Protocol 基类）
```python
# YOLOv8Detector 和 AnimalONNXEngine 遵循的隐式接口：
class Detector:
    def load(self) -> bool: ...
    def detect(self, frame: np.ndarray, ...) -> List[Detection]: ...
    def get_stats(self) -> dict: ...
```

**事件接口**
```python
# 所有事件必须遵循 AnimalEvent 契约
# 见 core/event_schema.py
```

## 5. Algorithm / Logic Contract

### 数据流

```
摄像头帧 → detector.detect() → tracker.update() → thermal_validator.validate()
                                      ↓
                              event_schema.build_intrusion_event()
                                      ↓
                              plugin.emit_event()
```

### 降级策略

| 组件失败 | 降级行为 |
|---------|---------|
| ONNX 模型加载失败 | 返回空检测，记录 ERROR |
| 热成像不可用 | 跳过热验证，仅视觉检测 |
| 跟踪器异常 | 重置跟踪状态，继续检测 |
| 驱离设备不可用 | 仅记录事件，不执行驱离 |

## 6. Validation Rules

```bash
# 架构合规检查
grep -rn "from.*standalone\|import.*standalone" core/ | grep -v __pycache__
# 期望: 无输出

grep -rn "from.*demo\|import.*demo" core/ | grep -v __pycache__
# 期望: 无输出
```

## 7. Failure Modes

| 违规 | 影响 | 修复 |
|------|------|------|
| core 依赖 standalone | 循环依赖风险 | 提取公共接口到 core |
| 跨层直接调用 | 测试困难 | 通过接口解耦 |
| 硬编码配置 | 部署困难 | 移至 configs/ |

## 8. Required Tests

- 架构合规检查通过（无禁止依赖）
- 所有公共接口有类型注解
