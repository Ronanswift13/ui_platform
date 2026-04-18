# 06 重构策略

## 1. When to use
在以下场景查阅本文件：
- 计划重构代码时
- 技术债务清理
- 性能优化
- 新增动物类型或检测链路

## 2. Inputs
- 现有代码
- 性能指标
- 技术债务清单

## 3. Outputs
- 重构计划
- 迁移步骤

## 4. Hard Constraints

### 重构原则
- 小步迭代，每次提交可独立运行
- 先写测试，再重构 (QR-2)
- 保持向后兼容（除非明确废弃）
- 配置兼容: 新参数必须有默认值 (QR-4)

### 禁止行为
- 重构时不添加新功能
- 不破坏现有测试
- 不删除公共 API（需先废弃）
- **不合并 plugin.py 与 core/ 模块** — 算法层与接口层职责分离 (HC-13)
- **不引入新的框架级依赖** — 最小依赖集: numpy, opencv, pydantic, pyyaml, onnxruntime (HC-7)
- **不改变目录结构** — core/, configs/, standalone/, tests/ 四层不可合并
- **不删除降级链路组件** — 降级策略是核心可靠性保障 (QR-3)

### 允许的重构
- 提取公共方法（≥ 3 处重复）
- 拆分大方法（> 80 行）
- 引入策略模式（当新增第 4 种检测链路时）
- 配置 schema 升级（需提供迁移脚本）
- 提取输入清洗为独立纯函数 (QR-13)
- 提取 metadata 构造为 builder 函数 (QR-12)

## 5. Refactor Workflow

```
1. 识别问题 / 技术债务
   ↓
2. 编写/补充测试 (确保重构前有覆盖)
   ↓
3. 小步重构 (每步 < 50 行变更)
   ↓
4. 运行测试验证 (pytest tests/ -v)
   ↓
5. 检查审查规则 (QR-1~QR-14)
   ↓
6. 提交 (每步独立提交)
```

### 重构前检查清单
- [ ] 受影响代码有 >= 80% 测试覆盖
- [ ] 降级链路测试存在且通过
- [ ] 枚举成员数断言存在 (QR-8)
- [ ] 输出 schema 字段断言存在 (QR-12)

## 6. 常见重构模式

### 提取纯函数 (QR-13)
```python
# Before: 清洗逻辑内嵌在业务方法中
def detect(self, frame):
    h, w = frame.shape[:2]
    if h > 4096 or w > 4096:
        raise ValueError("Too large")
    ...

# After: 独立纯函数 + 独立测试
def validate_frame(frame: np.ndarray) -> np.ndarray:
    """独立的帧验证函数，可单独测试"""
    h, w = frame.shape[:2]
    if h > 4096 or w > 4096:
        raise ValueError(f"Frame too large: {w}x{h}")
    return frame

def detect(self, frame):
    frame = validate_frame(frame)
    ...
```

### 提取 Metadata Builder (QR-12)
```python
# Before: 零散构造 metadata
result.metadata = {"animal_class": cls, "timestamp": time.time()}

# After: 集中构造
def _build_detection_metadata(cls, bbox, fallback_level=0):
    return {
        "animal_class": cls,
        "timestamp": time.time(),
        "fallback_level": fallback_level,
        "bbox_clamped": False,
    }
```

### 参数对象化 (QR-4)
```python
# Before: 多参数传递
def detect(frame, conf, iou, max_det, classes):
    pass

# After: 配置对象
@dataclass
class DetectConfig:
    conf: float = 0.5
    iou: float = 0.45
    max_det: int = 100
    classes: List[int] = None

def detect(frame, config: DetectConfig):
    pass
```

## 7. Deprecation Process

```python
import warnings

def old_function():
    warnings.warn(
        "old_function is deprecated, use new_function instead",
        DeprecationWarning,
        stacklevel=2
    )
    return new_function()
```

## 8. Validation Rules

```bash
# 重构前后测试必须通过
pytest tests/ -v

# 检查是否引入新依赖
pip freeze > requirements_after.txt
diff requirements.txt requirements_after.txt

# 检查审查规则
./scripts/run_quality_gate.sh
```
