# 05 安全边界

## 1. When to use
在以下场景查阅本文件：
- 处理外部输入时
- 添加新的 API 端点
- 配置文件处理
- 修改运行入口或依赖加载

## 2. Inputs
- 外部数据源（摄像头、热成像、配置、API）
- 用户输入

## 3. Outputs
- 安全的数据处理
- 审计日志

## 4. Hard Constraints

### 输入验证
- 所有外部输入必须验证
- 图像尺寸限制: 最大 4096x4096 (QR-13: 提取为独立验证函数)
- ROI 坐标必须裁剪到有效范围
- 配置值范围检查

### 敏感数据
- 禁止硬编码密钥
- 日志中不记录敏感信息
- 证据图片路径不暴露系统结构

### 网络安全
- standalone 模式默认仅监听 localhost
- 生产部署需配置认证

### 运行环境安全
- run_standalone.py 必须含 venv 自动激活守卫 (QR-6)
- `__init__.py` 顶层不得引入 C 扩展依赖 (QR-7)

## 5. Security Checklist

### 输入处理
```python
# 正确: 验证图像尺寸 (QR-13: 独立验证函数)
def validate_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame")
    h, w = frame.shape[:2]
    if h > 4096 or w > 4096:
        raise ValueError(f"Frame too large: {w}x{h}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected BGR frame, got shape {frame.shape}")
    return frame

# 错误: 直接处理未验证输入
def process_frame(frame):  # 无类型检查
    return cv2.resize(frame, (640, 640))  # 可能 OOM
```

### 配置处理
```python
# 正确: 使用 safe_load
config = yaml.safe_load(open("config.yaml"))

# 错误: 使用 load (允许任意代码执行)
config = yaml.load(open("config.yaml"))
```

### 路径处理
```python
# 正确: 验证路径
def save_evidence(filename: str, data: bytes):
    safe_name = os.path.basename(filename)  # 防止路径遍历
    path = os.path.join(EVIDENCE_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(data)
```

### 运行入口守卫 (QR-6)
```python
# run_standalone.py 顶部 — 必须在所有业务 import 之前
import os, sys
from pathlib import Path

VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
if VENV_PYTHON.exists() and sys.prefix == sys.base_prefix:
    # 当前不在虚拟环境中，提示用户
    print("ERROR: 请在虚拟环境中运行。执行: source .venv/bin/activate")
    sys.exit(1)
```

**关键点**:
- `sys.prefix == sys.base_prefix` 是判断"不在 venv 中"的标准方法
- 守卫在所有业务 import 之前执行，避免 import 副作用
- 检测方法: `grep -L 'sys.base_prefix' plugins/*/run_standalone.py`

### SDK 顶层导入安全 (QR-7)
```python
# 正确: 延迟导入
class AnimalPlugin:
    def detect(self, frame):
        import numpy as np  # 延迟导入，不在模块顶层
        ...

# 错误: __init__.py 顶层导入重量级依赖
# darkbreaker_sdk/__init__.py
import numpy as np  # 任何 from darkbreaker_sdk import ... 都触发 numpy
```

**检测方法**: `python -c "import darkbreaker_sdk"` 在仅安装 pydantic+pyyaml 的精简环境中必须成功。

## 6. 权限边界

| 组件 | 可读 | 可写 | 禁止 |
|------|------|------|------|
| core/ | configs/, models/ | logs/ (通过 logger) | 文件系统写入 |
| standalone/ | core/*, configs/ | evidence/, logs/ | 外部网络 |
| plugin.py | core/*, configs/ | 无 | 文件系统写入 |
| tests/ | 所有 | 临时文件 | 生产配置 |

## 7. Validation Rules

```bash
# 安全扫描
bandit -r . -ll --exclude __pycache__,tests

# 密钥扫描
grep -rn "password\s*=\|secret\s*=\|api_key\s*=" --include="*.py" . | grep -v test_

# venv 守卫检查 (QR-6)
grep -L 'sys.base_prefix' run_standalone.py

# 顶层 import 检查 (QR-7)
grep -rn "^import numpy\|^import cv2\|^import onnxruntime" core/__init__.py
```

## 8. Failure Modes

| 威胁 | 影响 | 缓解 |
|------|------|------|
| 路径遍历 | 任意文件读写 | basename 过滤 |
| YAML 注入 | 代码执行 | safe_load |
| 大图像 DoS | OOM | 尺寸限制 4096x4096 |
| 日志注入 | 日志污染 | 转义特殊字符 |
| 解释器漂移 | 依赖缺失 | venv guard (QR-6) |
| SDK 隐式依赖 | 环境不可用 | 延迟导入 (QR-7) |
