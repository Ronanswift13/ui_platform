# 00 项目上下文

## 1. When to use
在以下场景查阅本文件：
- 首次接触项目，需要了解整体结构
- 新成员加入开发
- 架构决策参考
- 确认依赖关系和配置加载方式

## 2. Inputs
- 仓库克隆: `git clone` 后的 `plugins/animal_detection/` 目录
- 配置文件: `configs/default.yaml`
- 依赖清单: `requirements.txt`

## 3. Outputs
- 理解项目目录结构
- 理解依赖关系（核心 vs 可选）
- 理解 Mock 策略（硬件/模型/测试）
- 理解配置加载优先级

## 4. Hard Constraints
- 核心依赖必须可安装: `pip install -r requirements.txt` 无报错
- `configs/default.yaml` 必须存在且可解析
- `plugin.py` 必须可被 DarkBreaker SDK 加载
- Python 版本: >= 3.9

## 5. Algorithm / Logic Contract

### 目录结构

```
animal_detection/
├── core/              # 核心算法模块
│   ├── detector.py    # YOLO 检测器
│   ├── tracker.py     # 目标跟踪 (ByteTrack)
│   ├── deterrent.py   # 驱离控制
│   ├── thermal_validator.py  # 热成像验证
│   ├── statistics.py  # 统计分析
│   ├── event_schema.py # 事件契约定义
│   └── onnx_inference.py # ONNX 推理引擎
├── configs/           # 配置文件
├── models/            # ONNX 模型文件
├── standalone/        # 独立运行支持（Web UI）
├── tests/             # 单元测试 + 集成测试
├── demo/              # 演示脚本
└── plugin.py          # DarkBreaker 插件接口
```

### 依赖关系

**核心依赖**
- `numpy`: 数值计算
- `opencv-python`: 图像处理
- `onnxruntime`: ONNX 模型推理
- `pyyaml`: 配置解析
- `darkbreaker_sdk`: 插件框架

**可选依赖（standalone 模式）**
- `flask`: Web 服务
- `ultralytics`: YOLO 训练（仅开发）

### Mock 策略

**硬件 Mock**
- 摄像头缺失: 使用测试图片或视频文件
- 热成像缺失: 跳过热验证，仅依赖视觉检测

**模型 Mock**
- ONNX 模型缺失: 返回空检测结果，记录警告
- 推理失败: 返回空结果，不阻塞主流程

**测试 Mock**
- `tests/` 中使用 `pytest.fixture` 提供标准化 mock 数据
- 所有检测器实现统一接口，便于替换

### 配置加载优先级

```
1. 命令行参数（最高优先级）
2. 环境变量 ANIMAL_DETECTION_CONFIG
3. configs/default.yaml（默认配置）
```

### 启动命令

```bash
# 首次运行 - 创建虚拟环境
python3.13 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python pyyaml flask onnxruntime pydantic fastapi uvicorn python-multipart

# 后续运行
source .venv/bin/activate
python run_standalone.py       # 独立运行（Web UI）
pytest tests/                  # 运行测试
```

**注意**: Homebrew Python 3.13+ 强制使用虚拟环境 (PEP 668)，不能直接 `pip install`

## 6. Validation Rules

```bash
# 依赖安装验证
pip install -r requirements.txt

# 模块可导入验证
python -c "import animal_detection"

# 配置可解析验证
python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))"

# 基础测试通过
pytest tests/ -q
```

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| 依赖缺失 | ImportError | `pip install -r requirements.txt` |
| 配置文件缺失 | FileNotFoundError | 回退到内置默认值 |
| SDK 版本不兼容 | 接口不匹配 | 检查 `darkbreaker_sdk` 版本 |
| Python 版本过低 | 语法错误 | 要求 >= 3.9 |
| ONNX 模型缺失 | 检测失败 | 返回空结果 + 警告日志 |
| Homebrew Python 禁止 pip | externally-managed-environment | 使用 `python -m venv .venv` 创建虚拟环境 |
| 端口占用 | Address already in use | `lsof -ti:8082 \| xargs kill -9` |

## 8. Required Tests

- `pytest tests/ -q` 全部通过
- `python -c "import animal_detection"` 无报错
