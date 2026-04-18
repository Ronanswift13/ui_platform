# CLAUDE.md — 表计读数插件 (meter_reading)

## 项目简介
变电站巡检平台 DarkBreaker 的表计读数插件。从摄像头帧中自动识别模拟表/数字表/LED指示灯的读数和状态。

## 快速上手
```bash
# 环境准备
cd plugins/meter_reading
pip install -r requirements.txt

# 冒烟测试
python -c "from plugin import MeterReadingPlugin; print('OK')"

# 启动 Standalone
python run_standalone.py  # → http://localhost:8091
```

## 关键文件
| 文件 | 职责 | 修改频率 |
|------|------|----------|
| `detector_enhanced.py` | 核心算法 (V3.0) | 高 — 算法迭代 |
| `plugin.py` | SDK 接口适配 | 低 — 接口稳定 |
| `configs/default.yaml` | 全量参数 | 中 — 调参 |
| `manifest.json` | 插件元数据 | 极低 |

## 开发规范

### 必读
在修改任何代码前，请先阅读:
1. `.agent_skills/08_task_routing.md` — 任务路由与脚本映射
2. `PROJECT_CARD.md` — 项目范围和约束
3. `.agent_skills/01_architecture_rules.md` — 架构规则
4. `.agent_skills/02_algorithm_contract.md` — 算法合约

### 编码规则
- **配置驱动**: 所有阈值走 `configs/default.yaml`，禁止硬编码
- **Fallback 不可删**: HRNet → HoughCircle → HoughLine 链路必须保留
- **类型注解**: 所有 public 方法必须有 type hints
- **异常处理**: 捕获具体异常，不用裸 `except:`
- **日志**: 关键路径 `logger.info()`，异常路径 `logger.warning()`
- **无残留标记**: 提交前消除所有 TODO/FIXME/HACK

### 测试要求
- 先跑模块化 targeted: `scripts/run_targeted_tests.sh <analog|digital|led|validation|plugin|contract>`
- 触及生产代码或配置契约 → 再跑 `scripts/run_regression_tests.sh`
- 提测 / 审计 → 跑 `scripts/run_quality_gate.sh`
- `tests/regression/` 与 `tests/fixtures/` 为空时，只能报告 skip，不能宣称完成数据集回归

### 禁止事项
- 禁止引入 PyTorch/TensorFlow (仅允许 ONNX Runtime)
- 禁止修改 BasePlugin 接口签名
- 禁止访问外部网络
- 禁止持久化原始图像

## 命令
| 命令 | 用途 |
|------|------|
| `/bootstrap` | 初始化开发环境 |
| `/implement` | 实现新功能 |
| `/repair` | 修复 bug |
| `/audit` | 全面质量审计 |
| `/propagate` | 经验传播到知识库 |

## 算法流水线概览
```
帧 (BGR np.ndarray)
  ↓
预处理: CLAHE + 去眩光 + 对比度增强
  ↓
类型分发:
  ├─ 模拟表: Keypoint → 透视校正 → 指针角度 → 量程插值
  ├─ 数字表: CRNN OCR → 数值解析
  └─ LED: HSV 色相分类
  ↓
置信度评估 (达到成功阈值 -> SUCCESS；否则进入 NEED_MANUAL_REVIEW)
  ↓
重试 (最多 3 次)
  ↓
RecognitionResult 输出
```

## 性能目标
| 指标 | 目标 |
|------|------|
| 单帧延迟 | ≤ 500ms (CPU P95) |
| 整体成功率 | ≥ 95% |
| 模拟表精度 | 满量程 ±2% |
| 数字表准确率 | ≥ 98% 字符级 |
| LED 准确率 | ≥ 99% |
