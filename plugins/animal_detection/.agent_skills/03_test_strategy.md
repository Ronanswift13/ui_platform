# 03 测试策略

## 1. When to use
在以下场景查阅本文件：
- 添加新功能后需要补充测试
- 修复 bug 后需要写防回归测试
- 不确定如何构造 mock 输入数据
- 需要确认测试覆盖率要求

## 2. Inputs
- 功能需求或 bug 报告
- 相关模块的源码
- `tests/test_matrix_template.md` 测试矩阵

## 3. Outputs
- 测试用例代码 (`tests/test_*.py`)
- Mock 数据 (`tests/fixtures/` 或内联)
- 覆盖率报告 (`htmlcov/index.html`)

## 4. Hard Constraints
- 单元测试覆盖率 >= 70%
- `core/` 模块覆盖率 >= 80%
- 所有降级路径必须有测试覆盖 (QR-3)
- 所有边界用例必须有测试覆盖
- 测试必须可在无硬件环境下运行（全部使用 mock）
- 枚举成员数必须有断言锁定 (QR-8)
- 输出 schema 必填字段必须有断言锁定 (QR-12)
- confidence 清洗函数必须有独立参数化测试 (QR-11)

## 5. 分级测试体系

### L0 — 单元测试 (快速, < 10s)

针对独立函数和类方法的测试，不依赖外部资源。

**覆盖范围**:
- `_sanitize_confidence()` — NaN/Inf/负数/超1 (QR-11)
- `_preprocess()` — 图像缩放和归一化
- `_parse_raw_outputs()` — YOLO 输出解析
- `_nms()` — 非极大值抑制
- `BoundingBox.from_xyxy()` — 坐标转换
- `build_intrusion_event()` — 事件构建
- 枚举成员数锁定 (QR-8)
- 输出字段完整性 (QR-12)

**运行命令**:
```bash
pytest tests/ -v --timeout=30 -k "not standalone and not integration"
```

### L1 — 集成测试 (中速, < 60s)

测试完整的处理链路：init → detect → track → validate → emit_event。

**覆盖范围**:
- 插件初始化和配置加载
- 检测 → 跟踪 → 事件生成的端到端流程
- 配置传递链路 (QR-4): 修改 YAML → 行为变化
- 降级链路 (QR-3): 模型缺失 → 空检测, 热成像缺失 → 跳过验证

**运行命令**:
```bash
pytest tests/ -v --timeout=60 -k "integration or plugin"
```

### L2 — 回归测试 (慢速, < 5min)

基于标定图片集的精度回归测试。测试代码已就位于 `tests/regression/`。

**覆盖范围**:
- 与基线比对，精度不劣化（召回率 ≥ 85%，精确率 ≥ 80%，误报率 < 5%）
- 特定场景 (小目标/遮挡/多目标/低光照) 的检测准确性

**状态**: 测试框架已就位，待标定数据集放入 `tests/regression/fixtures/` 后激活

**运行命令**:
```bash
pytest tests/regression/ -v -m regression
./scripts/run_regression_tests.sh    # 包含 L2 在内的全量回归
```

### L3 — 冒烟测试 (端口可达性)

验证 standalone 服务的基本可达性。

**覆盖范围**:
- 端口 8082 可达
- `/api/health` 返回 200 + `{"healthy": true}`
- 首页 `/` 返回 200

**运行命令**:
```bash
# 启动服务后
curl -s http://localhost:8082/api/health | python -m json.tool
```

## 6. Mock 输入构造

Mock 数据集中管理在 `tests/fixtures/` 目录：

| 模块 | 用途 |
|------|------|
| `tests/fixtures/frame_factories.py` | 帧数据工厂：blank/noise/thermal 等 |
| `tests/fixtures/detection_factories.py` | 检测结果工厂：make_mouse/snake/cat + track 工厂 |
| `tests/fixtures/config_factories.py` | 配置工厂：simulation/production/thermal_enabled 等 |

`tests/conftest.py` 注册了常用 pytest fixtures（从工厂导入），各测试文件通过 fixture 名直接使用：

```python
def test_example(blank_frame, mouse_detection, simulation_config):
    # blank_frame: 全黑 640x480 BGR
    # mouse_detection: AnimalDetectionResult(mouse, conf=0.85)
    # simulation_config: 仿真模式完整配置字典
    ...
```

## 7. 边界用例清单

### A. 检测边界

| 用例 ID | 场景 | 输入 | 期望输出 |
|---------|------|------|---------|
| DT-01 | 空帧 | 全黑图像 | 空检测列表 |
| DT-02 | 噪声帧 | 随机噪声 | 空检测列表 |
| DT-03 | 低置信度 | conf=0.3 | 被过滤 |
| DT-04 | 边界置信度 | conf=0.5 | 保留 |
| DT-05 | 多目标 | 5只鼠 | 5个检测 |

### B. 热验证边界

| 用例 ID | 场景 | 输入 | 期望输出 |
|---------|------|------|---------|
| TH-01 | 无温差 | diff=0°C | 验证失败 |
| TH-02 | 临界温差 | diff=2.0°C | 验证通过 |
| TH-03 | 高温差 | diff=10°C | 验证通过 |
| TH-04 | 热成像缺失 | None | 跳过验证 |

### C. 跟踪边界

| 用例 ID | 场景 | 输入 | 期望输出 |
|---------|------|------|---------|
| TR-01 | 新目标 | 首次出现 | 分配新 ID |
| TR-02 | 目标消失 | 连续 5 帧无检测 | 删除轨迹 |
| TR-03 | 目标重现 | 消失后重现 | 新 ID |
| TR-04 | ID 切换 | 两目标交叉 | 保持各自 ID |

### D. 事件生成边界

| 用例 ID | 场景 | 输入 | 期望输出 |
|---------|------|------|---------|
| EV-01 | 无检测 | 空列表 | CLEARED 事件 |
| EV-02 | 单检测 | 1只鼠 | DETECTED 事件 |
| EV-03 | 多类型 | 鼠+蛇 | 聚合事件 |
| EV-04 | 高风险 | 蛇 | risk=CRITICAL |

## 8. Validation Rules

```bash
# 运行测试 + 覆盖率
pytest tests/ --cov=. --cov-fail-under=70 --cov-report=html -v

# 运行质量闸门 (三步)
./scripts/run_targeted_tests.sh     # L0 单元测试
./scripts/run_regression_tests.sh   # L2 回归测试 (如 fixtures 就绪)
./scripts/run_quality_gate.sh       # 全量质量闸门

# 模块针对性测试
./scripts/run_targeted_tests.sh detector    # 检测模块
./scripts/run_targeted_tests.sh tracker     # 跟踪模块
./scripts/run_targeted_tests.sh thermal     # 热验证
./scripts/run_targeted_tests.sh event       # 事件生成
```

## 9. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| 覆盖率下降 | 质量退化 | 阻塞合并，补充测试 |
| Flaky 测试 | CI 不稳定 | 隔离随机种子，固定 fixture |
| Mock 与真实行为不符 | 假阳性通过 | 定期用真实数据校验 |
| 测试运行过慢 | 开发效率低 | 分层: L0 快速 + L2 慢速 |

## 10. Required Tests

覆盖率目标:
- 整体: >= 70%
- `core/detector.py`: >= 80%
- `core/tracker.py`: >= 80%
- `core/event_schema.py`: >= 90%
- `standalone/`: >= 60%

必须存在的测试:
- 每个降级路径至少一个测试 (QR-3)
- 每个边界用例至少一个测试
- 事件契约完整性测试 (QR-12)
- 枚举成员数锁定测试 (QR-8)
- 置信度清洗边界测试 (QR-11)
- 配置传递端到端测试 (QR-4)
