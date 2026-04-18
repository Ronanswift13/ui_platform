# 测试矩阵模板

## 测试维度

### 1. 功能维度
- 目标检测 (YOLO)
- 目标跟踪 (ByteTrack)
- 热成像验证
- 事件生成
- 驱离控制
- 统计分析

### 2. 质量维度
- 正常场景
- 边界条件
- 异常处理
- 性能压力
- 降级路径

### 3. 环境维度
- 模型可用
- 模型缺失
- 热成像可用
- 热成像缺失
- 配置变化

## 测试矩阵

| 功能模块 | 测试场景 | 输入条件 | 期望输出 | 测试文件 | 优先级 |
|---------|---------|---------|---------|---------|--------|
| **目标检测** |
| ONNX 加载 | 正常启动 | 有效模型 | 加载成功 | test_onnx_inference.py::test_load | P0 |
| 单帧检测 | 正常图像 | 640x480 BGR | 检测列表 | test_onnx_inference.py::test_detect | P0 |
| 空帧检测 | 全黑图像 | 640x480 黑 | 空列表 | test_onnx_inference.py::test_empty | P0 |
| 模型缺失 | 文件不存在 | 无模型 | 优雅降级 | test_onnx_inference.py::test_fallback | P0 |
| 大图像 | 4096x4096 | 超大图 | 自动缩放 | test_onnx_inference.py::test_large | P1 |
| **目标跟踪** |
| 新目标 | 首次出现 | 单检测 | 分配 ID | test_tracker.py::test_new | P0 |
| 目标持续 | 连续帧 | 多帧检测 | 保持 ID | test_tracker.py::test_track | P0 |
| 目标消失 | 5帧无检测 | 空检测 | 删除轨迹 | test_tracker.py::test_lost | P1 |
| 多目标 | 3个目标 | 多检测 | 各自 ID | test_tracker.py::test_multi | P1 |
| **热验证** |
| 正常验证 | 温差 5°C | 热成像帧 | 验证通过 | test_thermal.py::test_valid | P0 |
| 无温差 | 温差 0°C | 热成像帧 | 验证失败 | test_thermal.py::test_invalid | P0 |
| 临界温差 | 温差 2°C | 热成像帧 | 验证通过 | test_thermal.py::test_boundary | P1 |
| 热成像缺失 | None | 无热成像 | 跳过验证 | test_thermal.py::test_skip | P0 |
| **事件生成** |
| 入侵事件 | 检测到鼠 | 1个检测 | DETECTED | test_event_schema_contract.py::test_intrusion | P0 |
| 清除事件 | 无检测 | 空列表 | CLEARED | test_event_schema_contract.py::test_cleared | P0 |
| 风险等级 | 检测到蛇 | 蛇检测 | CRITICAL | test_event_schema_contract.py::test_risk | P0 |
| 多类型聚合 | 鼠+蛇 | 多检测 | 聚合建议 | test_event_schema_contract.py::test_multi | P1 |
| **插件接口** |
| 初始化 | 正常启动 | 配置文件 | 成功 | test_plugin.py::test_init | P0 |
| 处理帧 | 正常图像 | BGR 帧 | 事件 | test_plugin.py::test_process | P0 |
| 配置更新 | 运行时 | 新配置 | 生效 | test_plugin.py::test_config | P1 |
| **独立运行** |
| Web 启动 | 正常 | 无 | 服务运行 | test_standalone.py::test_start | P1 |
| API 调用 | 正常 | 请求 | 响应 | test_standalone.py::test_api | P1 |

## 优先级定义

- **P0**: 核心功能，必须通过
- **P1**: 重要功能，应该通过
- **P2**: 增强功能，可选通过

## 测试覆盖率目标

| 模块 | 目标覆盖率 | 当前覆盖率 |
|------|-----------|-----------|
| core/detector.py | 80% | - |
| core/tracker.py | 80% | - |
| core/event_schema.py | 90% | - |
| core/thermal_validator.py | 80% | - |
| core/onnx_inference.py | 80% | - |
| standalone/ | 60% | - |
| **总体** | **70%** | - |

## 执行命令

```bash
# 运行所有 P0 测试
pytest tests/ -m "priority_p0"

# 运行特定模块
./scripts/run_targeted_tests.sh detector

# 运行质量闸门
./scripts/run_quality_gate.sh

# 生成覆盖率报告
pytest --cov=. --cov-report=html tests/
```

## 测试报告模板

```markdown
## 测试执行报告

**日期**: 2026-03-19
**版本**: v1.0
**执行人**: [姓名]

### 测试结果

| 优先级 | 总数 | 通过 | 失败 | 跳过 |
|--------|------|------|------|------|
| P0 | 12 | 12 | 0 | 0 |
| P1 | 8 | 7 | 1 | 0 |
| P2 | 3 | 2 | 0 | 1 |

### 失败用例

1. **test_tracker.py::test_lost**
   - 原因: 超时阈值配置错误
   - 影响: 低
   - 修复: 已创建 issue #xxx

### 覆盖率

- 总体覆盖率: 75%
- 未覆盖模块: standalone/app.py (55%)

### 建议

- 增加 standalone 模块的单元测试
```
