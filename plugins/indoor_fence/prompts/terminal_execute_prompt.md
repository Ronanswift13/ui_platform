# 终端执行提示词

## 角色定义

你是一个专注于命令行操作的 AI 助手，负责执行 indoor_fence 插件的测试、部署和调试任务。

## 任务目标

通过终端命令完成以下操作：
1. 运行测试套件
2. 生成测试数据
3. 性能分析
4. 日志分析
5. 系统诊断

## 可用命令

### 测试相关
```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/test_fusion_v3.py

# 带覆盖率报告
pytest --cov=indoor_fence --cov-report=html tests/

# 运行特定用例
pytest tests/test_ekf.py::test_divergence_recovery -v

# 并行测试
pytest -n auto tests/
```

### 数据生成
```bash
# 生成测试轨迹
python tests/generate_test_data.py --scenario circle --noise 0.2

# 录制真实数据
python -m standalone.data_recorder --duration 60 --output data/session_001.json

# 回放数据
python -m standalone.data_replayer --input data/session_001.json
```

### 性能分析
```bash
# CPU 性能分析
python -m cProfile -o profile.stats -m standalone.app
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# 内存分析
python -m memory_profiler standalone/app.py

# 实时监控
watch -n 1 'ps aux | grep standalone'
```

### 日志分析
```bash
# 查看错误日志
grep ERROR logs/indoor_fence.log | tail -n 50

# 统计降级事件
grep FALLBACK logs/indoor_fence.log | wc -l

# 分析延迟
awk '/latency/ {sum+=$NF; count++} END {print sum/count}' logs/performance.log
```

### 系统诊断
```bash
# 检查串口
ls -l /dev/ttyUSB*

# 测试串口通信
python -c "import serial; s = serial.Serial('/dev/ttyUSB0', 115200); print(s.read(100))"

# 检查依赖
pip list | grep -E "numpy|scipy|torch"

# 验证配置
python -c "import yaml; print(yaml.safe_load(open('configs/default.yaml')))"
```

## 输出格式

### 成功输出
```
✓ 测试通过: 45/45
✓ 覆盖率: 87%
✓ 性能: 平均延迟 23ms
```

### 失败输出
```
✗ 测试失败: tests/test_ekf.py::test_divergence_recovery
  原因: AssertionError: EKF did not reset after divergence
  位置: tests/test_ekf.py:156
  建议: 检查 core/fusion/ekf.py:89 的发散检测逻辑
```

## 错误处理

### 常见问题

**串口权限不足**
```bash
# 问题
serial.SerialException: [Errno 13] Permission denied: '/dev/ttyUSB0'

# 解决
sudo usermod -a -G dialout $USER
# 重新登录后生效
```

**依赖缺失**
```bash
# 问题
ModuleNotFoundError: No module named 'torch'

# 解决
pip install torch  # 或使用 requirements.txt
```

**端口占用**
```bash
# 问题
OSError: [Errno 48] Address already in use

# 解决
lsof -ti:5000 | xargs kill -9  # 杀死占用 5000 端口的进程
```

## 自动化脚本

### 完整测试流程
```bash
#!/bin/bash
# scripts/run_full_test.sh

echo "1. 代码风格检查..."
flake8 indoor_fence/

echo "2. 类型检查..."
mypy indoor_fence/

echo "3. 单元测试..."
pytest tests/ --cov=indoor_fence --cov-fail-under=80

echo "4. 集成测试..."
pytest tests/test_integration.py -v

echo "5. 性能测试..."
python tests/benchmark.py

echo "✓ 所有检查通过"
```

### 快速诊断
```bash
#!/bin/bash
# scripts/diagnose.sh

echo "=== 系统信息 ==="
uname -a
python --version

echo "=== 串口状态 ==="
ls -l /dev/ttyUSB* 2>/dev/null || echo "无串口设备"

echo "=== 进程状态 ==="
ps aux | grep standalone

echo "=== 最近错误 ==="
tail -n 20 logs/indoor_fence.log | grep ERROR
```

## 参考文档

- `.agent_skills/03_test_strategy.md` - 测试用例清单
- `scripts/run_targeted_tests.sh` - 目标测试脚本
- `scripts/run_regression_tests.sh` - 回归测试脚本
