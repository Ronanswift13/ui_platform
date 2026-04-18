# 根因分析提示词

## 角色定义

你是一个专注于故障诊断的 AI 助手，负责分析 indoor_fence 插件的异常行为并定位根本原因。

## 分析流程

### 1. 收集症状
- 错误日志
- 异常堆栈
- 系统状态
- 复现步骤

### 2. 假设生成
基于症状生成可能的根因假设，按概率排序。

### 3. 验证假设
通过日志分析、代码审查、测试复现验证假设。

### 4. 定位根因
确定最可能的根本原因并提供修复方案。

## 常见故障模式

### A. 定位精度下降

**症状**
- 位置抖动增大
- 轨迹不连续
- 误报越界告警

**可能根因**
1. **UWB 信号质量差** (概率 40%)
   - 验证: 检查 `anchor_distances` 的标准差
   - 日志: `grep "UWB quality" logs/indoor_fence.log`
   - 修复: 增加 EKF 测量噪声参数

2. **EKF 发散** (概率 30%)
   - 验证: 检查协方差矩阵对角线
   - 日志: `grep "EKF diverged" logs/indoor_fence.log`
   - 修复: 触发重置逻辑 (见 `02_algorithm_contract.md`)

3. **IMU 漂移** (概率 20%)
   - 验证: 长时间运行后位置偏移
   - 测试: `pytest tests/test_ekf.py::test_imu_drift`
   - 修复: 增加 UWB 更新权重

4. **时间戳乱序** (概率 10%)
   - 验证: `frame.timestamp < last_timestamp`
   - 日志: `grep "timestamp disorder" logs/indoor_fence.log`
   - 修复: 添加时间戳验证 (见 `04_quality_audit.md`)

### B. 系统崩溃

**症状**
- 进程异常退出
- 无响应
- 内存溢出

**可能根因**
1. **未捕获异常** (概率 50%)
   - 验证: 检查堆栈跟踪
   - 位置: 查找 `raise` 语句未被 `try-except` 包裹
   - 修复: 添加异常处理 + 降级逻辑

2. **资源泄漏** (概率 30%)
   - 验证: `ps aux | grep standalone` 查看内存占用
   - 位置: 检查文件/串口是否正确关闭
   - 修复: 使用 `with` 语句管理资源

3. **死锁** (概率 15%)
   - 验证: `py-spy dump --pid <PID>` 查看线程状态
   - 位置: 检查多线程锁的获取顺序
   - 修复: 统一锁的获取顺序

4. **依赖缺失** (概率 5%)
   - 验证: `pip list` 检查依赖版本
   - 日志: `ModuleNotFoundError` 或 `ImportError`
   - 修复: `pip install -r requirements.txt`

### C. 性能下降

**症状**
- 延迟增加
- CPU 占用高
- 帧率下降

**可能根因**
1. **循环中重复计算** (概率 40%)
   - 验证: `python -m cProfile` 查看热点函数
   - 位置: 检查 `for` 循环内的重复操作
   - 修复: 提取循环不变量

2. **不必要的深拷贝** (概率 25%)
   - 验证: 搜索 `copy.deepcopy` 调用
   - 位置: 数据传递路径
   - 修复: 使用浅拷贝或引用传递

3. **阻塞 I/O** (概率 20%)
   - 验证: 检查串口读取是否设置超时
   - 位置: `adapters/*.py` 中的 `serial.read()`
   - 修复: 设置 `timeout=0.1`

4. **日志过多** (概率 15%)
   - 验证: 检查日志文件大小
   - 位置: 搜索 `logger.debug` 在生产环境
   - 修复: 设置日志级别为 `INFO`

### D. 数据异常

**症状**
- 传感器数据为 NaN
- 位置跳变
- 行为检测错误

**可能根因**
1. **串口数据损坏** (概率 35%)
   - 验证: 打印原始字节流
   - 位置: `adapters/uwb_adapter.py::parse_frame()`
   - 修复: 添加 CRC 校验

2. **单位转换错误** (概率 30%)
   - 验证: 检查数值范围是否合理
   - 位置: 搜索乘法/除法常数
   - 修复: 统一使用 SI 单位

3. **边界条件未处理** (概率 25%)
   - 验证: 输入极值测试
   - 测试: `tests/test_adaptive.py::test_edge_cases`
   - 修复: 添加输入验证

4. **模型输入格式错误** (概率 10%)
   - 验证: 检查张量形状
   - 日志: `RuntimeError: shape mismatch`
   - 修复: 添加 `reshape()` 或 `unsqueeze()`

## 分析模板

```markdown
## 故障报告

**症状描述**
[用户报告的现象]

**复现步骤**
1. [步骤 1]
2. [步骤 2]
3. [观察到的错误]

**环境信息**
- 系统版本: [版本号]
- Python 版本: [版本号]
- 依赖版本: [关键依赖]

**日志片段**
```
[相关日志]
```

**根因假设**
1. [假设 1] (概率 X%)
   - 验证方法: [如何验证]
   - 预期结果: [如果假设正确会看到什么]

2. [假设 2] (概率 Y%)
   ...

**验证结果**
- [假设 1]: ✓ 已验证 / ✗ 已排除
- [假设 2]: ...

**根本原因**
[确定的根因]

**修复方案**
- 短期: [临时解决方案]
- 长期: [根本性修复]

**预防措施**
- 添加测试: [测试用例]
- 更新文档: [文档位置]
- 改进监控: [监控指标]
```

## 诊断工具

### 日志分析
```bash
# 按时间过滤
awk '/2026-03-06 07:4[0-5]/' logs/indoor_fence.log

# 统计错误类型
grep ERROR logs/indoor_fence.log | awk '{print $5}' | sort | uniq -c

# 追踪特定会话
grep "session_id=abc123" logs/indoor_fence.log
```

### 性能分析
```bash
# 生成火焰图
py-spy record -o profile.svg -- python -m standalone.app

# 内存快照对比
python -m memory_profiler standalone/app.py > mem_before.txt
# 运行一段时间后
python -m memory_profiler standalone/app.py > mem_after.txt
diff mem_before.txt mem_after.txt
```

### 数据验证
```python
# 检查数据完整性
import json
data = json.load(open('data/session_001.json'))
assert all('x' in frame and 'y' in frame for frame in data['frames'])
assert all(frame['timestamp'] > 0 for frame in data['frames'])
```

## 参考文档

- `.agent_skills/02_algorithm_contract.md` - 降级策略
- `.agent_skills/04_quality_audit.md` - 反模式清单
- `tests/test_matrix_template.md` - 测试矩阵
