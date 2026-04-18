# 05 安全边界

## 1. When to use
- 修改串口、TCP/UDP、HTTP、文件 I/O、日志输出
- 修改 `run_standalone.py`、`plugin.py::get_standalone_routes()`
- 修改模型路径、zone / scenario 文件路径、训练/录制目录
- 新增依赖或扩大对外接口面

## 2. Inputs
- 代码 diff
- 新增依赖列表
- 配置变更内容
- `configs/default.yaml`、`standalone/configs/zone.yaml`、`configs/scenarios/*.json`

## 3. Outputs
- 安全合规结论
- 违规项与修复建议
- 需要补的测试或扫描命令

## 4. Hard Constraints
- 禁止在代码、配置、日志中硬编码密钥、令牌、密码
- 串口、IP、端口、模型路径、zone 路径必须来自配置或受控默认值，不能把设备地址写死在业务逻辑里
- `model_path`、`zone_config.path`、scenario 路径不得通过 `..` 跳出插件目录
- standalone 端点只能返回状态、结构化 JSON、MJPEG 或 JPEG bytes；不得暴露任意文件系统路径
- 不得把原始传感器二进制数据、认证信息、完整本机绝对路径直接写入日志或 HTTP 响应
- 新增网络依赖或新开放端口前，必须先确认 `PROJECT_CARD.md` 是否批准
- `run_standalone.py` 的 venv 切换逻辑属于启动安全边界，不要移除或后置
- 训练/录制目录只能放开发或仿真数据，禁止把生产采集数据直接提交到仓库

## 5. Algorithm / Logic Contract

### 串口 / 网络
- Camera / LiDAR / UWB / IMU 的连接参数走配置，不要把 `/dev/ttyUSB*`、测试 IP、测试端口写进算法逻辑
- LiDAR socket / serial 超时必须保留，避免 standalone 线程被永久阻塞
- `api_integration.py` 虽支持 WebSocket，但不是当前 standalone 默认 surface；不要在未经批准时扩大外网暴露面

### 文件与路径
- `configs/default.yaml` 只保存非敏感运行参数
- `standalone/configs/zone.yaml` 只保存区域布局，不保存凭据
- `configs/scenarios/*.json` 只保存仿真场景，不保存生产轨迹
- 所有新路径在落盘前都要做受控目录检查

### 日志与响应
- fallback 日志允许记录组件、原因、降级去向，但不要记录原始设备字节流
- snapshot / stream 响应只传图像 bytes，不回传本地文件路径
- config / events / tracking 接口返回结构化数据，不回传调试态系统信息

## 6. Validation Rules

```bash
# 可疑凭据扫描
rg -n "password|secret|api_key|token" . --glob '*.py' --glob '*.yaml' --glob '*.json' --glob '!tests/*'

# 非安全 YAML 读取
rg -n "yaml\\.load\\(" . --glob '*.py' --glob '!tests/*'

# 可疑路径拼接
rg -n "\\.\\./" plugin.py standalone adapters detection core configs

# 一键质量门禁
./scripts/run_quality_gate.sh
```

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| 模型/zone/scenario 路径可遍历 | 加载到非受控文件 | 做受控目录检查并补测试 |
| 启动 guard 被删 | 系统 Python 直接起服务缺依赖 | 恢复 venv 切换逻辑 |
| 路由返回调试路径或系统信息 | 信息泄露 | 只返回业务状态与图像 bytes |
| 设备地址硬编码 | 环境切换失败或泄露现场信息 | 回到配置字段 |
| 训练/录制数据误入仓库 | 数据合规风险 | 只保留仿真/开发样本 |

## 8. Required Tests
- `tests/test_camera_adapter.py`
- `tests/test_lidar_adapter.py`
- `tests/test_detection.py`
- `tests/test_config_updates.py`
- `tests/test_video_stream.py`
- `tests/test_api_routes.py`

