# 04 质量审计清单

## 1. When to use
- `/audit` 或提测前自查
- `/implement`、`/repair` 过程中提炼出新审计规则
- 修改 config/route/fallback/standalone 时确认没有引入结构性回归

## 2. Inputs
- 代码 diff
- `./scripts/run_quality_gate.sh` 输出
- 覆盖率报告与 targeted / regression 日志
- `.coveragerc`、`tests/`、`standalone/templates/indoor_fence.html`

## 3. Outputs
- `PASS / FAIL` 审计结论
- blocker / high-risk / debt 项列表
- 若发现新共性规则，回写到本文件

## 4. Hard Constraints
- 阻断项必须真实来源于已执行命令或已定位代码，不得“凭经验猜 PASS”
- 当前 blocking gate 以 `./scripts/run_quality_gate.sh` 为准，默认覆盖率门槛是 `70%`
- 变更触达的 active 模块不得新增 `except: pass`、裸 `print()`、未解释的 `TODO/FIXME`
- 变更若引入新的通用审计规则，必须在同次任务里回写本文件
- route / config / zone / scenario 变化必须在审计中显式检查契约同步

## 5. Algorithm / Logic Contract

### 审计维度

#### 1. 入口与回滚契约
- `manifest.json` 的 `entrypoint` / `plugin_class` 未漂移
- `plugin.py` 仍保留 `create_standalone()` 与 `Plugin = IndoorFencePlugin`
- `update_config()` 非法输入会回滚
- `update_zone_config(..., persist=False)` 行为未漂移

#### 2. 降级与可观测性
- Camera / LiDAR / YOLO / video stream 的 fallback 仍可用
- fallback 仍有结构化日志或可查询状态
- 无硬件 / 无模型 / 无 `cv2` 时仍有可测试的占位路径

#### 3. standalone surface
- `/api/indoor-fence/config`
- `/api/indoor-fence/zones`
- `/api/indoor-fence/events`
- `/api/indoor-fence/stream`
- `/api/indoor-fence/snapshot`
- `/api/indoor-fence/simulator/*`
- `/api/indoor-fence/tracking`
- 模板块名、前端元素 ID、视频流路径与后端保持一致

#### 4. 配置与资产一致性
- `configs/default.yaml`
- `standalone/configs/zone.yaml`
- `configs/scenarios/*.json`
- `manifest.json`
- `.coveragerc`

#### 5. Legacy 边界
- `detector.py`、`core/tracker_v3.py`、`core/enhanced_tracking.py` 目前仍属 legacy / 兼容面
- 这些文件不应成为新能力默认落点
- 若任务主动触碰 legacy 文件，审计必须说明是“债务清理”还是“兼容修复”

### 审计结论分级
- `BLOCKER`: 会破坏入口契约、质量门禁、主路由、配置回滚或核心 fallback
- `HIGH_RISK`: 现有测试未覆盖但高概率影响主链路或 standalone 可用性
- `DEBT`: 当前可交付但需要纳入 learning log / 后续清理

## 6. Validation Rules

```bash
# 一键质量闸门
./scripts/run_quality_gate.sh

# 如需缩小范围
./scripts/run_targeted_tests.sh <module>
./scripts/run_regression_tests.sh

# 架构检查
rg -n "from .*adapters|import .*adapters|from .*standalone|import .*standalone" core/

# route 合约
rg -n "/api/indoor-fence/(config|zones|events|stream|snapshot|tracking)" plugin.py
```

## 7. Failure Modes

| 故障 | 影响 | 处理 |
|------|------|------|
| 只审代码不审 route / config / assets | standalone 或配置热更新悄悄坏掉 | 审计结论必须覆盖契约面 |
| 把 warning 当成 PASS | 风险被吞掉 | 分级输出 blocker / high-risk / debt |
| 忽略 legacy 边界 | 在旧模块继续堆功能 | 明确标注落点并要求迁移说明 |
| 只看覆盖率总数 | 掩盖 `.coveragerc` 排除边界 | 结合 active 模块解释 |
| 发现共性问题但不回写 | 同类问题重复发生 | 同步更新本文件和 `07_learning_log.md` |

## 8. Required Tests
- `tests/test_config_updates.py`
- `tests/test_standalone.py`
- `tests/test_api_routes.py`
- `tests/test_video_stream.py`
- `tests/test_camera_adapter.py`
- `tests/test_lidar_adapter.py`
- `tests/test_detection.py`
- `tests/test_realtime_tracking.py`

## 9. indoor_fence 特有审计点
- 硬件或模型 fallback 后，模拟路径仍必须能产生有效数据或可视化帧，不能只是“连接成功但空输出”
- 模板块名、视频流元素和 MJPEG / snapshot 路由必须成套存在，否则容易出现“后端正常、前端全空白”的假象
- 主插件链路与 V3 演练链路并存；审计必须说明本次改动影响的是哪一条链路
- 3D Hungarian 关联、IMU/UWB 融合、scenario 重载是 indoor_fence 的领域重点，相关改动不能只跑通用 smoke test
