# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. 层级方向固定：接口层 -> 算法层 -> 配置层，禁止反向依赖。
2. 算法层纯业务：算法层不得依赖 SDK schema，不得直接拼装 `RecognitionResult`。
3. `standalone/` 隔离：仅做运行与展示，不承载读数决策。
4. 阈值单一来源：推理阈值、fallback 次数、预处理参数、LED 阈值必须来自配置；静态量程注册表必须被明示为契约的一部分。
5. 禁止循环依赖：`plugin.py`、`detector_enhanced.py`、`standalone/*` 之间不得形成循环 import。

## 2. 本项目差异规则（meter_reading）

### 2.1 依赖方向（必须满足）

```text
plugin.py ----------------> detector_enhanced.py
plugin.py ----------------> darkbreaker_sdk.*
standalone/* -------------> plugin.py / standalone services
tests/* ------------------> plugin.py / detector_enhanced.py

detector_enhanced.py -X-> darkbreaker_sdk.*
detector_enhanced.py -X-> standalone.*
```

### 2.2 模块职责（单一真相）

1. `plugin.py` 负责：ROI 提取、上下文透传、结果组装、告警生成、健康检查、UI/standalone 路由注册。
2. `detector_enhanced.py` 负责：输入校验、预处理、模拟表/数字表/LED 三条链路、置信度清洗、`reload_config()`。
3. `configs/default.yaml` 负责：`inference / fallback / preprocessing / pointer_detection / led_detection / performance` 等运行参数。
4. `detector_enhanced.py::METER_RANGES` 负责：当前基线模拟表量程注册表。任何量程或单位修改，必须同步 `02_algorithm_contract.md` 与 `tests/test_analog_meter.py`。
5. `standalone/` 负责：视频流、场景模拟、页面展示；禁止把业务判定从算法层搬进 Web 层。

### 2.3 架构不变量

1. `plugin.py` 不实现 OCR/Hough/HSV 判定细节。
2. `detector_enhanced.py` 不直接接触 `ROI`、`RecognitionResult`、`Alarm` 等 SDK 对象。
3. `ReadingStatus` 只能保持三态：`SUCCESS / FAILED / NEED_MANUAL_REVIEW`。
4. 模拟表三级降级链不可删除、不可跳级、不可隐藏 `fallback_level`。
5. 缺失量程、OCR 非法串、HSV 不可分离等场景必须返回可解释的失败或复核结果，不能用“看似正常”的默认值掩盖。

## 3. 强制反模式拦截

1. 禁止在生产模块新增 `print()`。
2. 禁止裸 `except:` 或 `except Exception: pass`。
3. 禁止在 detector 层写磁盘原始图像。
4. 禁止在 `plugin.py` 中新增算法分支，绕过 `detector_enhanced.py`。
5. 禁止修改 `manifest.json` 核心字段而不先人工确认。

## 4. 可执行架构校验

```bash
# A. detector 不得依赖 SDK/standalone
rg -n "darkbreaker_sdk|standalone" detector_enhanced.py

# B. 生产模块无 print()
rg -n "\bprint\(" plugin.py detector_enhanced.py

# C. 裸 except / silent pass
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py detector_enhanced.py

# D. 状态机与 metadata 锚点仍在
rg -n "ReadingStatus|fallback_level|timestamp_ms|need_manual_review" plugin.py detector_enhanced.py
```

## 5. 阻断条件

任一条件命中即阻断：

1. `detector_enhanced.py` 出现 SDK 或 `standalone` 依赖。
2. 生产模块出现 `print()` 或裸 `except`。
3. 模拟表降级链、三态状态集、metadata 必填字段被破坏。
4. 涉及量程、告警口径、对外接口的改动未同步契约和测试。
