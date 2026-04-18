# 二次设备动作监测 — 工程化收口说明

> 版本: 1.0.0-rc1 | 基准日期: 2026-04-07
> 对齐: 大理供电局二次设备智能管理方案

---

## 一、模块清单与状态

### 已落地(代码完整, 测试通过, 可独立运行)

| 模块 | 文件 | 说明 |
|------|------|------|
| 动作事件数据模型 | `platform_core/action_event_schema.py` | 30种ActionType, 16种ProtectionType, 13种SignalGroup, 14种SourceSystem, ~50条动作词映射 |
| 事件存储与查询 | `platform_core/action_event_store.py` | 内存存储, 支持多条件查询/时间索引/回调注册/LRU淘汰, 容量100k事件 |
| 动作链分析引擎 | `platform_core/action_sequence_analyzer.py` | 7条规则链(正常跳闸/重合闸失败/拒动/误动/控制回路异常/机构异常/信号抖动), 可配置阈值, 互斥门 |
| 根因分析服务 | `platform_core/root_cause_service.py` | 5阶段贝叶斯推理, 4道证据门禁(Gate A-D), 5类根因概率分布, 证据充分性评估, 人工复核项生成 |
| 设备关系服务 | `platform_core/device_correlation.py` | 厂站-间隔-一次设备-二次设备-信号点拓扑, 母线分段/保护区域/停电范围推导 |
| API路由 | `apps/action_event_api.py` | 8个端点: 事件CRUD, 动作链分析, 根因分析, 时间线, 设备关系, 故障归档 |
| 前端时间线页面 | `ui/templates/pages/action_timeline.html` | 筛选栏/统计卡片/时间线/跳闸范围/根因概率条/证据链/5种业务状态(空/错误/数据不足/未达门限/待复核) |
| 证据链组件 | `ui/templates/components/evidence_chain.html` | 7种节点类型(SOE/遥信变位/保护动作/断路器/一次设备/二次健康/人工复核), 筛选/导出/状态横幅 |
| 采集插件 | `plugins/action_event_monitoring/` | 插件生命周期管理, 协议适配(OPC UA/MQTT/Modbus/HTTP/IEC104/IEC61850), 自动触发分析 |
| 协议适配层 | `platform_core/data_import/protocol_adapters.py` | IEC104/IEC61850 适配器框架(含SOE订阅/GOOSE接收/SBO操作接口) |

### 待联调(接口已预留, 逻辑已实现, 需真实数据源验证)

| 模块 | 状态 | 依赖 |
|------|------|------|
| IEC 60870-5-104 适配器 | 框架完整, 当前为模拟模式 | 需安装 `iec104-python` 库 + 连接真实主站/RTU |
| IEC 61850 适配器 | 框架完整, 当前为模拟模式 | 需安装 `libiec61850-python` 库 + 连接真实IED |
| 保信子站数据接入 | 预留 SourceSystem.PROTECTION_INFO | 需对接保信子站文件/消息推送格式 |
| 故障录波关联 | schema 预留 `wave_record_id` 字段 | 需对接COMTRADE文件解析 |
| 历史故障关联 | root_cause_service 预留 `historical_faults` 输入 | 需接入OMS/PMS历史故障库 |

### 仅预留接口(未实现业务逻辑)

| 项目 | 预留位置 | 说明 |
|------|----------|------|
| 行波测距数据 | SourceSystem.WAVE_RANGING | 仅定义枚举, 无解析逻辑 |
| PT并列装置 | SourceSystem.PT_PARALLEL | 仅定义枚举 |
| 电能质量 | SignalGroup.POWER_QUALITY | 仅定义信号分组 |
| 一次设备巡视证据自动接入 | PrimaryDeviceEvidence.source_plugin | 需与 transformer_inspection/switch_inspection 等插件打通事件总线 |
| 二次设备健康指数自动接入 | SecondaryDeviceEvidence.health_index | 需与 device_monitoring 插件打通 |

---

## 二、验证层收口

### 测试覆盖

| 测试文件 | 用例数 | 覆盖范围 |
|----------|--------|----------|
| `tests/test_acceptance_closure.py` | 5 | 5类核心场景端到端(事件→分析→根因) |
| `tests/test_action_event_verification.py` | 14 | 7种动作链 + 互斥门 + 跨间隔隔离 + 保护非抖动 + 4种根因门禁 |
| `tests/test_iec_adapter_integration.py` | 6 | IEC104/61850实例化 + SOE/Report/GOOSE回调 + 全链路 |

### 5类核心场景预期结果

| # | 场景 | chain_type | confidence | is_real_trip | root_cause | sufficiency |
|---|------|-----------|-----------|-------------|------------|-------------|
| 1 | 正常跳闸(保护启动→出口→断路器分闸) | normal_trip | ≥0.90 | True | primary_equipment_fault | sufficient |
| 2 | 重合闸失败(跳闸→重合→再跳) | recloser_fail | ≥0.90 | True | primary_equipment_fault (≤0.55*) | partial |
| 3 | 控制回路异常(断线信号) | control_loop_abnormal | ≥0.75 | False | control_loop_issue | partial |
| 4 | 信号抖动(3秒4次变位) | signal_jitter | ≥0.50 | False | signal_anomaly | partial |
| 5 | 疑似误动(断路器分闸无保护前兆) | false_action | ≤0.70 | False | secondary_equipment_fault | partial |

> *Gate A: 无一次设备巡视证据时, PRIMARY后验概率上限55%(归一化后约47%)

---

## 三、边界层收口

### 当前版本能做什么

1. **接收标准化动作事件** — 通过API写入, 支持单条/批量, 自动归一化动作类型(50+关键词映射)
2. **7种动作链规则检测** — 正常跳闸/重合闸失败/拒动/误动/控制回路异常/机构异常/信号抖动, 含可配置时间窗阈值
3. **跳闸范围自动判定** — 基于设备拓扑推导一次/二次检修范围、停电范围、保护动作装置、跳开断路器
4. **根因概率分布推理** — 5类根因(一次设备故障/二次设备故障/控制回路/信号异常/外部原因), 4道证据门禁防止过拟合
5. **证据充分性评估** — 自动判定 sufficient/partial/insufficient, 生成证据缺口列表和人工复核项
6. **前端业务核查** — 厂站/间隔/设备/时间范围/动作序列/告警组名/跳闸范围/根因结果/证据链, 5种业务状态
7. **模拟全链路验证** — IEC104/61850模拟模式下可完成 采集→存储→分析→根因→展示 全流程

### 当前版本不能做什么

1. **不能自动闭环故障处置** — 根因分析结果仅为建议, 所有故障归档和检修工单必须人工确认
2. **不能替代人工复核** — evidence_sufficiency=partial/insufficient 时, 系统明确要求人工复核, 不允许自动结案
3. **不能接入真实站端协议** — IEC104/IEC61850适配器当前为模拟模式, 需安装对应Python库并配置真实连接参数
4. **不能自动获取一次设备巡视证据** — 需手动调用API传入PrimaryDeviceEvidence, 或后续打通巡视插件事件总线
5. **不能解析故障录波文件** — 预留了wave_record_id字段, 但无COMTRADE解析器
6. **不能查询历史故障关联** — root_cause_service支持historical_faults输入, 但无OMS/PMS数据源
7. **不能做实时流式处理** — 当前为请求-响应模式, 不支持CEP/流式窗口计算

### 需要真实数据联调的环节

| 环节 | 需要什么 | 联调要点 |
|------|----------|----------|
| IEC 60870-5-104 | 主站/RTU IP:Port + 公共地址 + IOA点表 | SOE带时标事件解析、总召唤、对时 |
| IEC 61850 | IED连接参数 + Dataset/RCB引用 + GOOSE AppID | MMS报告订阅、GOOSE帧解析、SBO操作 |
| 保信子站 | 文件推送格式/消息协议 + 告警信号映射表 | 保护动作报文→ActionEvent转换 |
| 故障录波 | COMTRADE文件路径/推送接口 | 录波文件→故障电流/故障类型/故障相别提取 |
| OMS/PMS | 历史故障查询API | 同类设备/同类故障历史记录关联 |
| 设备台账 | 厂站-间隔-设备-信号点映射表(Excel/JSON) | 加载到DeviceCorrelationService |

### 必须人工复核, 不允许自动闭环的场景

1. **根因置信度 < 60%** — 页面显示"未达判定门限"横幅, 结论仅供参考
2. **证据充分性 = partial/insufficient** — 缺少关键证据(一次巡视/录波/故障电流), 不能确定结论
3. **疑似误动(false_action)** — 置信度本身较低(≤0.70), 需人工现场确认断路器状态和保护装置定值
4. **所有故障归档** — 归档状态默认 draft, 必须人工审核后改为 confirmed
5. **停电范围判定** — 拓扑推导依赖设备台账完整性, 台账不全时停电范围可能遗漏

---

## 四、复用与新增说明

### 复用的现有模块

| 模块 | 复用方式 |
|------|----------|
| `platform_core/data_import/protocol_adapters.py` | 直接复用OPC UA/MQTT/Modbus/HTTP适配器; 在其基础上扩展IEC104/IEC61850适配器 |
| `platform_core/schema/indoor_fence_events.py` | 参考其Evidence/EvidenceItem模式设计EvidenceNode |
| `ui/templates/base.html` | 继承主布局模板, 保持深蓝色大屏主题一致性 |
| `apps/ui_server.py` | 在现有路由注册机制中集成action_event_api和action-timeline页面 |
| FastAPI路由风格 | 参考 `apps/api_server.py` 的 APIRouter + Pydantic 请求/响应模型模式 |

### 新增的模块

| 模块 | 为什么新增 |
|------|-----------|
| `platform_core/action_event_schema.py` | 现有schema无二次设备动作事件模型, 需对齐大理供电局业务字段(厂站/间隔/设备/信号/动作类型/电气属性) |
| `platform_core/action_event_store.py` | 现有存储无动作事件时间索引和trace_id聚合能力 |
| `platform_core/action_sequence_analyzer.py` | 现有分析引擎面向视觉检测, 不支持时序动作链规则匹配 |
| `platform_core/root_cause_service.py` | 现有multimodal_fusion的贝叶斯逻辑面向缺陷融合, 不支持4道证据门禁和5类根因概率分布 |
| `platform_core/device_correlation.py` | 现有device_adapter仅管理设备连接, 不支持厂站-间隔-设备-信号点拓扑查询和停电范围推导 |
| `plugins/action_event_monitoring/` | 需要独立的采集插件管理协议订阅、事件归一化、自动触发分析 |

### 为什么这样设计

1. **分层解耦** — schema/store/analyzer/root_cause/correlation 各司其职, 任一层可独立替换(如store从内存切换到PostgreSQL)
2. **证据门禁** — 防止在证据不足时给出高置信度结论, 这是大理供电局方案明确要求的(不能自动闭环)
3. **可配置阈值** — 所有时间窗、置信度门限都通过AnalyzerConfig暴露, 不同变电站可按实际调整
4. **动作词映射** — 各厂站、各厂商的动作描述文本不统一, 通过配置化关键词映射解决, 不硬编码
5. **trace_id贯穿** — 从采集→分析→根因→归档全程同一trace_id, 支持故障全过程追溯

### 如何继续扩展到真实站端数据

1. **准备设备台账** — 将厂站-间隔-设备-信号点映射表导入DeviceCorrelationService(支持JSON/YAML)
2. **安装协议库** — `pip install iec104-python` 或 `pip install libiec61850-python`
3. **配置协议参数** — 修改 `plugins/action_event_monitoring/configs/default.yaml` 中的protocol段
4. **配置信号点订阅** — 在subscriptions中填写IOA地址/DA引用, 绑定signal_group和action_type_hint
5. **补充动作词映射** — 在custom_action_keywords中添加本站特有的动作描述文本
6. **接入一次侧巡视** — 在巡视插件检测到缺陷时, 调用 `/api/root-cause/analyze` 附带primary_evidence
7. **接入二次设备健康** — 在device_monitoring检测到异常时, 附带secondary_evidence

---

## 五、文件变更清单(本次收口)

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `plugins/action_event_monitoring/plugin.py` | 修改 | 补齐AnalyzerConfig全部阈值从YAML加载, 兼容旧配置 |
| `plugins/action_event_monitoring/configs/default.yaml` | 修改 | 暴露全部8个分析规则阈值(原仅1个) |
| `ui/templates/components/evidence_chain.html` | 重写 | 从多模态巡视语义改造为动作事件域7种节点类型 |
| `ui/templates/pages/action_timeline.html` | 修改 | 新增3种业务状态(数据不足/根因未达门限/人工复核待确认) |
| `tests/test_acceptance_closure.py` | 新增 | 5类核心场景收口验收测试 |
| `docs/action_event_closure_notes.md` | 新增 | 本文档 |
