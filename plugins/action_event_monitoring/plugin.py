# -*- coding: utf-8 -*-
"""
二次设备动作事件监测插件
========================

功能:
1. 通过 OPC UA / MQTT / Modbus / HTTP 协议订阅二次设备动作/变位/SOE信号
2. 将原始信号归一化为 ActionEvent
3. 写入 ActionEventStore
4. 满足条件时自动触发动作链分析和根因分析
5. 分析结果通过 AlarmManager 推送

复用:
- platform_core/data_import/protocol_adapters.py (协议接入)
- platform_core/alarm_manager.py (告警推送)
- platform_core/device_adapter/base.py (设备状态管理)

遵循现有插件架构: 继承 BasePlugin 模式(init/process/shutdown)
"""

from __future__ import annotations
import logging
import time
import threading
import json
import hashlib
import importlib.util
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse as StarletteJSONResponse

from darkbreaker_sdk.interfaces import HealthStatus
from plugins._sensor_contract import (
    build_common_metadata,
    build_time_window,
    build_unified_temporal_output,
    build_virtual_result,
    clamp_confidence,
)


def _load_platform_core_module(module_name: str):
    """Load a platform_core submodule without executing platform_core.__init__."""
    root = Path(__file__).resolve().parents[2]
    package_name = "platform_core"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root / package_name)]
        sys.modules[package_name] = package

    full_name = f"{package_name}.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    module_path = root / package_name / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {full_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    setattr(package, module_name, module)
    spec.loader.exec_module(module)
    return module


_action_event_schema = _load_platform_core_module("action_event_schema")
_device_correlation = _load_platform_core_module("device_correlation")
_action_sequence_analyzer = _load_platform_core_module("action_sequence_analyzer")
_root_cause_service = _load_platform_core_module("root_cause_service")
_action_event_store = _load_platform_core_module("action_event_store")

ActionEvent = _action_event_schema.ActionEvent
ActionType = _action_event_schema.ActionType
SignalGroup = _action_event_schema.SignalGroup
SourceSystem = _action_event_schema.SourceSystem
SeverityHint = _action_event_schema.SeverityHint
normalize_action_type = _action_event_schema.normalize_action_type
normalize_protection_type = _action_event_schema.normalize_protection_type
ACTION_KEYWORD_MAP = _action_event_schema.ACTION_KEYWORD_MAP
PROTECTION_TYPE_KEYWORD_MAP = _action_event_schema.PROTECTION_TYPE_KEYWORD_MAP

ActionEventStore = _action_event_store.ActionEventStore
ActionSequenceAnalyzer = _action_sequence_analyzer.ActionSequenceAnalyzer
AnalyzerConfig = _action_sequence_analyzer.AnalyzerConfig
RootCauseService = _root_cause_service.RootCauseService
DeviceCorrelationService = _device_correlation.DeviceCorrelationService

logger = logging.getLogger(__name__)


class ActionEventMonitoringPlugin:
    """
    二次设备动作事件监测插件

    生命周期: init() -> start() -> [process()] -> stop() -> shutdown()
    """

    PLUGIN_ID = "action_event_monitoring"
    PLUGIN_NAME = "二次设备动作事件监测"
    VERSION = "1.0.0"

    def __init__(self, manifest=None, plugin_dir=None, config: Optional[Dict] = None):
        self.manifest = manifest
        self.plugin_dir = plugin_dir if plugin_dir else Path(__file__).parent
        self._initialized = False
        self._running = False
        self._config: Dict = config or {}

        # 核心依赖
        self.event_store: Optional[ActionEventStore] = None
        self.analyzer: Optional[ActionSequenceAnalyzer] = None
        self.root_cause_service: Optional[RootCauseService] = None
        self.correlation_service: Optional[DeviceCorrelationService] = None
        self.alarm_manager = None

        # 协议适配器
        self._protocol_adapter = None
        self._subscriptions: List[Dict] = []

        # 后台线程
        self._poll_thread: Optional[threading.Thread] = None

        # 统计
        self._stats = {
            "events_received": 0,
            "events_stored": 0,
            "analyses_triggered": 0,
            "errors": 0,
        }

    @property
    def id(self) -> str:
        return self.manifest.id if self.manifest and hasattr(self.manifest, "id") else self.PLUGIN_ID

    @property
    def name(self) -> str:
        return self.manifest.name if self.manifest and hasattr(self.manifest, "name") else self.PLUGIN_NAME

    @property
    def version(self) -> str:
        return self.manifest.version if self.manifest and hasattr(self.manifest, "version") else self.VERSION

    @property
    def status(self) -> str:
        if self._running:
            return "running"
        return "ready" if self._initialized else "unloaded"

    @property
    def code_hash(self) -> str:
        h = hashlib.sha256()
        plugin_file = Path(self.plugin_dir) / "plugin.py"
        if plugin_file.exists():
            h.update(plugin_file.read_bytes())
        return f"sha256:{h.hexdigest()[:12]}"

    def init(self, config: Optional[Dict] = None) -> bool:
        """
        初始化插件

        Args:
            config: 插件配置字典(或从configs/default.yaml加载)
        """
        try:
            self._config = config or {}

            # 初始化事件存储
            self.event_store = ActionEventStore()

            # 初始化设备关系服务
            self.correlation_service = DeviceCorrelationService()

            # 初始化分析器 — 从YAML加载所有可配置阈值
            analyzer_config = AnalyzerConfig()
            analysis_cfg = self._config.get('analysis', {})
            # 逐字段映射: YAML key 与 AnalyzerConfig 字段一一对应
            for cfg_key in (
                'trip_chain_window_s', 'recloser_window_s', 'refuse_action_timeout_s',
                'false_action_window_s', 'jitter_window_s', 'jitter_min_count',
                'control_loop_window_s', 'mechanism_window_s',
            ):
                if cfg_key in analysis_cfg:
                    setattr(analyzer_config, cfg_key, analysis_cfg[cfg_key])
            # 兼容旧配置: analysis_window_s → trip_chain_window_s
            if 'analysis_window_s' in analysis_cfg and 'trip_chain_window_s' not in analysis_cfg:
                analyzer_config.trip_chain_window_s = analysis_cfg['analysis_window_s']
            self.analyzer = ActionSequenceAnalyzer(
                config=analyzer_config,
                correlation_service=self.correlation_service,
            )

            # 初始化根因分析
            self.root_cause_service = RootCauseService(
                correlation_service=self.correlation_service,
                alarm_manager=self.alarm_manager,
            )

            # 加载自定义动作词
            custom_keywords = self._config.get('custom_action_keywords', {})
            if custom_keywords:
                ACTION_KEYWORD_MAP.update(custom_keywords)

            # 加载设备拓扑配置
            self._load_topology()

            self._initialized = True
            logger.info(f"[{self.PLUGIN_NAME}] 初始化成功")
            return True

        except Exception as e:
            logger.error(f"[{self.PLUGIN_NAME}] 初始化失败: {e}")
            return False

    def start(self) -> bool:
        """启动采集"""
        if not self._initialized:
            logger.error("插件未初始化")
            return False

        self._running = True

        # 初始化协议连接(如配置了)
        protocol_cfg = self._config.get('protocol', {})
        if protocol_cfg.get('type'):
            self._init_protocol(protocol_cfg)

        logger.info(f"[{self.PLUGIN_NAME}] 已启动")
        return True

    def stop(self):
        """停止采集"""
        self._running = False
        if self._protocol_adapter:
            try:
                self._protocol_adapter.disconnect()
            except Exception as exc:
                logger.debug("协议断开时发生可降级异常: %s", exc)
        logger.info(f"[{self.PLUGIN_NAME}] 已停止")

    def shutdown(self):
        """关闭插件"""
        self.stop()
        self._initialized = False

    # =========================================================================
    # 核心处理: 接收原始信号 -> 归一化 -> 存储 -> 触发分析
    # =========================================================================

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理输入数据(手动推送模式)

        input_data 可以是:
        1. 单条原始信号: {"signal_id": ..., "value": ..., "timestamp": ...}
        2. 预构建的 ActionEvent dict
        3. 批量事件: {"events": [...]}
        """
        try:
            input_data = input_data or {}
            if not self._initialized or not self.event_store:
                return self._contract_error(input_data, "插件未初始化")

            raw_events = self._extract_event_inputs(input_data)
            if not raw_events:
                return self._contract_error(input_data, "缺少动作事件或信号变位数据")

            events = [self._normalize_event(e) for e in raw_events]

            stored_ids = []
            for event in events:
                eid = self.event_store.add_event(event)
                stored_ids.append(eid)
                self._stats["events_stored"] += 1

            # 自动触发分析
            analysis_cfg = self._config.get('analysis', {})
            should_analyze = False
            for event in events:
                if analysis_cfg.get('auto_analyze_on_protection') and \
                   event.action_type in [ActionType.PROTECTION_START.value, ActionType.PROTECTION_TRIP.value]:
                    should_analyze = True
                if analysis_cfg.get('auto_analyze_on_breaker') and \
                   event.action_type in [ActionType.BREAKER_OPEN.value, ActionType.BREAKER_CLOSE.value]:
                    should_analyze = True

            analysis_result = None
            if should_analyze and events:
                analysis_result = self._trigger_analysis(events[0].trace_id)

            status = self._contract_status(events)
            label = self._contract_label(status, events)
            value = {
                "event_count": len(events),
                "stored_event_ids": stored_ids,
                "action_types": [event.action_type for event in events],
            }
            confidence = self._rule_confidence()
            metadata = self._build_contract_metadata(events)
            device_id = input_data.get("device_id") or events[0].secondary_device_id or events[0].primary_device_id or events[0].signal_id or "action_event_channel"
            time_window = build_time_window(
                input_data,
                window_size=metadata.get("window_size"),
                sample_interval=metadata.get("sample_interval"),
            )
            temporal_output = self._build_temporal_output(
                input_data=input_data,
                events=events,
                status=status,
                label=label,
                confidence=confidence,
                analysis_result=analysis_result,
                stored_ids=stored_ids,
                time_window=time_window,
                metadata=metadata,
            )
            virtual_result = build_virtual_result(
                payload=input_data,
                plugin_id=self.id,
                plugin_version=self.version,
                code_hash=self.code_hash,
                device_id=device_id,
                roi_id=input_data.get("roi_id") or events[0].signal_id or device_id,
                label=label,
                value=value,
                confidence=confidence,
                metadata=metadata,
                component_id="action_event_signal",
            )

            return {
                "success": True,
                "status": status,
                "label": label,
                "value": value,
                "confidence": clamp_confidence(confidence),
                "metadata": metadata,
                "results": [virtual_result],
                **temporal_output,
                "stored_event_ids": stored_ids,
                "analysis_triggered": should_analyze,
                "analysis_result": analysis_result,
            }

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"处理异常: {e}")
            return self._contract_error(input_data if isinstance(input_data, dict) else {}, str(e))

    def _extract_event_inputs(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "events" in input_data:
            raw = input_data.get("events") or []
            return raw if isinstance(raw, list) else [raw]
        if "signal_changes" in input_data or "state_change_events" in input_data:
            raw = input_data.get("signal_changes")
            if raw is None:
                raw = input_data.get("state_change_events") or []
            if isinstance(raw, dict):
                raw = [raw]
            events = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                event = dict(item)
                event.setdefault("timestamp", input_data.get("timestamp"))
                event.setdefault("source_ts", input_data.get("timestamp"))
                event.setdefault("secondary_device_id", input_data.get("device_id", ""))
                event.setdefault("signal_id", item.get("signal_id") or item.get("channel_id") or input_data.get("device_id", ""))
                event.setdefault("action_desc", item.get("action_desc") or item.get("description") or str(item.get("value_after", item.get("value", ""))))
                events.append(event)
            return events
        if "protocol_ingested_data" in input_data:
            raw = input_data.get("protocol_ingested_data") or []
            if isinstance(raw, dict):
                if "events" in raw:
                    raw = raw.get("events") or []
                elif "entries" in raw:
                    raw = raw.get("entries") or []
                else:
                    raw = [raw]
            return [item for item in raw if isinstance(item, dict)]
        signal_fields = {"signal_id", "action_type", "action_desc", "value", "value_after", "event_id"}
        if signal_fields.intersection(input_data.keys()):
            return [input_data]
        return []

    def _normalize_event(self, data: Dict[str, Any]) -> ActionEvent:
        """将原始数据归一化为 ActionEvent"""
        self._stats["events_received"] += 1

        # 如果已经是完整的 ActionEvent dict
        if "event_id" in data and "action_type" in data:
            event = ActionEvent.from_dict(data)
            # 补充推导字段
            if not event.protection_type and event.action_desc:
                event.protection_type = normalize_protection_type(event.action_desc)
            return event

        # 从原始信号构建
        action_desc = data.get("action_desc", data.get("description", ""))
        action_type = data.get("action_type", normalize_action_type(action_desc))
        protection_type = data.get("protection_type", normalize_protection_type(action_desc))

        # 通过 correlation 补充设备信息
        voltage_level = data.get("voltage_level", "")
        cabinet_id = data.get("cabinet_id", "")
        if self.correlation_service and not voltage_level:
            sec_id = data.get("secondary_device_id", "")
            sig_id = data.get("signal_id", "")
            if sec_id or sig_id:
                cr = self.correlation_service.correlate_by_signal(sec_id, sig_id)
                if cr.bay and not voltage_level:
                    voltage_level = cr.bay.voltage_level
                if cr.secondary_device and not cabinet_id:
                    cabinet_id = cr.secondary_device.cabinet_id

        # 数值型 value 解析
        value_numeric = data.get("value_numeric")
        fault_current_ka = data.get("fault_current_ka")
        if value_numeric is None:
            raw_val = data.get("value_after", data.get("value"))
            if raw_val is not None:
                try:
                    value_numeric = float(raw_val)
                except (ValueError, TypeError):
                    pass

        return ActionEvent(
            trace_id=data.get("trace_id", ""),
            station_id=data.get("station_id", ""),
            station_name=data.get("station_name", ""),
            bay_id=data.get("bay_id", ""),
            bay_name=data.get("bay_name", ""),
            primary_device_id=data.get("primary_device_id", ""),
            primary_device_type=data.get("primary_device_type", ""),
            secondary_device_id=data.get("secondary_device_id", ""),
            secondary_device_type=data.get("secondary_device_type", ""),
            signal_id=data.get("signal_id", ""),
            signal_name=data.get("signal_name", ""),
            signal_group=data.get("signal_group", SignalGroup.OTHER.value),
            action_type=action_type,
            action_desc=action_desc,
            protection_type=protection_type,
            value_before=data.get("value_before"),
            value_after=data.get("value_after"),
            value_numeric=value_numeric,
            voltage_level=voltage_level,
            phase=data.get("phase", ""),
            fault_current_ka=fault_current_ka,
            wave_record_id=data.get("wave_record_id"),
            cabinet_id=cabinet_id,
            seq_group_id=data.get("seq_group_id", ""),
            source_ts=data.get("source_ts", data.get("timestamp", datetime.now().isoformat())),
            source_system=data.get("source_system", SourceSystem.SIMULATED.value),
            seq_no=data.get("seq_no", 0),
            severity_hint=data.get("severity_hint", SeverityHint.INFO.value),
            raw_payload=data.get("raw_payload", data),
        )

    def _load_topology(self) -> None:
        """加载设备拓扑配置(YAML/JSON)"""
        topology_dir = Path(__file__).parent / "configs" / "topology"
        if not topology_dir.exists():
            logger.info(f"[{self.PLUGIN_NAME}] 拓扑目录不存在: {topology_dir}, 跳过")
            return

        loaded = 0
        for f in sorted(topology_dir.iterdir()):
            if f.suffix in ('.yaml', '.yml'):
                try:
                    self.correlation_service.load_from_yaml(str(f))
                    loaded += 1
                except Exception as e:
                    logger.error(f"加载拓扑 {f.name} 失败: {e}")
            elif f.suffix == '.json':
                try:
                    self.correlation_service.load_from_json(str(f))
                    loaded += 1
                except Exception as e:
                    logger.error(f"加载拓扑 {f.name} 失败: {e}")

        stats = self.correlation_service.get_statistics()
        logger.info(
            f"[{self.PLUGIN_NAME}] 加载 {loaded} 个拓扑文件, "
            f"共 {stats['stations']} 厂站, {stats['bays']} 间隔, "
            f"{stats['primary_devices']} 一次设备, {stats['secondary_devices']} 二次设备, "
            f"{stats['signal_points']} 信号点, {stats['bus_sections']} 母线段, "
            f"{stats['protection_zones']} 保护区域"
        )

    def _trigger_analysis(self, trace_id: str) -> Optional[Dict]:
        """触发动作链分析和根因分析"""
        try:
            events = self.event_store.get_events_by_trace(trace_id)
            if not events:
                return None

            self._stats["analyses_triggered"] += 1

            # 动作链分析
            chain_results = self.analyzer.analyze(events)

            # 根因分析
            root_cause = self.root_cause_service.analyze(
                chain_results=chain_results,
                action_events=events,
            )

            # 保存结果
            if chain_results and chain_results[0].trip_scope:
                self.event_store.save_trip_scope(chain_results[0].trip_scope)
            self.event_store.save_root_cause(root_cause)

            return root_cause.to_dict()

        except Exception as e:
            logger.error(f"分析触发异常: {e}")
            return None

    # =========================================================================
    # 协议接入
    # =========================================================================

    def _init_protocol(self, protocol_cfg: Dict) -> None:
        """
        初始化协议适配器

        支持: opcua / mqtt / modbus / http / iec104 / iec61850
        IEC104/IEC61850 有额外参数(common_address / ied_name 等),
        从 protocol_cfg 透传给 ProtocolFactory.create()。
        """
        try:
            from platform_core.data_import.protocol_adapters import ProtocolFactory

            proto_type = protocol_cfg['type']
            # 构造工厂参数: 基础字段 + 协议专属字段全部透传
            factory_kwargs = {k: v for k, v in protocol_cfg.items() if k != 'type' and v}

            adapter = ProtocolFactory.create(proto_type, **factory_kwargs)

            if adapter.connect():
                self._protocol_adapter = adapter
                logger.info(f"协议连接成功: {proto_type}")

                # ---- IEC104 额外订阅: SOE 全量回调 ----
                if proto_type == "iec104" and hasattr(adapter, 'subscribe_soe'):
                    adapter.subscribe_soe(self._on_iec104_soe)
                    logger.info("IEC104 SOE 全量回调已注册")
                    # 自动总召一次
                    if hasattr(adapter, 'general_interrogation'):
                        adapter.general_interrogation()

                # ---- IEC61850 额外订阅: Report + GOOSE ----
                if proto_type == "iec61850":
                    if hasattr(adapter, 'subscribe_report'):
                        adapter.subscribe_report(self._on_iec61850_report)
                        logger.info("IEC61850 Report 回调已注册")
                    if hasattr(adapter, 'subscribe_goose'):
                        adapter.subscribe_goose(self._on_iec61850_goose)
                        logger.info("IEC61850 GOOSE 回调已注册")

                # 通用信号点订阅
                for sub in self._config.get('subscriptions', []):
                    self._setup_subscription(sub)
            else:
                logger.warning(f"协议连接失败: {proto_type}")
        except Exception as e:
            logger.warning(f"协议初始化跳过: {e}")

    def _setup_subscription(self, sub_config: Dict) -> None:
        """
        设置信号订阅 (通用)

        sub_config 字段:
          node_id      - OPC UA node / MQTT topic / IEC104 IOA / IEC61850 DA ref
          signal_name  - 可读名称
          signal_group - 信号组
          action_type_hint - 预设动作类型
          station_id / bay_id / ... - 可选关联字段
        """
        if not self._protocol_adapter:
            return

        node_id = sub_config.get('node_id', '')
        if not node_id:
            return

        # 捕获 sub_config 到闭包
        extra_fields = {k: v for k, v in sub_config.items()
                        if k not in ('node_id',) and v}

        def on_value_change(nid, value):
            # value 可能是 dict (IEC104/IEC61850) 或原始值
            if isinstance(value, dict):
                action_desc = value.get('action_desc', '') or str(value.get('value', ''))
                source_ts = value.get('timestamp', '') or datetime.now().isoformat()
                raw_payload = value
            else:
                action_desc = str(value)
                source_ts = datetime.now().isoformat()
                raw_payload = {"raw": value}

            event_data = {
                "signal_id": nid,
                "signal_name": extra_fields.get('signal_name', nid),
                "signal_group": extra_fields.get('signal_group', SignalGroup.OTHER.value),
                "action_desc": action_desc,
                "action_type": extra_fields.get('action_type_hint', ''),
                "source_ts": source_ts,
                "source_system": self._config.get('protocol', {}).get('type', 'opcua'),
                "raw_payload": raw_payload,
            }
            # 透传关联字段
            for k in ('station_id', 'station_name', 'bay_id', 'bay_name',
                       'primary_device_id', 'primary_device_type',
                       'secondary_device_id', 'secondary_device_type',
                       'severity_hint'):
                if k in extra_fields:
                    event_data[k] = extra_fields[k]

            self.process(event_data)

        if hasattr(self._protocol_adapter, 'subscribe'):
            self._protocol_adapter.subscribe(node_id, on_value_change)

    # =========================================================================
    # IEC 104 专用回调
    # =========================================================================

    def _on_iec104_soe(self, point_data: Dict[str, Any]) -> None:
        """
        IEC 104 SOE 全量回调

        每收到一条 SOE (带时标单/双点) 自动归一化为 ActionEvent 并存储。
        point_data 格式: {"ioa": int, "value": Any, "quality": int,
                          "timestamp": str, "type_id": str, "common_address": int}
        """
        ioa = point_data.get("ioa", 0)
        value = point_data.get("value")
        ts = point_data.get("timestamp", datetime.now().isoformat())
        quality = point_data.get("quality", 0)

        # 通过 correlation service 尝试丰富信号信息
        signal_name = f"IOA_{ioa}"
        signal_group = SignalGroup.SOE.value
        station_id = ""
        bay_id = ""
        secondary_device_id = ""
        if self.correlation_service:
            sig = self.correlation_service._signal_points.get(str(ioa))
            if sig:
                signal_name = sig.signal_name or signal_name
                signal_group = sig.signal_group or signal_group
                station_id = sig.station_id
                bay_id = sig.bay_id
                secondary_device_id = sig.secondary_device_id

        self.process({
            "signal_id": str(ioa),
            "signal_name": signal_name,
            "signal_group": signal_group,
            "action_desc": f"SOE: IOA={ioa} value={value}",
            "value_before": None,
            "value_after": str(value),
            "source_ts": ts,
            "source_system": SourceSystem.PROTECTION_INFO.value,
            "severity_hint": SeverityHint.WARNING.value if quality == 0 else SeverityHint.ALARM.value,
            "station_id": station_id,
            "bay_id": bay_id,
            "secondary_device_id": secondary_device_id,
            "raw_payload": point_data,
            "seq_no": ioa,
        })

    # =========================================================================
    # IEC 61850 专用回调
    # =========================================================================

    def _on_iec61850_report(self, report_data: Dict[str, Any]) -> None:
        """
        IEC 61850 Report 回调

        report_data 可以是单条 {"ref": ..., "value": ..., "quality": ..., "timestamp": ...}
        或 {"entries": [...]}.
        """
        if "entries" in report_data:
            for entry in report_data["entries"]:
                self._process_61850_entry(entry)
        else:
            self._process_61850_entry(report_data)

    def _on_iec61850_goose(self, goose_data: Dict[str, Any]) -> None:
        """
        IEC 61850 GOOSE 报文回调

        GOOSE 报文通常包含跳闸/保护动作等关键快速信号。
        goose_data: {"appid": str, "st_num": int, "sq_num": int,
                     "timestamp": str, "data_set": [{"ref": ..., "value": ...}]}
        """
        ts = goose_data.get("timestamp", datetime.now().isoformat())
        appid = goose_data.get("appid", "")
        st_num = goose_data.get("st_num", 0)

        for item in goose_data.get("data_set", []):
            ref = item.get("ref", "")
            value = item.get("value")

            # GOOSE 跳闸信号通常严重程度较高
            severity = SeverityHint.CRITICAL.value if st_num > 0 else SeverityHint.WARNING.value

            self.process({
                "signal_id": ref,
                "signal_name": ref.split("$")[-1] if "$" in ref else ref,
                "signal_group": SignalGroup.PROTECTION.value,
                "action_desc": f"GOOSE: {ref}={value} (AppID={appid}, StNum={st_num})",
                "value_after": str(value),
                "source_ts": ts,
                "source_system": SourceSystem.PROTECTION_DEVICE.value,
                "severity_hint": severity,
                "raw_payload": goose_data,
            })

    def _process_61850_entry(self, entry: Dict[str, Any]) -> None:
        """处理单条 IEC 61850 数据条目"""
        ref = entry.get("ref", "")
        value = entry.get("value")
        quality = entry.get("quality", 0)
        ts = entry.get("timestamp", datetime.now().isoformat())

        # 根据引用路径推断信号组
        signal_group = self._infer_61850_signal_group(ref)

        # 严重程度推断
        severity = SeverityHint.INFO.value
        if signal_group in (SignalGroup.PROTECTION.value, SignalGroup.BREAKER.value):
            severity = SeverityHint.ALARM.value
        elif signal_group == SignalGroup.RECLOSER.value:
            severity = SeverityHint.WARNING.value

        # 从 correlation service 尝试补充
        station_id = ""
        bay_id = ""
        secondary_device_id = ""
        if self.correlation_service:
            sig = self.correlation_service._signal_points.get(ref)
            if sig:
                station_id = sig.station_id
                bay_id = sig.bay_id
                secondary_device_id = sig.secondary_device_id

        self.process({
            "signal_id": ref,
            "signal_name": ref.split("$")[-1] if "$" in ref else ref,
            "signal_group": signal_group,
            "action_desc": f"61850: {ref}={value}",
            "value_after": str(value),
            "source_ts": ts if isinstance(ts, str) else str(ts),
            "source_system": SourceSystem.PROTECTION_DEVICE.value,
            "severity_hint": severity,
            "station_id": station_id,
            "bay_id": bay_id,
            "secondary_device_id": secondary_device_id,
            "raw_payload": entry,
        })

    @staticmethod
    def _infer_61850_signal_group(ref: str) -> str:
        """根据 IEC 61850 逻辑节点类推断信号组"""
        ref_upper = ref.upper()
        # PTRC / PTOC / PDIS / PIOC ... -> 保护
        if any(ln in ref_upper for ln in ("PTRC", "PTOC", "PDIS", "PIOC", "PDIF", "PTUV", "PTOV")):
            return SignalGroup.PROTECTION.value
        # XCBR -> 断路器
        if "XCBR" in ref_upper:
            return SignalGroup.BREAKER.value
        # RREC -> 重合闸
        if "RREC" in ref_upper:
            return SignalGroup.RECLOSER.value
        # RSYN -> 备自投
        if "RSYN" in ref_upper or "RBRF" in ref_upper:
            return SignalGroup.BACKUP_AUTO.value
        # MMXU / MSQI -> 遥测
        if any(ln in ref_upper for ln in ("MMXU", "MSQI", "MMTR")):
            return SignalGroup.TELEMETRY.value
        # CSWI -> 控制
        if "CSWI" in ref_upper:
            return SignalGroup.CONTROL_LOOP.value
        return SignalGroup.OTHER.value

    # =========================================================================
    # 状态
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.PLUGIN_ID,
            "plugin_name": self.PLUGIN_NAME,
            "version": self.VERSION,
            "initialized": self._initialized,
            "running": self._running,
            "stats": dict(self._stats),
            "store_stats": self.event_store.get_statistics() if self.event_store else {},
        }

    def healthcheck(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._initialized,
            message="OK" if self._initialized else "未初始化",
            details={"running": self._running, "stats": dict(self._stats)},
        )

    def _threshold_snapshot(self) -> Dict[str, Any]:
        return {
            "event_count_warning": self._config.get("thresholds", {}).get("event_count_warning", 3),
            "analysis": dict(self._config.get("analysis", {})),
            "alarm": dict(self._config.get("alarm", {})),
        }

    def _rule_confidence(self) -> float:
        return float(self._config.get("runtime", {}).get("rule_confidence", 1.0))

    def _build_contract_metadata(self, events: Optional[List[ActionEvent]] = None) -> Dict[str, Any]:
        sampling = self._config.get("sampling", {})
        window = self._config.get("window", {})
        analysis = self._config.get("analysis", {})
        runtime = self._config.get("runtime", {})
        return build_common_metadata(
            modality="action_event",
            sensor_type="secondary_action_signal",
            sample_interval=sampling.get("sample_interval_seconds", self._config.get("poll_interval_s", 1.0)),
            window_size=window.get("size", analysis.get("trip_chain_window_s", 10.0)),
            threshold_snapshot=self._threshold_snapshot(),
            runtime_mode=runtime.get("mode", "standalone"),
            algorithm_stage="event_normalization_with_rule_analysis",
            model_status="unavailable",
            fallback_level="rules",
            trend_prediction_available=False,
            upgrade_placeholders=self._config.get("upgrade_placeholders", {
                "prediction_model": "action sequence prediction model hook",
                "anomaly_detection_model": "SOE/action anomaly model hook",
                "protocol_adapter": "OPC UA/MQTT/IEC adapter hook",
                "online_learning": "online sequence baseline hook",
            }),
            extra={
                "event_count": len(events or []),
                "protocol_mode": self._config.get("protocol", {}).get("type", "disabled"),
            },
        )

    def _contract_status(self, events: List[ActionEvent]) -> str:
        thresholds = self._config.get("thresholds", {})
        event_count_warning = int(thresholds.get("event_count_warning", 3))
        alarm_severities = {SeverityHint.ALARM.value, SeverityHint.CRITICAL.value}
        if any(event.severity_hint in alarm_severities for event in events):
            return "alarm"
        if len(events) >= event_count_warning:
            return "warning"
        watched_actions = {
            ActionType.PROTECTION_START.value,
            ActionType.PROTECTION_TRIP.value,
            ActionType.BREAKER_OPEN.value,
            ActionType.BREAKER_CLOSE.value,
        }
        if any(event.action_type in watched_actions for event in events):
            return "warning"
        return "normal"

    def _contract_label(self, status: str, events: List[ActionEvent]) -> str:
        if status == "normal":
            return "normal"
        if status == "warning":
            return "warning"
        if status == "alarm":
            return "abnormal"
        return status or "normal"

    def _build_temporal_output(
        self,
        *,
        input_data: Dict[str, Any],
        events: List[ActionEvent],
        status: str,
        label: str,
        confidence: float,
        analysis_result: Optional[Dict[str, Any]],
        stored_ids: List[str],
        time_window: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        anomaly_events = []
        abnormal_intervals = []
        reason_codes = []
        for event, stored_id in zip(events, stored_ids):
            event_reasons = [f"ACTION_{str(event.action_type).upper()}"]
            if event.severity_hint in (SeverityHint.WARNING.value, SeverityHint.ALARM.value, SeverityHint.CRITICAL.value):
                event_reasons.append(f"SEVERITY_{str(event.severity_hint).upper()}")
            reason_codes.extend(event_reasons)
            severity = "alarm" if event.severity_hint in (SeverityHint.ALARM.value, SeverityHint.CRITICAL.value) else ("warning" if event.action_type != ActionType.SIGNAL_CHANGE.value else "normal")
            if severity != "normal":
                anomaly_events.append({
                    "event_id": stored_id,
                    "event_type": "action_event",
                    "label": label,
                    "severity": severity,
                    "confidence": clamp_confidence(confidence),
                    "reason_codes": event_reasons,
                    "value": {
                        "action_type": event.action_type,
                        "signal_id": event.signal_id,
                        "value_numeric": event.value_numeric,
                    },
                    "metric_name": event.signal_id or "action_signal",
                    "time_window": time_window,
                    "evidence": {
                        "action_desc": event.action_desc,
                        "source_system": event.source_system,
                    },
                })
                abnormal_intervals.append({
                    "start": time_window.get("start"),
                    "end": time_window.get("end"),
                    "severity": severity,
                    "reason_codes": event_reasons,
                    "metrics": [event.signal_id or "action_signal"],
                })

        root_reason = (analysis_result or {}).get("root_cause_reason")
        if root_reason:
            reason_codes.append("ROOT_CAUSE_ANALYSIS_AVAILABLE")

        return build_unified_temporal_output(
            plugin_name=self.PLUGIN_NAME,
            task_type="event_monitoring",
            payload=input_data,
            status=status,
            label=label,
            severity=status,
            confidence=confidence,
            summary={
                "event_count": len(events),
                "stored_event_ids": stored_ids,
                "analysis_triggered": bool(analysis_result),
                "root_cause_category": (analysis_result or {}).get("root_cause_category"),
            },
            anomaly_events=anomaly_events,
            abnormal_intervals=abnormal_intervals,
            reason_codes=reason_codes or ["ACTION_EVENT_NORMAL"],
            recommended_actions=(analysis_result or {}).get("next_actions") or ["继续采集动作事件并保留回溯链路"],
            trend_diagnosis={
                "available": False,
                "direction": "event_sequence",
                "confidence": clamp_confidence((analysis_result or {}).get("confidence", 0.0), default=0.0),
                "reason": root_reason or "event sequence model reserved for second phase",
            },
            evidence=[
                {
                    "type": "action_event_store",
                    "event_ids": stored_ids,
                    "signals": [event.signal_id for event in events],
                    "analysis_result": analysis_result,
                }
            ],
            review_required=status in ("warning", "alarm") or bool((analysis_result or {}).get("manual_review_items")),
            model_info={
                "model_status": metadata.get("model_status"),
                "algorithm_stage": metadata.get("algorithm_stage"),
                "fallback_level": metadata.get("fallback_level"),
            },
            placeholders={
                "model_features_placeholder": ["action_type", "signal_group", "value_numeric", "source_ts"],
                "sequence_embedding_placeholder": None,
                "temporal_pattern_placeholder": "action_sequence_window",
                "anomaly_score_trace_placeholder": [1.0 if event.severity_hint in (SeverityHint.ALARM.value, SeverityHint.CRITICAL.value) else 0.5 for event in events],
                "root_cause_feature_placeholder": analysis_result,
            },
            time_window=time_window,
            input_protocol={
                "metric_names": [event.signal_id for event in events],
                "sampling_or_timestamp": metadata.get("sample_interval"),
            },
        )

    def _contract_error(self, input_data: Dict[str, Any], message: str) -> Dict[str, Any]:
        device_id = input_data.get("device_id", "action_event_channel") if isinstance(input_data, dict) else "action_event_channel"
        metadata = self._build_contract_metadata([])
        time_window = build_time_window(
            input_data if isinstance(input_data, dict) else {},
            window_size=metadata.get("window_size"),
            sample_interval=metadata.get("sample_interval"),
        )
        temporal_output = build_unified_temporal_output(
            plugin_name=self.PLUGIN_NAME,
            task_type="event_monitoring",
            payload=input_data if isinstance(input_data, dict) else {},
            status="error",
            label="error",
            severity="error",
            confidence=0.0,
            summary={"device_id": device_id, "status": "error", "message": message},
            reason_codes=["INPUT_VALIDATION_ERROR"],
            recommended_actions=["检查 events/state_change_events/protocol_ingested_data 输入"],
            trend_diagnosis={"available": False, "direction": "unknown", "confidence": 0.0, "reason": message},
            evidence=[{"type": "validation_error", "message": message}],
            review_required=True,
            model_info={"model_status": metadata.get("model_status"), "fallback_level": metadata.get("fallback_level")},
            time_window=time_window,
        )
        virtual_result = build_virtual_result(
            payload=input_data if isinstance(input_data, dict) else {},
            plugin_id=self.id,
            plugin_version=self.version,
            code_hash=self.code_hash,
            device_id=device_id,
            roi_id=(input_data or {}).get("roi_id") or (input_data or {}).get("signal_id") or device_id if isinstance(input_data, dict) else device_id,
            label="error",
            value=None,
            confidence=0.0,
            metadata=metadata,
            component_id="action_event_signal",
            failure_reason=message,
        )
        return {
            "success": False,
            "status": "error",
            "label": "error",
            "value": None,
            "confidence": 0.0,
            "metadata": metadata,
            "results": [virtual_result],
            **temporal_output,
            "error": message,
            "error_message": message,
        }

    def get_standalone_routes(self) -> list:
        plugin = self

        async def action_event_smoke(request: StarletteRequest):
            body = {}
            try:
                body = await request.json()
            except Exception as exc:
                logger.debug("action event smoke request has no JSON body: %s", exc)
            sample = {
                "device_id": "protection_relay_smoke",
                "timestamp": datetime.now().isoformat(),
                "signal_changes": [{
                    "signal_id": "SOE-SMOKE-1",
                    "signal_name": "Protection Trip Smoke",
                    "signal_group": SignalGroup.PROTECTION.value,
                    "action_type": ActionType.PROTECTION_TRIP.value,
                    "action_desc": "protection trip smoke sample",
                    "value_after": "1",
                    "severity_hint": SeverityHint.WARNING.value,
                }],
                "context": {"task_id": "action-event-smoke", "site_id": "standalone"},
            }
            sample.update(body)
            if not sample.get("events") and not sample.get("signal_changes"):
                sample["signal_changes"] = [{
                    "signal_id": "SOE-SMOKE-1",
                    "signal_name": "Protection Trip Smoke",
                    "signal_group": SignalGroup.PROTECTION.value,
                    "action_type": ActionType.PROTECTION_TRIP.value,
                    "action_desc": "protection trip smoke sample",
                    "value_after": "1",
                    "severity_hint": SeverityHint.WARNING.value,
                }]
            return StarletteJSONResponse(plugin.process(sample))

        return [
            {"path": "/api/action-event/smoke", "endpoint": action_event_smoke, "methods": ["GET", "POST"], "summary": "Run action event smoke sample"},
        ]

    @classmethod
    def create_standalone(cls, config: Optional[Dict] = None) -> 'ActionEventMonitoringPlugin':
        """工厂方法: 创建独立实例"""
        plugin_dir = Path(__file__).resolve().parent
        if config is None:
            from darkbreaker_sdk.utils import load_plugin_config
            config = load_plugin_config(plugin_dir / "configs" / "default.yaml")
        plugin = cls(plugin_dir=plugin_dir)
        plugin.init(config or {})
        return plugin


Plugin = ActionEventMonitoringPlugin
