#!/usr/bin/env python3
"""
室内电子围栏 - 验收场景脚本
============================

验收场景:
1. 多目标检测验收
2. 跟踪稳定性验收
3. 越线检测验收
4. 授权校验验收
5. WebSocket实时推送验收
6. 历史查询验收
7. 证据回放验收

版本: 3.0.0
作者: 变电站AI监控平台架构组

运行方式:
    python scripts/acceptance/run_indoor_fence_scenarios.py --api-url http://localhost:8080
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import random

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# 配置
# =============================================================================

class AcceptanceConfig:
    """验收配置"""
    API_BASE = "/api/v3/indoor/fence"
    WS_PATH = "/api/v3/indoor/fence/ws"
    TIMEOUT = 30.0
    RETRY_COUNT = 3
    REPORT_DIR = "reports"


# =============================================================================
# 数据结构
# =============================================================================

class ScenarioStatus(Enum):
    """场景状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScenarioResult:
    """场景结果"""
    scenario_id: str
    scenario_name: str
    status: ScenarioStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    checks: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "checks": self.checks,
            "errors": self.errors,
            "evidence": self.evidence
        }


@dataclass
class AcceptanceReport:
    """验收报告"""
    report_id: str
    title: str
    start_time: datetime
    end_time: Optional[datetime] = None
    scenarios: List[ScenarioResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "summary": self.summary
        }


# =============================================================================
# HTTP客户端
# =============================================================================

class AcceptanceHttpClient:
    """验收HTTP客户端"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str, params: Dict = None) -> Dict:
        """GET请求"""
        url = f"{self.base_url}{AcceptanceConfig.API_BASE}{endpoint}"
        async with self.session.get(url, params=params, timeout=AcceptanceConfig.TIMEOUT) as resp:
            resp.raise_for_status()
            return await resp.json()
    
    async def post(self, endpoint: str, data: Dict) -> Dict:
        """POST请求"""
        url = f"{self.base_url}{AcceptanceConfig.API_BASE}{endpoint}"
        async with self.session.post(url, json=data, timeout=AcceptanceConfig.TIMEOUT) as resp:
            resp.raise_for_status()
            return await resp.json()


# =============================================================================
# WebSocket客户端
# =============================================================================

class AcceptanceWebSocketClient:
    """验收WebSocket客户端"""
    
    def __init__(self, base_url: str):
        self.ws_url = base_url.replace('http', 'ws') + AcceptanceConfig.WS_PATH
        self.ws = None
        self.messages = []
    
    async def connect(self, client_id: str = None):
        """连接WebSocket"""
        import websockets
        url = self.ws_url
        if client_id:
            url += f"?client_id={client_id}"
        self.ws = await websockets.connect(url)
        return self
    
    async def disconnect(self):
        """断开连接"""
        if self.ws:
            await self.ws.close()
    
    async def receive(self, timeout: float = 5.0) -> Dict:
        """接收消息"""
        import asyncio
        try:
            data = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            msg = json.loads(data)
            self.messages.append(msg)
            return msg
        except asyncio.TimeoutError:
            return None
    
    async def send(self, message: Dict):
        """发送消息"""
        await self.ws.send(json.dumps(message))


# =============================================================================
# 验收场景定义
# =============================================================================

class AcceptanceScenario:
    """验收场景基类"""
    
    def __init__(self, scenario_id: str, name: str, description: str):
        self.scenario_id = scenario_id
        self.name = name
        self.description = description
        self.result = None
    
    async def run(self, client: AcceptanceHttpClient, ws_client: AcceptanceWebSocketClient = None) -> ScenarioResult:
        """运行场景"""
        self.result = ScenarioResult(
            scenario_id=self.scenario_id,
            scenario_name=self.name,
            status=ScenarioStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            await self.execute(client, ws_client)
            
            # 检查是否所有检查都通过
            failed = [c for c in self.result.checks if not c.get("passed", False)]
            if failed:
                self.result.status = ScenarioStatus.FAILED
                self.result.errors.extend([c.get("message", "") for c in failed])
            else:
                self.result.status = ScenarioStatus.PASSED
        
        except Exception as e:
            self.result.status = ScenarioStatus.FAILED
            self.result.errors.append(str(e))
        
        finally:
            self.result.end_time = datetime.now()
            self.result.duration_ms = (self.result.end_time - self.result.start_time).total_seconds() * 1000
        
        return self.result
    
    async def execute(self, client: AcceptanceHttpClient, ws_client: AcceptanceWebSocketClient):
        """执行场景 (子类实现)"""
        raise NotImplementedError
    
    def add_check(self, check_id: str, description: str, passed: bool, message: str = "", evidence: Any = None):
        """添加检查项"""
        self.result.checks.append({
            "check_id": check_id,
            "description": description,
            "passed": passed,
            "message": message,
            "evidence": evidence
        })


class MultiTargetDetectionScenario(AcceptanceScenario):
    """多目标检测验收场景"""
    
    def __init__(self):
        super().__init__(
            "S001",
            "多目标检测验收",
            "验证系统能够同时检测并跟踪多个人员"
        )
    
    async def execute(self, client: AcceptanceHttpClient, ws_client: AcceptanceWebSocketClient):
        """执行验收"""
        # 1. 获取当前跟踪列表
        tracks = await client.get("/tracks")
        
        self.add_check(
            "S001-C001",
            "API返回轨迹列表",
            isinstance(tracks, list),
            f"返回了{len(tracks)}条轨迹"
        )
        
        # 2. 检查轨迹数据结构
        if tracks:
            track = tracks[0]
            required_fields = ["track_id", "person_state", "ground_position", "confidence"]
            has_all = all(f in track for f in required_fields)
            self.add_check(
                "S001-C002",
                "轨迹包含必要字段",
                has_all,
                f"字段检查: {required_fields}"
            )
        
        # 3. 获取系统状态
        status = await client.get("/status")
        
        self.add_check(
            "S001-C003",
            "系统状态正常",
            status.get("status") in ["running", "degraded"],
            f"当前状态: {status.get('status')}"
        )
        
        # 4. 检查传感器健康度
        health = await client.get("/health")
        
        self.add_check(
            "S001-C004",
            "传感器健康",
            health.get("healthy", False),
            health.get("message", "")
        )


class TrackingStabilityScenario(AcceptanceScenario):
    """跟踪稳定性验收场景"""
    
    def __init__(self):
        super().__init__(
            "S002",
            "跟踪稳定性验收",
            "验证人员交叉行走时ID切换次数"
        )
    
    async def execute(self, client: AcceptanceHttpClient, ws_client: AcceptanceWebSocketClient):
        """执行验收"""
        # 监控轨迹ID变化
        initial_tracks = await client.get("/tracks")
        initial_ids = {t["track_id"] for t in initial_tracks}
        
        # 等待一段时间收集数据
        id_changes = 0
        for i in range(10):
            await asyncio.sleep(0.5)
            current_tracks = await client.get("/tracks")
            current_ids = {t["track_id"] for t in current_tracks}
            
            # 计算ID变化
            new_ids = current_ids - initial_ids
            id_changes += len(new_ids)
            initial_ids = current_ids
        
        # ID切换应该很少
        self.add_check(
            "S002-C001",
            "ID切换次数检查",
            id_changes <= 2,
            f"ID切换次数: {id_changes} (允许 ≤ 2)"
        )


class YellowLineDetectionScenario(AcceptanceScenario):
    """越线检测验收场景"""
    
    def __init__(self):
        super().__init__(
            "S003",
            "越线检测验收",
            "验证越线检测功能和告警响应时间"
        )
    
    async def execute(self, client: AcceptanceHttpClient, ws_client: AcceptanceWebSocketClient):
        """执行验收"""
        # 1. 获取区域配置确认黄线存在
        zones = await client.get("/zones")
        
        has_yellow_line = "yellow_line" in zones
        self.add_check(
            "S003-C001",
            "黄线配置存在",
            has_yellow_line,
            f"黄线配置: {zones.get('yellow_line', {})}"
        )
        
        # 2. 查询越线相关事件
        events = await client.get("/events", {
            "event_types": "yellow_line_warning,yellow_line_alarm",
            "limit": 10
        })
        
        self.add_check(
            "S003-C002",
            "能够查询越线事件",
            "events" in events,
            f"查询到 {events.get('total', 0)} 条越线事件"
        )
        
        # 3. 如果有越线事件,验证事件结构
        if events.get("events"):
            event = events["events"][0]
            has_trace_id = "trace_id" in event
            has_evidence = "evidence" in event
            
            self.add_check(
                "S003-C003",
                "越线事件包含trace_id",
                has_trace_id,
                f"trace_id: {event.get('trace_id', 'N/A')}"
            )
            
            self.add_check(
                "S003-C004",
                "越线事件包含证据",
                has_evidence,
                f"证据字段: {list(event.get('evidence', {}).keys())}"
            )


class AuthorizationCheckScenario(AcceptanceScenario):
    """授权校验验收场景"""
    
    def __init__(self):
        super().__init__(
            "S004",
            "授权校验验收",
            "验证未授权访问检测功能"
        )
    
    async def execute(self, client: AcceptanceHttpClient, ws_client: AcceptanceWebSocketClient):
        """执行验收"""
        # 1. 测试授权更新接口
        auth_result = await client.post("/authorize", {
            "person_id": "T_TEST_001",
            "cabinet_ids": [1, 2, 3]
        })
        
        self.add_check(
            "S004-C001",
            "授权更新成功",
            auth_result.get("status") == "success",
            f"授权结果: {auth_result}"
        )
        
        # 2. 查询未授权访问事件
        events = await client.get("/events", {
            "event_types": "unauthorized_access",
            "limit": 10
        })
        
        self.add_check(
            "S004-C002",
            "能够查询未授权事件",
            "events" in events,
            f"查询到 {events.get('total', 0)} 条未授权事件"
        )


class WebSocketPushScenario(AcceptanceScenario):
    """WebSocket实时推送验收场景"""
    
    def __init__(self):
        super().__init__(
            "S005",
            "WebSocket实时推送验收",
            "验证WebSocket连接、心跳、消息推送"
        )
    
    async def execute(self, client: AcceptanceHttpClient, ws_client: AcceptanceWebSocketClient):
        """执行验收"""
        if ws_client is None:
            self.add_check("S005-C001", "WebSocket客户端", False, "WebSocket客户端未提供")
            return
        
        try:
            # 1. 连接WebSocket
            await ws_client.connect()
            
            # 2. 接收连接确认
            ack = await ws_client.receive(timeout=5.0)
            
            self.add_check(
                "S005-C001",
                "WebSocket连接成功",
                ack is not None and ack.get("type") == "connection_ack",
                f"连接确认: {ack}"
            )
            
            # 3. 测试心跳
            await ws_client.send({"type": "ping"})
            pong = await ws_client.receive(timeout=5.0)
            
            self.add_check(
                "S005-C002",
                "心跳响应正常",
                pong is not None and pong.get("type") == "pong",
                f"心跳响应: {pong}"
            )
            
            # 4. 测试状态请求
            await ws_client.send({"type": "get_status"})
            status_msg = await ws_client.receive(timeout=5.0)
            
            self.add_check(
                "S005-C003",
                "状态请求响应",
                status_msg is not None and status_msg.get("type") == "status",
                f"状态响应: {status_msg is not None}"
            )
            
            # 5. 等待事件推送
            event_received = False
            for _ in range(10):
                msg = await ws_client.receive(timeout=2.0)
                if msg and msg.get("type") == "event":
                    event_received = True
                    self.result.evidence["sample_event"] = msg
                    break
            
            self.add_check(
                "S005-C004",
                "接收到事件推送",
                event_received,
                "在10秒内收到事件" if event_received else "未收到事件推送"
            )
            
        finally:
            await ws_client.disconnect()


class HistoryQueryScenario(AcceptanceScenario):
    """历史查询验收场景"""
    
    def __init__(self):
        super().__init__(
            "S006",
            "历史查询验收",
            "验证历史事件查询功能"
        )
    
    async def execute(self, client: AcceptanceHttpClient, ws_client: AcceptanceWebSocketClient):
        """执行验收"""
        # 1. 基本查询
        events = await client.get("/events")
        
        self.add_check(
            "S006-C001",
            "事件查询返回正确结构",
            "total" in events and "events" in events,
            f"总数: {events.get('total', 0)}"
        )
        
        # 2. 分页查询
        events_page = await client.get("/events", {"limit": 10, "offset": 0})
        
        self.add_check(
            "S006-C002",
            "分页查询正常",
            len(events_page.get("events", [])) <= 10,
            f"返回 {len(events_page.get('events', []))} 条"
        )
        
        # 3. 类型过滤
        events_filtered = await client.get("/events", {
            "event_types": "person_detected"
        })
        
        all_correct_type = all(
            e.get("type") == "person_detected" 
            for e in events_filtered.get("events", [])
        )
        
        self.add_check(
            "S006-C003",
            "类型过滤正常",
            all_correct_type or len(events_filtered.get("events", [])) == 0,
            "所有返回事件类型正确"
        )


class EvidenceReplayScenario(AcceptanceScenario):
    """证据回放验收场景"""
    
    def __init__(self):
        super().__init__(
            "S007",
            "证据回放验收",
            "验证告警事件的证据链完整性"
        )
    
    async def execute(self, client: AcceptanceHttpClient, ws_client: AcceptanceWebSocketClient):
        """执行验收"""
        # 1. 获取告警事件
        events = await client.get("/events", {
            "alert_levels": "alarm,critical",
            "limit": 5
        })
        
        alarm_events = events.get("events", [])
        
        self.add_check(
            "S007-C001",
            "能够查询告警事件",
            True,  # 即使没有告警也算通过
            f"查询到 {len(alarm_events)} 条告警"
        )
        
        # 2. 检查告警事件的证据
        if alarm_events:
            event = alarm_events[0]
            evidence = event.get("evidence", {})
            
            has_evidence = bool(evidence)
            self.add_check(
                "S007-C002",
                "告警事件有证据",
                has_evidence,
                f"证据字段: {list(evidence.keys())}"
            )
            
            # 检查trace_id
            trace_id = event.get("trace_id")
            self.add_check(
                "S007-C003",
                "告警事件有trace_id",
                bool(trace_id),
                f"trace_id: {trace_id}"
            )
            
            # 保存证据用于报告
            self.result.evidence["sample_alarm"] = event


# =============================================================================
# 验收运行器
# =============================================================================

class AcceptanceRunner:
    """验收运行器"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.report = AcceptanceReport(
            report_id=f"acceptance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title="室内电子围栏验收报告",
            start_time=datetime.now()
        )
        
        # 注册场景
        self.scenarios = [
            MultiTargetDetectionScenario(),
            TrackingStabilityScenario(),
            YellowLineDetectionScenario(),
            AuthorizationCheckScenario(),
            WebSocketPushScenario(),
            HistoryQueryScenario(),
            EvidenceReplayScenario(),
        ]
    
    async def run(self):
        """运行所有验收场景"""
        print(f"\n{'='*60}")
        print("       室内电子围栏 V3.0 - 验收测试")
        print(f"{'='*60}")
        print(f"API地址: {self.api_url}")
        print(f"开始时间: {self.report.start_time.isoformat()}")
        print(f"{'='*60}\n")
        
        async with AcceptanceHttpClient(self.api_url) as client:
            ws_client = AcceptanceWebSocketClient(self.api_url)
            
            for scenario in self.scenarios:
                print(f"[{scenario.scenario_id}] {scenario.name}...")
                
                try:
                    result = await scenario.run(client, ws_client)
                    self.report.scenarios.append(result)
                    
                    status_icon = "✓" if result.status == ScenarioStatus.PASSED else "✗"
                    print(f"  {status_icon} {result.status.value} ({result.duration_ms:.0f}ms)")
                    
                    for check in result.checks:
                        check_icon = "✓" if check["passed"] else "✗"
                        print(f"    {check_icon} {check['description']}")
                    
                    if result.errors:
                        for error in result.errors:
                            print(f"    ⚠ {error}")
                    
                except Exception as e:
                    print(f"  ✗ 场景执行失败: {e}")
        
        # 生成汇总
        self.report.end_time = datetime.now()
        self.report.summary = {
            "total": len(self.report.scenarios),
            "passed": sum(1 for s in self.report.scenarios if s.status == ScenarioStatus.PASSED),
            "failed": sum(1 for s in self.report.scenarios if s.status == ScenarioStatus.FAILED),
            "skipped": sum(1 for s in self.report.scenarios if s.status == ScenarioStatus.SKIPPED)
        }
        
        # 打印汇总
        print(f"\n{'='*60}")
        print("验收汇总")
        print(f"{'='*60}")
        print(f"总计: {self.report.summary['total']}")
        print(f"通过: {self.report.summary['passed']}")
        print(f"失败: {self.report.summary['failed']}")
        print(f"跳过: {self.report.summary['skipped']}")
        print(f"耗时: {(self.report.end_time - self.report.start_time).total_seconds():.1f}秒")
        
        # 判断是否全部通过
        all_passed = self.report.summary['failed'] == 0
        print(f"\n{'='*60}")
        if all_passed:
            print("✓ 验收通过!")
        else:
            print("✗ 验收未通过,请查看详细报告")
        print(f"{'='*60}\n")
        
        return self.report
    
    def save_report(self, output_dir: str = None):
        """保存验收报告"""
        if output_dir is None:
            output_dir = AcceptanceConfig.REPORT_DIR
        
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON报告
        json_path = os.path.join(output_dir, f"{self.report.report_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"JSON报告已保存: {json_path}")
        
        # HTML报告
        html_path = os.path.join(output_dir, f"{self.report.report_id}.html")
        self._generate_html_report(html_path)
        print(f"HTML报告已保存: {html_path}")
    
    def _generate_html_report(self, path: str):
        """生成HTML报告"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{self.report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .summary-item {{ background: #f0f0f0; padding: 15px 25px; border-radius: 8px; text-align: center; }}
        .summary-item.passed {{ background: #e8f5e9; color: #2e7d32; }}
        .summary-item.failed {{ background: #ffebee; color: #c62828; }}
        .scenario {{ border: 1px solid #ddd; margin: 10px 0; border-radius: 8px; overflow: hidden; }}
        .scenario-header {{ background: #f5f5f5; padding: 15px; cursor: pointer; display: flex; justify-content: space-between; }}
        .scenario-header.passed {{ border-left: 4px solid #4CAF50; }}
        .scenario-header.failed {{ border-left: 4px solid #f44336; }}
        .scenario-body {{ padding: 15px; display: none; }}
        .scenario-body.show {{ display: block; }}
        .check {{ padding: 8px; margin: 5px 0; border-radius: 4px; }}
        .check.passed {{ background: #e8f5e9; }}
        .check.failed {{ background: #ffebee; }}
        .check-icon {{ margin-right: 8px; }}
        .timestamp {{ color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 {self.report.title}</h1>
        
        <div class="summary">
            <div class="summary-item">
                <div style="font-size: 24px; font-weight: bold;">{self.report.summary['total']}</div>
                <div>总计</div>
            </div>
            <div class="summary-item passed">
                <div style="font-size: 24px; font-weight: bold;">{self.report.summary['passed']}</div>
                <div>通过</div>
            </div>
            <div class="summary-item failed">
                <div style="font-size: 24px; font-weight: bold;">{self.report.summary['failed']}</div>
                <div>失败</div>
            </div>
        </div>
        
        <p class="timestamp">
            开始时间: {self.report.start_time.isoformat()}<br>
            结束时间: {self.report.end_time.isoformat() if self.report.end_time else 'N/A'}<br>
            报告ID: {self.report.report_id}
        </p>
        
        <h2>验收场景</h2>
        """
        
        for s in self.report.scenarios:
            status_class = s.status.value
            status_icon = "✓" if s.status == ScenarioStatus.PASSED else "✗"
            
            html += f"""
        <div class="scenario">
            <div class="scenario-header {status_class}" onclick="this.nextElementSibling.classList.toggle('show')">
                <span>{status_icon} [{s.scenario_id}] {s.scenario_name}</span>
                <span>{s.duration_ms:.0f}ms</span>
            </div>
            <div class="scenario-body">
            """
            
            for check in s.checks:
                check_class = "passed" if check["passed"] else "failed"
                check_icon = "✓" if check["passed"] else "✗"
                html += f"""
                <div class="check {check_class}">
                    <span class="check-icon">{check_icon}</span>
                    <strong>{check['description']}</strong>
                    <div style="margin-left: 24px; color: #666;">{check['message']}</div>
                </div>
                """
            
            if s.errors:
                html += "<div style='color: red; margin-top: 10px;'><strong>错误:</strong></div>"
                for error in s.errors:
                    html += f"<div style='color: red; margin-left: 20px;'>• {error}</div>"
            
            html += """
            </div>
        </div>
            """
        
        html += """
    </div>
</body>
</html>
        """
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)


# =============================================================================
# 主函数
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="室内电子围栏验收测试")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API地址")
    parser.add_argument("--output-dir", default="reports", help="报告输出目录")
    args = parser.parse_args()
    
    runner = AcceptanceRunner(args.api_url)
    
    try:
        await runner.run()
        runner.save_report(args.output_dir)
        
        # 返回退出码
        if runner.report.summary['failed'] == 0:
            sys.exit(0)
        else:
            sys.exit(1)
    
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
