/**
 * 室内电子围栏监控中心 V3.0
 * ===========================
 * 
 * 生产级前端实现:
 * - 移除模拟数据，仅从API获取真实数据
 * - WebSocket实时推送 (断线重连、消息幂等)
 * - 统一事件契约解析
 * - 证据回放功能
 * 
 * 版本: 3.0.0
 * 作者: 变电站AI监控平台架构组
 */

// =============================================================================
// 配置常量
// =============================================================================

const IndoorFenceConfig = {
    // API配置
    API_BASE: '/api/v3/indoor/fence',
    WS_URL: `ws://${window.location.host}/api/v3/indoor/fence/ws`,
    
    // 刷新配置
    REFRESH_INTERVAL_MS: 1000,      // API刷新间隔
    WS_RECONNECT_DELAY_MS: 3000,    // WebSocket重连延迟
    WS_MAX_RECONNECT_ATTEMPTS: 10,  // 最大重连次数
    WS_HEARTBEAT_INTERVAL_MS: 25000, // 心跳间隔
    
    // 仿真模式 (生产环境必须为false)
    SIMULATION_MODE: false,
    SIMULATION_FALLBACK_ONLY: true,  // 仅API不可用时启用仿真
    
    // 显示配置
    TRACK_HISTORY_LENGTH: 50,       // 轨迹历史长度
    EVENT_DISPLAY_LIMIT: 100,       // 事件显示数量限制
    
    // 坐标系配置 (与后端统一)
    COORD_ORIGIN: { x: 0, y: 0 },
    COORD_SCALE: { x: 10, y: 6 },   // 监测区域尺寸 (米)
};


// =============================================================================
// 状态管理
// =============================================================================

const IndoorFenceState = {
    version: '3.0.0',
    
    // 连接状态
    wsConnected: false,
    wsReconnectAttempts: 0,
    lastReceivedTraceId: null,
    clientId: null,
    
    // 数据状态
    tracks: [],                     // 当前跟踪列表
    events: [],                     // 历史事件
    zones: null,                    // 区域配置
    systemStatus: null,             // 系统状态
    
    // UI状态
    selectedTrack: null,
    alertSoundEnabled: true,
    autoScroll: true,
    
    // 统计
    stats: {
        totalPersons: 0,
        activeAlarms: 0,
        processedFrames: 0,
        lastUpdateTime: null
    },
    
    // 仿真模式 (仅作为fallback)
    simulationMode: false,
    simulationActive: false
};


// =============================================================================
// API 客户端
// =============================================================================

class IndoorFenceApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.requestTimeout = 5000;
    }
    
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), this.requestTimeout);
        
        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });
            
            if (!response.ok) {
                throw new Error(`API错误: ${response.status} ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('请求超时');
            }
            throw error;
        } finally {
            clearTimeout(timeout);
        }
    }
    
    // 获取系统状态
    async getStatus() {
        return this.request('/status');
    }
    
    // 获取当前跟踪列表
    async getTracks() {
        return this.request('/tracks');
    }
    
    // 获取历史事件
    async getEvents(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/events?${query}`);
    }
    
    // 获取区域配置
    async getZones() {
        return this.request('/zones');
    }
    
    // 更新授权
    async updateAuthorization(personId, cabinetIds) {
        return this.request('/authorize', {
            method: 'POST',
            body: JSON.stringify({
                person_id: personId,
                cabinet_ids: cabinetIds
            })
        });
    }
    
    // 健康检查
    async healthCheck() {
        return this.request('/health');
    }
}


// =============================================================================
// WebSocket 客户端
// =============================================================================

class IndoorFenceWebSocket {
    constructor(url, handlers) {
        this.url = url;
        this.handlers = handlers;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.heartbeatTimer = null;
        this.reconnectTimer = null;
    }
    
    connect(clientId = null, lastTraceId = null) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('[WS] 已连接，跳过');
            return;
        }
        
        // 构建连接URL
        let wsUrl = this.url;
        const params = [];
        if (clientId) params.push(`client_id=${clientId}`);
        if (lastTraceId) params.push(`last_trace_id=${lastTraceId}`);
        if (params.length > 0) {
            wsUrl += `?${params.join('&')}`;
        }
        
        console.log(`[WS] 连接: ${wsUrl}`);
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('[WS] 连接成功');
            this.reconnectAttempts = 0;
            IndoorFenceState.wsConnected = true;
            IndoorFenceState.simulationActive = false;
            
            this.startHeartbeat();
            this.handlers.onConnect?.();
        };
        
        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (error) {
                console.error('[WS] 消息解析失败:', error);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('[WS] 错误:', error);
            this.handlers.onError?.(error);
        };
        
        this.ws.onclose = (event) => {
            console.log(`[WS] 断开: code=${event.code}, reason=${event.reason}`);
            IndoorFenceState.wsConnected = false;
            this.stopHeartbeat();
            
            this.handlers.onDisconnect?.();
            this.scheduleReconnect();
        };
    }
    
    handleMessage(message) {
        const { type, data, trace_id, client_id } = message;
        
        // 保存trace_id用于重连
        if (trace_id) {
            IndoorFenceState.lastReceivedTraceId = trace_id;
        }
        
        // 保存client_id
        if (client_id) {
            IndoorFenceState.clientId = client_id;
        }
        
        switch (type) {
            case 'connection_ack':
                console.log('[WS] 连接确认:', message);
                IndoorFenceState.clientId = message.client_id;
                break;
                
            case 'event':
                this.handlers.onEvent?.(data);
                break;
                
            case 'status':
                this.handlers.onStatus?.(data);
                break;
                
            case 'tracks':
                this.handlers.onTracks?.(data);
                break;
                
            case 'pong':
                // 心跳响应
                break;
                
            default:
                console.log('[WS] 未知消息类型:', type);
        }
    }
    
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }
    
    sendPing() {
        this.send({ type: 'ping' });
    }
    
    requestStatus() {
        this.send({ type: 'get_status' });
    }
    
    requestTracks() {
        this.send({ type: 'get_tracks' });
    }
    
    startHeartbeat() {
        this.stopHeartbeat();
        this.heartbeatTimer = setInterval(() => {
            this.sendPing();
        }, IndoorFenceConfig.WS_HEARTBEAT_INTERVAL_MS);
    }
    
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts >= IndoorFenceConfig.WS_MAX_RECONNECT_ATTEMPTS) {
            console.error('[WS] 达到最大重连次数，启用仿真模式');
            if (IndoorFenceConfig.SIMULATION_FALLBACK_ONLY) {
                IndoorFenceState.simulationActive = true;
                this.handlers.onSimulationFallback?.();
            }
            return;
        }
        
        this.reconnectAttempts++;
        const delay = IndoorFenceConfig.WS_RECONNECT_DELAY_MS * Math.min(this.reconnectAttempts, 5);
        
        console.log(`[WS] ${delay}ms后重连 (尝试 ${this.reconnectAttempts}/${IndoorFenceConfig.WS_MAX_RECONNECT_ATTEMPTS})`);
        
        this.reconnectTimer = setTimeout(() => {
            this.connect(
                IndoorFenceState.clientId,
                IndoorFenceState.lastReceivedTraceId
            );
        }, delay);
    }
    
    disconnect() {
        this.stopHeartbeat();
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}


// =============================================================================
// 事件处理
// =============================================================================

class IndoorFenceEventProcessor {
    constructor() {
        this.eventHandlers = new Map();
    }
    
    registerHandler(eventType, handler) {
        if (!this.eventHandlers.has(eventType)) {
            this.eventHandlers.set(eventType, []);
        }
        this.eventHandlers.get(eventType).push(handler);
    }
    
    processEvent(event) {
        // 统一事件契约解析
        const {
            event_id,
            timestamp,
            type,
            location,
            value,
            confidence,
            evidence,
            suggestion,
            trace_id,
            alert_level,
            person_id,
            person_state
        } = event;
        
        // 添加到事件历史
        IndoorFenceState.events.unshift(event);
        if (IndoorFenceState.events.length > IndoorFenceConfig.EVENT_DISPLAY_LIMIT) {
            IndoorFenceState.events.pop();
        }
        
        // 调用对应处理器
        const handlers = this.eventHandlers.get(type) || [];
        for (const handler of handlers) {
            try {
                handler(event);
            } catch (error) {
                console.error(`[Event] 处理器错误 [${type}]:`, error);
            }
        }
        
        // 通用处理
        this.handleCommon(event);
    }
    
    handleCommon(event) {
        // 更新UI
        updateEventList(event);
        
        // 告警级别处理
        if (event.alert_level === 'alarm' || event.alert_level === 'critical') {
            this.handleAlarm(event);
        }
        
        // 处置建议自动执行
        if (event.suggestion?.auto_execute) {
            this.executeSuggestion(event.suggestion);
        }
    }
    
    handleAlarm(event) {
        // 增加告警计数
        IndoorFenceState.stats.activeAlarms++;
        
        // 播放告警声音
        if (IndoorFenceState.alertSoundEnabled) {
            playAlertSound(event.alert_level);
        }
        
        // 显示告警通知
        showAlertNotification(event);
        
        // 高亮相关轨迹
        if (event.person_id) {
            highlightTrack(event.person_id);
        }
    }
    
    executeSuggestion(suggestion) {
        console.log('[Suggestion] 执行:', suggestion);
        
        switch (suggestion.suggestion_type) {
            case 'verbal_warning':
                speakWarning(suggestion.message);
                break;
            case 'light_warning':
                // 通过API控制灯光 (实际实现)
                console.log('[Light] 灯光警示:', suggestion.parameters);
                break;
            case 'sound_alarm':
                playAlertSound('critical');
                break;
        }
    }
}


// =============================================================================
// 画布渲染
// =============================================================================

class IndoorFenceRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.trackHistory = new Map(); // track_id -> position history
    }
    
    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }
    
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    // 坐标转换: 世界坐标 -> 画布坐标
    worldToCanvas(worldX, worldY) {
        const scale = IndoorFenceConfig.COORD_SCALE;
        const canvasX = (worldX / scale.x) * this.canvas.width;
        const canvasY = (1 - worldY / scale.y) * this.canvas.height; // Y轴翻转
        return { x: canvasX, y: canvasY };
    }
    
    render(tracks, zones) {
        this.clear();
        
        // 绘制区域
        if (zones) {
            this.renderZones(zones);
        }
        
        // 绘制黄线
        if (zones?.yellow_line) {
            this.renderYellowLine(zones.yellow_line.y);
        }
        
        // 绘制机柜
        if (zones?.cabinets) {
            this.renderCabinets(zones.cabinets);
        }
        
        // 绘制轨迹
        for (const track of tracks) {
            this.renderTrack(track);
        }
        
        // 绘制信息面板
        this.renderInfoPanel(tracks.length);
    }
    
    renderZones(zones) {
        for (const zone of zones.zones || []) {
            const { polygon, zone_type, name } = zone;
            if (!polygon || polygon.length < 3) continue;
            
            this.ctx.beginPath();
            const first = this.worldToCanvas(polygon[0].x, polygon[0].y);
            this.ctx.moveTo(first.x, first.y);
            
            for (let i = 1; i < polygon.length; i++) {
                const pt = this.worldToCanvas(polygon[i].x, polygon[i].y);
                this.ctx.lineTo(pt.x, pt.y);
            }
            this.ctx.closePath();
            
            // 根据区域类型设置颜色
            this.ctx.fillStyle = zone_type === 'danger' ? 
                'rgba(255, 0, 0, 0.1)' : 'rgba(0, 255, 0, 0.1)';
            this.ctx.fill();
            
            this.ctx.strokeStyle = zone_type === 'danger' ? 
                'rgba(255, 0, 0, 0.5)' : 'rgba(0, 255, 0, 0.5)';
            this.ctx.stroke();
        }
    }
    
    renderYellowLine(y) {
        const left = this.worldToCanvas(0, y);
        const right = this.worldToCanvas(IndoorFenceConfig.COORD_SCALE.x, y);
        
        this.ctx.beginPath();
        this.ctx.moveTo(left.x, left.y);
        this.ctx.lineTo(right.x, right.y);
        this.ctx.strokeStyle = '#FFD700';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([10, 5]);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
        
        // 标签
        this.ctx.fillStyle = '#FFD700';
        this.ctx.font = '12px Arial';
        this.ctx.fillText('黄线', left.x + 5, left.y - 5);
    }
    
    renderCabinets(cabinets) {
        for (const cabinet of cabinets) {
            const { cabinet_id, name, x_start, x_end, y_line } = cabinet;
            
            const topLeft = this.worldToCanvas(x_start, y_line + 1);
            const bottomRight = this.worldToCanvas(x_end, y_line);
            const width = bottomRight.x - topLeft.x;
            const height = bottomRight.y - topLeft.y;
            
            // 机柜框
            this.ctx.fillStyle = 'rgba(100, 100, 100, 0.3)';
            this.ctx.fillRect(topLeft.x, topLeft.y, width, height);
            this.ctx.strokeStyle = '#666';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(topLeft.x, topLeft.y, width, height);
            
            // 标签
            this.ctx.fillStyle = '#333';
            this.ctx.font = '10px Arial';
            this.ctx.fillText(name || `机柜${cabinet_id}`, topLeft.x + 2, topLeft.y + 12);
        }
    }
    
    renderTrack(track) {
        const { track_id, ground_position, person_state, confidence, is_authorized } = track;
        const pos = this.worldToCanvas(ground_position.x, ground_position.y);
        
        // 更新轨迹历史
        if (!this.trackHistory.has(track_id)) {
            this.trackHistory.set(track_id, []);
        }
        const history = this.trackHistory.get(track_id);
        history.push(pos);
        if (history.length > IndoorFenceConfig.TRACK_HISTORY_LENGTH) {
            history.shift();
        }
        
        // 绘制轨迹线
        if (history.length > 1) {
            this.ctx.beginPath();
            this.ctx.moveTo(history[0].x, history[0].y);
            for (let i = 1; i < history.length; i++) {
                this.ctx.lineTo(history[i].x, history[i].y);
            }
            this.ctx.strokeStyle = 'rgba(0, 100, 255, 0.5)';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        }
        
        // 根据状态确定颜色
        let color = '#00FF00'; // 安全-绿色
        if (person_state === 'on_line') color = '#FFFF00'; // 压线-黄色
        if (person_state === 'cross_line') color = '#FF0000'; // 越线-红色
        if (person_state === 'unauthorized') color = '#FF6600'; // 未授权-橙色
        if (person_state === 'high_risk') color = '#FF00FF'; // 高风险-紫色
        
        // 绘制人员标记
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, 12, 0, Math.PI * 2);
        this.ctx.fillStyle = color;
        this.ctx.fill();
        this.ctx.strokeStyle = '#FFF';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // 置信度环
        this.ctx.beginPath();
        this.ctx.arc(pos.x, pos.y, 15, 0, Math.PI * 2 * confidence);
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();
        
        // 标签
        this.ctx.fillStyle = '#FFF';
        this.ctx.font = 'bold 10px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(track_id.substring(0, 6), pos.x, pos.y + 4);
        
        // 授权状态标记
        if (!is_authorized) {
            this.ctx.fillStyle = '#FF0000';
            this.ctx.font = 'bold 14px Arial';
            this.ctx.fillText('⚠', pos.x + 15, pos.y - 10);
        }
        
        // 选中高亮
        if (IndoorFenceState.selectedTrack === track_id) {
            this.ctx.beginPath();
            this.ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
            this.ctx.strokeStyle = '#00FFFF';
            this.ctx.lineWidth = 3;
            this.ctx.setLineDash([5, 3]);
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }
    }
    
    renderInfoPanel(trackCount) {
        const padding = 10;
        const panelHeight = 60;
        
        // 背景
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(padding, padding, 200, panelHeight);
        
        // 文字
        this.ctx.fillStyle = '#FFF';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'left';
        
        const statusText = IndoorFenceState.wsConnected ? '🟢 在线' : '🔴 离线';
        const modeText = IndoorFenceState.simulationActive ? '(仿真)' : '';
        
        this.ctx.fillText(`状态: ${statusText} ${modeText}`, padding + 10, padding + 20);
        this.ctx.fillText(`跟踪人数: ${trackCount}`, padding + 10, padding + 35);
        this.ctx.fillText(`告警数: ${IndoorFenceState.stats.activeAlarms}`, padding + 10, padding + 50);
    }
}


// =============================================================================
// UI 更新函数
// =============================================================================

function updateEventList(event) {
    const eventList = document.getElementById('event-list');
    if (!eventList) return;
    
    const eventItem = document.createElement('div');
    eventItem.className = `event-item event-${event.alert_level}`;
    eventItem.dataset.eventId = event.event_id;
    eventItem.dataset.traceId = event.trace_id;
    
    const time = new Date(event.timestamp).toLocaleTimeString();
    const typeMap = {
        'person_detected': '检测到人员',
        'yellow_line_warning': '⚠️ 压线预警',
        'yellow_line_alarm': '🚨 越线告警',
        'unauthorized_access': '🔒 未授权访问',
        'track_created': '轨迹创建',
        'track_deleted': '轨迹删除',
        'system_status': '系统状态'
    };
    
    eventItem.innerHTML = `
        <div class="event-time">${time}</div>
        <div class="event-type">${typeMap[event.type] || event.type}</div>
        <div class="event-person">${event.person_id || '-'}</div>
        <div class="event-confidence">${(event.confidence * 100).toFixed(0)}%</div>
        <div class="event-trace" title="${event.trace_id}">${event.trace_id.substring(0, 8)}...</div>
    `;
    
    // 点击查看详情
    eventItem.addEventListener('click', () => showEventDetail(event));
    
    // 插入到列表顶部
    eventList.insertBefore(eventItem, eventList.firstChild);
    
    // 限制列表长度
    while (eventList.children.length > IndoorFenceConfig.EVENT_DISPLAY_LIMIT) {
        eventList.removeChild(eventList.lastChild);
    }
    
    // 自动滚动
    if (IndoorFenceState.autoScroll) {
        eventList.scrollTop = 0;
    }
}

function showEventDetail(event) {
    const modal = document.getElementById('event-detail-modal');
    if (!modal) return;
    
    document.getElementById('event-detail-content').innerHTML = `
        <h3>事件详情</h3>
        <table class="event-detail-table">
            <tr><td>事件ID</td><td>${event.event_id}</td></tr>
            <tr><td>追踪ID</td><td>${event.trace_id}</td></tr>
            <tr><td>时间</td><td>${event.timestamp}</td></tr>
            <tr><td>类型</td><td>${event.type}</td></tr>
            <tr><td>告警级别</td><td>${event.alert_level}</td></tr>
            <tr><td>人员ID</td><td>${event.person_id || '-'}</td></tr>
            <tr><td>人员状态</td><td>${event.person_state || '-'}</td></tr>
            <tr><td>位置</td><td>(${event.location?.ground_position?.x?.toFixed(2)}, ${event.location?.ground_position?.y?.toFixed(2)})</td></tr>
            <tr><td>置信度</td><td>${(event.confidence * 100).toFixed(1)}%</td></tr>
            <tr><td>处置建议</td><td>${event.suggestion?.message || '-'}</td></tr>
        </table>
        
        <h4>证据链</h4>
        <div class="evidence-section">
            ${event.evidence?.frame_annotated ? 
                `<img src="${event.evidence.frame_annotated}" alt="标注帧" class="evidence-image" />` : 
                '<p>无图像证据</p>'}
        </div>
        
        <h4>原始数据</h4>
        <pre class="event-raw">${JSON.stringify(event, null, 2)}</pre>
    `;
    
    modal.style.display = 'block';
}

function showAlertNotification(event) {
    // 使用浏览器通知API
    if (Notification.permission === 'granted') {
        new Notification('室内电子围栏告警', {
            body: event.suggestion?.message || event.type,
            icon: '/static/images/alert-icon.png',
            tag: event.event_id
        });
    }
    
    // 页面内通知
    const notification = document.createElement('div');
    notification.className = `notification notification-${event.alert_level}`;
    notification.innerHTML = `
        <div class="notification-title">🚨 ${event.type}</div>
        <div class="notification-body">${event.suggestion?.message || ''}</div>
        <button class="notification-close">&times;</button>
    `;
    
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.remove();
    });
    
    document.getElementById('notification-container')?.appendChild(notification);
    
    // 5秒后自动关闭
    setTimeout(() => notification.remove(), 5000);
}

function highlightTrack(trackId) {
    IndoorFenceState.selectedTrack = trackId;
    // 触发重绘
}

function playAlertSound(level) {
    const audio = document.getElementById(`alert-sound-${level}`) || 
                  document.getElementById('alert-sound-default');
    if (audio) {
        audio.currentTime = 0;
        audio.play().catch(e => console.log('无法播放声音:', e));
    }
}

function speakWarning(message) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(message);
        utterance.lang = 'zh-CN';
        utterance.rate = 1.2;
        speechSynthesis.speak(utterance);
    }
}


// =============================================================================
// 主初始化
// =============================================================================

let apiClient = null;
let wsClient = null;
let eventProcessor = null;
let renderer = null;
let refreshTimer = null;

async function initIndoorFence() {
    console.log(`[IndoorFence] 初始化 V${IndoorFenceState.version}`);
    
    // 初始化API客户端
    apiClient = new IndoorFenceApiClient(IndoorFenceConfig.API_BASE);
    
    // 初始化事件处理器
    eventProcessor = new IndoorFenceEventProcessor();
    
    // 注册事件处理器
    eventProcessor.registerHandler('yellow_line_alarm', (event) => {
        console.log('[Alarm] 越线告警:', event);
    });
    eventProcessor.registerHandler('unauthorized_access', (event) => {
        console.log('[Alarm] 未授权访问:', event);
    });
    
    // 初始化画布渲染器
    const canvas = document.getElementById('fence-canvas');
    if (canvas) {
        renderer = new IndoorFenceRenderer(canvas);
        renderer.resize();
        window.addEventListener('resize', () => renderer.resize());
    }
    
    // 初始化WebSocket
    wsClient = new IndoorFenceWebSocket(IndoorFenceConfig.WS_URL, {
        onConnect: () => {
            updateConnectionStatus(true);
        },
        onDisconnect: () => {
            updateConnectionStatus(false);
        },
        onEvent: (event) => {
            eventProcessor.processEvent(event);
        },
        onStatus: (status) => {
            IndoorFenceState.systemStatus = status;
            updateStatusPanel(status);
        },
        onTracks: (tracks) => {
            IndoorFenceState.tracks = tracks;
            if (renderer) {
                renderer.render(tracks, IndoorFenceState.zones);
            }
        },
        onError: (error) => {
            console.error('[WS] 错误:', error);
        },
        onSimulationFallback: () => {
            console.warn('[WS] 启用仿真模式作为fallback');
            startSimulationFallback();
        }
    });
    
    // 加载初始数据
    try {
        // 加载区域配置
        IndoorFenceState.zones = await apiClient.getZones();
        
        // 加载初始状态
        IndoorFenceState.systemStatus = await apiClient.getStatus();
        updateStatusPanel(IndoorFenceState.systemStatus);
        
        // 加载初始轨迹
        IndoorFenceState.tracks = await apiClient.getTracks();
        if (renderer) {
            renderer.render(IndoorFenceState.tracks, IndoorFenceState.zones);
        }
        
        // 连接WebSocket
        wsClient.connect();
        
    } catch (error) {
        console.error('[IndoorFence] 初始化失败:', error);
        
        // API不可用时启用仿真fallback
        if (IndoorFenceConfig.SIMULATION_FALLBACK_ONLY) {
            console.warn('[IndoorFence] API不可用，启用仿真模式');
            startSimulationFallback();
        }
    }
    
    // 启动定时刷新 (作为WebSocket的备份)
    startRefreshTimer();
    
    // 请求通知权限
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
    
    console.log('[IndoorFence] 初始化完成');
}

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('connection-indicator');
    if (indicator) {
        indicator.className = connected ? 'connected' : 'disconnected';
        indicator.title = connected ? '已连接' : '已断开';
    }
}

function updateStatusPanel(status) {
    const panel = document.getElementById('status-panel');
    if (!panel) return;
    
    panel.innerHTML = `
        <div class="status-item">
            <span class="status-label">系统状态</span>
            <span class="status-value status-${status.status}">${status.status}</span>
        </div>
        <div class="status-item">
            <span class="status-label">跟踪人数</span>
            <span class="status-value">${status.tracked_persons}</span>
        </div>
        <div class="status-item">
            <span class="status-label">活跃告警</span>
            <span class="status-value ${status.active_alarms > 0 ? 'alarm' : ''}">${status.active_alarms}</span>
        </div>
        <div class="status-item">
            <span class="status-label">处理帧数</span>
            <span class="status-value">${status.frame_count}</span>
        </div>
        <div class="status-item">
            <span class="status-label">处理延迟</span>
            <span class="status-value">${status.last_process_time_ms?.toFixed(1) || 0}ms</span>
        </div>
    `;
    
    IndoorFenceState.stats.totalPersons = status.tracked_persons;
    IndoorFenceState.stats.activeAlarms = status.active_alarms;
}

function startRefreshTimer() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
    }
    
    refreshTimer = setInterval(async () => {
        // 仅在WebSocket断开时使用API轮询
        if (!IndoorFenceState.wsConnected && !IndoorFenceState.simulationActive) {
            try {
                const tracks = await apiClient.getTracks();
                IndoorFenceState.tracks = tracks;
                if (renderer) {
                    renderer.render(tracks, IndoorFenceState.zones);
                }
            } catch (error) {
                console.error('[Refresh] API轮询失败:', error);
            }
        }
    }, IndoorFenceConfig.REFRESH_INTERVAL_MS);
}

function startSimulationFallback() {
    // 仿真模式仅作为紧急fallback
    // 生产环境应当显示错误而非仿真数据
    console.warn('[Simulation] 启用仿真模式 - 仅用于调试/演示');
    
    IndoorFenceState.simulationActive = true;
    IndoorFenceState.simulationMode = true;
    
    // 显示仿真模式警告
    const warning = document.createElement('div');
    warning.id = 'simulation-warning';
    warning.className = 'simulation-warning';
    warning.innerHTML = `
        <span>⚠️ 仿真模式 - API连接失败，显示模拟数据</span>
        <button onclick="retryConnection()">重试连接</button>
    `;
    document.body.prepend(warning);
}

function retryConnection() {
    const warning = document.getElementById('simulation-warning');
    if (warning) warning.remove();
    
    IndoorFenceState.simulationActive = false;
    wsClient.reconnectAttempts = 0;
    wsClient.connect();
}


// =============================================================================
// 页面加载
// =============================================================================

document.addEventListener('DOMContentLoaded', initIndoorFence);


// =============================================================================
// 导出 (供外部调用)
// =============================================================================

window.IndoorFence = {
    state: IndoorFenceState,
    config: IndoorFenceConfig,
    api: () => apiClient,
    ws: () => wsClient,
    renderer: () => renderer,
    
    // 公开方法
    selectTrack: (trackId) => {
        IndoorFenceState.selectedTrack = trackId;
    },
    authorize: async (personId, cabinetIds) => {
        return apiClient.updateAuthorization(personId, cabinetIds);
    },
    getEvents: async (params) => {
        return apiClient.getEvents(params);
    }
};
