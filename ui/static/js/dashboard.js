/**
 * 激光星芒破夜绘明监测中心 Dashboard JavaScript
 * 输变电监测平台 V3.0
 *
 * 功能:
 * - 统一模块切换与管理
 * - 实时数据轮询与WebSocket推送
 * - 检测结果可视化
 * - 告警管理与联动
 * - 多摄像头切换
 * - 模块功能面板集成
 */

// =============================================================================
// 全局状态
// =============================================================================
const DashboardState = {
    version: '3.0.0',
    currentModule: 'transformer',
    currentCamera: 'cam1',
    currentVoltage: '110kv',
    currentSite: null,

    // WebSocket连接
    ws: null,
    wsConnected: false,
    wsReconnectTimer: null,

    // 监测状态
    isPatrolling: false,
    isPaused: false,

    // 检测结果
    detections: [],
    regions: [],
    alarms: [],

    // 性能指标
    metrics: {
        fps: 0,
        latency: 0,
        gpuUsage: 0,
        memoryUsage: 0
    },

    // 模块配置 (V3.0重组)
    modules: {
        // 室外巡视模块
        transformer: { name: '主变巡视', icon: 'box', type: 'outdoor', loaded: false },
        switch: { name: '开关巡视', icon: 'toggle-on', type: 'outdoor', loaded: false },
        busbar: { name: '母线巡视', icon: 'diagram-3', type: 'outdoor', loaded: false },
        capacitor: { name: '电容器巡视', icon: 'battery-charging', type: 'outdoor', loaded: false },
        meter: { name: '表计读数', icon: 'speedometer2', type: 'outdoor', loaded: false },
        bird: { name: '鸟类监控', icon: 'feather', type: 'outdoor', loaded: false },
        // 高级监测模块
        acoustic: { name: '声学监测', icon: 'soundwave', type: 'advanced', loaded: false },
        gas: { name: '气体检测', icon: 'cloud-haze2', type: 'advanced', loaded: false },
        hyperspectral: { name: '高光谱检测', icon: 'rainbow', type: 'advanced', loaded: false },
        slam: { name: 'SLAM建图', icon: 'map', type: 'advanced', loaded: false },
        fusion: { name: '多模态融合', icon: 'diagram-3-fill', type: 'advanced', loaded: false }
    },

    // 轮询定时器
    pollTimer: null,
    pollInterval: 1000, // 1秒
};


// =============================================================================
// 初始化
// =============================================================================
function initDashboard() {
    console.log(`[Dashboard] 初始化 V${DashboardState.version}`);

    // 初始化视频画布
    initVideoCanvas();

    // 加载站点列表
    loadSites();

    // 加载初始模块状态
    loadModuleStatus();

    // 连接WebSocket
    connectWebSocket();

    // 启动数据轮询
    startDataPolling();

    // 更新时间显示
    startTimeUpdate();

    // 绑定事件
    bindEvents();

    // 加载初始区域和告警
    loadRegions();
    loadAlarms();

    // V3.0: 初始化模块功能面板
    loadModuleControlPanel(DashboardState.currentModule);

    console.log('[Dashboard] 初始化完成');
}


// =============================================================================
// 视频画布
// =============================================================================
function initVideoCanvas() {
    const canvas = document.getElementById('main-video-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    // 设置画布尺寸
    canvas.width = 1920;
    canvas.height = 1080;

    // 绘制占位图
    ctx.fillStyle = '#0f0f1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#444';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('等待视频流连接...', canvas.width / 2, canvas.height / 2);
}


// =============================================================================
// WebSocket 连接
// =============================================================================
function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/api/advanced/ws/monitoring`;

    console.log(`[WS] 连接到 ${wsUrl}`);

    try {
        DashboardState.ws = new WebSocket(wsUrl);

        DashboardState.ws.onopen = () => {
            console.log('[WS] 连接成功');
            DashboardState.wsConnected = true;
            updateConnectionStatus(true);

            // 订阅当前模块
            subscribeModule(DashboardState.currentModule);
        };

        DashboardState.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error('[WS] 消息解析失败:', e);
            }
        };

        DashboardState.ws.onclose = () => {
            console.log('[WS] 连接断开');
            DashboardState.wsConnected = false;
            updateConnectionStatus(false);

            // 自动重连
            if (!DashboardState.wsReconnectTimer) {
                DashboardState.wsReconnectTimer = setTimeout(() => {
                    DashboardState.wsReconnectTimer = null;
                    connectWebSocket();
                }, 5000);
            }
        };

        DashboardState.ws.onerror = (error) => {
            console.error('[WS] 错误:', error);
        };

    } catch (e) {
        console.error('[WS] 创建连接失败:', e);
    }
}

function subscribeModule(moduleId) {
    if (DashboardState.ws && DashboardState.ws.readyState === WebSocket.OPEN) {
        DashboardState.ws.send(JSON.stringify({
            type: 'subscribe',
            device_id: DashboardState.currentCamera,
            modalities: [moduleId]
        }));
    }
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'subscribed':
            console.log('[WS] 已订阅:', data.modalities);
            break;

        case 'detection':
            handleDetectionUpdate(data.payload);
            break;

        case 'alarm':
            handleAlarmUpdate(data.payload);
            break;

        case 'frame':
            handleFrameUpdate(data.payload);
            break;

        case 'metrics':
            handleMetricsUpdate(data.payload);
            break;

        case 'pong':
            // 心跳响应
            break;

        default:
            console.log('[WS] 未知消息类型:', data.type);
    }
}


// =============================================================================
// 数据轮询 (备用方案，当WebSocket不可用时)
// =============================================================================
function startDataPolling() {
    if (DashboardState.pollTimer) {
        clearInterval(DashboardState.pollTimer);
    }

    DashboardState.pollTimer = setInterval(async () => {
        // 如果WebSocket已连接，减少轮询频率
        if (DashboardState.wsConnected) {
            return;
        }

        try {
            await pollMonitorData();
        } catch (e) {
            console.error('[Poll] 轮询失败:', e);
        }
    }, DashboardState.pollInterval);
}

async function pollMonitorData() {
    try {
        const params = new URLSearchParams({
            camera_id: DashboardState.currentCamera,
            module: DashboardState.currentModule,
            voltage: DashboardState.currentVoltage
        });

        const response = await fetch(`/api/ui/monitor?${params}`);

        if (response.ok) {
            const data = await response.json();
            processMonitorData(data);
        }
    } catch (e) {
        console.error('[Poll] 请求失败:', e);
    }
}

function processMonitorData(data) {
    // 处理各模块数据
    if (data.transformer) {
        updateModuleData('transformer', data.transformer);
    }
    if (data.switch) {
        updateModuleData('switch', data.switch);
    }
    if (data.acoustic) {
        updateModuleData('acoustic', data.acoustic);
    }
    if (data.gas) {
        updateModuleData('gas', data.gas);
    }
    // ... 其他模块

    // 更新检测结果
    if (data.detections) {
        updateDetections(data.detections);
    }

    // 更新告警
    if (data.alarms) {
        mergeAlarms(data.alarms);
    }
}


// =============================================================================
// 模块切换
// =============================================================================
function switchModule(moduleId) {
    if (moduleId === DashboardState.currentModule) return;

    console.log(`[Module] 切换到: ${moduleId}`);

    // 更新UI
    document.querySelectorAll('.module-item').forEach(item => {
        item.classList.remove('active');
    });

    const targetItem = document.querySelector(`.module-item[data-module="${moduleId}"]`);
    if (targetItem) {
        targetItem.classList.add('active');
    }

    // 更新状态
    DashboardState.currentModule = moduleId;

    // 重新订阅WebSocket
    subscribeModule(moduleId);

    // 加载模块数据
    loadModuleData(moduleId);

    // 更新区域列表
    loadRegions();

    // V3.0: 切换功能控制面板
    loadModuleControlPanel(moduleId);
}

/**
 * V3.0新增: 加载模块功能控制面板
 */
function loadModuleControlPanel(moduleId) {
    // 隐藏所有模块控制面板
    document.querySelectorAll('.module-control-panel').forEach(panel => {
        panel.style.display = 'none';
    });

    // 显示当前模块的控制面板
    const targetPanel = document.getElementById(`panel-${moduleId}`);
    if (targetPanel) {
        targetPanel.style.display = 'block';
        console.log(`[Controls] 显示模块面板: ${moduleId}`);
    } else {
        console.log(`[Controls] 模块 ${moduleId} 无专属控制面板`);
    }
}

/**
 * V3.0新增: 应用模块设置
 */
function applyModuleSettings() {
    const moduleId = DashboardState.currentModule;
    const settings = collectModuleSettings(moduleId);

    console.log(`[Settings] 应用 ${moduleId} 设置:`, settings);

    // 发送设置到后端
    fetch(`/api/module/${moduleId}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('设置已应用', 'success');
        } else {
            showNotification('设置应用失败: ' + (data.message || ''), 'error');
        }
    })
    .catch(error => {
        console.error('[Settings] 应用失败:', error);
        showNotification('设置应用失败', 'error');
    });
}

/**
 * V3.0新增: 收集模块设置
 */
function collectModuleSettings(moduleId) {
    const settings = {};
    const panel = document.getElementById(`panel-${moduleId}`);

    if (!panel) return settings;

    // 收集所有checkbox设置
    panel.querySelectorAll('input[type="checkbox"]').forEach(input => {
        settings[input.id] = input.checked;
    });

    // 收集所有number/text输入
    panel.querySelectorAll('input[type="number"], input[type="text"]').forEach(input => {
        settings[input.id || input.name] = input.value;
    });

    return settings;
}

/**
 * V3.0新增: 显示通知
 */
function showNotification(message, type = 'info') {
    console.log(`[Notification] ${type}: ${message}`);
    // 可以在这里添加Toast通知UI
}

async function loadModuleData(moduleId) {
    try {
        const response = await fetch(`/api/advanced/${moduleId}/status`);

        if (response.ok) {
            const data = await response.json();
            updateModuleDisplay(moduleId, data);
        }
    } catch (e) {
        console.error(`[Module] 加载 ${moduleId} 数据失败:`, e);
    }
}

function updateModuleData(moduleId, data) {
    DashboardState.modules[moduleId].data = data;

    // 更新状态显示
    const statusEl = document.querySelector(`.module-item[data-module="${moduleId}"] .module-item-status`);
    if (statusEl) {
        statusEl.textContent = data.status || '运行中';
        statusEl.className = `module-item-status ${data.loaded ? 'ready' : ''}`;
    }

    // 如果是当前模块，更新主显示
    if (moduleId === DashboardState.currentModule) {
        updateMainDisplay(data);
    }
}


// =============================================================================
// 检测结果可视化
// =============================================================================
function updateDetections(detections) {
    DashboardState.detections = detections;
    renderDetectionBoxes(detections);
}

function renderDetectionBoxes(detections) {
    const overlay = document.getElementById('detection-overlay');
    if (!overlay) return;

    // 清除现有检测框
    overlay.innerHTML = '';

    const canvas = document.getElementById('main-video-canvas');
    const canvasRect = canvas.getBoundingClientRect();

    detections.forEach((det, idx) => {
        const box = document.createElement('div');
        box.className = `detection-box ${det.level || 'normal'}`;

        // 计算相对位置 (假设坐标是归一化的 0-1)
        const left = det.x * canvasRect.width;
        const top = det.y * canvasRect.height;
        const width = det.width * canvasRect.width;
        const height = det.height * canvasRect.height;

        box.style.cssText = `
            left: ${left}px;
            top: ${top}px;
            width: ${width}px;
            height: ${height}px;
        `;

        // 添加标签
        const label = document.createElement('div');
        label.className = 'detection-label';
        label.textContent = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
        box.appendChild(label);

        // 点击事件
        box.addEventListener('click', () => {
            selectRegion(idx);
        });

        overlay.appendChild(box);
    });
}

function handleDetectionUpdate(payload) {
    if (payload.detections) {
        updateDetections(payload.detections);
    }
}


// =============================================================================
// 识别区域管理
// =============================================================================
async function loadRegions() {
    try {
        const response = await fetch(`/api/ui/regions?module=${DashboardState.currentModule}&camera=${DashboardState.currentCamera}`);

        if (response.ok) {
            const data = await response.json();
            DashboardState.regions = data.regions || [];
            renderRegionList(DashboardState.regions);
        } else {
            // 使用模拟数据
            renderRegionList(getMockRegions());
        }
    } catch (e) {
        console.error('[Regions] 加载失败:', e);
        renderRegionList(getMockRegions());
    }
}

function getMockRegions() {
    return [
        { id: 1, name: '储油柜', type: '设备外观查看', status: 'normal', confidence: 0.95 },
        { id: 2, name: '冷却系统', type: '设备外观查看', status: 'normal', confidence: 0.92 },
        { id: 3, name: '取气阀', type: '设备外观查看', status: 'warning', confidence: 0.88 },
        { id: 4, name: '地层及接地引', type: '设备外观查看', status: 'normal', confidence: 0.91 },
        { id: 5, name: '本体油位窗A侧', type: '表计读数', status: 'normal', confidence: 0.89 },
        { id: 6, name: '本体顶部绝缘件', type: '设备外观查看', status: 'normal', confidence: 0.93 },
        { id: 7, name: '风扇/端子箱/5', type: '设备外观查看', status: 'normal', confidence: 0.90 },
        { id: 8, name: '套压输入附着物', type: '表计读数', status: 'normal', confidence: 0.87 }
    ];
}

function renderRegionList(regions) {
    const listEl = document.getElementById('region-list');
    const countEl = document.getElementById('region-count');

    if (countEl) {
        countEl.textContent = regions.length;
    }

    if (!listEl) return;

    listEl.innerHTML = regions.map((region, idx) => `
        <li class="region-item" data-id="${region.id}" onclick="selectRegion(${idx})">
            <div class="region-icon ${region.status}">
                <i class="bi bi-${getRegionIcon(region.type)}"></i>
            </div>
            <div class="region-info">
                <div class="region-name">${region.name}</div>
                <div class="region-status">${region.type}</div>
            </div>
            <div class="region-confidence">${(region.confidence * 100).toFixed(0)}%</div>
        </li>
    `).join('');
}

function getRegionIcon(type) {
    const icons = {
        '设备外观查看': 'eye',
        '表计读数': 'speedometer',
        '状态识别': 'check-circle',
        '缺陷检测': 'exclamation-triangle'
    };
    return icons[type] || 'square';
}

function selectRegion(idx) {
    // 高亮选中的区域
    document.querySelectorAll('.region-item').forEach((item, i) => {
        item.classList.toggle('selected', i === idx);
    });

    // 聚焦到对应的检测框
    const boxes = document.querySelectorAll('.detection-box');
    boxes.forEach((box, i) => {
        box.style.zIndex = i === idx ? 100 : 1;
        if (i === idx) {
            box.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
}

function configureRegions() {
    // 打开区域配置模态框
    alert('区域配置功能开发中...');
}


// =============================================================================
// 告警管理
// =============================================================================
async function loadAlarms() {
    try {
        const response = await fetch('/api/advanced/alarms?limit=50');

        if (response.ok) {
            const data = await response.json();
            DashboardState.alarms = data.alarms || [];
            renderAlarmList(DashboardState.alarms);
        } else {
            renderAlarmList(getMockAlarms());
        }
    } catch (e) {
        console.error('[Alarms] 加载失败:', e);
        renderAlarmList(getMockAlarms());
    }
}

function getMockAlarms() {
    return [
        { id: 1, type: '声学异常', level: 'critical', message: '检测到局部放电信号', time: '16:52:30', module: 'acoustic' },
        { id: 2, type: '温度告警', level: 'alarm', message: '套管温升超过阈值', time: '16:45:12', module: 'transformer' },
        { id: 3, type: '油位注意', level: 'warning', message: '油位略低于正常范围', time: '16:30:05', module: 'transformer' }
    ];
}

function renderAlarmList(alarms) {
    const listEl = document.getElementById('alarm-list');
    const countEl = document.getElementById('alarm-count');

    const criticalCount = alarms.filter(a => a.level === 'critical').length;
    if (countEl) {
        countEl.textContent = criticalCount || alarms.length;
        countEl.className = `badge ${criticalCount > 0 ? 'bg-danger' : 'bg-warning'}`;
    }

    if (!listEl) return;

    listEl.innerHTML = alarms.map(alarm => `
        <li class="alarm-item ${alarm.level}">
            <div class="alarm-header">
                <span class="alarm-type">
                    <i class="bi bi-${getAlarmIcon(alarm.level)}"></i>
                    ${alarm.type}
                </span>
                <span class="alarm-time">${alarm.time}</span>
            </div>
            <div class="alarm-message">${alarm.message}</div>
        </li>
    `).join('');
}

function getAlarmIcon(level) {
    const icons = {
        critical: 'exclamation-octagon-fill',
        alarm: 'exclamation-triangle-fill',
        warning: 'exclamation-circle-fill'
    };
    return icons[level] || 'info-circle';
}

function handleAlarmUpdate(payload) {
    // 添加新告警到列表顶部
    DashboardState.alarms.unshift(payload);
    renderAlarmList(DashboardState.alarms);

    // 显示通知
    showAlarmNotification(payload);

    // 更新模块告警状态
    if (payload.module) {
        const moduleItem = document.querySelector(`.module-item[data-module="${payload.module}"]`);
        if (moduleItem) {
            moduleItem.classList.add('has-alarm');
        }
    }
}

function mergeAlarms(newAlarms) {
    const existingIds = new Set(DashboardState.alarms.map(a => a.id));
    const toAdd = newAlarms.filter(a => !existingIds.has(a.id));

    if (toAdd.length > 0) {
        DashboardState.alarms = [...toAdd, ...DashboardState.alarms].slice(0, 100);
        renderAlarmList(DashboardState.alarms);
    }
}

function showAlarmNotification(alarm) {
    // 可以使用 Notification API 或 Toast
    console.log(`[Alarm] ${alarm.level}: ${alarm.message}`);
}

function clearAlarms() {
    DashboardState.alarms = DashboardState.alarms.filter(a => a.level === 'critical');
    renderAlarmList(DashboardState.alarms);
}


// =============================================================================
// 摄像头切换
// =============================================================================
function switchCamera(cameraId) {
    if (cameraId === DashboardState.currentCamera) return;

    console.log(`[Camera] 切换到: ${cameraId}`);

    // 更新UI
    document.querySelectorAll('.thumbnail').forEach(item => {
        item.classList.toggle('active', item.dataset.camera === cameraId);
    });

    // 更新状态
    DashboardState.currentCamera = cameraId;

    // 重新订阅
    subscribeModule(DashboardState.currentModule);

    // 重新加载区域
    loadRegions();
}


// =============================================================================
// 性能指标
// =============================================================================
function handleMetricsUpdate(payload) {
    DashboardState.metrics = { ...DashboardState.metrics, ...payload };
    updateMetricsDisplay();
}

function updateMetricsDisplay() {
    const { fps, latency, gpuUsage } = DashboardState.metrics;

    const fpsEl = document.getElementById('metric-fps');
    const latencyEl = document.getElementById('metric-latency');
    const gpuEl = document.getElementById('metric-gpu');

    if (fpsEl) fpsEl.textContent = fps.toFixed(1);
    if (latencyEl) latencyEl.textContent = `${latency.toFixed(0)}ms`;
    if (gpuEl) gpuEl.textContent = `${gpuUsage.toFixed(0)}%`;
}


// =============================================================================
// 模块状态加载
// =============================================================================
async function loadModuleStatus() {
    try {
        const response = await fetch('/api/advanced/plugins/status');

        if (response.ok) {
            const data = await response.json();

            for (const [pluginId, status] of Object.entries(data.plugins || {})) {
                // 映射插件ID到模块ID
                const moduleId = mapPluginToModule(pluginId);
                if (moduleId && DashboardState.modules[moduleId]) {
                    DashboardState.modules[moduleId].loaded = status.loaded;
                    updateModuleStatusUI(moduleId, status);
                }
            }
        }
    } catch (e) {
        console.error('[Status] 加载模块状态失败:', e);
    }
}

function mapPluginToModule(pluginId) {
    const mapping = {
        'acoustic_monitoring': 'acoustic',
        'gas_detection': 'gas',
        'hyperspectral_detection': 'hyperspectral',
        'slam_mapping': 'slam',
        'multimodal_fusion': 'fusion',
        'transformer_inspection': 'transformer',
        'switch_inspection': 'switch',
        'busbar_inspection': 'busbar',
        'capacitor_inspection': 'capacitor',
        'meter_reading': 'meter'
    };
    return mapping[pluginId] || pluginId;
}

function updateModuleStatusUI(moduleId, status) {
    const item = document.querySelector(`.module-item[data-module="${moduleId}"]`);
    if (!item) return;

    const statusEl = item.querySelector('.module-item-status');
    if (statusEl) {
        if (status.loaded) {
            statusEl.textContent = '已就绪';
            statusEl.classList.add('ready');
        } else if (status.error) {
            statusEl.textContent = '错误';
            statusEl.classList.remove('ready');
        } else {
            statusEl.textContent = '未加载';
            statusEl.classList.remove('ready');
        }
    }
}


// =============================================================================
// 站点加载
// =============================================================================
async function loadSites() {
    try {
        const response = await fetch('/api/sites');

        if (response.ok) {
            const sites = await response.json();
            const select = document.getElementById('site-selector');

            if (select) {
                select.innerHTML = '<option value="">选择站点...</option>' +
                    sites.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
            }
        }
    } catch (e) {
        console.error('[Sites] 加载失败:', e);
    }
}


// =============================================================================
// 面板切换
// =============================================================================
function switchPanel(panelId) {
    // 更新Tab
    document.querySelectorAll('.panel-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.panel === panelId);
    });

    // 更新内容
    document.querySelectorAll('.panel-section').forEach(section => {
        section.classList.toggle('active', section.id === `panel-${panelId}`);
    });
}


// =============================================================================
// 时间更新
// =============================================================================
function startTimeUpdate() {
    function updateTime() {
        const now = new Date();
        const weekdays = ['星期日', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六'];

        const dateStr = `${now.getFullYear()}年${String(now.getMonth() + 1).padStart(2, '0')}月${String(now.getDate()).padStart(2, '0')}日`;
        const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

        const el = document.getElementById('current-datetime');
        if (el) {
            el.textContent = `${dateStr} ${weekdays[now.getDay()]} ${timeStr}`;
        }
    }

    updateTime();
    setInterval(updateTime, 1000);
}


// =============================================================================
// 连接状态
// =============================================================================
function updateConnectionStatus(connected) {
    const indicators = document.querySelectorAll('.status-item');

    indicators.forEach(item => {
        if (item.textContent.includes('WebSocket')) {
            const dot = item.querySelector('.status-dot');
            if (dot) {
                dot.classList.toggle('error', !connected);
            }
            item.querySelector('span:last-child').textContent = `WebSocket: ${connected ? '已连接' : '已断开'}`;
        }
    });
}


// =============================================================================
// 事件绑定
// =============================================================================
function bindEvents() {
    // 电压等级选择
    const voltageSelector = document.getElementById('voltage-selector');
    if (voltageSelector) {
        voltageSelector.addEventListener('change', (e) => {
            DashboardState.currentVoltage = e.target.value;
            loadRegions();
        });
    }

    // 站点选择
    const siteSelector = document.getElementById('site-selector');
    if (siteSelector) {
        siteSelector.addEventListener('change', (e) => {
            DashboardState.currentSite = e.target.value;
            loadModuleData(DashboardState.currentModule);
        });
    }
}


// =============================================================================
// 全局操作函数
// =============================================================================
function openSettings() {
    window.location.href = '/settings';
}

function openTraining() {
    window.location.href = '/training';
}

function startAutoPatrol() {
    if (DashboardState.isPatrolling) {
        stopPatrol();
    } else {
        startPatrol();
    }
}

async function startPatrol() {
    try {
        const response = await fetch('/api/patrol/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                modules: Object.keys(DashboardState.modules),
                voltage: DashboardState.currentVoltage,
                site: DashboardState.currentSite
            })
        });

        if (response.ok) {
            DashboardState.isPatrolling = true;
            updatePatrolButton();
            console.log('[Patrol] 开始巡检');
        }
    } catch (e) {
        console.error('[Patrol] 启动失败:', e);
    }
}

async function stopPatrol() {
    try {
        await fetch('/api/patrol/stop', { method: 'POST' });
        DashboardState.isPatrolling = false;
        updatePatrolButton();
        console.log('[Patrol] 停止巡检');
    } catch (e) {
        console.error('[Patrol] 停止失败:', e);
    }
}

function updatePatrolButton() {
    const btn = document.querySelector('.toolbar-controls .btn-primary');
    if (btn) {
        if (DashboardState.isPatrolling) {
            btn.innerHTML = '<i class="bi bi-stop-fill"></i> 停止巡检';
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-danger');
        } else {
            btn.innerHTML = '<i class="bi bi-play-fill"></i> 开始巡检';
            btn.classList.remove('btn-danger');
            btn.classList.add('btn-primary');
        }
    }
}


// =============================================================================
// 帧更新处理
// =============================================================================
function handleFrameUpdate(payload) {
    if (payload.image) {
        renderFrame(payload.image);
    }
    if (payload.detections) {
        updateDetections(payload.detections);
    }
}

function renderFrame(imageData) {
    const canvas = document.getElementById('main-video-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };

    img.src = `data:image/jpeg;base64,${imageData}`;
}


// =============================================================================
// 导出全局函数 (供HTML调用)
// =============================================================================
window.initDashboard = initDashboard;
window.switchModule = switchModule;
window.switchCamera = switchCamera;
window.switchPanel = switchPanel;
window.selectRegion = selectRegion;
window.configureRegions = configureRegions;
window.clearAlarms = clearAlarms;
window.openSettings = openSettings;
window.openTraining = openTraining;
window.startAutoPatrol = startAutoPatrol;
