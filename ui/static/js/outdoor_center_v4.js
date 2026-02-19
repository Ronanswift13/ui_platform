/**
 * 室外监测中心 V4.0 - 完整JavaScript实现
 * 输变电激光星芒破夜绘明监测平台
 * 版本: 4.0.0
 */

const OutdoorState = {
    version: '4.0.0',
    currentModule: 'transformer',
    isPatrolling: false,
    wsConnection: null,
    refreshTimer: null,
    
    modules: {
        transformer: { name: '主变巡视', icon: 'box', api: '/api/outdoor/transformer' },
        switch: { name: '开关巡视', icon: 'toggle-on', api: '/api/outdoor/switch' },
        busbar: { name: '母线巡视', icon: 'diagram-3', api: '/api/outdoor/busbar' },
        capacitor: { name: '电容器', icon: 'battery-charging', api: '/api/outdoor/capacitor' },
        meter: { name: '表计读数', icon: 'speedometer2', api: '/api/outdoor/meter' },
        bird: { name: '鸟类监控', icon: 'feather', api: '/api/outdoor/bird' },
        acoustic: { name: '声学监测', icon: 'soundwave', api: '/api/outdoor/acoustic' },
        gas: { name: '气体检测', icon: 'cloud-haze2', api: '/api/outdoor/gas' },
        hyperspectral: { name: '高光谱检测', icon: 'rainbow', api: '/api/outdoor/hyperspectral' },
        slam: { name: 'SLAM建图', icon: 'map', api: '/api/outdoor/slam' },
        fusion: { name: '多模态融合', icon: 'diagram-3-fill', api: '/api/outdoor/fusion' }
    },
    
    currentData: null,
    allModulesData: {}
};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log(`[OutdoorCenter] 初始化 V${OutdoorState.version}`);
    initCanvas();
    initWebSocket();
    loadModuleData(OutdoorState.currentModule);
    updateDateTime();
    setInterval(updateDateTime, 1000);
    startAutoRefresh();
    renderPluginList();
});

// 画布初始化
function initCanvas() {
    const canvas = document.getElementById('main-video-canvas');
    if (!canvas) return;
    
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    drawDefaultScene(canvas);
    
    window.addEventListener('resize', () => {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        drawDefaultScene(canvas);
    });
}

function drawDefaultScene(canvas) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    
    // 背景
    const gradient = ctx.createLinearGradient(0, 0, w, h);
    gradient.addColorStop(0, '#0a1628');
    gradient.addColorStop(1, '#1a2744');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);
    
    // 网格
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 50) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 50) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    
    drawDeviceVisualization(ctx, w, h);
}

function drawDeviceVisualization(ctx, w, h) {
    const cx = w / 2, cy = h / 2;
    const colors = {
        transformer: '#f59e0b', switch: '#10b981', busbar: '#3b82f6',
        capacitor: '#06b6d4', meter: '#ef4444', bird: '#8b5cf6',
        acoustic: '#6366f1', gas: '#eab308', hyperspectral: '#14b8a6',
        slam: '#22c55e', fusion: '#f43f5e'
    };
    const color = colors[OutdoorState.currentModule] || '#00d4ff';
    
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 30;
    
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.3;
    ctx.fillRect(cx - 80, cy - 100, 160, 200);
    
    ctx.globalAlpha = 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(cx - 80, cy - 100, 160, 200);
    
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.arc(cx, cy - 120, 20, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillRect(cx - 60, cy + 100, 20, 40);
    ctx.fillRect(cx + 40, cy + 100, 20, 40);
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx - 150, cy - 50); ctx.lineTo(cx - 80, cy - 50); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx + 80, cy - 50); ctx.lineTo(cx + 150, cy - 50); ctx.stroke();
    
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.globalAlpha = 0.8;
    ctx.fillText(OutdoorState.modules[OutdoorState.currentModule]?.name || '', cx, cy + 170);
    ctx.restore();
}

// WebSocket
function initWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    try {
        OutdoorState.wsConnection = new WebSocket(`${protocol}//${location.host}/ws/outdoor`);
        OutdoorState.wsConnection.onopen = () => updateSystemStatus('connected');
        OutdoorState.wsConnection.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                if (data.type === 'outdoor_update' && data.data.module_id === OutdoorState.currentModule) {
                    OutdoorState.currentData = data.data;
                    updateUI(data.data);
                }
            } catch (err) {}
        };
        OutdoorState.wsConnection.onclose = () => {
            updateSystemStatus('disconnected');
            setTimeout(initWebSocket, 5000);
        };
    } catch (e) {}
}

// 模块切换
function switchModule(moduleId) {
    if (moduleId === OutdoorState.currentModule) return;
    OutdoorState.currentModule = moduleId;
    
    document.querySelectorAll('.module-item').forEach(item => {
        item.classList.toggle('active', item.dataset.module === moduleId);
    });
    
    document.getElementById('current-module-name').textContent = 
        OutdoorState.modules[moduleId]?.name || moduleId;
    
    const canvas = document.getElementById('main-video-canvas');
    if (canvas) drawDefaultScene(canvas);
    
    loadModuleData(moduleId);
}

// 数据加载
async function loadModuleData(moduleId) {
    const module = OutdoorState.modules[moduleId];
    if (!module) return;
    
    try {
        const response = await fetch(module.api);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        OutdoorState.currentData = data;
        updateUI(data);
        drawDetections(data);
    } catch (error) {
        console.error(`加载${moduleId}失败:`, error);
    }
}

function startAutoRefresh() {
    if (OutdoorState.refreshTimer) clearInterval(OutdoorState.refreshTimer);
    OutdoorState.refreshTimer = setInterval(() => {
        if (OutdoorState.isPatrolling) loadModuleData(OutdoorState.currentModule);
    }, 2000);
}

// UI更新
function updateUI(data) {
    if (!data) return;
    updateResultList(data.detections || []);
    updateAlarmList(data.alarms || []);
    document.getElementById('detection-count').textContent = data.detections?.length || 0;
    document.getElementById('alarm-count').textContent = data.alarms?.length || 0;
    if (data.metrics) {
        document.getElementById('latency-value').textContent = 
            `${Math.round(data.metrics.processing_time_ms || 45)}ms`;
    }
}

function updateResultList(detections) {
    const container = document.getElementById('result-list');
    if (!container) return;
    
    if (detections.length === 0) {
        container.innerHTML = `<li class="result-item">
            <div class="result-icon normal"><i class="bi bi-check"></i></div>
            <div class="result-info"><div class="result-name">无检测结果</div></div>
        </li>`;
        return;
    }
    
    container.innerHTML = detections.map(d => `
        <li class="result-item">
            <div class="result-icon ${d.status || 'normal'}">
                <i class="bi bi-${d.status === 'alarm' ? 'exclamation-triangle' : d.status === 'warning' ? 'exclamation' : 'check'}"></i>
            </div>
            <div class="result-info">
                <div class="result-name">${d.label}</div>
                <div class="result-detail">${d.details || ''}</div>
            </div>
            <span class="result-confidence">${Math.round((d.confidence || 0) * 100)}%</span>
        </li>
    `).join('');
}

function updateAlarmList(alarms) {
    const container = document.getElementById('alarm-list');
    if (!container) return;
    
    if (alarms.length === 0) {
        container.innerHTML = `<div class="text-muted text-center" style="padding:1rem;font-size:0.75rem">暂无告警</div>`;
        return;
    }
    
    container.innerHTML = alarms.map(a => `
        <div class="alarm-item ${a.level === 'critical' || a.level === 'error' ? 'critical' : ''}">
            <div class="alarm-header">
                <span class="alarm-type">${a.type}</span>
                <span class="alarm-time">${a.timestamp}</span>
            </div>
            <div class="alarm-message">${a.message}</div>
        </div>
    `).join('');
}

// 绘制检测框
function drawDetections(data) {
    const canvas = document.getElementById('main-video-canvas');
    if (!canvas || !data?.detections) return;
    
    const ctx = canvas.getContext('2d');
    drawDefaultScene(canvas);
    
    data.detections.forEach(det => {
        if (!det.bbox) return;
        const x = det.bbox.x * canvas.width;
        const y = det.bbox.y * canvas.height;
        const w = det.bbox.width * canvas.width;
        const h = det.bbox.height * canvas.height;
        
        const color = det.status === 'alarm' ? '#ef4444' : det.status === 'warning' ? '#f59e0b' : '#10b981';
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
        
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.8;
        const label = `${det.label} ${Math.round(det.confidence * 100)}%`;
        ctx.fillRect(x, y - 18, ctx.measureText(label).width + 8, 16);
        ctx.fillStyle = '#fff';
        ctx.globalAlpha = 1;
        ctx.font = '11px Arial';
        ctx.fillText(label, x + 4, y - 5);
    });
}

// Map outdoor module IDs to plugin IDs in the installer
const OUTDOOR_MODULE_TO_PLUGIN = {
    busbar: 'busbar_inspection',
    switch: 'switch_inspection',
    transformer: 'transformer_inspection',
    capacitor: 'capacitor_inspection',
    meter: 'meter_reading',
    bird: 'bird_monitoring',
    acoustic: 'acoustic_monitoring',
    gas: 'gas_detection',
    hyperspectral: 'hyperspectral_detection',
    slam: 'slam_mapping',
    fusion: 'multimodal_fusion',
};

// Enabled plugins cache
let _enabledOutdoorPlugins = null;

/**
 * Load enabled plugins from the API and filter the plugin list
 */
async function fetchEnabledPlugins() {
    try {
        const resp = await fetch('/api/plugins/enabled');
        if (resp.ok) {
            const data = await resp.json();
            _enabledOutdoorPlugins = new Set(data.enabled || []);
        }
    } catch (e) {
        console.warn('[OutdoorCenter] Could not load enabled plugins, showing all:', e.message);
    }
}

function isPluginEnabled(moduleId) {
    if (!_enabledOutdoorPlugins) return true; // show all if API unavailable
    const pluginId = OUTDOOR_MODULE_TO_PLUGIN[moduleId];
    return !pluginId || _enabledOutdoorPlugins.has(pluginId);
}

// 插件列表 (dynamic loading - only show enabled plugins)
async function renderPluginList() {
    const container = document.getElementById('plugin-list');
    if (!container) return;

    // Fetch enabled plugins first
    await fetchEnabledPlugins();

    const allPlugins = ['busbar', 'switch', 'transformer', 'capacitor', 'meter',
                        'bird', 'acoustic', 'gas', 'hyperspectral', 'slam', 'fusion'];

    // Filter to only enabled plugins
    const plugins = allPlugins.filter(id => isPluginEnabled(id));

    container.innerHTML = plugins.map(id => `
        <div class="plugin-panel" data-plugin="${id}">
            <div class="plugin-header" onclick="togglePlugin('${id}')">
                <div class="plugin-title">
                    <i class="bi bi-${OutdoorState.modules[id]?.icon || 'gear'}"></i>
                    ${OutdoorState.modules[id]?.name || id}
                </div>
                <div class="plugin-status">
                    <span class="status-dot"></span>
                    <i class="bi bi-chevron-down"></i>
                </div>
            </div>
            <div class="plugin-content" id="plugin-content-${id}">
                <div class="plugin-metric"><span class="metric-label">状态</span><span class="metric-value good">就绪</span></div>
                <div class="plugin-metric"><span class="metric-label">检测数</span><span class="metric-value">0</span></div>
            </div>
        </div>
    `).join('');

    // Log disabled plugins
    const disabled = allPlugins.filter(id => !isPluginEnabled(id));
    if (disabled.length > 0) {
        console.log(`[OutdoorCenter] Disabled plugins hidden: ${disabled.join(', ')}`);
    }
}

function togglePlugin(pluginId) {
    const content = document.getElementById(`plugin-content-${pluginId}`);
    if (content) {
        content.classList.toggle('show');
        if (content.classList.contains('show')) loadPluginData(pluginId);
    }
}

async function loadPluginData(pluginId) {
    const module = OutdoorState.modules[pluginId];
    if (!module) return;
    
    try {
        const response = await fetch(module.api);
        const data = await response.json();
        const content = document.getElementById(`plugin-content-${pluginId}`);
        if (content) {
            const statusClass = data.status === 'normal' ? 'good' : data.status === 'warning' ? 'warning' : 'alarm';
            content.innerHTML = `
                <div class="plugin-metric"><span class="metric-label">状态</span><span class="metric-value ${statusClass}">${data.status || '正常'}</span></div>
                <div class="plugin-metric"><span class="metric-label">检测数</span><span class="metric-value">${data.detections?.length || 0}</span></div>
                <div class="plugin-metric"><span class="metric-label">告警数</span><span class="metric-value ${data.alarms?.length > 0 ? 'warning' : ''}">${data.alarms?.length || 0}</span></div>
                <div class="plugin-metric"><span class="metric-label">处理时间</span><span class="metric-value">${Math.round(data.metrics?.processing_time_ms || 0)}ms</span></div>
            `;
        }
    } catch (e) {}
}

// 操作函数
function startPatrol() {
    OutdoorState.isPatrolling = !OutdoorState.isPatrolling;
    const btn = document.getElementById('btn-start-patrol');
    if (btn) btn.innerHTML = OutdoorState.isPatrolling ? 
        '<i class="bi bi-stop-fill"></i> 停止巡检' : '<i class="bi bi-play-fill"></i> 开始巡检';
    showToast(OutdoorState.isPatrolling ? '巡检已开始' : '巡检已停止', 'success');
    if (OutdoorState.isPatrolling) loadModuleData(OutdoorState.currentModule);
}

function startDetection() {
    loadModuleData(OutdoorState.currentModule);
    showToast('检测已执行', 'success');
}

function captureImage() {
    const canvas = document.getElementById('main-video-canvas');
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `outdoor_${OutdoorState.currentModule}_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
    showToast('截图已保存', 'success');
}

function setView(viewType) {
    document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
}

function zoomIn() { console.log('放大'); }
function zoomOut() { console.log('缩小'); }
function resetView() { 
    const canvas = document.getElementById('main-video-canvas');
    if (canvas) drawDefaultScene(canvas);
}
function openSettings() { window.location.href = '/settings'; }

// 辅助函数
function updateDateTime() {
    const el = document.getElementById('current-datetime');
    if (el) el.textContent = new Date().toLocaleString('zh-CN');
}

function updateSystemStatus(status) {
    const dot = document.getElementById('system-status-dot');
    const text = document.getElementById('system-status-text');
    if (dot && text) {
        dot.className = status === 'connected' ? 'status-dot' : 'status-dot warning';
        text.textContent = status === 'connected' ? '系统正常' : '连接断开';
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    const alertType = type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info';
    toast.className = `alert alert-${alertType} position-fixed`;
    toast.style.cssText = 'top:80px;right:20px;z-index:9999;min-width:200px;font-size:0.85rem;';
    toast.innerHTML = `<i class="bi bi-${type === 'success' ? 'check-circle' : 'info-circle'}"></i> ${message}`;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 2500);
}

// 全局导出
window.switchModule = switchModule;
window.startPatrol = startPatrol;
window.startDetection = startDetection;
window.captureImage = captureImage;
window.setView = setView;
window.zoomIn = zoomIn;
window.zoomOut = zoomOut;
window.resetView = resetView;
window.openSettings = openSettings;
window.togglePlugin = togglePlugin;

console.log(`[OutdoorCenter] V${OutdoorState.version} 加载完成`);
