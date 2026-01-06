/**
 * 输变电监测平台 - 主JavaScript文件
 */

// 全局配置
const CONFIG = {
    API_BASE: '/api',
    WS_BASE: 'ws://' + window.location.host + '/ws',
    REFRESH_INTERVAL: 5000,
};

// 全局状态
const AppState = {
    currentSite: null,
    currentPosition: null,
    currentDevice: null,
    currentTask: null,
    plugins: [],
    devices: [],
    ws: null,
};

// ============== 工具函数 ==============

function formatTime(date) {
    return new Date(date).toLocaleString('zh-CN');
}

function formatDuration(ms) {
    if (ms < 1000) return ms + 'ms';
    if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
    return (ms / 60000).toFixed(1) + 'min';
}

function showToast(message, type = 'info') {
    // 简单的toast通知
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed top-0 end-0 m-3`;
    toast.style.zIndex = '9999';
    toast.innerHTML = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ============== API 调用 ==============

const API = {
    async get(endpoint) {
        const response = await fetch(CONFIG.API_BASE + endpoint);
        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        return response.json();
    },

    async post(endpoint, data) {
        const response = await fetch(CONFIG.API_BASE + endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        return response.json();
    },

    async delete(endpoint) {
        const response = await fetch(CONFIG.API_BASE + endpoint, {
            method: 'DELETE',
        });
        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        return response.json();
    },

    // 站点相关
    getSites: () => API.get('/sites'),
    getSite: (id) => API.get(`/sites/${id}`),

    // 插件相关
    getPlugins: () => API.get('/plugins'),
    getPlugin: (id) => API.get(`/plugins/${id}`),
    reloadPlugin: (id) => API.post(`/plugins/${id}/reload`),

    // 任务相关
    runTask: (data) => API.post('/tasks/run', data),
    getTask: (id) => API.get(`/tasks/${id}`),
    cancelTask: (id) => API.post(`/tasks/${id}/cancel`),

    // 证据相关
    getEvidenceRuns: (params) => API.get('/evidence/runs?' + new URLSearchParams(params)),
    getEvidenceRun: (id) => API.get(`/evidence/runs/${id}`),

    // 系统相关
    getHealth: () => API.get('/health'),
    getStatus: () => API.get('/status'),
};

// ============== WebSocket ==============

function connectWebSocket() {
    if (AppState.ws) {
        AppState.ws.close();
    }

    AppState.ws = new WebSocket(CONFIG.WS_BASE + '/events');

    AppState.ws.onopen = () => {
        console.log('WebSocket connected');
        showToast('实时连接已建立', 'success');
    };

    AppState.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    AppState.ws.onclose = () => {
        console.log('WebSocket disconnected');
        // 自动重连
        setTimeout(connectWebSocket, 5000);
    };

    AppState.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'task_progress':
            updateTaskProgress(data.payload);
            break;
        case 'task_complete':
            handleTaskComplete(data.payload);
            break;
        case 'alarm':
            handleNewAlarm(data.payload);
            break;
        case 'video_frame':
            updateVideoFrame(data.payload);
            break;
        default:
            console.log('Unknown message type:', data.type);
    }
}

// ============== UI 更新函数 ==============

function updateTaskProgress(data) {
    const progressBar = document.getElementById('task-progress');
    if (progressBar) {
        progressBar.style.width = data.progress + '%';
        progressBar.textContent = data.progress + '%';
    }
}

function handleTaskComplete(data) {
    const btn = document.getElementById('btn-run');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-play-fill"></i> 运行任务';
    }

    if (data.success) {
        showToast('任务执行完成', 'success');
        updateResults(data.results);
    } else {
        showToast('任务执行失败: ' + data.error, 'danger');
    }
}

function handleNewAlarm(data) {
    const alarmList = document.getElementById('alarm-list');
    if (!alarmList) return;

    // 清除空提示
    if (alarmList.querySelector('.text-muted')) {
        alarmList.innerHTML = '';
    }

    const alarmItem = document.createElement('div');
    alarmItem.className = `alarm-item alarm-${data.level}`;
    alarmItem.innerHTML = `
        <div class="d-flex justify-content-between">
            <strong>${data.title}</strong>
            <small>${formatTime(data.timestamp)}</small>
        </div>
        <small class="text-muted">${data.message}</small>
    `;
    alarmList.prepend(alarmItem);

    // 更新计数
    updateAlarmCounts();
}

function updateResults(results) {
    if (!results) return;

    document.getElementById('result-normal').textContent = results.normal || 0;
    document.getElementById('result-warning').textContent = results.warning || 0;
    document.getElementById('result-error').textContent = results.error || 0;
}

function updateAlarmCounts() {
    const alarmList = document.getElementById('alarm-list');
    if (!alarmList) return;

    const warnings = alarmList.querySelectorAll('.alarm-warning').length;
    const errors = alarmList.querySelectorAll('.alarm-error').length +
                   alarmList.querySelectorAll('.alarm-critical').length;

    document.getElementById('result-warning').textContent = warnings;
    document.getElementById('result-error').textContent = errors;
}

function updateVideoFrame(data) {
    const canvas = document.getElementById('video-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
    };
    img.src = 'data:image/jpeg;base64,' + data.frame;
}

// ============== 时钟更新 ==============

function updateClock() {
    const clockEl = document.getElementById('current-time');
    if (clockEl) {
        clockEl.textContent = new Date().toLocaleString('zh-CN');
    }
}

// ============== 插件管理 ==============

async function loadPlugins() {
    try {
        const plugins = await API.getPlugins();
        AppState.plugins = plugins;
        updatePluginTable(plugins);
        updatePluginCount(plugins.length);
    } catch (error) {
        console.error('加载插件失败:', error);
    }
}

function updatePluginTable(plugins) {
    const table = document.getElementById('plugin-table');
    if (!table) return;

    table.innerHTML = plugins.map(plugin => `
        <tr>
            <td><code>${plugin.id}</code></td>
            <td>${plugin.name}</td>
            <td>${plugin.version}</td>
            <td>
                <span class="badge bg-${plugin.status === 'ready' ? 'success' : 'secondary'}">
                    ${plugin.status}
                </span>
            </td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="reloadPlugin('${plugin.id}')">
                    <i class="bi bi-arrow-repeat"></i>
                </button>
                <button class="btn btn-sm btn-outline-danger" onclick="disablePlugin('${plugin.id}')">
                    <i class="bi bi-x"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

function updatePluginCount(count) {
    const countEl = document.getElementById('plugin-count');
    if (countEl) {
        countEl.textContent = count;
    }
}

async function reloadPlugin(id) {
    try {
        await API.reloadPlugin(id);
        showToast(`插件 ${id} 已重新加载`, 'success');
        loadPlugins();
    } catch (error) {
        showToast(`重载失败: ${error.message}`, 'danger');
    }
}

// ============== 初始化 ==============

document.addEventListener('DOMContentLoaded', function() {
    // 更新时钟
    updateClock();
    setInterval(updateClock, 1000);

    // 加载插件信息
    loadPlugins();

    // WebSocket连接 (可选)
    // connectWebSocket();

    console.log('输变电监测平台 UI 已初始化');
});
