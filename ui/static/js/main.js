/**
 * 输变电监测平台 - 主 JavaScript (V4.0)
 * =========================================
 * 职责:
 *  - 全局 AppState 状态中心 + EventBus
 *  - WebSocket 连接管理（含断线重连 + 轮询降级）
 *  - 全局 API 工具
 *  - 性能埋点 PerfTracker
 *  - 时钟更新 / 插件加载 / Toast 通知
 */

'use strict';

// ============== 全局配置 ==============
const CONFIG = {
    API_BASE: '/api',
    WS_BASE: 'ws://' + window.location.host + '/ws',
    REFRESH_INTERVAL: 5000,
    WS_RECONNECT_MIN: 1000,
    WS_RECONNECT_MAX: 30000,
    FALLBACK_POLL_INTERVAL: 10000,
    ALARM_BUFFER_SIZE: 200,
    PLUGIN_STALE_TIMEOUT: 60000, // 60s 未更新视为"通信中断"
};

// ============== 全局状态中心 + 事件总线 ==============
const AppState = {
    // 数据
    plugins: new Map(),
    alarms: [],
    wsConnected: false,

    // 事件系统
    _listeners: new Map(),

    /**
     * 订阅事件
     * @param {string} event - 事件名，如 'plugin:updated', 'alarm:new', 'ws:connected'
     * @param {Function} fn - 回调函数
     */
    on(event, fn) {
        if (!this._listeners.has(event)) this._listeners.set(event, []);
        this._listeners.get(event).push(fn);
    },

    /**
     * 取消订阅
     */
    off(event, fn) {
        const fns = this._listeners.get(event);
        if (fns) {
            const idx = fns.indexOf(fn);
            if (idx !== -1) fns.splice(idx, 1);
        }
    },

    /**
     * 触发事件
     */
    emit(event, data) {
        const fns = this._listeners.get(event) || [];
        fns.forEach(fn => {
            try { fn(data); }
            catch (e) { console.error(`[AppState] Event handler error for ${event}:`, e); }
        });
    },

    /**
     * 更新单个插件状态
     */
    updatePlugin(id, patch) {
        const current = this.plugins.get(id) || {};
        const updated = { ...current, ...patch, lastSeen: Date.now() };
        this.plugins.set(id, updated);
        this.emit('plugin:updated', { id, ...updated });
    },

    /**
     * 批量设置插件数据
     */
    setPlugins(pluginArray) {
        this.plugins.clear();
        if (Array.isArray(pluginArray)) {
            pluginArray.forEach(p => {
                this.plugins.set(p.id, { ...p, lastSeen: Date.now() });
            });
        }
        this.emit('plugins:loaded', Array.from(this.plugins.values()));
    },

    /**
     * 新增告警
     */
    addAlarm(alarm) {
        this.alarms.unshift({ ...alarm, receivedAt: Date.now() });
        if (this.alarms.length > CONFIG.ALARM_BUFFER_SIZE) {
            this.alarms.length = CONFIG.ALARM_BUFFER_SIZE;
        }
        this.emit('alarm:new', alarm);
    },

    /**
     * 获取所有插件列表
     */
    getPluginList() {
        return Array.from(this.plugins.values());
    },

    // 兼容旧版属性
    currentSite: null,
    currentPosition: null,
    currentDevice: null,
    currentTask: null,
    devices: [],
    ws: null,
    _fallbackTimer: null,
};

// ============== 性能埋点 ==============
const PerfTracker = {
    mark(name) {
        try { performance.mark(name); }
        catch (e) { /* ignore */ }
    },

    measure(name, startMark, endMark) {
        try {
            if (endMark) {
                performance.measure(name, startMark, endMark);
            } else {
                performance.measure(name, startMark);
            }
            const entry = performance.getEntriesByName(name).pop();
            if (entry) {
                console.info(`[PERF] ${name}: ${entry.duration.toFixed(0)}ms`);
            }
        } catch (e) { /* ignore */ }
    },
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
    const colorMap = {
        info: 'var(--ck-accent)',
        success: 'var(--ck-success)',
        warning: 'var(--ck-warning)',
        danger: 'var(--ck-danger)',
    };
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; top: 16px; right: 16px; z-index: 9999;
        padding: 10px 18px; border-radius: 6px;
        background: var(--ck-bg-secondary); color: var(--ck-text-primary);
        border: 1px solid var(--ck-border);
        border-left: 3px solid ${colorMap[type] || colorMap.info};
        font-size: 0.85rem; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        animation: fadeIn 0.3s ease-in-out;
    `;
    toast.innerHTML = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * 三态面板渲染器
 * @param {HTMLElement} container
 * @param {'loading'|'error'|'empty'|'ready'} state
 * @param {Function} renderFn - state='ready' 时调用
 * @param {Object} options - { emptyText, errorText, retryFn }
 */
function renderPanel(container, state, renderFn, options = {}) {
    if (!container) return;
    const {
        emptyText = '暂无数据',
        errorText = '数据加载失败',
        retryFn = null,
    } = options;

    switch (state) {
        case 'loading':
            container.innerHTML = '<div class="panel-skeleton">加载中...</div>';
            break;
        case 'error':
            container.innerHTML = `
                <div class="panel-error">
                    <i class="bi bi-exclamation-triangle"></i>
                    <span>${errorText}</span>
                    ${retryFn ? '<button class="retry-btn" data-action="retry"><i class="bi bi-arrow-clockwise"></i> 重试</button>' : ''}
                </div>`;
            if (retryFn) {
                const btn = container.querySelector('[data-action="retry"]');
                if (btn) btn.addEventListener('click', retryFn);
            }
            break;
        case 'empty':
            container.innerHTML = `
                <div class="panel-empty">
                    <i class="bi bi-inbox"></i>
                    <span>${emptyText}</span>
                </div>`;
            break;
        default:
            renderFn(container);
    }
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
    getPlugins: () => API.get('/plugins/list'),
    getPlugin: (id) => API.get(`/plugins/${id}/info`),
    reloadPlugin: (id) => API.post(`/plugins/${id}/reload`),

    // 驾驶舱聚合
    getCockpitOverview: () => API.get('/cockpit/overview'),

    // 插件注册表
    getPluginRegistry: () => API.get('/plugins/registry'),

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

// ============== WebSocket 连接管理 ==============

const WSManager = {
    _ws: null,
    _reconnectDelay: CONFIG.WS_RECONNECT_MIN,
    _reconnectTimer: null,
    _intentionalClose: false,

    connect() {
        if (this._ws && this._ws.readyState <= 1) return; // CONNECTING or OPEN

        this._intentionalClose = false;
        this._updateUI('connecting');

        try {
            this._ws = new WebSocket(CONFIG.WS_BASE + '/events');
        } catch (e) {
            console.warn('[WS] WebSocket creation failed:', e.message);
            this._scheduleReconnect();
            return;
        }

        this._ws.onopen = () => {
            console.info('[WS] Connected');
            this._reconnectDelay = CONFIG.WS_RECONNECT_MIN;
            AppState.wsConnected = true;
            AppState.ws = this._ws;
            AppState.emit('ws:connected');
            this._updateUI('connected');
            this._stopFallbackPolling();
        };

        this._ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                this._handleMessage(msg);
            } catch (e) {
                console.warn('[WS] Parse error:', e.message);
            }
        };

        this._ws.onclose = (event) => {
            console.info(`[WS] Disconnected (code=${event.code})`);
            AppState.wsConnected = false;
            AppState.ws = null;
            AppState.emit('ws:disconnected');
            this._updateUI('disconnected');

            if (!this._intentionalClose) {
                this._startFallbackPolling();
                this._scheduleReconnect();
            }
        };

        this._ws.onerror = (error) => {
            console.warn('[WS] Error:', error);
        };
    },

    disconnect() {
        this._intentionalClose = true;
        if (this._ws) this._ws.close();
        if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
        this._stopFallbackPolling();
    },

    _handleMessage(msg) {
        switch (msg.type) {
            case 'plugin_status':
                if (msg.plugin_id && msg.data) {
                    // 时间戳校验：丢弃过时消息
                    const current = AppState.plugins.get(msg.plugin_id);
                    if (current && msg.timestamp && current._wsTimestamp > msg.timestamp) {
                        return; // 丢弃过时消息
                    }
                    AppState.updatePlugin(msg.plugin_id, {
                        ...msg.data,
                        _wsTimestamp: msg.timestamp || Date.now(),
                    });
                }
                break;
            case 'alarm':
                if (msg.data) AppState.addAlarm(msg.data);
                break;
            case 'device_status':
                AppState.emit('device:status', msg.data);
                break;
            case 'task_progress':
                updateTaskProgress(msg.payload || msg.data);
                break;
            case 'task_complete':
                handleTaskComplete(msg.payload || msg.data);
                break;
            case 'video_frame':
                updateVideoFrame(msg.payload || msg.data);
                break;
            default:
                console.debug('[WS] Unknown message type:', msg.type);
        }
    },

    _scheduleReconnect() {
        if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
        console.info(`[WS] Reconnecting in ${this._reconnectDelay}ms...`);
        this._reconnectTimer = setTimeout(() => {
            this.connect();
        }, this._reconnectDelay);
        this._reconnectDelay = Math.min(this._reconnectDelay * 2, CONFIG.WS_RECONNECT_MAX);
    },

    _startFallbackPolling() {
        if (AppState._fallbackTimer) return;
        console.info('[WS] Starting fallback polling');
        AppState._fallbackTimer = setInterval(async () => {
            try {
                const plugins = await API.getPlugins();
                if (Array.isArray(plugins)) {
                    plugins.forEach(p => AppState.updatePlugin(p.id, p));
                }
            } catch (e) {
                console.warn('[Poll] Fallback polling error:', e.message);
            }
        }, CONFIG.FALLBACK_POLL_INTERVAL);
    },

    _stopFallbackPolling() {
        if (AppState._fallbackTimer) {
            clearInterval(AppState._fallbackTimer);
            AppState._fallbackTimer = null;
            console.info('[WS] Stopped fallback polling');
        }
    },

    _updateUI(state) {
        // 更新导航栏 WebSocket 指示
        const navDot = document.getElementById('nav-ws-dot');
        const navLabel = document.getElementById('nav-ws-status');
        const footerDot = document.getElementById('footer-ws-dot');
        const footerLabel = document.getElementById('footer-ws-label');

        const dotClass = {
            connected: 'ws-dot ws-connected',
            disconnected: 'ws-dot ws-disconnected',
            connecting: 'ws-dot ws-connecting',
        }[state] || 'ws-dot ws-disconnected';

        const labelText = {
            connected: '系统正常',
            disconnected: '连接断开',
            connecting: '正在连接...',
        }[state] || '未知';

        const wsText = {
            connected: 'WebSocket: 已连接',
            disconnected: 'WebSocket: 已断开',
            connecting: 'WebSocket: 连接中...',
        }[state] || 'WebSocket: 未知';

        if (navDot) navDot.className = dotClass;
        if (navLabel) {
            navLabel.style.color = state === 'connected' ? 'var(--ck-success)' :
                                   state === 'connecting' ? 'var(--ck-warning)' : 'var(--ck-danger)';
            navLabel.innerHTML = `<span class="${dotClass}"></span> ${labelText}`;
        }
        if (footerDot) footerDot.className = dotClass;
        if (footerLabel) footerLabel.textContent = wsText;

        // cockpit footer 的 ws-status
        const wsStatus = document.getElementById('ws-status');
        if (wsStatus) {
            wsStatus.textContent = state === 'connected' ? '已连接' :
                                   state === 'connecting' ? '连接中...' : '已断开';
        }
    },
};

// ============== UI 更新函数 (兼容旧版) ==============

function updateTaskProgress(data) {
    if (!data) return;
    const progressBar = document.getElementById('task-progress');
    if (progressBar) {
        progressBar.style.width = data.progress + '%';
        progressBar.textContent = data.progress + '%';
    }
}

function handleTaskComplete(data) {
    if (!data) return;
    const btn = document.getElementById('btn-run');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<i class="bi bi-play-fill"></i> 运行任务';
    }

    if (data.success) {
        showToast('任务执行完成', 'success');
        updateResults(data.results);
    } else {
        showToast('任务执行失败: ' + (data.error || '未知错误'), 'danger');
    }
}

function handleNewAlarm(data) {
    if (!data) return;
    AppState.addAlarm(data);
}

function updateResults(results) {
    if (!results) return;
    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val || 0;
    };
    set('result-normal', results.normal);
    set('result-warning', results.warning);
    set('result-error', results.error);
}

function updateAlarmCounts() {
    const alarmList = document.getElementById('alarm-list');
    if (!alarmList) return;

    const warnings = alarmList.querySelectorAll('.alarm-warning').length;
    const errors = alarmList.querySelectorAll('.alarm-error').length +
                   alarmList.querySelectorAll('.alarm-critical').length;

    const wEl = document.getElementById('result-warning');
    const eEl = document.getElementById('result-error');
    if (wEl) wEl.textContent = warnings;
    if (eEl) eEl.textContent = errors;
}

function updateVideoFrame(data) {
    if (!data) return;
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
        if (Array.isArray(plugins)) {
            AppState.setPlugins(plugins);
            updatePluginTable(plugins);
            updatePluginCount(plugins.length);
        }
    } catch (error) {
        console.warn('加载插件失败:', error.message);
    }
}

function updatePluginTable(plugins) {
    const table = document.getElementById('plugin-table');
    if (!table) return;

    table.innerHTML = plugins.map(plugin => `
        <tr>
            <td><code>${plugin.id}</code></td>
            <td>${plugin.name}</td>
            <td>${plugin.version || '--'}</td>
            <td>
                <span class="badge bg-${plugin.status === 'ready' ? 'success' : 'secondary'}">
                    ${plugin.status || 'unknown'}
                </span>
            </td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="reloadPlugin('${plugin.id}')">
                    <i class="bi bi-arrow-repeat"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

function updatePluginCount(count) {
    const countEl = document.getElementById('plugin-count');
    if (countEl) countEl.textContent = count;
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
    PerfTracker.mark('app-init-start');

    // 更新时钟
    updateClock();
    setInterval(updateClock, 1000);

    // 加载插件信息
    loadPlugins();

    // WebSocket 连接
    WSManager.connect();

    PerfTracker.measure('app-init', 'app-init-start');
    console.info('输变电监测平台 UI V4.0 已初始化');
});
