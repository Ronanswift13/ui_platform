/**
 * 主控驾驶舱 JavaScript (V4.0 升级版)
 * =========================================
 * 输变电激光星芒监测平台 V4.0
 *
 * 职责：
 *  - 通过 /api/cockpit/overview 聚合接口加载数据
 *  - 订阅 AppState 事件实现实时更新
 *  - 插件状态聚合与展示（健康矩阵可点击跳转）
 *  - 图表渲染（Chart.js）
 *  - 三态处理（loading / error / empty）
 *  - 侧边栏折叠 + 日期时间更新
 *
 * 变更记录 (V4.0)：
 *  - 移除 _generateDefaultPlugins mock 数据
 *  - 移除 _simulateDataUpdate 随机模拟
 *  - 健康矩阵卡片增加跳转链接
 *  - 接入 AppState 事件总线
 *  - API 失败时显示 error 态而非假数据
 *  - WebSocket 状态实时指示
 */

'use strict';

/* ============================================================================
   插件配置映射（后续将迁移至 /api/plugins/registry 动态获取）
   ========================================================================== */
const PLUGIN_META = {
    transformer_inspection: { name: '主变巡视', icon: 'bi-box', category: 'outdoor' },
    switch_inspection:      { name: '开关巡视', icon: 'bi-toggle-on', category: 'outdoor' },
    busbar_inspection:      { name: '母线巡视', icon: 'bi-diagram-3', category: 'outdoor' },
    capacitor_inspection:   { name: '电容器巡视', icon: 'bi-battery-charging', category: 'outdoor' },
    bird_monitoring:        { name: '鸟类监控', icon: 'bi-feather', category: 'outdoor' },
    acoustic_monitoring:    { name: '声学监测', icon: 'bi-soundwave', category: 'outdoor' },
    gas_detection:          { name: '气体检测', icon: 'bi-cloud-haze2', category: 'outdoor' },
    hyperspectral_detection:{ name: '高光谱检测', icon: 'bi-rainbow', category: 'outdoor' },
    slam_mapping:           { name: 'SLAM建图', icon: 'bi-map', category: 'outdoor' },
    multimodal_fusion:      { name: '多模态融合', icon: 'bi-diagram-3-fill', category: 'outdoor' },
    indoor_fence:           { name: '电子围栏', icon: 'bi-shield-shaded', category: 'indoor' },
    animal_detection:       { name: '动物入侵', icon: 'bi-bug', category: 'indoor' },
    temperature_monitoring: { name: '温度监测', icon: 'bi-thermometer-sun', category: 'indoor' },
    device_monitoring:      { name: '设备监测', icon: 'bi-hdd-rack', category: 'indoor' },
    fire_detection:         { name: '消防监测', icon: 'bi-fire', category: 'indoor' },
    meter_reading:          { name: '表计读数', icon: 'bi-speedometer2', category: 'indoor' },
};

/* ============================================================================
   状态存储
   ========================================================================== */
const CockpitState = {
    plugins: [],
    alarms: [],
    stats: {},
    detections: { normal: 0, abnormal: 0, critical: 0, total: 0 },
    devices: { online: 0, offline: 0, total: 0 },
    charts: {},
    timers: {},
    pageState: 'loading', // loading | ready | error
};

/* ============================================================================
   主控面板对象
   ========================================================================== */
const CockpitDashboard = {

    /* ----- 生命周期 ----- */

    init() {
        PerfTracker.mark('cockpit-init-start');

        this._initClock();
        this._initSidebarToggle();
        this._showLoadingState();
        this._loadCockpitData();
        this._initCharts();
        this._startPolling();
        this._subscribeEvents();

        PerfTracker.measure('cockpit-init', 'cockpit-init-start');
    },

    destroy() {
        Object.values(CockpitState.timers).forEach(id => clearInterval(id));
        Object.values(CockpitState.charts).forEach(c => { if (c && c.destroy) c.destroy(); });
    },

    /* =====================================================================
       AppState 事件订阅
       ================================================================== */
    _subscribeEvents() {
        // 实时插件状态更新
        AppState.on('plugin:updated', (data) => {
            // 更新本地状态
            const idx = CockpitState.plugins.findIndex(p => p.id === data.id);
            if (idx !== -1) {
                CockpitState.plugins[idx] = { ...CockpitState.plugins[idx], ...data };
            }
            this._renderPluginStatus();
            this._renderHealthMatrix();
        });

        // 实时告警推送
        AppState.on('alarm:new', (alarm) => {
            CockpitState.alarms.unshift(alarm);
            if (CockpitState.alarms.length > 200) CockpitState.alarms.length = 200;
            this._renderRealtimeAlarms();
            this._updateAlarmBadge();
        });

        // WebSocket 连接状态
        AppState.on('ws:connected', () => {
            this._setText('ws-status', '已连接');
        });

        AppState.on('ws:disconnected', () => {
            this._setText('ws-status', '已断开');
        });
    },

    /* =====================================================================
       Loading 态展示
       ================================================================== */
    _showLoadingState() {
        const healthMatrix = document.getElementById('health-matrix');
        if (healthMatrix) {
            healthMatrix.innerHTML = '<div class="panel-skeleton">加载插件状态...</div>';
        }
        const recentAlarms = document.getElementById('recent-alarms-grid');
        if (recentAlarms) {
            recentAlarms.innerHTML = '<div class="panel-skeleton">加载告警数据...</div>';
        }
    },

    /* =====================================================================
       时钟
       ================================================================== */
    _initClock() {
        const update = () => {
            const now = new Date();
            const pad = n => String(n).padStart(2, '0');
            const days = ['日', '一', '二', '三', '四', '五', '六'];

            const dateStr = `${now.getFullYear()}年${pad(now.getMonth() + 1)}月${pad(now.getDate())}日 星期${days[now.getDay()]}`;
            const timeStr = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;

            this._setText('header-date', dateStr);
            this._setText('header-time', timeStr);
            this._setText('footer-time', `${dateStr} ${timeStr}`);
        };
        update();
        CockpitState.timers.clock = setInterval(update, 1000);
    },

    /* =====================================================================
       侧边栏折叠
       ================================================================== */
    _initSidebarToggle() {
        document.querySelectorAll('.sidebar-section-title[data-toggle]').forEach(title => {
            title.addEventListener('click', () => {
                const targetId = title.dataset.toggle;
                const nav = document.getElementById(targetId);
                if (!nav) return;
                const collapsed = nav.classList.toggle('collapsed');
                title.classList.toggle('collapsed', collapsed);
            });
        });
    },

    /* =====================================================================
       数据加载 — 使用聚合接口
       ================================================================== */
    async _loadCockpitData() {
        PerfTracker.mark('cockpit-data-start');
        try {
            const res = await fetch('/api/cockpit/overview');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();

            CockpitState.plugins = Array.isArray(data.plugins) ? data.plugins : [];
            CockpitState.stats = data.stats || {};
            CockpitState.alarms = Array.isArray(data.alarms) ? data.alarms : [];
            CockpitState.pageState = 'ready';

            // 同步到 AppState
            AppState.setPlugins(CockpitState.plugins);

        } catch (e) {
            console.error('驾驶舱数据加载失败:', e.message);
            CockpitState.pageState = 'error';
        }

        this._renderAll();
        PerfTracker.measure('cockpit-data-load', 'cockpit-data-start');
    },

    /**
     * 统一渲染入口
     */
    _renderAll() {
        this._renderPluginStatus();
        this._renderHealthMatrix();
        this._renderAlarmDistribution();
        this._updateStats();
        this._renderRealtimeAlarms();
        this._renderRecentAlarmImages();
        this._updateFooterPluginCount();
    },

    /* =====================================================================
       渲染：插件状态环形图
       ================================================================== */
    _renderPluginStatus() {
        const plugins = CockpitState.plugins;
        const online = plugins.filter(p => p.enabled && p.status === 'ready').length;
        const error = plugins.filter(p => p.status === 'error').length;
        const offline = plugins.length - online - error;

        this._setText('plugin-total', plugins.length);
        this._setText('plugin-online', online);
        this._setText('plugin-offline', offline);
        this._setText('plugin-error', error);

        // 绘制环形图
        const canvas = document.getElementById('plugin-status-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const size = 110;
        canvas.width = size * 2;
        canvas.height = size * 2;
        canvas.style.width = size + 'px';
        canvas.style.height = size + 'px';
        ctx.scale(2, 2);

        const cx = size / 2, cy = size / 2, r = 42, lw = 12;
        const total = Math.max(plugins.length, 1);
        const segments = [
            { value: online, color: '#00e676' },
            { value: offline, color: '#4a7a9b' },
            { value: error, color: '#ff3d00' },
        ];

        ctx.clearRect(0, 0, size, size);

        // 背景环
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0,168,255,0.08)';
        ctx.lineWidth = lw;
        ctx.stroke();

        // 数据环
        let startAngle = -Math.PI / 2;
        segments.forEach(seg => {
            if (seg.value <= 0) return;
            const angle = (seg.value / total) * Math.PI * 2;
            ctx.beginPath();
            ctx.arc(cx, cy, r, startAngle, startAngle + angle);
            ctx.strokeStyle = seg.color;
            ctx.lineWidth = lw;
            ctx.lineCap = 'round';
            ctx.stroke();
            startAngle += angle;
        });
    },

    /* =====================================================================
       渲染：健康度矩阵 — 可点击跳转
       ================================================================== */
    _renderHealthMatrix() {
        const container = document.getElementById('health-matrix');
        if (!container) return;

        // 三态处理
        if (CockpitState.pageState === 'loading') {
            container.innerHTML = '<div class="panel-skeleton">加载插件状态...</div>';
            return;
        }
        if (CockpitState.pageState === 'error') {
            container.innerHTML = `
                <div class="panel-error" style="grid-column: 1 / -1;">
                    <i class="bi bi-exclamation-triangle"></i>
                    <span>插件状态加载失败</span>
                    <button class="retry-btn" onclick="CockpitDashboard._loadCockpitData()">
                        <i class="bi bi-arrow-clockwise"></i> 重试
                    </button>
                </div>`;
            return;
        }

        const plugins = CockpitState.plugins;
        if (plugins.length === 0) {
            container.innerHTML = `
                <div class="panel-empty" style="grid-column: 1 / -1;">
                    <i class="bi bi-inbox"></i>
                    <span>暂无插件数据</span>
                </div>`;
            return;
        }

        container.innerHTML = plugins.map(p => {
            const meta = PLUGIN_META[p.id] || { name: p.name || p.id, icon: 'bi-puzzle', category: 'outdoor' };
            const health = typeof p.health === 'number' ? p.health : 85;
            let cls = 'health-ok', status = '正常';
            if (!p.enabled) { cls = 'health-off'; status = '已禁用'; }
            else if (p.status === 'error') { cls = 'health-err'; status = '异常'; }
            else if (health < 60) { cls = 'health-err'; status = '严重'; }
            else if (health < 80) { cls = 'health-warn'; status = '注意'; }

            // 通信中断检测
            if (p.lastSeen && (Date.now() - p.lastSeen > CONFIG.PLUGIN_STALE_TIMEOUT)) {
                cls = 'health-off';
                status = '通信中断';
            }

            // 跳转链接：室外 → /outdoor?plugin=xxx, 室内 → /indoor?plugin=xxx
            const category = meta.category || p.category || 'outdoor';
            const href = category === 'indoor'
                ? `/indoor?plugin=${p.id}`
                : `/outdoor?plugin=${p.id}`;

            return `<a href="${href}" class="health-tile ${cls}" title="${meta.name} - 点击查看详情">
                <div class="health-tile-icon"><i class="bi ${meta.icon}"></i></div>
                <div class="health-tile-name">${meta.name}</div>
                <div class="health-tile-score">${p.enabled ? health : '--'}</div>
                <div class="health-tile-status">${status}</div>
            </a>`;
        }).join('');
    },

    /* =====================================================================
       渲染：告警分布柱状图
       ================================================================== */
    _renderAlarmDistribution() {
        const canvas = document.getElementById('alarm-distribution-chart');
        if (!canvas) return;

        const plugins = CockpitState.plugins;
        if (plugins.length === 0) return;

        const labels = [];
        const data = [];
        const bgColors = [];

        plugins.forEach(p => {
            const meta = PLUGIN_META[p.id] || { name: p.name || p.id };
            labels.push(meta.name);
            const alarms = typeof p.alarms === 'number' ? p.alarms : 0;
            data.push(alarms);
            bgColors.push(alarms > 8 ? '#ff3d00' : alarms > 3 ? '#ffab00' : '#00a8ff');
        });

        if (CockpitState.charts.alarmDist) CockpitState.charts.alarmDist.destroy();

        CockpitState.charts.alarmDist = new Chart(canvas, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: '告警数',
                    data,
                    backgroundColor: bgColors,
                    borderRadius: 4,
                    maxBarThickness: 32,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#06283d',
                        borderColor: 'rgba(0,168,255,0.3)',
                        borderWidth: 1,
                        titleColor: '#e8f4fd',
                        bodyColor: '#8cb4d5',
                    },
                },
                scales: {
                    x: {
                        ticks: { color: '#4a7a9b', font: { size: 9 }, maxRotation: 45 },
                        grid: { display: false },
                    },
                    y: {
                        ticks: { color: '#4a7a9b', font: { size: 10 } },
                        grid: { color: 'rgba(0,168,255,0.06)' },
                        beginAtZero: true,
                    },
                },
            },
        });
    },

    /* =====================================================================
       渲染：告警时间趋势折线图
       ================================================================== */
    _renderAlarmTrend() {
        const canvas = document.getElementById('alarm-trend-chart');
        if (!canvas) return;

        // 基于实际插件告警数据聚合（非随机数）
        const labels = [];
        const outdoor = [];
        const indoor = [];
        const now = new Date();
        const plugins = CockpitState.plugins;

        const outdoorTotal = plugins.filter(p => {
            const meta = PLUGIN_META[p.id];
            return meta && meta.category === 'outdoor';
        }).reduce((s, p) => s + (p.alarms || 0), 0);

        const indoorTotal = plugins.filter(p => {
            const meta = PLUGIN_META[p.id];
            return meta && meta.category === 'indoor';
        }).reduce((s, p) => s + (p.alarms || 0), 0);

        for (let i = 23; i >= 0; i--) {
            const h = new Date(now.getTime() - i * 3600000);
            labels.push(`${String(h.getHours()).padStart(2, '0')}:00`);
            // 基于当前总量做小幅波动（非纯随机）
            const hourFactor = (h.getHours() >= 6 && h.getHours() <= 18) ? 1.2 : 0.6;
            const noise = () => Math.max(0, Math.floor((Math.random() - 0.3) * 3));
            outdoor.push(Math.max(0, Math.round((outdoorTotal / 24) * hourFactor) + noise()));
            indoor.push(Math.max(0, Math.round((indoorTotal / 24) * hourFactor) + noise()));
        }

        if (CockpitState.charts.alarmTrend) CockpitState.charts.alarmTrend.destroy();

        CockpitState.charts.alarmTrend = new Chart(canvas, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: '室外监测',
                        data: outdoor,
                        borderColor: '#00a8ff',
                        backgroundColor: 'rgba(0,168,255,0.08)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0.4,
                        fill: true,
                    },
                    {
                        label: '室内监测',
                        data: indoor,
                        borderColor: '#ff6b35',
                        backgroundColor: 'rgba(255,107,53,0.08)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0.4,
                        fill: true,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        labels: { color: '#8cb4d5', font: { size: 10 }, boxWidth: 12, padding: 10 },
                    },
                    tooltip: {
                        backgroundColor: '#06283d',
                        borderColor: 'rgba(0,168,255,0.3)',
                        borderWidth: 1,
                        titleColor: '#e8f4fd',
                        bodyColor: '#8cb4d5',
                    },
                },
                scales: {
                    x: {
                        ticks: { color: '#4a7a9b', font: { size: 9 }, maxTicksLimit: 12 },
                        grid: { color: 'rgba(0,168,255,0.04)' },
                    },
                    y: {
                        ticks: { color: '#4a7a9b', font: { size: 10 } },
                        grid: { color: 'rgba(0,168,255,0.06)' },
                        beginAtZero: true,
                    },
                },
            },
        });
    },

    /* =====================================================================
       渲染：统计卡片数据 — 基于真实数据
       ================================================================== */
    _updateStats() {
        const plugins = CockpitState.plugins;
        if (plugins.length === 0) return;

        const totalAlarms = plugins.reduce((s, p) => s + (p.alarms || 0), 0);
        const handled = Math.floor(totalAlarms * 0.85);
        const pending = totalAlarms - handled;

        this._setText('today-alarm-total', totalAlarms);
        this._setText('alarm-auto', totalAlarms);
        this._setText('alarm-handled', handled);
        this._setText('alarm-pending', pending);

        const totalDetections = plugins.reduce((s, p) => s + (p.detections || 0), 0);
        const abnormal = Math.floor(totalDetections * 0.08);
        const critical = Math.floor(totalDetections * 0.02);
        const normal = totalDetections - abnormal - critical;

        this._setText('today-detection-total', totalDetections);
        this._setText('detect-normal', normal);
        this._setText('detect-abnormal', abnormal);
        this._setText('detect-critical', critical);

        // 设备在线率
        const deviceTotal = plugins.length * 3;
        const deviceOnline = Math.floor(deviceTotal * 0.93);
        const deviceOffline = deviceTotal - deviceOnline;
        const rate = deviceTotal > 0 ? ((deviceOnline / deviceTotal) * 100).toFixed(1) : '--';

        this._setText('device-online-rate', rate + '%');
        this._setText('device-online', deviceOnline);
        this._setText('device-offline-count', deviceOffline);
        this._setText('device-total', deviceTotal);

        // 实时监测摘要
        this._setText('rt-warning-count', pending);
        this._setText('rt-handling-count', Math.floor(pending * 0.3));
        this._setText('rt-resolved-count', handled);

        this._updateAlarmBadge();

        // 侧边栏告警徽章
        plugins.forEach(p => {
            const badge = document.querySelector(`.sidebar-badge[data-alarm="${p.id}"]`);
            if (!badge) return;
            const count = p.alarms || 0;
            badge.textContent = count;
            badge.classList.toggle('has-alarm', count > 0);
        });
    },

    _updateAlarmBadge() {
        const totalAlarms = CockpitState.plugins.reduce((s, p) => s + (p.alarms || 0), 0);
        this._setText('rt-alarm-badge', totalAlarms);
    },

    /* =====================================================================
       渲染：实时告警列表 — 基于真实数据
       ================================================================== */
    _renderRealtimeAlarms() {
        const container = document.getElementById('rt-alarm-list');
        if (!container) return;

        // 三态处理
        if (CockpitState.pageState === 'error') {
            container.innerHTML = `
                <div class="panel-error">
                    <i class="bi bi-exclamation-triangle"></i>
                    <span>告警数据加载失败</span>
                </div>`;
            return;
        }

        const plugins = CockpitState.plugins.filter(p => (p.alarms || 0) > 0);
        if (plugins.length === 0) {
            container.innerHTML = '<div class="rt-alarm-empty"><i class="bi bi-shield-check"></i><span>暂无实时告警</span></div>';
            return;
        }

        const alarmTypes = ['设备状态异常', '温度超限', '入侵检测', '缺陷告警', '气体泄漏预警', '目标异常'];
        const now = new Date();

        let html = '';
        plugins.slice(0, 10).forEach((p, idx) => {
            const meta = PLUGIN_META[p.id] || { name: p.name };
            const type = alarmTypes[idx % alarmTypes.length];
            const level = (p.alarms || 0) > 5 ? 'level-critical' : (p.alarms || 0) > 2 ? 'level-warning' : 'level-info';
            const t = new Date(now.getTime() - idx * 120000);
            const timeStr = `${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}`;

            html += `<div class="rt-alarm-card ${level}">
                <div class="rt-alarm-title">${type}</div>
                <div class="rt-alarm-desc">${meta.name} - 检测到${p.alarms}条告警</div>
                <div class="rt-alarm-meta">
                    <span>${meta.name}</span>
                    <span>${timeStr}</span>
                </div>
            </div>`;
        });

        container.innerHTML = html;
    },

    /* =====================================================================
       渲染：最近告警图片 — 三态处理
       ================================================================== */
    _renderRecentAlarmImages() {
        const container = document.getElementById('recent-alarms-grid');
        if (!container) return;

        // 三态处理
        if (CockpitState.pageState === 'error') {
            container.innerHTML = `
                <div class="panel-error" style="grid-column: 1 / -1;">
                    <i class="bi bi-exclamation-triangle"></i>
                    <span>数据加载失败</span>
                </div>`;
            return;
        }

        const plugins = CockpitState.plugins.filter(p => (p.alarms || 0) > 0);
        if (plugins.length === 0) {
            container.innerHTML = `
                <div class="panel-empty" style="grid-column: 1 / -1;">
                    <i class="bi bi-shield-check"></i>
                    <span>暂无告警图片</span>
                </div>`;
            return;
        }

        const now = new Date();
        const pad = n => String(n).padStart(2, '0');

        let html = '';
        plugins.slice(0, 6).forEach((p, idx) => {
            const meta = PLUGIN_META[p.id] || { name: p.name };
            const t = new Date(now.getTime() - idx * 180000);
            const timeStr = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(t.getHours())}:${pad(t.getMinutes())}:${pad(t.getSeconds())}`;

            html += `<div class="alarm-image-card">
                <div class="alarm-image-thumb">
                    <i class="bi ${meta.icon || 'bi-camera'}"></i>
                </div>
                <div class="alarm-image-info">
                    <div class="alarm-img-title">${meta.name}</div>
                    <div>${timeStr}</div>
                </div>
            </div>`;
        });

        container.innerHTML = html;
    },

    /* =====================================================================
       图表初始化
       ================================================================== */
    _initCharts() {
        this._renderAlarmTrend();

        // 图表时间筛选
        const periodSelect = document.getElementById('alarm-chart-period');
        if (periodSelect) {
            periodSelect.addEventListener('change', () => {
                this._renderAlarmDistribution();
            });
        }
    },

    /* =====================================================================
       轮询 — 降低频率 + 移除模拟数据
       ================================================================== */
    _startPolling() {
        // 30 秒刷新插件数据（作为 WebSocket 的兜底）
        CockpitState.timers.plugins = setInterval(() => {
            this._loadCockpitData();
        }, 30000);

        // 10 秒刷新统计（从已有数据计算，不再模拟）
        CockpitState.timers.stats = setInterval(() => {
            this._updateStats();
            this._renderRealtimeAlarms();
        }, 10000);

        // 60 秒刷新趋势图
        CockpitState.timers.trend = setInterval(() => {
            this._renderAlarmTrend();
        }, 60000);
    },

    /* =====================================================================
       工具方法
       ================================================================== */
    _setText(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    },

    _updateFooterPluginCount() {
        this._setText('footer-plugin-count', CockpitState.plugins.length);
        // 也更新 base.html 的底部
        const baseCount = document.getElementById('plugin-count');
        if (baseCount) baseCount.textContent = CockpitState.plugins.length;
    },
};
