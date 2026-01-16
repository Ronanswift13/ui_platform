/**
 * 统一仪表盘 - 增强版
 * =====================
 *
 * 基于UPDATE.md优化方案实现:
 * - 综合监控页面，按设备/区域展示实时状态
 * - 趋势图、告警、健康指标
 * - 模块过滤、层级下钻
 * - 深度学习检测结果可视化
 *
 * 输变电激光星芒破夜绘明监测平台 V3.0
 */

// =============================================================================
// 统一仪表盘状态管理
// =============================================================================
const UnifiedDashboard = {
    version: '3.0.0',

    // 状态数据
    state: {
        // 按设备分类的监测数据
        devices: {},

        // 按区域分类的监测数据
        areas: {},

        // 各模块健康状态
        moduleHealth: {},

        // 实时告警
        activeAlarms: [],

        // 历史趋势数据
        trendData: {
            timestamps: [],
            transformer: [],
            switch: [],
            busbar: [],
            capacitor: [],
            meter: [],
            acoustic: [],
            gas: []
        },

        // 过滤器状态
        filters: {
            voltage: 'all',
            area: 'all',
            module: 'all',
            alarmLevel: 'all'
        },

        // 深度学习检测统计
        dlStats: {
            totalInferences: 0,
            avgLatency: 0,
            accuracy: 0,
            modelStatus: {}
        }
    },

    // 配置
    config: {
        refreshInterval: 5000,       // 刷新间隔(ms)
        trendPoints: 60,             // 趋势图点数
        maxAlarms: 100,              // 最大告警数
        enableDeepLearning: true     // 启用深度学习
    },

    // 定时器
    timers: {
        refresh: null,
        trend: null
    }
};


// =============================================================================
// 初始化
// =============================================================================
function initUnifiedDashboard() {
    console.log(`[UnifiedDashboard] 初始化 V${UnifiedDashboard.version}`);

    // 初始化图表
    initCharts();

    // 加载初始数据
    loadDashboardData();

    // 启动定时刷新
    startAutoRefresh();

    // 绑定事件
    bindDashboardEvents();

    // 初始化深度学习状态
    if (UnifiedDashboard.config.enableDeepLearning) {
        initDeepLearningStatus();
    }

    console.log('[UnifiedDashboard] 初始化完成');
}


// =============================================================================
// 数据加载
// =============================================================================
async function loadDashboardData() {
    try {
        // 并行加载所有数据
        const [
            deviceData,
            areaData,
            healthData,
            alarmData,
            dlStats
        ] = await Promise.all([
            fetchDeviceStatus(),
            fetchAreaStatus(),
            fetchModuleHealth(),
            fetchActiveAlarms(),
            fetchDeepLearningStats()
        ]);

        // 更新状态
        UnifiedDashboard.state.devices = deviceData;
        UnifiedDashboard.state.areas = areaData;
        UnifiedDashboard.state.moduleHealth = healthData;
        UnifiedDashboard.state.activeAlarms = alarmData;
        UnifiedDashboard.state.dlStats = dlStats;

        // 刷新UI
        refreshDashboardUI();

    } catch (error) {
        console.error('[UnifiedDashboard] 数据加载失败:', error);
    }
}

async function fetchDeviceStatus() {
    try {
        const response = await fetch('/api/advanced/devices/status');
        if (response.ok) {
            return await response.json();
        }
    } catch (e) {
        console.error('[Devices] 加载失败:', e);
    }
    return getSimulatedDeviceData();
}

async function fetchAreaStatus() {
    try {
        const response = await fetch('/api/advanced/areas/status');
        if (response.ok) {
            return await response.json();
        }
    } catch (e) {
        console.error('[Areas] 加载失败:', e);
    }
    return getSimulatedAreaData();
}

async function fetchModuleHealth() {
    try {
        const response = await fetch('/api/advanced/modules/health');
        if (response.ok) {
            return await response.json();
        }
    } catch (e) {
        console.error('[Health] 加载失败:', e);
    }
    return getSimulatedHealthData();
}

async function fetchActiveAlarms() {
    try {
        const response = await fetch('/api/advanced/alarms/active');
        if (response.ok) {
            return (await response.json()).alarms || [];
        }
    } catch (e) {
        console.error('[Alarms] 加载失败:', e);
    }
    return getSimulatedAlarmData();
}

async function fetchDeepLearningStats() {
    try {
        const response = await fetch('/api/advanced/dl/stats');
        if (response.ok) {
            return await response.json();
        }
    } catch (e) {
        console.error('[DL Stats] 加载失败:', e);
    }
    return getSimulatedDLStats();
}


// =============================================================================
// 模拟数据生成 (开发/演示用)
// =============================================================================
function getSimulatedDeviceData() {
    return {
        transformer: [
            { id: 'T1', name: '1号主变', status: 'normal', health: 95, lastCheck: new Date().toISOString() },
            { id: 'T2', name: '2号主变', status: 'warning', health: 82, lastCheck: new Date().toISOString() },
            { id: 'T3', name: '3号主变', status: 'normal', health: 91, lastCheck: new Date().toISOString() }
        ],
        switch: [
            { id: 'S1', name: '断路器组1', status: 'normal', health: 98, position: 'closed' },
            { id: 'S2', name: '断路器组2', status: 'normal', health: 96, position: 'open' },
            { id: 'S3', name: '隔离开关组', status: 'normal', health: 94, position: 'closed' }
        ],
        busbar: [
            { id: 'B1', name: '110kV母线I', status: 'normal', health: 97 },
            { id: 'B2', name: '110kV母线II', status: 'normal', health: 95 }
        ],
        capacitor: [
            { id: 'C1', name: '电容器组1', status: 'normal', health: 93 },
            { id: 'C2', name: '电容器组2', status: 'alarm', health: 65 }
        ]
    };
}

function getSimulatedAreaData() {
    return {
        outdoor: {
            name: '室外场地',
            deviceCount: 24,
            normalCount: 21,
            warningCount: 2,
            alarmCount: 1,
            health: 88
        },
        indoor: {
            name: '室内环境',
            deviceCount: 12,
            normalCount: 12,
            warningCount: 0,
            alarmCount: 0,
            health: 100
        },
        control: {
            name: '控制室',
            deviceCount: 8,
            normalCount: 8,
            warningCount: 0,
            alarmCount: 0,
            health: 100
        }
    };
}

function getSimulatedHealthData() {
    return {
        transformer: { healthy: true, score: 92, lastUpdate: new Date().toISOString(), dlEnabled: true },
        switch: { healthy: true, score: 96, lastUpdate: new Date().toISOString(), dlEnabled: true },
        busbar: { healthy: true, score: 94, lastUpdate: new Date().toISOString(), dlEnabled: true },
        capacitor: { healthy: false, score: 78, lastUpdate: new Date().toISOString(), dlEnabled: true },
        meter: { healthy: true, score: 89, lastUpdate: new Date().toISOString(), dlEnabled: true },
        acoustic: { healthy: true, score: 91, lastUpdate: new Date().toISOString(), dlEnabled: true },
        gas: { healthy: true, score: 95, lastUpdate: new Date().toISOString(), dlEnabled: true },
        bird: { healthy: true, score: 88, lastUpdate: new Date().toISOString(), dlEnabled: true }
    };
}

function getSimulatedAlarmData() {
    return [
        {
            id: 'A001',
            level: 'critical',
            module: 'capacitor',
            device: 'C2',
            title: '电容器鼓包告警',
            message: '检测到2号电容器组外壳鼓包变形',
            timestamp: new Date(Date.now() - 300000).toISOString(),
            acknowledged: false
        },
        {
            id: 'A002',
            level: 'warning',
            module: 'transformer',
            device: 'T2',
            title: '油位偏低',
            message: '2号主变油位略低于正常范围',
            timestamp: new Date(Date.now() - 1800000).toISOString(),
            acknowledged: false
        },
        {
            id: 'A003',
            level: 'info',
            module: 'acoustic',
            device: 'T1',
            title: '局部放电检测',
            message: '1号主变检测到轻微局放信号',
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            acknowledged: true
        }
    ];
}

function getSimulatedDLStats() {
    return {
        totalInferences: 15847,
        avgLatency: 45.2,
        accuracy: 0.947,
        modelStatus: {
            'YOLOv8-ViT': { loaded: true, version: '1.0.0', lastInference: new Date().toISOString() },
            'GL-TransLSTM': { loaded: true, version: '1.0.0', lastInference: new Date().toISOString() },
            'CNN-LSTM-Acoustic': { loaded: true, version: '1.0.0', lastInference: new Date().toISOString() },
            'KeypointDetector': { loaded: true, version: '1.0.0', lastInference: new Date().toISOString() },
            'DeepSORT': { loaded: true, version: '1.0.0', lastInference: new Date().toISOString() }
        },
        gpuUsage: 62,
        memoryUsage: 4.8
    };
}


// =============================================================================
// UI刷新
// =============================================================================
function refreshDashboardUI() {
    renderDeviceOverview();
    renderAreaSummary();
    renderHealthIndicators();
    renderAlarmList();
    renderDLStatus();
    updateTrendChart();
}

function renderDeviceOverview() {
    const container = document.getElementById('device-overview');
    if (!container) return;

    const { devices } = UnifiedDashboard.state;
    let html = '';

    // 渲染每种设备类型
    const deviceTypes = [
        { key: 'transformer', name: '主变', icon: 'box', color: 'primary' },
        { key: 'switch', name: '开关', icon: 'toggle-on', color: 'success' },
        { key: 'busbar', name: '母线', icon: 'diagram-3', color: 'warning' },
        { key: 'capacitor', name: '电容器', icon: 'battery-charging', color: 'info' }
    ];

    for (const type of deviceTypes) {
        const deviceList = devices[type.key] || [];
        const normalCount = deviceList.filter(d => d.status === 'normal').length;
        const avgHealth = deviceList.length > 0
            ? Math.round(deviceList.reduce((sum, d) => sum + d.health, 0) / deviceList.length)
            : 0;

        html += `
            <div class="device-type-card" data-type="${type.key}">
                <div class="device-type-header">
                    <i class="bi bi-${type.icon} text-${type.color}"></i>
                    <span class="device-type-name">${type.name}</span>
                    <span class="device-type-count">${deviceList.length}台</span>
                </div>
                <div class="device-type-stats">
                    <div class="stat-item">
                        <span class="stat-label">正常</span>
                        <span class="stat-value text-success">${normalCount}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">健康度</span>
                        <span class="stat-value ${getHealthClass(avgHealth)}">${avgHealth}%</span>
                    </div>
                </div>
                <div class="device-list">
                    ${deviceList.map(d => renderDeviceItem(d)).join('')}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function renderDeviceItem(device) {
    const statusClass = getStatusClass(device.status);
    const healthClass = getHealthClass(device.health);

    return `
        <div class="device-item ${statusClass}" data-id="${device.id}" onclick="showDeviceDetail('${device.id}')">
            <div class="device-name">${device.name}</div>
            <div class="device-health">
                <div class="health-bar">
                    <div class="health-fill ${healthClass}" style="width: ${device.health}%"></div>
                </div>
                <span class="health-value">${device.health}%</span>
            </div>
            <div class="device-status">
                <span class="status-dot ${statusClass}"></span>
                ${getStatusText(device.status)}
            </div>
        </div>
    `;
}

function renderAreaSummary() {
    const container = document.getElementById('area-summary');
    if (!container) return;

    const { areas } = UnifiedDashboard.state;
    let html = '';

    for (const [key, area] of Object.entries(areas)) {
        const healthClass = getHealthClass(area.health);

        html += `
            <div class="area-card" data-area="${key}">
                <div class="area-header">
                    <span class="area-name">${area.name}</span>
                    <span class="area-health ${healthClass}">${area.health}%</span>
                </div>
                <div class="area-stats">
                    <div class="area-stat">
                        <i class="bi bi-check-circle text-success"></i>
                        <span>${area.normalCount}</span>
                    </div>
                    <div class="area-stat">
                        <i class="bi bi-exclamation-triangle text-warning"></i>
                        <span>${area.warningCount}</span>
                    </div>
                    <div class="area-stat">
                        <i class="bi bi-x-circle text-danger"></i>
                        <span>${area.alarmCount}</span>
                    </div>
                </div>
                <div class="area-progress">
                    <div class="progress" style="height: 6px;">
                        <div class="progress-bar bg-success" style="width: ${(area.normalCount / area.deviceCount * 100)}%"></div>
                        <div class="progress-bar bg-warning" style="width: ${(area.warningCount / area.deviceCount * 100)}%"></div>
                        <div class="progress-bar bg-danger" style="width: ${(area.alarmCount / area.deviceCount * 100)}%"></div>
                    </div>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function renderHealthIndicators() {
    const container = document.getElementById('health-indicators');
    if (!container) return;

    const { moduleHealth } = UnifiedDashboard.state;
    let html = '';

    const moduleNames = {
        transformer: '主变检测',
        switch: '开关检测',
        busbar: '母线检测',
        capacitor: '电容器检测',
        meter: '表计读数',
        acoustic: '声学监测',
        gas: '气体检测',
        bird: '鸟类监测'
    };

    for (const [key, health] of Object.entries(moduleHealth)) {
        const healthClass = getHealthClass(health.score);
        const dlBadge = health.dlEnabled
            ? '<span class="badge bg-info ms-1">DL</span>'
            : '';

        html += `
            <div class="health-card ${health.healthy ? '' : 'unhealthy'}" data-module="${key}">
                <div class="health-header">
                    <span class="module-name">${moduleNames[key] || key}</span>
                    ${dlBadge}
                </div>
                <div class="health-score ${healthClass}">${health.score}</div>
                <div class="health-status">
                    <span class="status-indicator ${health.healthy ? 'healthy' : 'unhealthy'}"></span>
                    ${health.healthy ? '正常' : '异常'}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function renderAlarmList() {
    const container = document.getElementById('unified-alarm-list');
    if (!container) return;

    const { activeAlarms } = UnifiedDashboard.state;
    const { alarmLevel } = UnifiedDashboard.state.filters;

    // 过滤告警
    let filteredAlarms = activeAlarms;
    if (alarmLevel !== 'all') {
        filteredAlarms = activeAlarms.filter(a => a.level === alarmLevel);
    }

    // 按时间排序
    filteredAlarms.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    if (filteredAlarms.length === 0) {
        container.innerHTML = '<div class="no-alarms">暂无告警</div>';
        return;
    }

    let html = '';
    for (const alarm of filteredAlarms) {
        const timeAgo = getTimeAgo(alarm.timestamp);
        const levelClass = getLevelClass(alarm.level);
        const ackClass = alarm.acknowledged ? 'acknowledged' : '';

        html += `
            <div class="alarm-card ${levelClass} ${ackClass}" data-id="${alarm.id}">
                <div class="alarm-icon">
                    <i class="bi bi-${getAlarmIcon(alarm.level)}"></i>
                </div>
                <div class="alarm-content">
                    <div class="alarm-title">${alarm.title}</div>
                    <div class="alarm-message">${alarm.message}</div>
                    <div class="alarm-meta">
                        <span class="alarm-module">${alarm.module}</span>
                        <span class="alarm-device">${alarm.device}</span>
                        <span class="alarm-time">${timeAgo}</span>
                    </div>
                </div>
                <div class="alarm-actions">
                    ${!alarm.acknowledged ? `
                        <button class="btn btn-sm btn-outline-light" onclick="acknowledgeAlarm('${alarm.id}')">
                            <i class="bi bi-check"></i>
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;

    // 更新告警计数
    updateAlarmCount(activeAlarms);
}

function renderDLStatus() {
    const container = document.getElementById('dl-status-panel');
    if (!container) return;

    const { dlStats } = UnifiedDashboard.state;

    let modelsHtml = '';
    for (const [name, status] of Object.entries(dlStats.modelStatus || {})) {
        modelsHtml += `
            <div class="dl-model-item ${status.loaded ? 'loaded' : 'unloaded'}">
                <span class="model-name">${name}</span>
                <span class="model-status">
                    <span class="status-dot ${status.loaded ? 'active' : 'inactive'}"></span>
                    ${status.loaded ? 'v' + status.version : '未加载'}
                </span>
            </div>
        `;
    }

    container.innerHTML = `
        <div class="dl-stats-header">
            <h6><i class="bi bi-cpu text-primary"></i> 深度学习状态</h6>
        </div>
        <div class="dl-stats-overview">
            <div class="dl-stat-item">
                <span class="stat-label">总推理次数</span>
                <span class="stat-value">${formatNumber(dlStats.totalInferences)}</span>
            </div>
            <div class="dl-stat-item">
                <span class="stat-label">平均延迟</span>
                <span class="stat-value">${dlStats.avgLatency.toFixed(1)}ms</span>
            </div>
            <div class="dl-stat-item">
                <span class="stat-label">准确率</span>
                <span class="stat-value">${(dlStats.accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="dl-stat-item">
                <span class="stat-label">GPU使用</span>
                <span class="stat-value">${dlStats.gpuUsage}%</span>
            </div>
        </div>
        <div class="dl-models-list">
            <h6 class="models-title">已加载模型</h6>
            ${modelsHtml}
        </div>
    `;
}


// =============================================================================
// 图表
// =============================================================================
let trendChart = null;

function initCharts() {
    const ctx = document.getElementById('trend-chart');
    if (!ctx) return;

    // 初始化趋势数据
    const now = new Date();
    for (let i = UnifiedDashboard.config.trendPoints - 1; i >= 0; i--) {
        const time = new Date(now - i * 60000);
        UnifiedDashboard.state.trendData.timestamps.push(formatTime(time));
    }

    // 初始化各模块数据
    const modules = ['transformer', 'switch', 'busbar', 'capacitor', 'meter', 'acoustic', 'gas'];
    for (const module of modules) {
        UnifiedDashboard.state.trendData[module] = Array(UnifiedDashboard.config.trendPoints).fill(0).map(() =>
            Math.floor(80 + Math.random() * 20)
        );
    }

    // 创建Chart (如果Chart.js可用)
    if (typeof Chart !== 'undefined') {
        trendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: UnifiedDashboard.state.trendData.timestamps,
                datasets: [
                    {
                        label: '主变',
                        data: UnifiedDashboard.state.trendData.transformer,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: '开关',
                        data: UnifiedDashboard.state.trendData.switch,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: '声学',
                        data: UnifiedDashboard.state.trendData.acoustic,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#888' }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#888', maxTicksLimit: 10 }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#888' }
                    }
                }
            }
        });
    }
}

function updateTrendChart() {
    if (!trendChart) return;

    // 添加新数据点
    const now = new Date();
    const { trendData } = UnifiedDashboard.state;
    const { moduleHealth } = UnifiedDashboard.state;

    // 移除最旧的点
    trendData.timestamps.shift();
    trendData.timestamps.push(formatTime(now));

    // 更新各模块数据
    const modules = ['transformer', 'switch', 'busbar', 'capacitor', 'meter', 'acoustic', 'gas'];
    for (const module of modules) {
        trendData[module].shift();
        const health = moduleHealth[module];
        trendData[module].push(health ? health.score : Math.floor(80 + Math.random() * 20));
    }

    // 更新图表
    trendChart.data.labels = trendData.timestamps;
    trendChart.data.datasets[0].data = trendData.transformer;
    trendChart.data.datasets[1].data = trendData.switch;
    trendChart.data.datasets[2].data = trendData.acoustic;
    trendChart.update('none');
}


// =============================================================================
// 事件处理
// =============================================================================
function bindDashboardEvents() {
    // 过滤器事件
    const voltageFilter = document.getElementById('voltage-filter');
    if (voltageFilter) {
        voltageFilter.addEventListener('change', (e) => {
            UnifiedDashboard.state.filters.voltage = e.target.value;
            loadDashboardData();
        });
    }

    const areaFilter = document.getElementById('area-filter');
    if (areaFilter) {
        areaFilter.addEventListener('change', (e) => {
            UnifiedDashboard.state.filters.area = e.target.value;
            refreshDashboardUI();
        });
    }

    const moduleFilter = document.getElementById('module-filter');
    if (moduleFilter) {
        moduleFilter.addEventListener('change', (e) => {
            UnifiedDashboard.state.filters.module = e.target.value;
            refreshDashboardUI();
        });
    }

    const alarmFilter = document.getElementById('alarm-level-filter');
    if (alarmFilter) {
        alarmFilter.addEventListener('change', (e) => {
            UnifiedDashboard.state.filters.alarmLevel = e.target.value;
            renderAlarmList();
        });
    }
}

function showDeviceDetail(deviceId) {
    console.log(`[Dashboard] 显示设备详情: ${deviceId}`);
    // 打开设备详情模态框或跳转页面
    // 这里可以实现层级下钻功能
}

async function acknowledgeAlarm(alarmId) {
    try {
        const response = await fetch(`/api/advanced/alarms/${alarmId}/acknowledge`, {
            method: 'POST'
        });

        if (response.ok) {
            // 更新本地状态
            const alarm = UnifiedDashboard.state.activeAlarms.find(a => a.id === alarmId);
            if (alarm) {
                alarm.acknowledged = true;
            }
            renderAlarmList();
        }
    } catch (error) {
        console.error('[Alarm] 确认失败:', error);
    }
}


// =============================================================================
// 自动刷新
// =============================================================================
function startAutoRefresh() {
    // 数据刷新
    UnifiedDashboard.timers.refresh = setInterval(() => {
        loadDashboardData();
    }, UnifiedDashboard.config.refreshInterval);

    // 趋势图更新
    UnifiedDashboard.timers.trend = setInterval(() => {
        updateTrendChart();
    }, 60000); // 每分钟更新
}

function stopAutoRefresh() {
    if (UnifiedDashboard.timers.refresh) {
        clearInterval(UnifiedDashboard.timers.refresh);
    }
    if (UnifiedDashboard.timers.trend) {
        clearInterval(UnifiedDashboard.timers.trend);
    }
}


// =============================================================================
// 深度学习状态监控
// =============================================================================
function initDeepLearningStatus() {
    // 检查深度学习模型状态
    checkModelStatus();

    // 定期检查
    setInterval(checkModelStatus, 30000);
}

async function checkModelStatus() {
    try {
        const response = await fetch('/api/advanced/dl/models/status');
        if (response.ok) {
            const data = await response.json();
            UnifiedDashboard.state.dlStats.modelStatus = data.models || {};
            renderDLStatus();
        }
    } catch (error) {
        console.log('[DL] 模型状态检查失败 (可能未部署)');
    }
}


// =============================================================================
// 辅助函数
// =============================================================================
function getStatusClass(status) {
    const classes = {
        normal: 'status-normal',
        warning: 'status-warning',
        alarm: 'status-alarm',
        error: 'status-error'
    };
    return classes[status] || 'status-unknown';
}

function getStatusText(status) {
    const texts = {
        normal: '正常',
        warning: '注意',
        alarm: '告警',
        error: '故障'
    };
    return texts[status] || '未知';
}

function getHealthClass(health) {
    if (health >= 90) return 'health-excellent';
    if (health >= 80) return 'health-good';
    if (health >= 60) return 'health-warning';
    return 'health-danger';
}

function getLevelClass(level) {
    const classes = {
        critical: 'level-critical',
        alarm: 'level-alarm',
        warning: 'level-warning',
        info: 'level-info'
    };
    return classes[level] || 'level-info';
}

function getAlarmIcon(level) {
    const icons = {
        critical: 'exclamation-octagon-fill',
        alarm: 'exclamation-triangle-fill',
        warning: 'exclamation-circle-fill',
        info: 'info-circle-fill'
    };
    return icons[level] || 'info-circle';
}

function getTimeAgo(timestamp) {
    const now = new Date();
    const time = new Date(timestamp);
    const diff = Math.floor((now - time) / 1000);

    if (diff < 60) return '刚刚';
    if (diff < 3600) return `${Math.floor(diff / 60)}分钟前`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}小时前`;
    return `${Math.floor(diff / 86400)}天前`;
}

function formatTime(date) {
    return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

function updateAlarmCount(alarms) {
    const criticalCount = alarms.filter(a => a.level === 'critical' && !a.acknowledged).length;
    const warningCount = alarms.filter(a => a.level === 'warning' && !a.acknowledged).length;
    const totalCount = alarms.filter(a => !a.acknowledged).length;

    const countEl = document.getElementById('total-alarm-count');
    if (countEl) {
        countEl.textContent = totalCount;
        countEl.className = `badge ${criticalCount > 0 ? 'bg-danger' : (warningCount > 0 ? 'bg-warning' : 'bg-secondary')}`;
    }

    const criticalEl = document.getElementById('critical-alarm-count');
    if (criticalEl) {
        criticalEl.textContent = criticalCount;
    }
}


// =============================================================================
// 导出
// =============================================================================
window.initUnifiedDashboard = initUnifiedDashboard;
window.showDeviceDetail = showDeviceDetail;
window.acknowledgeAlarm = acknowledgeAlarm;
window.stopAutoRefresh = stopAutoRefresh;
