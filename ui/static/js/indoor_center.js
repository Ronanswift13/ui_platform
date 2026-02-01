/**
 * 室内监测中心 V3.5 - 完整JavaScript实现
 * =========================================
 * 
 * 输变电激光星芒破夜绘明监测平台
 * 
 * 功能模块:
 * 1. 电子围栏 - 多人安全监测、越线检测
 * 2. 动物入侵检测 - 小动物识别与驱离
 * 3. 温度监测 - 热力图可视化
 * 4. 设备状态监测 - 设备健康检查
 * 5. 消防监测 - 火灾预警
 * 6. 环境监测 - SF6/O2/CO浓度
 * 
 * 版本: 3.5.0
 * 更新: 2026/01/22
 */

// =============================================================================
// 全局状态管理
// =============================================================================
const IndoorMonitorState = {
    version: '3.5.0',
    
    // 模块配置
    modules: {
        fence: {
            id: 'fence',
            name: '电子围栏',
            icon: 'bi-shield-shaded',
            api: '/api/indoor/fence',
            status: 'online',
            canvas: null,
            ctx: null,
            data: null,
            refreshInterval: null
        },
        animal: {
            id: 'animal',
            name: '动物入侵检测',
            icon: 'bi-bug',
            api: '/api/indoor/animal',
            status: 'online',
            canvas: null,
            ctx: null,
            data: null,
            refreshInterval: null
        },
        temperature: {
            id: 'temperature',
            name: '温度监测',
            icon: 'bi-thermometer-half',
            api: '/api/indoor/temperature',
            status: 'attention',
            canvas: null,
            ctx: null,
            data: null,
            refreshInterval: null
        },
        device: {
            id: 'device',
            name: '设备状态监测',
            icon: 'bi-hdd-stack',
            api: '/api/indoor/device',
            status: 'online',
            canvas: null,
            ctx: null,
            data: null,
            refreshInterval: null
        },
        fire: {
            id: 'fire',
            name: '消防监测',
            icon: 'bi-fire',
            api: '/api/indoor/fire',
            status: 'online',
            canvas: null,
            ctx: null,
            data: null,
            refreshInterval: null
        },
        environment: {
            id: 'environment',
            name: '环境监测',
            icon: 'bi-cloud-haze2',
            api: '/api/indoor/environment',
            status: 'good',
            canvas: null,
            ctx: null,
            data: null,
            refreshInterval: null
        }
    },
    
    // WebSocket连接
    ws: null,
    wsConnected: false,
    wsReconnectTimer: null,
    
    // 告警状态
    alarms: [],
    totalAlarms: 0,
    
    // 刷新配置
    refreshRate: 1000, // 1秒
    autoRefresh: true,
    
    // 模拟数据模式（当API不可用时）
    simulationMode: true
};

// =============================================================================
// 3D数字孪生状态
// =============================================================================
const Indoor3DState = {
    viewer: null,
    alarmManager: null,
    alarms: [],
    alarmStats: {},
    activeAlarmId: null,
    refreshTimer: null,
    lastSceneSync: 0,
    lastOverlaySync: 0,
    lastAlarmSync: 0,
    sceneSyncInterval: 15000,
    overlaySyncInterval: 4000,
    alarmSyncInterval: 4000,
    trackHistory: new Map(),
    useServerTrajectories: false,
    pointCloudLoaded: false,
    modelLoaded: false,
    controlsBound: false,
    targetCounts: {
        persons: 0,
        animals: 0,
        fires: 0
    }
};

// =============================================================================
// 初始化
// =============================================================================
document.addEventListener('DOMContentLoaded', function() {
    console.log(`[IndoorMonitor] 初始化 V${IndoorMonitorState.version}`);
    
    // 初始化所有模块画布
    initModuleCanvases();
    
    // 加载初始数据
    loadAllModulesData();
    
    // 启动自动刷新
    startAutoRefresh();
    
    // 尝试连接WebSocket
    connectWebSocket();
    
    // 绑定事件
    bindEvents();

    // 初始化3D数字孪生联动
    initIndoor3DIntegration();

    // 初始化融合证据链
    updateFusionEvidence();

    // 加载默认设备的控制面板
    selectDevice('fence');

    console.log('[IndoorMonitor] 初始化完成');
});

/**
 * 初始化所有模块的画布
 */
function initModuleCanvases() {
    // 电子围栏画布
    const fenceCanvas = document.getElementById('fence-canvas');
    if (fenceCanvas) {
        IndoorMonitorState.modules.fence.canvas = fenceCanvas;
        IndoorMonitorState.modules.fence.ctx = fenceCanvas.getContext('2d');
        resizeCanvas(fenceCanvas);
    }
    
    // 动物入侵画布
    const animalCanvas = document.getElementById('animal-canvas');
    if (animalCanvas) {
        IndoorMonitorState.modules.animal.canvas = animalCanvas;
        IndoorMonitorState.modules.animal.ctx = animalCanvas.getContext('2d');
        resizeCanvas(animalCanvas);
    }
    
    // 温度监测画布
    const tempCanvas = document.getElementById('temp-canvas');
    if (tempCanvas) {
        IndoorMonitorState.modules.temperature.canvas = tempCanvas;
        IndoorMonitorState.modules.temperature.ctx = tempCanvas.getContext('2d');
        resizeCanvas(tempCanvas);
    }
    
    // 设备状态画布
    const deviceCanvas = document.getElementById('device-canvas');
    if (deviceCanvas) {
        IndoorMonitorState.modules.device.canvas = deviceCanvas;
        IndoorMonitorState.modules.device.ctx = deviceCanvas.getContext('2d');
        resizeCanvas(deviceCanvas);
    }
    
    // 消防监测画布
    const fireCanvas = document.getElementById('fire-canvas');
    if (fireCanvas) {
        IndoorMonitorState.modules.fire.canvas = fireCanvas;
        IndoorMonitorState.modules.fire.ctx = fireCanvas.getContext('2d');
        resizeCanvas(fireCanvas);
    }
    
    // 环境监测画布
    const envCanvas = document.getElementById('env-canvas');
    if (envCanvas) {
        IndoorMonitorState.modules.environment.canvas = envCanvas;
        IndoorMonitorState.modules.environment.ctx = envCanvas.getContext('2d');
        resizeCanvas(envCanvas);
    }
    
    // 监听窗口大小变化
    window.addEventListener('resize', debounce(handleResize, 250));
}

/**
 * 调整画布大小
 */
function resizeCanvas(canvas) {
    if (!canvas) return;
    const container = canvas.parentElement;
    if (container) {
        canvas.width = container.clientWidth || 300;
        canvas.height = container.clientHeight || 200;
    }
}

/**
 * 处理窗口大小变化
 */
function handleResize() {
    Object.values(IndoorMonitorState.modules).forEach(module => {
        if (module.canvas) {
            resizeCanvas(module.canvas);
            // 重新渲染
            renderModule(module.id);
        }
    });
}

// =============================================================================
// 数据加载
// =============================================================================

/**
 * 加载所有模块数据
 */
function loadAllModulesData() {
    Object.keys(IndoorMonitorState.modules).forEach(moduleId => {
        loadModuleData(moduleId);
    });
}

/**
 * 加载单个模块数据
 */
async function loadModuleData(moduleId) {
    const module = IndoorMonitorState.modules[moduleId];
    if (!module) return;
    
    try {
        // 尝试从API获取数据
        const response = await fetch(module.api);
        if (response.ok) {
            const data = await response.json();
            module.data = data;
            module.status = data.status || 'online';
            IndoorMonitorState.simulationMode = false;
        } else {
            throw new Error('API not available');
        }
    } catch (error) {
        // API不可用，使用模拟数据
        console.log(`[${moduleId}] 使用模拟数据`);
        module.data = generateSimulationData(moduleId);
        IndoorMonitorState.simulationMode = true;
    }
    
    // 渲染模块
    renderModule(moduleId);
    updateModuleStatus(moduleId);

    // 同步到3D场景
    syncIndoor3DWithModule(moduleId, module.data);
}

/**
 * 生成模拟数据
 */
function generateSimulationData(moduleId) {
    const timestamp = Date.now();
    
    switch (moduleId) {
        case 'fence':
            return generateFenceData(timestamp);
        case 'animal':
            return generateAnimalData(timestamp);
        case 'temperature':
            return generateTemperatureData(timestamp);
        case 'device':
            return generateDeviceData(timestamp);
        case 'fire':
            return generateFireData(timestamp);
        case 'environment':
            return generateEnvironmentData(timestamp);
        default:
            return { timestamp, status: 'unknown' };
    }
}

/**
 * 电子围栏模拟数据
 */
function generateFenceData(timestamp) {
    // 模拟人员位置
    const persons = [
        { id: 1, name: '张工', x: 0.3, y: 0.5, cabinet: 2, status: 'safe', authorized: true },
        { id: 2, name: '李工', x: 0.7, y: 0.2, cabinet: 5, status: 'danger', authorized: true },
    ];
    
    // 模拟机柜状态
    const cabinets = [];
    for (let i = 1; i <= 6; i++) {
        const person = persons.find(p => p.cabinet === i);
        cabinets.push({
            id: i,
            name: `机柜-${String(i).padStart(2, '0')}`,
            status: person ? (person.status === 'danger' ? 'alarm' : 'occupied') : 'idle',
            person: person ? person.name : null
        });
    }
    
    // 黄线位置
    const yellowLine = { y: 0.15 };
    
    return {
        timestamp,
        status: persons.some(p => p.status === 'danger') ? 'alarm' : 'online',
        persons,
        cabinets,
        yellowLine,
        totalPersons: persons.length,
        alarmCount: persons.filter(p => p.status === 'danger').length
    };
}

/**
 * 动物入侵模拟数据
 */
function generateAnimalData(timestamp) {
    // 随机检测到动物
    const hasAnimal = Math.random() < 0.2;
    const detections = hasAnimal ? [
        {
            id: 1,
            type: Math.random() < 0.5 ? 'mouse' : 'bird',
            confidence: 0.75 + Math.random() * 0.2,
            bbox: {
                x: 0.3 + Math.random() * 0.4,
                y: 0.3 + Math.random() * 0.4,
                width: 0.1,
                height: 0.08
            }
        }
    ] : [];
    
    return {
        timestamp,
        status: hasAnimal ? 'alarm' : 'online',
        detections,
        todayCount: Math.floor(Math.random() * 5),
        deterrentActive: hasAnimal
    };
}

/**
 * 温度监测模拟数据
 */
function generateTemperatureData(timestamp) {
    // 生成热力图数据 (8x6 网格)
    const gridWidth = 8;
    const gridHeight = 6;
    const heatmap = [];
    
    for (let y = 0; y < gridHeight; y++) {
        const row = [];
        for (let x = 0; x < gridWidth; x++) {
            // 基础温度 + 随机波动 + 热点
            let temp = 25 + Math.random() * 5;
            // 创建一些热点
            if ((x === 3 && y === 2) || (x === 6 && y === 4)) {
                temp += 15 + Math.random() * 10;
            }
            row.push(Math.round(temp * 10) / 10);
        }
        heatmap.push(row);
    }
    
    const maxTemp = Math.max(...heatmap.flat());
    const minTemp = Math.min(...heatmap.flat());
    const avgTemp = heatmap.flat().reduce((a, b) => a + b, 0) / (gridWidth * gridHeight);
    
    return {
        timestamp,
        status: maxTemp > 45 ? 'warning' : 'attention',
        heatmap,
        maxTemp: Math.round(maxTemp * 10) / 10,
        minTemp: Math.round(minTemp * 10) / 10,
        avgTemp: Math.round(avgTemp * 10) / 10,
        hotspots: maxTemp > 40 ? [{ x: 3, y: 2, temp: maxTemp }] : []
    };
}

/**
 * 设备状态模拟数据
 */
function generateDeviceData(timestamp) {
    const devices = [
        { id: 'cam-01', name: '摄像机-01', type: 'camera', status: 'online', health: 98 },
        { id: 'cam-02', name: '摄像机-02', type: 'camera', status: 'online', health: 95 },
        { id: 'lidar-01', name: '激光雷达', type: 'lidar', status: 'online', health: 100 },
        { id: 'sensor-01', name: '温湿度传感器', type: 'sensor', status: 'online', health: 92 },
        { id: 'relay-01', name: '继电器模块', type: 'relay', status: 'online', health: 100 },
        { id: 'speaker-01', name: '声光报警器', type: 'alarm', status: 'standby', health: 100 }
    ];
    
    const onlineCount = devices.filter(d => d.status === 'online' || d.status === 'standby').length;
    
    return {
        timestamp,
        status: onlineCount === devices.length ? 'online' : 'warning',
        devices,
        totalDevices: devices.length,
        onlineDevices: onlineCount,
        avgHealth: Math.round(devices.reduce((a, d) => a + d.health, 0) / devices.length)
    };
}

/**
 * 消防监测模拟数据
 */
function generateFireData(timestamp) {
    const zones = [
        { id: 'zone-a', name: 'A区-主控室', status: 'normal', smokeLevel: 0.02, temp: 24 },
        { id: 'zone-b', name: 'B区-GIS室', status: 'normal', smokeLevel: 0.01, temp: 23 },
        { id: 'zone-c', name: 'C区-继保室', status: 'normal', smokeLevel: 0.015, temp: 25 },
        { id: 'zone-d', name: 'D区-蓄电池室', status: 'normal', smokeLevel: 0.03, temp: 26 }
    ];
    
    return {
        timestamp,
        status: 'online',
        zones,
        lastTest: '2026-01-20 09:00:00',
        sprinklerStatus: 'ready',
        alarmStatus: 'armed'
    };
}

/**
 * 环境监测模拟数据
 */
function generateEnvironmentData(timestamp) {
    // 模拟气体浓度时间序列
    const historyLength = 30;
    const sf6History = [];
    const o2History = [];
    const coHistory = [];
    
    for (let i = 0; i < historyLength; i++) {
        sf6History.push(5 + Math.random() * 3);
        o2History.push(20.5 + Math.random() * 0.5);
        coHistory.push(Math.random() * 2);
    }
    
    return {
        timestamp,
        status: 'good',
        sf6: {
            current: sf6History[sf6History.length - 1],
            unit: 'ppm',
            threshold: 1000,
            history: sf6History
        },
        o2: {
            current: o2History[o2History.length - 1],
            unit: '%',
            threshold: { min: 19.5, max: 23.5 },
            history: o2History
        },
        co: {
            current: coHistory[coHistory.length - 1],
            unit: 'ppm',
            threshold: 50,
            history: coHistory
        },
        humidity: 45 + Math.random() * 10,
        pressure: 1013 + Math.random() * 5
    };
}

// =============================================================================
// 3D数字孪生联动
// =============================================================================

function initIndoor3DIntegration() {
    const viewer = ensureIndoor3DViewer();
    if (!viewer) {
        console.warn('[Indoor3D] Three.js未就绪，跳过3D联动');
        return;
    }

    Indoor3DState.viewer = viewer;

    viewer.onAlarmClick((alarm) => {
        if (alarm && alarm.id) {
            // 只更新UI状态，不打开模态框（避免重复打开）
            selectIndoorAlarm(alarm.id, { focus: false, openModal: false });
        }
    });

    bindIndoor3DControls();
    updateAlarmProcessUI(null);
    renderIndoorAlarmList();
    loadIndoor3DScene();
    scheduleIndoor3DRefresh();
}

function ensureIndoor3DViewer() {
    if (Indoor3DState.viewer) {
        return Indoor3DState.viewer;
    }
    if (window.indoor3DViewer) {
        Indoor3DState.viewer = window.indoor3DViewer;
        return Indoor3DState.viewer;
    }

    const container = document.getElementById('indoor-3d-container');
    if (!container || !window.Indoor3DViewer || !window.THREE) {
        return null;
    }

    window.indoor3DViewer = new Indoor3DViewer('indoor-3d-container');
    container.classList.add('viewer-ready');
    Indoor3DState.viewer = window.indoor3DViewer;
    return Indoor3DState.viewer;
}

function bindIndoor3DControls() {
    if (Indoor3DState.controlsBound) return;

    const resetBtn = document.getElementById('indoor-3d-reset');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            const viewer = ensureIndoor3DViewer();
            if (viewer) {
                viewer.resetCamera();
            }
        });
    }

    const alarmList = document.getElementById('indoor-alarm-list');
    if (alarmList) {
        alarmList.addEventListener('click', (event) => {
            const item = event.target.closest('[data-alarm-id]');
            if (!item) return;
            selectIndoorAlarm(item.dataset.alarmId, { openModal: true });
        });
    }

    const actionButtons = document.querySelectorAll('[data-alarm-action]');
    actionButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            handleAlarmAction(btn.dataset.alarmAction);
        });
    });

    Indoor3DState.controlsBound = true;
}

function scheduleIndoor3DRefresh() {
    if (Indoor3DState.refreshTimer) return;

    Indoor3DState.refreshTimer = setInterval(() => {
        if (!IndoorMonitorState.autoRefresh) return;

        const now = Date.now();
        if (now - Indoor3DState.lastOverlaySync > Indoor3DState.overlaySyncInterval) {
            refreshIndoor3DOverlays();
            Indoor3DState.lastOverlaySync = now;
        }

        if (now - Indoor3DState.lastAlarmSync > Indoor3DState.alarmSyncInterval) {
            refreshIndoor3DAlarms();
            Indoor3DState.lastAlarmSync = now;
        }

        if (now - Indoor3DState.lastSceneSync > Indoor3DState.sceneSyncInterval) {
            loadIndoor3DScene();
            Indoor3DState.lastSceneSync = now;
        }
    }, 1000);
}

async function loadIndoor3DScene() {
    const viewer = ensureIndoor3DViewer();
    if (!viewer) return;

    Indoor3DState.lastSceneSync = Date.now();

    const container = document.getElementById('indoor-3d-container');
    const modelUrl = container?.dataset?.modelUrl;
    if (modelUrl && !Indoor3DState.modelLoaded) {
        viewer.loadModel(modelUrl, { scale: 1 });
        Indoor3DState.modelLoaded = true;
    }

    try {
        const data = await fetchJson('/api/indoor/3d/scene');
        if (!data) return;

        const sceneModelUrl = data.model_url || data.modelUrl || data.pointcloud?.url;
        if (sceneModelUrl && !Indoor3DState.modelLoaded) {
            viewer.loadModel(sceneModelUrl, { scale: 1 });
            Indoor3DState.modelLoaded = true;
        }

        if (data.pointcloud?.points?.length && !Indoor3DState.pointCloudLoaded) {
            viewer.loadPointCloud(data.pointcloud, { colorMode: 'height' });
            Indoor3DState.pointCloudLoaded = true;
        }

        if (Array.isArray(data.zones)) {
            viewer.updateZones(data.zones);
        }

        if (Array.isArray(data.trajectories) && data.trajectories.length) {
            viewer.updateTrajectories(data.trajectories);
            Indoor3DState.useServerTrajectories = true;
        } else {
            Indoor3DState.useServerTrajectories = false;
            updateLocalTrajectories();
        }

        updateHeatmapLayers(data.heatmaps);
        if (Array.isArray(data.alarms)) {
            syncIndoorAlarms(data.alarms, data.statistics);
        }
    } catch (error) {
        console.warn('[Indoor3D] 场景数据加载失败:', error);
    }
}

async function refreshIndoor3DOverlays() {
    const viewer = ensureIndoor3DViewer();
    if (!viewer) return;

    Indoor3DState.lastOverlaySync = Date.now();

    const [tempRes, gasRes, trajRes] = await Promise.allSettled([
        fetchJson('/api/indoor/3d/heatmap/temperature'),
        fetchJson('/api/indoor/3d/heatmap/gas'),
        fetchJson('/api/indoor/3d/trajectories')
    ]);

    const tempData = tempRes.status === 'fulfilled' ? tempRes.value : null;
    const gasData = gasRes.status === 'fulfilled' ? gasRes.value : null;
    const trajData = trajRes.status === 'fulfilled' ? trajRes.value : null;

    updateHeatmapLayers({
        temperature: tempData?.heatmap,
        gas: gasData?.heatmap
    });

    if (trajData?.trajectories?.length) {
        viewer.updateTrajectories(trajData.trajectories);
        Indoor3DState.useServerTrajectories = true;
    } else {
        Indoor3DState.useServerTrajectories = false;
        updateLocalTrajectories();
    }
}

async function refreshIndoor3DAlarms() {
    Indoor3DState.lastAlarmSync = Date.now();
    const data = await fetchJson('/api/indoor/3d/alarms');
    if (data?.alarms) {
        syncIndoorAlarms(data.alarms, data.statistics);
    }
}

function updateHeatmapLayers(heatmaps) {
    const viewer = ensureIndoor3DViewer();
    if (!viewer || !heatmaps) return;

    const temperature = heatmaps.temperature?.heatmap || heatmaps.temperature;
    const gas = heatmaps.gas?.heatmap || heatmaps.gas;

    let cleared = false;
    if (temperature?.data?.length) {
        viewer.updateHeatmap(temperature, {
            type: 'temperature',
            heightOffset: 0,
            clear: true,
            layerId: 'temperature',
            opacity: 0.55
        });
        cleared = true;
    }

    if (gas?.data?.length) {
        viewer.updateHeatmap(gas, {
            type: 'gas',
            heightOffset: 0.6,
            clear: !cleared,
            layerId: 'gas',
            opacity: 0.45
        });
        cleared = true;
    }

    if (!cleared) {
        const tempModule = IndoorMonitorState.modules.temperature?.data;
        if (tempModule?.heatmap) {
            viewer.updateHeatmap({ type: 'temperature', data: tempModule.heatmap }, {
                type: 'temperature',
                heightOffset: 0,
                clear: true,
                layerId: 'temperature',
                opacity: 0.55
            });
        }
    }
}

function syncIndoor3DWithModule(moduleId, data) {
    const viewer = ensureIndoor3DViewer();
    if (!viewer || !data) return;

    if (moduleId === 'fence' && Array.isArray(data.persons)) {
        const persons = normalizeEntityPositions(data.persons);
        viewer.updatePersons(persons);
        Indoor3DState.targetCounts.persons = persons.length;
        recordTrackPoints(persons, 'person');
    }

    if (moduleId === 'animal' && Array.isArray(data.detections)) {
        const animals = normalizeEntityPositions(mapAnimalDetections(data.detections));
        viewer.updateAnimals(animals);
        Indoor3DState.targetCounts.animals = animals.length;
        recordTrackPoints(animals, 'animal');
    }

    if (moduleId === 'fire') {
        const fires = normalizeEntityPositions(mapFireDetections(data));
        viewer.updateFires(fires);
        Indoor3DState.targetCounts.fires = fires.length;
        recordTrackPoints(fires, 'fire');
    }

    updateIndoor3DTargets();
    updateLocalTrajectories();
}

function updateIndoor3DTargets() {
    const total = Indoor3DState.targetCounts.persons +
                  Indoor3DState.targetCounts.animals +
                  Indoor3DState.targetCounts.fires;
    const targetsEl = document.getElementById('indoor-3d-targets');
    if (targetsEl) {
        targetsEl.textContent = total;
    }
}

function updateLocalTrajectories() {
    if (Indoor3DState.useServerTrajectories) return;
    const viewer = ensureIndoor3DViewer();
    if (!viewer) return;

    const trajectories = buildTrajectoriesFromTracks();
    viewer.updateTrajectories(trajectories);
}

function recordTrackPoints(objects, type) {
    if (!Array.isArray(objects) || objects.length === 0) return;

    const now = Date.now();
    const viewer = ensureIndoor3DViewer();
    const gridSize = viewer?.options?.gridSize || 20;

    objects.forEach(obj => {
        if (obj.x === undefined || obj.y === undefined) return;

        const trackId = `${type}-${obj.id}`;
        const track = Indoor3DState.trackHistory.get(trackId) || {
            id: trackId,
            type,
            points: [],
            lastTimestamp: 0,
            speed: 0
        };

        track.points.push({ x: clamp01(obj.x), y: clamp01(obj.y), timestamp: now });
        if (track.points.length > 30) {
            track.points.shift();
        }

        if (track.points.length > 1) {
            const prev = track.points[track.points.length - 2];
            const distance = Math.hypot(obj.x - prev.x, obj.y - prev.y) * gridSize;
            const delta = (now - prev.timestamp) / 1000;
            if (delta > 0) {
                track.speed = distance / delta;
            }
        }

        track.lastTimestamp = now;
        Indoor3DState.trackHistory.set(trackId, track);
    });

    Indoor3DState.trackHistory.forEach((track, key) => {
        if (now - track.lastTimestamp > 20000) {
            Indoor3DState.trackHistory.delete(key);
        }
    });
}

function buildTrajectoriesFromTracks() {
    const trajectories = [];
    Indoor3DState.trackHistory.forEach(track => {
        if (track.points.length < 2) return;
        trajectories.push({
            id: `traj-${track.id}`,
            type: track.type,
            points: track.points.map(p => ({ x: p.x, y: p.y })),
            speed: track.speed
        });
    });
    return trajectories;
}

function mapAnimalDetections(detections) {
    return detections.map((det, index) => ({
        id: det.id || `animal-${index}`,
        type: det.type || 'animal',
        x: det.bbox?.x ?? det.x ?? 0.5,
        y: det.bbox?.y ?? det.y ?? 0.5,
        status: det.confidence >= 0.85 ? 'danger' : 'warning',
        confidence: det.confidence
    })).filter(item => Number.isFinite(item.x) && Number.isFinite(item.y));
}

function mapFireDetections(data) {
    const fires = [];
    if (!data) return fires;

    if (Array.isArray(data.recentDetections) && data.recentDetections.length) {
        data.recentDetections.forEach((det, index) => {
            fires.push({
                id: det.id || `fire-${index}`,
                x: det.x ?? det.position?.x ?? 0.5,
                y: det.y ?? det.position?.y ?? 0.5,
                level: det.level || 'alarm'
            });
        });
        return fires;
    }

    if (Array.isArray(data.zones)) {
        const activeZones = data.zones.filter(zone => {
            const status = typeof zone.status === 'string' ? zone.status.toLowerCase() : zone.status;
            return (status && status !== 'normal') || zone.fireDetected || zone.smokeDetected;
        });

        activeZones.forEach((zone, index) => {
            const pos = deriveGridPosition(index, activeZones.length || 1);
            const status = typeof zone.status === 'string' ? zone.status.toLowerCase() : zone.status;
            fires.push({
                id: zone.id || `fire-${index}`,
                x: pos.x,
                y: pos.y,
                level: status || 'alarm'
            });
        });
    }

    return fires.filter(item => Number.isFinite(item.x) && Number.isFinite(item.y));
}

function deriveGridPosition(index, total) {
    const columns = Math.ceil(Math.sqrt(total));
    const rows = Math.ceil(total / columns);
    const col = index % columns;
    const row = Math.floor(index / columns);
    return {
        x: (col + 1) / (columns + 1),
        y: (row + 1) / (rows + 1)
    };
}

function normalizeEntityPositions(items, xKey = 'x', yKey = 'y') {
    if (!Array.isArray(items)) return [];

    const xs = items.map(item => item[xKey]).filter(Number.isFinite);
    const ys = items.map(item => item[yKey]).filter(Number.isFinite);
    if (!xs.length || !ys.length) return items;

    const needsNormalize = xs.some(x => x < 0 || x > 1) || ys.some(y => y < 0 || y > 1);
    if (!needsNormalize) {
        return items.map(item => ({
            ...item,
            x: clamp01(item[xKey]),
            y: clamp01(item[yKey])
        }));
    }

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    return items.map(item => ({
        ...item,
        x: normalizeValue(item[xKey], minX, maxX),
        y: normalizeValue(item[yKey], minY, maxY)
    }));
}

function normalizeValue(value, min, max) {
    if (!Number.isFinite(value)) return 0.5;
    if (max === min) return 0.5;
    return clamp01((value - min) / (max - min));
}

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function syncIndoorAlarms(alarms, stats) {
    // 处理告警数据，确保每个告警都有位置信息
    Indoor3DState.alarms = Array.isArray(alarms) ? alarms.map(alarm => {
        // 如果告警没有位置信息，根据类型和区域生成默认位置
        if (alarm.x === undefined || alarm.y === undefined) {
            const position = getAlarmDefaultPosition(alarm);
            return { ...alarm, x: position.x, y: position.y };
        }
        return alarm;
    }) : [];
    IndoorMonitorState.alarms = Indoor3DState.alarms;
    Indoor3DState.alarmStats = stats || {};

    updateAlarmIndicators();
    renderIndoorAlarmList();

    // 同步告警到3D视图
    syncAlarmsTo3DView();

    if (Indoor3DState.activeAlarmId) {
        const active = getAlarmById(Indoor3DState.activeAlarmId);
        updateAlarmProcessUI(active);
    }
}

/**
 * 根据告警类型和区域获取默认位置
 */
function getAlarmDefaultPosition(alarm) {
    // 区域位置映射
    const zonePositions = {
        'gis室': { x: 0.3, y: 0.4 },
        'gis': { x: 0.3, y: 0.4 },
        '主控室': { x: 0.5, y: 0.5 },
        '控制室': { x: 0.5, y: 0.5 },
        '继保室': { x: 0.7, y: 0.4 },
        '蓄电池室': { x: 0.7, y: 0.6 },
        '配电室': { x: 0.3, y: 0.6 },
        '黄线警戒区': { x: 0.4, y: 0.5 },
        '高压危险区': { x: 0.5, y: 0.3 }
    };

    // 告警类型位置映射
    const typePositions = {
        'fence_violation': { x: 0.4, y: 0.5 },
        'temperature_high': { x: 0.5, y: 0.5 },
        'animal_intrusion': { x: 0.6, y: 0.4 },
        'fire_detected': { x: 0.5, y: 0.6 },
        'gas_leak': { x: 0.4, y: 0.6 }
    };

    // 尝试从告警详情中获取区域
    const zone = alarm.details?.zone || alarm.details?.area || alarm.zone || '';
    const zoneLower = zone.toLowerCase();

    for (const [key, pos] of Object.entries(zonePositions)) {
        if (zoneLower.includes(key.toLowerCase())) {
            // 添加一些随机偏移避免重叠
            return {
                x: pos.x + (Math.random() - 0.5) * 0.1,
                y: pos.y + (Math.random() - 0.5) * 0.1
            };
        }
    }

    // 根据告警类型获取位置
    const alarmType = typeof alarm.type === 'string' ? alarm.type.toLowerCase() : '';
    if (typePositions[alarmType]) {
        const pos = typePositions[alarmType];
        return {
            x: pos.x + (Math.random() - 0.5) * 0.1,
            y: pos.y + (Math.random() - 0.5) * 0.1
        };
    }

    // 默认位置（中心附近）
    return {
        x: 0.5 + (Math.random() - 0.5) * 0.2,
        y: 0.5 + (Math.random() - 0.5) * 0.2
    };
}

/**
 * 同步告警到3D视图
 */
function syncAlarmsTo3DView() {
    const viewer = ensureIndoor3DViewer();
    if (!viewer) return;

    // 获取所有未解决的告警
    const activeAlarms = Indoor3DState.alarms.filter(a =>
        String(a.status).toLowerCase() !== 'resolved'
    );

    // 在3D视图中显示所有告警标记
    if (typeof viewer.updateAlarmMarkers === 'function') {
        viewer.updateAlarmMarkers(activeAlarms);
    }

    // 更新告警区域高亮
    const pendingAlarms = activeAlarms.filter(a =>
        String(a.status).toLowerCase() === 'pending'
    );

    if (pendingAlarms.length > 0) {
        const alarmZones = pendingAlarms.map(alarm => ({
            id: `alarm-zone-${alarm.id}`,
            name: alarm.message || '告警区域',
            type: 'danger',
            polygon: generateAlarmZonePolygon(alarm.x, alarm.y, 0.05)
        }));

        // 合并现有区域和告警区域
        const existingZones = viewer.state?.zones ?
            Array.from(viewer.state.zones.values())
                .map(z => z.userData)
                .filter(z => !z.id?.startsWith('alarm-zone-')) : [];

        viewer.updateZones([...existingZones, ...alarmZones]);
    } else {
        // 如果没有待处理告警，清除告警区域
        const existingZones = viewer.state?.zones ?
            Array.from(viewer.state.zones.values())
                .map(z => z.userData)
                .filter(z => !z.id?.startsWith('alarm-zone-')) : [];
        viewer.updateZones(existingZones);
    }

    console.log('[Indoor3D] 同步告警到3D视图:', activeAlarms.length, '个活动告警');
}

/**
 * 生成告警区域多边形
 */
function generateAlarmZonePolygon(x, y, radius) {
    const points = [];
    const segments = 6;
    for (let i = 0; i < segments; i++) {
        const angle = (i / segments) * Math.PI * 2;
        points.push([
            x + Math.cos(angle) * radius,
            y + Math.sin(angle) * radius
        ]);
    }
    return points;
}

function updateAlarmIndicators() {
    const alarms = Indoor3DState.alarms;
    const pending = alarms.filter(a => String(a.status).toLowerCase() === 'pending').length;
    const total = Indoor3DState.alarmStats?.total ?? alarms.length;

    IndoorMonitorState.totalAlarms = pending;

    const badge = document.getElementById('total-alarms');
    if (badge) badge.textContent = pending;

    const bottomCount = document.getElementById('alarm-count');
    if (bottomCount) bottomCount.textContent = pending;

    const panelCount = document.getElementById('indoor-alarm-count-badge');
    if (panelCount) panelCount.textContent = pending;

    const viewerCount = document.getElementById('indoor-3d-alarms');
    if (viewerCount) viewerCount.textContent = total;
}

function renderIndoorAlarmList() {
    const listEl = document.getElementById('indoor-alarm-list');
    if (!listEl) return;

    if (!Indoor3DState.alarms.length) {
        listEl.innerHTML = `
            <div class="alarm-item notice">
                <div class="alarm-header">
                    <span>暂无告警</span>
                    <span>--</span>
                </div>
                <div class="alarm-title">当前室内监测运行正常</div>
                <div class="alarm-meta">
                    <span><i class="bi bi-check-circle"></i> 全部安全</span>
                </div>
            </div>
        `;
        return;
    }

    listEl.innerHTML = Indoor3DState.alarms.map(alarm => {
        const level = typeof alarm.level === 'string' ? alarm.level.toLowerCase() : alarm.level;
        const levelClass = level === 'critical' || level === 'alarm' ? 'critical' :
            level === 'warning' ? '' : 'notice';
        const activeClass = alarm.id === Indoor3DState.activeAlarmId ? 'active' : '';
        const zone = alarm.details?.zone || alarm.details?.area;
        const person = alarm.details?.person_name || alarm.details?.person;
        const locationMeta = zone || person ? `
            ${zone ? `<span><i class="bi bi-geo-alt"></i> ${zone}</span>` : ''}
            ${person ? `<span><i class="bi bi-person"></i> ${person}</span>` : ''}
        ` : '';

        return `
            <div class="alarm-item ${levelClass} ${activeClass}" data-alarm-id="${alarm.id}">
                <div class="alarm-header">
                    <span>${formatAlarmTypeLabel(alarm.type)}</span>
                    <span>${formatAlarmTime(alarm.timestamp)}</span>
                </div>
                <div class="alarm-title">${alarm.message || '未命名告警'}</div>
                <div class="alarm-meta">
                    <span><i class="bi bi-exclamation-circle"></i> ${formatAlarmLevelLabel(alarm.level)}</span>
                    <span><i class="bi bi-activity"></i> ${formatAlarmStatusLabel(alarm.status)}</span>
                    ${locationMeta}
                </div>
            </div>
        `;
    }).join('');
}

function selectIndoorAlarm(alarmId, options = {}) {
    const { focus = true, openModal = false } = options;
    const alarm = getAlarmById(alarmId);
    if (!alarm) {
        console.warn('[Indoor3D] 告警不存在:', alarmId);
        return;
    }

    console.log('[Indoor3D] 选中告警:', alarm.id, alarm.message);

    Indoor3DState.activeAlarmId = alarmId;
    renderIndoorAlarmList();
    updateAlarmProcessUI(alarm);

    // 高亮对应的监测模块
    highlightRelatedModule(alarm);

    if (focus) {
        const viewer = ensureIndoor3DViewer();
        if (viewer) {
            // 确保告警有位置信息
            if (alarm.x === undefined || alarm.y === undefined) {
                const position = getAlarmDefaultPosition(alarm);
                alarm.x = position.x;
                alarm.y = position.y;
            }
            viewer.focusOnAlarm(alarm);
            console.log('[Indoor3D] 聚焦到告警位置:', alarm.x, alarm.y);
        }
    }

    if (openModal) {
        openAlarmDetailModal(alarm);
    }
}

/**
 * 高亮与告警相关的监测模块
 */
function highlightRelatedModule(alarm) {
    // 移除所有模块的高亮
    document.querySelectorAll('.monitor-card').forEach(card => {
        card.classList.remove('alarm-highlight');
    });

    // 根据告警类型确定相关模块
    const typeModuleMap = {
        'fence_violation': 'fence',
        'temperature_high': 'temperature',
        'animal_intrusion': 'animal',
        'fire_detected': 'fire',
        'gas_leak': 'environment',
        'device_fault': 'device'
    };

    const alarmType = typeof alarm.type === 'string' ? alarm.type.toLowerCase() : '';
    const moduleId = typeModuleMap[alarmType];

    if (moduleId) {
        const card = document.getElementById(`card-${moduleId}`);
        if (card) {
            card.classList.add('alarm-highlight');
            // 滚动到可见区域
            card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
}

function updateAlarmProcessUI(alarm) {
    const statusEl = document.getElementById('indoor-process-status');
    const steps = document.querySelectorAll('#indoor-process-steps .process-step');

    if (!alarm) {
        if (statusEl) {
            statusEl.className = 'status-badge attention';
            statusEl.textContent = '待选择';
        }
        steps.forEach(step => step.classList.remove('active'));
        return;
    }

    if (statusEl) {
        const status = typeof alarm.status === 'string' ? alarm.status.toLowerCase() : alarm.status;
        const statusClass = status === 'pending' ? 'alarm' :
            status === 'confirmed' || status === 'dispersing' ? 'warning' :
            status === 'reported' ? 'online' : 'good';
        statusEl.className = `status-badge ${statusClass}`;
        statusEl.textContent = formatAlarmStatusLabel(status);
    }

    const stepMap = {
        pending: 1,
        confirmed: 2,
        dispersing: 2,
        reported: 3,
        resolved: 4
    };
    const activeStep = stepMap[typeof alarm.status === 'string' ? alarm.status.toLowerCase() : alarm.status] || 1;
    steps.forEach(step => {
        const stepIndex = parseInt(step.dataset.step, 10);
        if (stepIndex <= activeStep) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });
}

async function handleAlarmAction(action) {
    const alarm = getAlarmById(Indoor3DState.activeAlarmId);
    if (!alarm) {
        showToast('请先选择一条告警', 'warning');
        return;
    }

    const actionMap = {
        confirm: 'confirm',
        disperse: 'disperse',
        report: 'report',
        resolve: 'resolve'
    };
    const endpoint = actionMap[action];
    if (!endpoint) return;

    try {
        const response = await fetch(`/api/indoor/3d/alarms/${alarm.id}/${endpoint}`, {
            method: 'POST'
        });
        if (!response.ok) {
            throw new Error('请求失败');
        }

        if (action === 'confirm') alarm.status = 'confirmed';
        if (action === 'disperse') alarm.status = 'dispersing';
        if (action === 'report') alarm.status = 'reported';
        if (action === 'resolve') alarm.status = 'resolved';

        updateAlarmIndicators();
        renderIndoorAlarmList();
        updateAlarmProcessUI(alarm);
        showToast('告警处理已提交', 'success');
    } catch (error) {
        console.error('[Alarm] 处理失败:', error);
        showToast('告警处理失败，请稍后再试', 'error');
    }
}

async function confirmAllAlarms() {
    const pendingAlarms = Indoor3DState.alarms.filter(alarm => String(alarm.status).toLowerCase() === 'pending');
    if (!pendingAlarms.length) {
        showToast('暂无待处理告警', 'info');
        return;
    }

    try {
        await Promise.all(pendingAlarms.map(async alarm => {
            const response = await fetch(`/api/indoor/3d/alarms/${alarm.id}/confirm`, { method: 'POST' });
            if (!response.ok) {
                throw new Error(`确认失败: ${alarm.id}`);
            }
        }));
        pendingAlarms.forEach(alarm => {
            alarm.status = 'confirmed';
        });
        updateAlarmIndicators();
        renderIndoorAlarmList();
        showToast('已批量确认告警', 'success');
    } catch (error) {
        console.error('[Alarm] 批量确认失败:', error);
        showToast('批量确认失败，请稍后再试', 'error');
    }
}

function openAlarmDetailModal(alarm) {
    let modal = document.getElementById('alarm-detail-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'alarm-detail-modal';
        modal.setAttribute('tabindex', '-1');
        modal.innerHTML = `
            <div class="modal-dialog modal-xl modal-dialog-centered">
                <div class="modal-content bg-dark text-white">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title"><i class="bi bi-exclamation-triangle text-warning"></i> 告警详情</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" id="alarm-detail-body"></div>
                    <div class="modal-footer border-secondary">
                        <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">关闭</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    const detailBody = modal.querySelector('#alarm-detail-body');
    if (detailBody) {
        const videoUrl = alarm.details?.video_url || alarm.details?.videoUrl || '';
        const historySeries = alarm.details?.history || alarm.details?.series || [];
        detailBody.innerHTML = `
            <div class="row g-3">
                <div class="col-lg-4">
                    <div class="p-3 rounded bg-secondary bg-opacity-10">
                        <div class="small text-muted">告警信息</div>
                        <div class="fw-semibold mt-1">${formatAlarmTypeLabel(alarm.type)}</div>
                        <div class="text-muted small mt-2">${alarm.message || '-'}</div>
                        <div class="small mt-3">
                            <div>级别: ${formatAlarmLevelLabel(alarm.level)}</div>
                            <div>状态: ${formatAlarmStatusLabel(alarm.status)}</div>
                            <div>时间: ${formatAlarmTimestamp(alarm.timestamp)}</div>
                            <div>位置: (${alarm.x?.toFixed?.(2) ?? '--'}, ${alarm.y?.toFixed?.(2) ?? '--'})</div>
                        </div>
                    </div>
                    <div class="p-3 rounded bg-secondary bg-opacity-10 mt-3">
                        <div class="small text-muted">处理建议</div>
                        <ul class="small mb-0 mt-2">
                            <li>确认现场人员与区域权限</li>
                            <li>必要时启动驱离或隔离流程</li>
                            <li>完成处置后提交上报记录</li>
                        </ul>
                    </div>
                </div>
                <div class="col-lg-8">
                    <div class="ratio ratio-16x9 bg-black rounded d-flex align-items-center justify-content-center">
                        ${videoUrl ? `<video src="${videoUrl}" controls style="width: 100%; height: 100%;"></video>` :
                            `<div class="text-muted small">暂无回放视频</div>`}
                    </div>
                    <div class="mt-3 p-2 rounded bg-secondary bg-opacity-10">
                        <canvas id="alarm-history-chart" height="140" style="width: 100%;"></canvas>
                        <div id="alarm-history-empty" class="text-muted small text-center mt-2"></div>
                    </div>
                </div>
            </div>
        `;

        requestAnimationFrame(() => renderAlarmHistoryChart(historySeries));
    }

    // 使用 getOrCreateInstance 避免重复创建 Modal 实例导致界面卡死
    let bsModal = bootstrap.Modal.getInstance(modal);
    if (!bsModal) {
        bsModal = new bootstrap.Modal(modal);
    }
    bsModal.show();
}

function renderAlarmHistoryChart(series) {
    const canvas = document.getElementById('alarm-history-chart');
    const emptyEl = document.getElementById('alarm-history-empty');
    if (!canvas) return;

    if (!Array.isArray(series) || series.length < 2) {
        if (emptyEl) {
            emptyEl.textContent = '暂无历史曲线';
        }
        return;
    }

    const values = series.map(item => {
        if (typeof item === 'number') return item;
        if (item && typeof item.value === 'number') return item.value;
        return 0;
    });

    if (emptyEl) emptyEl.textContent = '';
    const ctx = canvas.getContext('2d');
    const width = canvas.parentElement?.clientWidth || canvas.clientWidth || 400;
    const height = canvas.height;
    canvas.width = width;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    drawTrendLine(ctx, values, {
        x: 20,
        y: 15,
        width: width - 30,
        height: height - 30,
        color: '#3b82f6'
    });
}

function getAlarmById(alarmId) {
    return Indoor3DState.alarms.find(alarm => String(alarm.id) === String(alarmId));
}

function formatAlarmTime(timestamp) {
    const ts = formatAlarmTimestamp(timestamp, true);
    const diff = Date.now() - ts;
    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;
    const date = new Date(ts);
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function formatAlarmLevelLabel(level) {
    const key = typeof level === 'string' ? level.toLowerCase() : level;
    const map = {
        info: '提示',
        warning: '预警',
        alarm: '告警',
        critical: '严重'
    };
    return map[key] || level || '未知';
}

function formatAlarmTypeLabel(type) {
    const key = typeof type === 'string' ? type.toLowerCase() : type;
    const map = {
        fence_violation: '越线告警',
        temperature_high: '温度异常',
        animal_intrusion: '动物入侵',
        fire_detected: '火情告警',
        gas_leak: '气体泄漏',
        device_fault: '设备故障'
    };
    return map[key] || type || '告警';
}

function formatAlarmStatusLabel(status) {
    const key = typeof status === 'string' ? status.toLowerCase() : status;
    const map = {
        pending: '待处理',
        confirmed: '已确认',
        dispersing: '驱离中',
        reported: '已上报',
        resolved: '已关闭'
    };
    return map[key] || status || '未知';
}

function formatAlarmTimestamp(timestamp, raw = false) {
    const numeric = Number(timestamp);
    if (!Number.isFinite(numeric)) {
        return raw ? Date.now() : '--';
    }
    const ts = numeric < 1000000000000 ? numeric * 1000 : numeric;
    if (raw) return ts;
    return new Date(ts).toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

async function fetchJson(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.warn('[Fetch] 获取失败:', url, error);
        return null;
    }
}

// =============================================================================
// 渲染模块
// =============================================================================

/**
 * 渲染指定模块
 */
function renderModule(moduleId) {
    const module = IndoorMonitorState.modules[moduleId];
    if (!module || !module.data) return;
    
    switch (moduleId) {
        case 'fence':
            renderFenceModule(module);
            break;
        case 'animal':
            renderAnimalModule(module);
            break;
        case 'temperature':
            renderTemperatureModule(module);
            break;
        case 'device':
            renderDeviceModule(module);
            break;
        case 'fire':
            renderFireModule(module);
            break;
        case 'environment':
            renderEnvironmentModule(module);
            break;
    }
}

/**
 * 渲染电子围栏模块
 */
function renderFenceModule(module) {
    const { canvas, ctx, data } = module;
    if (!canvas || !ctx || !data) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // 清空画布
    ctx.clearRect(0, 0, width, height);
    
    // 绘制背景网格
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);
    
    // 绘制网格线
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.5)';
    ctx.lineWidth = 1;
    const gridSize = 20;
    for (let x = 0; x <= width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y <= height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
    
    // 绘制黄线警戒区
    const yellowLineY = height * data.yellowLine.y;
    ctx.fillStyle = 'rgba(234, 179, 8, 0.2)';
    ctx.fillRect(0, 0, width, yellowLineY);
    
    // 绘制黄线
    ctx.strokeStyle = '#eab308';
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    ctx.moveTo(0, yellowLineY);
    ctx.lineTo(width, yellowLineY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // 绘制黄线标签
    ctx.fillStyle = '#eab308';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText('⚠ 黄线警戒区', 10, yellowLineY - 5);
    
    // 绘制机柜区域
    const cabinetWidth = width / 6;
    data.cabinets.forEach((cabinet, index) => {
        const x = index * cabinetWidth;
        
        // 机柜背景
        let bgColor = 'rgba(30, 41, 59, 0.8)';
        if (cabinet.status === 'occupied') {
            bgColor = 'rgba(59, 130, 246, 0.3)';
        } else if (cabinet.status === 'alarm') {
            bgColor = 'rgba(239, 68, 68, 0.4)';
        }
        
        ctx.fillStyle = bgColor;
        ctx.fillRect(x + 2, height - 60, cabinetWidth - 4, 55);
        
        // 机柜边框
        ctx.strokeStyle = cabinet.status === 'alarm' ? '#ef4444' : 
                          cabinet.status === 'occupied' ? '#3b82f6' : '#334155';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 2, height - 60, cabinetWidth - 4, 55);
        
        // 机柜编号
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`#${cabinet.id}`, x + cabinetWidth / 2, height - 42);
        
        // 机柜状态
        ctx.font = '10px sans-serif';
        ctx.fillStyle = cabinet.status === 'alarm' ? '#ef4444' : 
                        cabinet.status === 'occupied' ? '#3b82f6' : '#94a3b8';
        const statusText = cabinet.status === 'alarm' ? '越线!' :
                          cabinet.status === 'occupied' ? '作业中' : '空闲';
        ctx.fillText(statusText, x + cabinetWidth / 2, height - 25);
    });
    
    // 绘制人员
    data.persons.forEach(person => {
        const px = person.x * width;
        const py = person.y * height;
        
        // 人员圆圈
        ctx.beginPath();
        ctx.arc(px, py, 15, 0, Math.PI * 2);
        ctx.fillStyle = person.status === 'danger' ? 'rgba(239, 68, 68, 0.8)' : 'rgba(34, 197, 94, 0.8)';
        ctx.fill();
        ctx.strokeStyle = person.status === 'danger' ? '#ef4444' : '#22c55e';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // 人员图标
        ctx.fillStyle = '#fff';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('👤', px, py);
        
        // 人员标签
        ctx.fillStyle = '#fff';
        ctx.font = '10px sans-serif';
        ctx.fillText(person.name, px, py - 22);
        
        // 危险标记
        if (person.status === 'danger') {
            ctx.fillStyle = '#ef4444';
            ctx.font = 'bold 10px sans-serif';
            ctx.fillText('⚠ 越线', px, py + 25);
        }
    });
    
    // 绘制状态信息
    ctx.fillStyle = '#fff';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`在场人员: ${data.totalPersons}`, 10, 20);
    if (data.alarmCount > 0) {
        ctx.fillStyle = '#ef4444';
        ctx.fillText(`告警: ${data.alarmCount}`, 100, 20);
    }
    
    // 更新机柜网格UI
    updateCabinetGrid(data.cabinets);
    
    // 更新人员列表UI
    updatePersonList(data.persons);
}

/**
 * 更新机柜网格UI
 */
function updateCabinetGrid(cabinets) {
    cabinets.forEach(cabinet => {
        const cell = document.getElementById(`cabinet-${cabinet.id}`);
        if (cell) {
            cell.className = 'cabinet-cell';
            if (cabinet.status === 'occupied') {
                cell.classList.add('occupied');
            } else if (cabinet.status === 'alarm') {
                cell.classList.add('alarm');
            }
            
            const statusEl = cell.querySelector('.cabinet-status');
            if (statusEl) {
                statusEl.textContent = cabinet.status === 'alarm' ? '越线!' :
                                       cabinet.status === 'occupied' ? '作业中' : '空闲';
            }
        }
    });
}

/**
 * 更新人员列表UI
 */
function updatePersonList(persons) {
    const listEl = document.getElementById('person-list');
    if (!listEl) return;
    
    listEl.innerHTML = persons.map(person => `
        <div class="person-item">
            <span class="person-info">
                <i class="bi bi-person"></i> ${person.name} - 机柜#${person.cabinet}
            </span>
            <span class="person-status ${person.status}">
                ${person.status === 'danger' ? '越线' : '安全'}
            </span>
        </div>
    `).join('');
}

/**
 * 渲染动物入侵模块
 */
function renderAnimalModule(module) {
    const { canvas, ctx, data } = module;
    if (!canvas || !ctx || !data) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // 清空画布
    ctx.clearRect(0, 0, width, height);
    
    // 绘制背景 (模拟摄像头画面)
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);
    
    // 绘制网格
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= width; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y <= height; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
    
    // 绘制检测结果
    data.detections.forEach(det => {
        const x = det.bbox.x * width;
        const y = det.bbox.y * height;
        const w = det.bbox.width * width;
        const h = det.bbox.height * height;
        
        // 检测框
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.strokeRect(x - w/2, y - h/2, w, h);
        
        // 标签背景
        ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
        const label = `${det.type === 'mouse' ? '🐭 老鼠' : '🐦 鸟类'} ${Math.round(det.confidence * 100)}%`;
        ctx.font = '12px sans-serif';
        const labelWidth = ctx.measureText(label).width + 10;
        ctx.fillRect(x - w/2, y - h/2 - 20, labelWidth, 18);
        
        // 标签文字
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x - w/2 + 5, y - h/2 - 6);
        
        // 跟踪线
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 30, y + 20);
        ctx.stroke();
        ctx.setLineDash([]);
    });
    
    // 如果没有检测到动物，显示安全状态
    if (data.detections.length === 0) {
        ctx.fillStyle = 'rgba(34, 197, 94, 0.8)';
        ctx.font = 'bold 14px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('✓ 未检测到入侵', width / 2, height / 2);
    }
    
    // 状态信息
    ctx.textAlign = 'left';
    ctx.fillStyle = '#fff';
    ctx.font = '11px sans-serif';
    ctx.fillText(`今日检测: ${data.todayCount}次`, 10, 20);
    
    if (data.deterrentActive) {
        ctx.fillStyle = '#f59e0b';
        ctx.fillText('🔊 驱离中', 10, 35);
    }
}

/**
 * 渲染温度监测模块
 */
function renderTemperatureModule(module) {
    const { canvas, ctx, data } = module;
    if (!canvas || !ctx || !data) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // 清空画布
    ctx.clearRect(0, 0, width, height);
    
    // 绘制热力图
    const heatmap = data.heatmap;
    const cellWidth = (width - 30) / heatmap[0].length;
    const cellHeight = height / heatmap.length;
    
    heatmap.forEach((row, yi) => {
        row.forEach((temp, xi) => {
            // 根据温度计算颜色
            const normalizedTemp = (temp - data.minTemp) / (data.maxTemp - data.minTemp);
            const color = getHeatmapColor(normalizedTemp);
            
            ctx.fillStyle = color;
            ctx.fillRect(xi * cellWidth, yi * cellHeight, cellWidth - 1, cellHeight - 1);
            
            // 高温标记
            if (temp > 40) {
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.strokeRect(xi * cellWidth, yi * cellHeight, cellWidth - 1, cellHeight - 1);
            }
        });
    });
    
    // 绘制温度刻度
    const scaleX = width - 25;
    const gradient = ctx.createLinearGradient(scaleX, 0, scaleX + 15, height);
    gradient.addColorStop(0, '#ef4444');
    gradient.addColorStop(0.33, '#f59e0b');
    gradient.addColorStop(0.66, '#22c55e');
    gradient.addColorStop(1, '#3b82f6');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(scaleX, 10, 15, height - 20);
    
    // 刻度标签
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${data.maxTemp}°C`, scaleX - 35, 18);
    ctx.fillText(`${Math.round((data.maxTemp + data.minTemp) / 2)}°C`, scaleX - 35, height / 2);
    ctx.fillText(`${data.minTemp}°C`, scaleX - 35, height - 8);
    
    // 热点标记
    data.hotspots.forEach(spot => {
        const hx = spot.x * cellWidth + cellWidth / 2;
        const hy = spot.y * cellHeight + cellHeight / 2;
        
        ctx.beginPath();
        ctx.arc(hx, hy, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();
        
        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 10px sans-serif';
        ctx.fillText(`${spot.temp}°C`, hx + 8, hy + 4);
    });
}

/**
 * 获取热力图颜色
 */
function getHeatmapColor(value) {
    // 从蓝色到红色的渐变
    if (value < 0.25) {
        return `rgba(59, 130, 246, ${0.5 + value * 2})`;
    } else if (value < 0.5) {
        return `rgba(34, 197, 94, ${0.5 + value})`;
    } else if (value < 0.75) {
        return `rgba(234, 179, 8, ${0.6 + value * 0.4})`;
    } else {
        return `rgba(239, 68, 68, ${0.7 + value * 0.3})`;
    }
}

/**
 * 渲染设备状态模块
 */
function renderDeviceModule(module) {
    const { canvas, ctx, data } = module;
    if (!canvas || !ctx || !data) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // 清空画布
    ctx.clearRect(0, 0, width, height);
    
    // 绘制背景
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);
    
    // 绘制设备状态
    const cols = 3;
    const rows = Math.ceil(data.devices.length / cols);
    const cellWidth = width / cols;
    const cellHeight = height / rows;
    
    data.devices.forEach((device, index) => {
        const col = index % cols;
        const row = Math.floor(index / cols);
        const x = col * cellWidth + 10;
        const y = row * cellHeight + 10;
        
        // 设备图标背景
        const iconSize = Math.min(cellWidth, cellHeight) * 0.4;
        ctx.fillStyle = device.status === 'online' ? 'rgba(34, 197, 94, 0.2)' :
                        device.status === 'standby' ? 'rgba(59, 130, 246, 0.2)' :
                        'rgba(239, 68, 68, 0.2)';
        ctx.beginPath();
        ctx.arc(x + cellWidth / 2 - 5, y + cellHeight / 2 - 15, iconSize / 2, 0, Math.PI * 2);
        ctx.fill();
        
        // 设备图标
        ctx.font = `${iconSize * 0.5}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillStyle = '#fff';
        const icons = { camera: '📷', lidar: '📡', sensor: '🌡️', relay: '🔌', alarm: '🔔' };
        ctx.fillText(icons[device.type] || '📦', x + cellWidth / 2 - 5, y + cellHeight / 2 - 10);
        
        // 设备名称
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#fff';
        ctx.fillText(device.name, x + cellWidth / 2 - 5, y + cellHeight / 2 + 15);
        
        // 健康度指示
        ctx.fillStyle = device.health > 90 ? '#22c55e' : 
                        device.health > 70 ? '#f59e0b' : '#ef4444';
        ctx.font = '9px sans-serif';
        ctx.fillText(`${device.health}%`, x + cellWidth / 2 - 5, y + cellHeight / 2 + 28);
        
        // 状态指示点
        ctx.beginPath();
        ctx.arc(x + cellWidth - 20, y + 15, 4, 0, Math.PI * 2);
        ctx.fillStyle = device.status === 'online' ? '#22c55e' :
                        device.status === 'standby' ? '#3b82f6' : '#ef4444';
        ctx.fill();
    });
    
    // 统计信息
    ctx.fillStyle = '#fff';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`在线: ${data.onlineDevices}/${data.totalDevices}`, 10, 15);
    ctx.fillText(`健康度: ${data.avgHealth}%`, width - 80, 15);
}

/**
 * 渲染消防监测模块
 */
function renderFireModule(module) {
    const { canvas, ctx, data } = module;
    if (!canvas || !ctx || !data) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // 清空画布
    ctx.clearRect(0, 0, width, height);
    
    // 绘制背景
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);
    
    // 绘制区域状态
    const zoneWidth = width / 2;
    const zoneHeight = height / 2;
    
    data.zones.forEach((zone, index) => {
        const col = index % 2;
        const row = Math.floor(index / 2);
        const x = col * zoneWidth + 5;
        const y = row * zoneHeight + 5;
        
        // 区域背景
        ctx.fillStyle = zone.status === 'normal' ? 'rgba(34, 197, 94, 0.15)' :
                        zone.status === 'warning' ? 'rgba(234, 179, 8, 0.2)' :
                        'rgba(239, 68, 68, 0.3)';
        ctx.fillRect(x, y, zoneWidth - 10, zoneHeight - 10);
        
        // 区域边框
        ctx.strokeStyle = zone.status === 'normal' ? '#22c55e' :
                          zone.status === 'warning' ? '#f59e0b' : '#ef4444';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, zoneWidth - 10, zoneHeight - 10);
        
        // 区域名称
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 11px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(zone.name, x + (zoneWidth - 10) / 2, y + 18);
        
        // 状态图标
        ctx.font = '16px sans-serif';
        ctx.fillText(zone.status === 'normal' ? '✓' : '⚠', x + (zoneWidth - 10) / 2, y + 42);
        
        // 烟雾浓度
        ctx.font = '9px sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`烟: ${(zone.smokeLevel * 100).toFixed(1)}%`, x + (zoneWidth - 10) / 2, y + 58);
        ctx.fillText(`温: ${zone.temp}°C`, x + (zoneWidth - 10) / 2, y + 70);
    });
    
    // 系统状态
    ctx.fillStyle = '#22c55e';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`喷淋: ${data.sprinklerStatus === 'ready' ? '就绪' : '关闭'}`, width - 10, height - 5);
}

/**
 * 渲染环境监测模块
 */
function renderEnvironmentModule(module) {
    const { canvas, ctx, data } = module;
    if (!canvas || !ctx || !data) return;
    
    const width = canvas.width;
    const height = canvas.height;
    
    // 清空画布
    ctx.clearRect(0, 0, width, height);
    
    // 绘制背景
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);
    
    // 绘制趋势图
    const padding = { top: 20, right: 10, bottom: 30, left: 40 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // 绘制坐标轴
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();
    
    // 绘制SF6曲线
    drawTrendLine(ctx, data.sf6.history, {
        x: padding.left,
        y: padding.top,
        width: chartWidth,
        height: chartHeight,
        color: '#3b82f6',
        label: 'SF6'
    });
    
    // 绘制O2曲线
    drawTrendLine(ctx, data.o2.history, {
        x: padding.left,
        y: padding.top,
        width: chartWidth,
        height: chartHeight,
        color: '#22c55e',
        label: 'O2',
        scale: { min: 19, max: 22 }
    });
    
    // 绘制图例
    ctx.font = '10px sans-serif';
    ctx.fillStyle = '#3b82f6';
    ctx.fillText('● SF6', padding.left + 10, padding.top + 10);
    ctx.fillStyle = '#22c55e';
    ctx.fillText('● O2', padding.left + 60, padding.top + 10);
    ctx.fillStyle = '#ef4444';
    ctx.fillText('● CO', padding.left + 100, padding.top + 10);
    
    // 当前值标签
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillStyle = '#3b82f6';
    ctx.fillText(`SF6: ${data.sf6.current.toFixed(1)}ppm`, width - 10, padding.top + 10);
    ctx.fillStyle = '#22c55e';
    ctx.fillText(`O2: ${data.o2.current.toFixed(1)}%`, width - 10, padding.top + 25);
    ctx.fillStyle = '#ef4444';
    ctx.fillText(`CO: ${data.co.current.toFixed(1)}ppm`, width - 10, padding.top + 40);
}

/**
 * 绘制趋势线
 */
function drawTrendLine(ctx, data, options) {
    const { x, y, width, height, color, scale } = options;
    const min = scale?.min ?? Math.min(...data);
    const max = scale?.max ?? Math.max(...data);
    const range = max - min || 1;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    data.forEach((value, index) => {
        const px = x + (index / (data.length - 1)) * width;
        const py = y + height - ((value - min) / range) * height;
        
        if (index === 0) {
            ctx.moveTo(px, py);
        } else {
            ctx.lineTo(px, py);
        }
    });
    
    ctx.stroke();
}

// =============================================================================
// 状态更新
// =============================================================================

/**
 * 更新模块状态显示
 */
function updateModuleStatus(moduleId) {
    const module = IndoorMonitorState.modules[moduleId];
    if (!module) return;
    
    // 更新状态徽章
    const card = document.getElementById(`card-${moduleId}`);
    if (card) {
        const badge = card.querySelector('.status-badge');
        if (badge) {
            badge.className = 'status-badge ' + module.status;
            badge.textContent = getStatusText(module.status);
        }
    }
}

/**
 * 获取状态文本
 */
function getStatusText(status) {
    const statusMap = {
        'online': '在线',
        'offline': '离线',
        'alarm': '告警',
        'warning': '警告',
        'attention': '注意',
        'good': '良好',
        'normal': '正常',
        'standby': '待机'
    };
    return statusMap[status] || status;
}

/**
 * 更新最后刷新时间
 */
function updateLastTime() {
    const lastTimeEl = document.getElementById('last-update-time');
    if (lastTimeEl) {
        const now = new Date();
        lastTimeEl.textContent = now.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

// =============================================================================
// 自动刷新
// =============================================================================

/**
 * 启动自动刷新
 */
function startAutoRefresh() {
    if (IndoorMonitorState.autoRefresh) {
        setInterval(() => {
            loadAllModulesData();
            updateLastTime();
        }, IndoorMonitorState.refreshRate);
    }
}

// =============================================================================
// WebSocket连接
// =============================================================================

/**
 * 连接WebSocket
 */
function connectWebSocket() {
    try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/indoor`;
        
        IndoorMonitorState.ws = new WebSocket(wsUrl);
        
        IndoorMonitorState.ws.onopen = () => {
            console.log('[WebSocket] 已连接');
            IndoorMonitorState.wsConnected = true;
            updateConnectionStatus(true);
        };
        
        IndoorMonitorState.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            } catch (e) {
                console.error('[WebSocket] 消息解析错误:', e);
            }
        };
        
        IndoorMonitorState.ws.onclose = () => {
            console.log('[WebSocket] 连接关闭');
            IndoorMonitorState.wsConnected = false;
            updateConnectionStatus(false);
            
            // 尝试重连
            IndoorMonitorState.wsReconnectTimer = setTimeout(connectWebSocket, 5000);
        };
        
        IndoorMonitorState.ws.onerror = (error) => {
            console.error('[WebSocket] 错误:', error);
        };
    } catch (e) {
        console.log('[WebSocket] 不可用，使用轮询模式');
    }
}

/**
 * 处理WebSocket消息
 */
function handleWebSocketMessage(message) {
    if (message.type === 'update' && message.module) {
        const module = IndoorMonitorState.modules[message.module];
        if (module) {
            module.data = message.data;
            module.status = message.data.status || module.status;
            renderModule(message.module);
            updateModuleStatus(message.module);
            syncIndoor3DWithModule(message.module, message.data);
        }
    } else if (message.type === 'alarm') {
        handleAlarm(message);
    }
}

/**
 * 更新连接状态显示
 */
function updateConnectionStatus(connected) {
    const statusEl = document.querySelector('.status-item .status-dot');
    if (statusEl) {
        statusEl.className = 'status-dot ' + (connected ? '' : 'warning');
    }
}

// =============================================================================
// 告警处理
// =============================================================================

/**
 * 处理告警
 */
function handleAlarm(alarm) {
    if (!alarm) return;

    if (!alarm.id) {
        alarm.id = `alarm-${Date.now()}`;
    }

    Indoor3DState.alarms = [alarm, ...Indoor3DState.alarms].slice(0, 50);
    IndoorMonitorState.alarms = Indoor3DState.alarms;

    updateAlarmIndicators();
    renderIndoorAlarmList();
    selectIndoorAlarm(alarm.id, { openModal: false });
    showAlarmNotification(alarm);
}

/**
 * 显示告警通知
 */
function showAlarmNotification(alarm) {
    // 可以使用浏览器通知API或自定义Toast
    console.log('[告警]', alarm);
}

// =============================================================================
// 模态框功能 (放大功能)
// =============================================================================

/**
 * 打开模块详情模态框
 */
function openModuleDetail(moduleId) {
    const module = IndoorMonitorState.modules[moduleId];
    if (!module) return;
    
    console.log(`[Modal] 打开模块详情: ${moduleId}`);
    
    // 检查是否已存在模态框，如果没有则创建
    let modal = document.getElementById('module-detail-modal');
    if (!modal) {
        modal = createDetailModal();
        document.body.appendChild(modal);
    }
    
    // 更新模态框内容
    const modalTitle = modal.querySelector('.modal-title');
    const modalBody = modal.querySelector('.modal-body');
    
    if (modalTitle) {
        modalTitle.innerHTML = `<i class="bi ${module.icon}"></i> ${module.name}`;
    }
    
    if (modalBody) {
        // 创建大画布
        modalBody.innerHTML = `
            <div class="modal-canvas-container" style="width: 100%; height: 500px; position: relative;">
                <canvas id="modal-canvas-${moduleId}" style="width: 100%; height: 100%;"></canvas>
            </div>
            <div class="modal-info mt-3">
                <div class="row">
                    <div class="col-md-4">
                        <div class="info-card bg-dark p-3 rounded">
                            <h6 class="text-muted mb-2">状态</h6>
                            <span class="status-badge ${module.status}">${getStatusText(module.status)}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="info-card bg-dark p-3 rounded">
                            <h6 class="text-muted mb-2">更新时间</h6>
                            <span class="text-white">${new Date().toLocaleTimeString()}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="info-card bg-dark p-3 rounded">
                            <h6 class="text-muted mb-2">模式</h6>
                            <span class="text-white">${IndoorMonitorState.simulationMode ? '模拟' : '实时'}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 初始化大画布
        setTimeout(() => {
            const largeCanvas = document.getElementById(`modal-canvas-${moduleId}`);
            if (largeCanvas) {
                const container = largeCanvas.parentElement;
                largeCanvas.width = container.clientWidth;
                largeCanvas.height = container.clientHeight;
                
                // 创建临时模块对象用于渲染
                const tempModule = {
                    ...module,
                    canvas: largeCanvas,
                    ctx: largeCanvas.getContext('2d')
                };
                
                // 根据模块类型渲染
                switch (moduleId) {
                    case 'fence':
                        renderFenceModule(tempModule);
                        break;
                    case 'animal':
                        renderAnimalModule(tempModule);
                        break;
                    case 'temperature':
                        renderTemperatureModule(tempModule);
                        break;
                    case 'device':
                        renderDeviceModule(tempModule);
                        break;
                    case 'fire':
                        renderFireModule(tempModule);
                        break;
                    case 'environment':
                        renderEnvironmentModule(tempModule);
                        break;
                }
            }
        }, 100);
    }

    // 使用 getOrCreateInstance 避免重复创建 Modal 实例导致界面卡死
    let bsModal = bootstrap.Modal.getInstance(modal);
    if (!bsModal) {
        bsModal = new bootstrap.Modal(modal);
    }
    bsModal.show();
}

/**
 * 创建详情模态框
 */
function createDetailModal() {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'module-detail-modal';
    modal.setAttribute('tabindex', '-1');
    modal.innerHTML = `
        <div class="modal-dialog modal-xl modal-dialog-centered">
            <div class="modal-content bg-dark text-white">
                <div class="modal-header border-secondary">
                    <h5 class="modal-title"></h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                </div>
                <div class="modal-footer border-secondary">
                    <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">关闭</button>
                    <button type="button" class="btn btn-primary" onclick="exportModuleData()">
                        <i class="bi bi-download"></i> 导出数据
                    </button>
                </div>
            </div>
        </div>
    `;
    return modal;
}

// =============================================================================
// 用户交互功能
// =============================================================================

/**
 * 刷新所有模块
 */
function refreshAll() {
    console.log('[Action] 刷新所有模块');
    loadAllModulesData();
    updateLastTime();
    loadIndoor3DScene();
    refreshIndoor3DOverlays();
    refreshIndoor3DAlarms();

    // 显示刷新动画
    const refreshBtn = document.querySelector('[onclick="refreshAll()"] i');
    if (refreshBtn) {
        refreshBtn.classList.add('spin');
        setTimeout(() => refreshBtn.classList.remove('spin'), 1000);
    }
}

/**
 * 切换所有模块的启用/禁用状态
 */
function toggleAllModules() {
    console.log('[Action] 切换所有模块状态');

    const modules = Object.keys(IndoorMonitorState.modules);
    const allOnline = modules.every(key => IndoorMonitorState.modules[key].status === 'online');

    modules.forEach(moduleId => {
        const module = IndoorMonitorState.modules[moduleId];
        if (allOnline) {
            module.status = 'offline';
            if (module.refreshInterval) {
                clearInterval(module.refreshInterval);
                module.refreshInterval = null;
            }
        } else {
            module.status = 'online';
            startModuleRefresh(moduleId);
        }

        // 更新UI状态
        const card = document.getElementById(`card-${moduleId}`);
        if (card) {
            const badge = card.querySelector('.status-badge');
            if (badge) {
                badge.className = `status-badge ${module.status}`;
                badge.textContent = module.status === 'online' ? '在线' : '离线';
            }
        }
    });

    showToast(allOnline ? '已暂停所有模块' : '已启动所有模块', allOnline ? 'warning' : 'success');
}

/**
 * 刷新单个模块
 */
function refreshModule(moduleId) {
    console.log(`[Action] 刷新模块: ${moduleId}`);
    loadModuleData(moduleId);
}

/**
 * 查看历史记录
 */
function viewHistory(moduleId) {
    console.log(`[Action] 查看历史: ${moduleId}`);
    
    // 创建历史记录模态框
    let historyModal = document.getElementById('history-modal');
    if (!historyModal) {
        historyModal = document.createElement('div');
        historyModal.className = 'modal fade';
        historyModal.id = 'history-modal';
        historyModal.setAttribute('tabindex', '-1');
        historyModal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content bg-dark text-white">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title"><i class="bi bi-clock-history"></i> 历史记录</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">时间范围</label>
                            <div class="btn-group w-100">
                                <button class="btn btn-outline-primary active">1小时</button>
                                <button class="btn btn-outline-primary">6小时</button>
                                <button class="btn btn-outline-primary">24小时</button>
                                <button class="btn btn-outline-primary">7天</button>
                            </div>
                        </div>
                        <div class="history-chart" style="height: 300px; background: #1e293b; border-radius: 8px;">
                            <div class="text-center text-muted pt-5">
                                <i class="bi bi-graph-up" style="font-size: 3rem;"></i>
                                <p class="mt-2">历史数据图表</p>
                            </div>
                        </div>
                        <div class="history-list mt-3">
                            <h6>事件记录</h6>
                            <div class="list-group list-group-flush">
                                <div class="list-group-item bg-transparent text-white border-secondary">
                                    <div class="d-flex justify-content-between">
                                        <span><i class="bi bi-info-circle text-info"></i> 系统启动</span>
                                        <small class="text-muted">14:20:00</small>
                                    </div>
                                </div>
                                <div class="list-group-item bg-transparent text-white border-secondary">
                                    <div class="d-flex justify-content-between">
                                        <span><i class="bi bi-exclamation-triangle text-warning"></i> 越线告警</span>
                                        <small class="text-muted">14:15:32</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(historyModal);
    }

    // 使用 getOrCreateInstance 避免重复创建 Modal 实例导致界面卡死
    let bsModal = bootstrap.Modal.getInstance(historyModal);
    if (!bsModal) {
        bsModal = new bootstrap.Modal(historyModal);
    }
    bsModal.show();
}

/**
 * 启动全部监测
 */
function startMonitoring() {
    console.log('[Action] 启动全部监测');
    IndoorMonitorState.autoRefresh = true;
    
    // 更新按钮状态
    const btn = document.querySelector('[onclick="startMonitoring()"]');
    if (btn) {
        btn.innerHTML = '<i class="bi bi-pause-fill"></i> 暂停';
        btn.setAttribute('onclick', 'stopMonitoring()');
    }
    
    showToast('已启动全部室内监测模块', 'success');
}

/**
 * 停止监测
 */
function stopMonitoring() {
    console.log('[Action] 停止监测');
    IndoorMonitorState.autoRefresh = false;
    
    const btn = document.querySelector('[onclick="stopMonitoring()"]');
    if (btn) {
        btn.innerHTML = '<i class="bi bi-play-fill"></i> 全部启动';
        btn.setAttribute('onclick', 'startMonitoring()');
    }
    
    showToast('已暂停监测', 'warning');
}

/**
 * 查看告警
 */
function viewAlarms() {
    console.log('[Action] 查看告警');

    const alarms = Indoor3DState.alarms.length ? Indoor3DState.alarms : IndoorMonitorState.alarms;
    const pending = alarms.filter(a => String(a.status).toLowerCase() === 'pending').length;
    const resolved = alarms.filter(a => String(a.status).toLowerCase() === 'resolved').length;

    let alarmModal = document.getElementById('alarm-modal');
    if (!alarmModal) {
        alarmModal = document.createElement('div');
        alarmModal.className = 'modal fade';
        alarmModal.id = 'alarm-modal';
        alarmModal.setAttribute('tabindex', '-1');
        document.body.appendChild(alarmModal);
    }

    const iconMap = {
        fence_violation: 'bi-exclamation-octagon',
        temperature_high: 'bi-thermometer-high',
        animal_intrusion: 'bi-bug',
        fire_detected: 'bi-fire',
        gas_leak: 'bi-cloud-haze2',
        device_fault: 'bi-hdd-stack'
    };

    const listHtml = alarms.length ? alarms.map(alarm => {
        const level = typeof alarm.level === 'string' ? alarm.level.toLowerCase() : alarm.level;
        const levelClass = level === 'critical' || level === 'alarm' ? 'danger' :
            level === 'warning' ? 'warning' : 'info';
        const typeKey = typeof alarm.type === 'string' ? alarm.type.toLowerCase() : alarm.type;
        const icon = iconMap[typeKey] || 'bi-exclamation-triangle';
        return `
            <div class="alert alert-${levelClass} d-flex align-items-center" data-alarm-id="${alarm.id}">
                <i class="bi ${icon} me-3" style="font-size: 1.5rem;"></i>
                <div class="flex-grow-1">
                    <strong>${formatAlarmTypeLabel(alarm.type)}</strong>
                    <p class="mb-0 small">${alarm.message || '-'}</p>
                    <small class="text-muted">${formatAlarmTimestamp(alarm.timestamp)}</small>
                </div>
                <div class="d-flex gap-2">
                    <button class="btn btn-sm btn-outline-light" data-alarm-action="focus">定位</button>
                    <button class="btn btn-sm btn-outline-light" data-alarm-action="detail">详情</button>
                </div>
            </div>
        `;
    }).join('') : `
        <div class="text-muted text-center py-3">暂无告警</div>
    `;

    alarmModal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content bg-dark text-white">
                <div class="modal-header border-secondary bg-danger">
                    <h5 class="modal-title"><i class="bi bi-exclamation-triangle"></i> 告警管理</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="d-flex justify-content-between mb-3">
                        <div>
                            <span class="badge bg-danger me-2">待处理: ${pending}</span>
                            <span class="badge bg-secondary">已处理: ${resolved}</span>
                        </div>
                        <button class="btn btn-sm btn-outline-success" onclick="confirmAllAlarms()">
                            <i class="bi bi-check-all"></i> 全部确认
                        </button>
                    </div>
                    <div class="alarm-list" id="alarm-modal-list">
                        ${listHtml}
                    </div>
                </div>
            </div>
        </div>
    `;

    const listEl = alarmModal.querySelector('#alarm-modal-list');
    if (listEl) {
        listEl.onclick = (event) => {
            const item = event.target.closest('[data-alarm-id]');
            if (!item) return;

            const action = event.target.closest('[data-alarm-action]')?.dataset?.alarmAction;
            if (action === 'focus') {
                selectIndoorAlarm(item.dataset.alarmId, { openModal: false });
                return;
            }
            if (action === 'detail') {
                const alarm = getAlarmById(item.dataset.alarmId);
                if (alarm) openAlarmDetailModal(alarm);
                return;
            }

            selectIndoorAlarm(item.dataset.alarmId, { openModal: true });
        };
    }

    // 使用 getOrCreateInstance 避免重复创建 Modal 实例导致界面卡死
    let bsModal = bootstrap.Modal.getInstance(alarmModal);
    if (!bsModal) {
        bsModal = new bootstrap.Modal(alarmModal);
    }
    bsModal.show();
}

/**
 * 配置电子围栏
 */
function toggleFence(moduleId) {
    console.log('[Action] 配置电子围栏');
    
    let configModal = document.getElementById('fence-config-modal');
    if (!configModal) {
        configModal = document.createElement('div');
        configModal.className = 'modal fade';
        configModal.id = 'fence-config-modal';
        configModal.setAttribute('tabindex', '-1');
        configModal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content bg-dark text-white">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title"><i class="bi bi-shield-check"></i> 围栏配置</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="mb-3">
                            <label class="form-label">黄线位置 (距机柜距离)</label>
                            <div class="input-group">
                                <input type="number" class="form-control bg-dark text-white" value="0.5" step="0.1">
                                <span class="input-group-text bg-secondary text-white">米</span>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">告警灵敏度</label>
                            <select class="form-select bg-dark text-white">
                                <option value="high">高 - 立即告警</option>
                                <option value="medium" selected>中 - 延迟1秒</option>
                                <option value="low">低 - 延迟3秒</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">授权机柜</label>
                            <div class="row g-2">
                                ${[1,2,3,4,5,6].map(i => `
                                    <div class="col-4">
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" id="auth-cabinet-${i}" ${i <= 3 ? 'checked' : ''}>
                                            <label class="form-check-label" for="auth-cabinet-${i}">机柜 #${i}</label>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        <div class="form-check form-switch mb-3">
                            <input class="form-check-input" type="checkbox" id="sound-alarm" checked>
                            <label class="form-check-label" for="sound-alarm">启用声音告警</label>
                        </div>
                        <div class="form-check form-switch mb-3">
                            <input class="form-check-input" type="checkbox" id="light-alarm" checked>
                            <label class="form-check-label" for="light-alarm">启用灯光告警</label>
                        </div>
                    </div>
                    <div class="modal-footer border-secondary">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                        <button type="button" class="btn btn-primary" onclick="saveFenceConfig()">
                            <i class="bi bi-check-lg"></i> 保存配置
                        </button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(configModal);
    }

    // 使用 getOrCreateInstance 避免重复创建 Modal 实例导致界面卡死
    let bsModal = bootstrap.Modal.getInstance(configModal);
    if (!bsModal) {
        bsModal = new bootstrap.Modal(configModal);
    }
    bsModal.show();
}

/**
 * 保存围栏配置
 */
function saveFenceConfig() {
    console.log('[Action] 保存围栏配置');
    showToast('围栏配置已保存', 'success');
    bootstrap.Modal.getInstance(document.getElementById('fence-config-modal')).hide();
}

/**
 * 启动驱离设备
 */
function activateDeterrent() {
    console.log('[Action] 启动驱离设备');
    showToast('超声波驱离设备已启动', 'info');
}

/**
 * 显示温度详情
 */
function showTempDetail() {
    openModuleDetail('temperature');
}

/**
 * 扫描设备
 */
function scanDevices() {
    console.log('[Action] 扫描设备');
    showToast('正在扫描设备...', 'info');
    
    setTimeout(() => {
        showToast('设备扫描完成，发现6个设备', 'success');
    }, 2000);
}

/**
 * 查看设备列表
 */
function viewDeviceList() {
    openModuleDetail('device');
}

/**
 * 测试消防报警
 */
function testFireAlarm() {
    console.log('[Action] 测试消防报警');
    
    if (confirm('确定要测试消防报警系统吗？')) {
        showToast('消防报警测试中...', 'warning');
        setTimeout(() => {
            showToast('消防报警系统正常', 'success');
        }, 3000);
    }
}

/**
 * 查看传感器数据
 */
function viewSensorData() {
    openModuleDetail('environment');
}

/**
 * 导出模块数据
 */
function exportModuleData() {
    console.log('[Action] 导出数据');
    showToast('数据导出中...', 'info');
    
    // 模拟导出
    setTimeout(() => {
        showToast('数据已导出', 'success');
    }, 1000);
}

// =============================================================================
// 工具函数
// =============================================================================

/**
 * 显示Toast通知
 */
function showToast(message, type = 'info') {
    // 检查是否已存在toast容器
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    
    const toastId = 'toast-' + Date.now();
    const bgClass = {
        'success': 'bg-success',
        'error': 'bg-danger',
        'warning': 'bg-warning text-dark',
        'info': 'bg-info text-dark'
    }[type] || 'bg-secondary';
    
    const toastHtml = `
        <div id="${toastId}" class="toast ${bgClass} text-white" role="alert">
            <div class="toast-body d-flex align-items-center">
                <i class="bi ${type === 'success' ? 'bi-check-circle' : type === 'error' ? 'bi-x-circle' : type === 'warning' ? 'bi-exclamation-triangle' : 'bi-info-circle'} me-2"></i>
                ${message}
                <button type="button" class="btn-close btn-close-white ms-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastEl = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
    toast.show();
    
    toastEl.addEventListener('hidden.bs.toast', () => {
        toastEl.remove();
    });
}

/**
 * 防抖函数
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * 绑定事件
 */
function bindEvents() {
    // 区域选择器变化
    const areaSelector = document.getElementById('area-selector');
    if (areaSelector) {
        areaSelector.addEventListener('change', (e) => {
            console.log('[Event] 区域切换:', e.target.value);
            loadAllModulesData();
        });
    }
}

// =============================================================================
// CSS动画样式注入
// =============================================================================
const styleSheet = document.createElement('style');
styleSheet.textContent = `
    .spin {
        animation: spin 1s linear;
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .status-badge.online { background: #22c55e; }
    .status-badge.offline { background: #6b7280; }
    .status-badge.alarm { background: #ef4444; animation: pulse 1s infinite; }
    .status-badge.warning { background: #f59e0b; }
    .status-badge.attention { background: #eab308; }
    .status-badge.good { background: #22c55e; }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .person-status.safe { background: #22c55e; }
    .person-status.warning { background: #f59e0b; color: #000; }
    .person-status.danger { background: #ef4444; }
    
    .modal-canvas-container canvas {
        background: #0f172a;
        border-radius: 8px;
    }
`;
document.head.appendChild(styleSheet);

// =============================================================================
// 设备选择和动态控制面板
// =============================================================================
let currentSelectedDevice = 'fence';

// 设备ID到插件ID的映射
const deviceToPluginMap = {
    'fence': 'indoor_fence',
    'animal': 'animal_detection',
    'temperature': 'temperature_monitoring',
    'device': 'device_monitoring',
    'fire': 'fire_detection',
    'environment': 'gas_detection'
};

async function selectDevice(moduleId) {
    currentSelectedDevice = moduleId;

    // 更新设备列表选中状态
    document.querySelectorAll('.device-item').forEach(item => {
        item.classList.remove('active');
    });
    const selectedItem = document.querySelector(`.device-item[data-module="${moduleId}"]`);
    if (selectedItem) {
        selectedItem.classList.add('active');
    }

    // 更新主视频画面
    updateMainVideo(moduleId);

    // 加载动态控制面板（使用正确的插件ID）
    const pluginId = deviceToPluginMap[moduleId] || moduleId;
    await loadControlPanel(pluginId);
}

function updateMainVideo(moduleId) {
    const moduleNames = {
        'fence': '电子围栏',
        'animal': '动物入侵检测',
        'temperature': '温度监测',
        'device': '设备状态监测',
        'fire': '消防监测',
        'environment': '环境监测'
    };

    const moduleNameEl = document.getElementById('current-module-name');
    if (moduleNameEl) {
        moduleNameEl.textContent = moduleNames[moduleId] || moduleId;
    }

    // 更新主画布
    const canvas = document.getElementById('main-video-canvas');
    if (canvas && IndoorMonitorState.modules[moduleId]) {
        const module = IndoorMonitorState.modules[moduleId];
        if (module.canvas && module.ctx) {
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;

            // 复制模块画布内容到主画布
            try {
                ctx.drawImage(module.canvas, 0, 0, canvas.width, canvas.height);
            } catch (e) {
                console.warn('无法复制画布内容:', e);
            }
        }
    }
}

async function loadControlPanel(moduleId) {
    const panelContent = document.getElementById('control-panel-content');
    const panelTitle = document.getElementById('control-panel-module-name');

    if (!panelContent) return;

    // 显示加载状态
    panelContent.innerHTML = '<div class="control-placeholder"><i class="bi bi-hourglass-split"></i><div>加载中...</div></div>';

    try {
        // 获取插件能力配置
        const response = await fetch(`/api/indoor/plugin/${moduleId}/capabilities`);
        if (!response.ok) {
            throw new Error('获取插件能力失败');
        }

        const capabilities = await response.json();

        // 更新标题
        if (panelTitle) {
            panelTitle.textContent = capabilities.name || '控制面板';
        }

        // 渲染控制面板
        renderControlPanel(capabilities, panelContent);

    } catch (error) {
        console.error('加载控制面板失败:', error);
        panelContent.innerHTML = `
            <div class="control-placeholder">
                <i class="bi bi-exclamation-triangle text-warning"></i>
                <div>控制面板加载失败</div>
                <div style="font-size: 0.7rem; margin-top: 0.5rem;">${error.message}</div>
            </div>
        `;
    }
}

function renderControlPanel(capabilities, container) {
    let html = '';

    // 渲染控制项
    if (capabilities.controls && capabilities.controls.length > 0) {
        capabilities.controls.forEach(control => {
            html += '<div class="control-group">';
            html += `<label class="control-label">${control.label}</label>`;

            if (control.type === 'slider') {
                const value = control.default || control.min || 0;
                html += `
                    <input type="range"
                           class="control-slider"
                           id="control-${control.id}"
                           min="${control.min}"
                           max="${control.max}"
                           step="${control.step || 1}"
                           value="${value}"
                           data-control-id="${control.id}">
                    <div class="control-value" id="value-${control.id}">${value}</div>
                `;
            } else if (control.type === 'select') {
                html += `<select class="control-select" id="control-${control.id}" data-control-id="${control.id}">`;
                control.options.forEach(option => {
                    const selected = option === control.default ? 'selected' : '';
                    html += `<option value="${option}" ${selected}>${option}</option>`;
                });
                html += '</select>';
            }

            html += '</div>';
        });
    }

    // 渲染操作按钮
    if (capabilities.operations && capabilities.operations.length > 0) {
        html += '<div class="control-group"><label class="control-label">操作</label>';
        html += '<div class="control-operations">';

        capabilities.operations.forEach(operation => {
            html += `
                <button class="control-operation-btn"
                        data-operation="${operation.id}"
                        onclick="executePluginOperation('${capabilities.plugin_id}', '${operation.id}')">
                    <i class="bi bi-${operation.icon}"></i>
                    ${operation.label}
                </button>
            `;
        });

        html += '</div></div>';
    }

    if (!html) {
        html = '<div class="control-placeholder"><i class="bi bi-info-circle"></i><div>该设备暂无可配置项</div></div>';
    }

    container.innerHTML = html;

    // 绑定滑块事件
    container.querySelectorAll('.control-slider').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const valueEl = document.getElementById(`value-${e.target.dataset.controlId}`);
            if (valueEl) {
                valueEl.textContent = e.target.value;
            }
        });

        slider.addEventListener('change', (e) => {
            updatePluginControl(capabilities.plugin_id, e.target.dataset.controlId, e.target.value);
        });
    });

    // 绑定下拉框事件
    container.querySelectorAll('.control-select').forEach(select => {
        select.addEventListener('change', (e) => {
            updatePluginControl(capabilities.plugin_id, e.target.dataset.controlId, e.target.value);
        });
    });
}

async function updatePluginControl(pluginId, controlId, value) {
    try {
        const response = await fetch(`/api/indoor/plugin/${pluginId}/command`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                operation: 'update_control',
                control_id: controlId,
                value: value
            })
        });

        if (response.ok) {
            console.log(`✓ 控制参数已更新: ${controlId} = ${value}`);
        }
    } catch (error) {
        console.error('更新控制参数失败:', error);
    }
}

async function executePluginOperation(pluginId, operationId) {
    try {
        const response = await fetch(`/api/indoor/plugin/${pluginId}/command`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                operation: operationId
            })
        });

        const result = await response.json();

        if (result.success) {
            console.log(`✓ 操作执行成功: ${operationId}`);
            // 可以添加成功提示
        } else {
            console.error(`✗ 操作执行失败: ${result.error}`);
            // 可以添加错误提示
        }
    } catch (error) {
        console.error('执行操作失败:', error);
    }
}

// 定期更新主视频画面
setInterval(() => {
    if (currentSelectedDevice) {
        updateMainVideo(currentSelectedDevice);
    }
}, 100);

// =============================================================================
// 融合证据链更新
// =============================================================================
async function updateFusionEvidence() {
    try {
        const response = await fetch('/api/indoor/fusion/evidence');
        if (!response.ok) {
            console.warn('融合证据链API返回错误');
            return;
        }

        const data = await response.json();

        // 更新置信度
        const confidenceEl = document.getElementById('fusion-confidence');
        if (confidenceEl && data.confidence !== undefined) {
            const confidence = (data.confidence * 100).toFixed(0);
            confidenceEl.textContent = `${confidence}%`;

            // 根据置信度设置颜色
            if (data.confidence >= 0.8) {
                confidenceEl.style.background = 'var(--accent-success)';
            } else if (data.confidence >= 0.6) {
                confidenceEl.style.background = 'var(--accent-warning)';
                confidenceEl.style.color = '#000';
            } else {
                confidenceEl.style.background = 'var(--accent-danger)';
            }
        }

        // 更新模态列表
        const modalitiesEl = document.getElementById('fusion-modalities');
        if (modalitiesEl && data.modalities) {
            modalitiesEl.innerHTML = data.modalities.map(m => {
                const statusClass = m.result === '正常' ? 'normal' :
                                   m.result === '注意' ? 'warning' : 'danger';
                const iconMap = {
                    'vision': 'bi-eye',
                    'acoustic': 'bi-soundwave',
                    'gas': 'bi-cloud-haze2',
                    'thermal': 'bi-thermometer-half'
                };
                const icon = iconMap[m.type] || 'bi-circle';
                const confidence = (m.confidence * 100).toFixed(0);

                return `
                    <div class="modality-item">
                        <div class="modality-info">
                            <div class="modality-icon ${m.type}">
                                <i class="bi ${icon}"></i>
                            </div>
                            <span>${m.name || m.type}</span>
                        </div>
                        <div class="modality-result">
                            <span class="modality-status ${statusClass}">${m.result}</span>
                            <span class="modality-confidence">${confidence}%</span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // 更新综合判定
        const resultValueEl = document.getElementById('fusion-result-value');
        if (resultValueEl && data.fusion_result) {
            resultValueEl.textContent = data.fusion_result;

            // 根据结果设置样式
            const resultContainer = document.getElementById('fusion-result');
            if (resultContainer) {
                if (data.fusion_result === '正常') {
                    resultContainer.style.background = 'rgba(34, 197, 94, 0.1)';
                    resultContainer.style.borderColor = 'rgba(34, 197, 94, 0.3)';
                } else if (data.fusion_result === '注意') {
                    resultContainer.style.background = 'rgba(234, 179, 8, 0.1)';
                    resultContainer.style.borderColor = 'rgba(234, 179, 8, 0.3)';
                } else {
                    resultContainer.style.background = 'rgba(239, 68, 68, 0.1)';
                    resultContainer.style.borderColor = 'rgba(239, 68, 68, 0.3)';
                }
            }
        }

        // 更新建议
        const recommendationEl = document.getElementById('fusion-recommendation');
        if (recommendationEl && data.recommendation) {
            recommendationEl.textContent = data.recommendation;
        }

    } catch (error) {
        console.error('更新融合证据链失败:', error);
    }
}

// 定期更新融合证据链
setInterval(updateFusionEvidence, 2000);

// =============================================================================
// 导出全局函数（确保HTML中的onclick可以调用）
// =============================================================================
window.selectDevice = selectDevice;
window.loadControlPanel = loadControlPanel;
window.executePluginOperation = executePluginOperation;
window.openModuleDetail = openModuleDetail;
window.refreshAll = refreshAll;
window.refreshModule = refreshModule;
window.viewHistory = viewHistory;
window.startMonitoring = startMonitoring;
window.stopMonitoring = stopMonitoring;
window.viewAlarms = viewAlarms;
window.confirmAllAlarms = confirmAllAlarms;
window.toggleFence = toggleFence;
window.saveFenceConfig = saveFenceConfig;
window.activateDeterrent = activateDeterrent;
window.showTempDetail = showTempDetail;
window.scanDevices = scanDevices;
window.viewDeviceList = viewDeviceList;
window.testFireAlarm = testFireAlarm;
window.viewSensorData = viewSensorData;
window.exportModuleData = exportModuleData;
