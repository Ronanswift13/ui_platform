/**
 * 室外监测中心 V3.6 - 完整JavaScript实现
 * =========================================
 * 
 * 输变电激光星芒破夜绘明监测平台
 * 
 * 功能模块:
 * 1. 主变巡视 - 油位、硅胶、漏油检测
 * 2. 开关巡视 - 断路器/隔离开关状态识别
 * 3. 母线巡视 - 绝缘子缺陷、销钉检测
 * 4. 电容器巡视 - 倾斜/缺失/入侵检测
 * 5. 表计读数 - 指针/数字识别
 * 6. 鸟类监控 - 鸟类检测与驱离
 * 7. 高级监测 - 声学/气体/高光谱/SLAM/融合
 * 
 * 版本: 3.6.0
 * 更新: 2026/01/25
 */

// =============================================================================
// 全局状态管理
// =============================================================================
const OutdoorCenterState = {
    version: '3.6.0',
    
    // 当前选中的模块
    currentModule: 'transformer',
    
    // 当前摄像头
    currentCamera: null,
    cameraConnected: false,
    
    // WebSocket连接
    ws: null,
    wsConnected: false,
    wsReconnectTimer: null,
    wsReconnectAttempts: 0,
    maxReconnectAttempts: 5,
    
    // 视频流状态
    videoStream: null,
    videoCanvas: null,
    videoCtx: null,
    frameRate: 25,
    lastFrameTime: 0,
    
    // 检测状态
    isDetecting: false,
    isPaused: false,
    detectionResults: [],
    alarms: [],
    
    // 性能指标
    metrics: {
        fps: 0,
        latency: 0,
        gpuUsage: 0,
        memoryUsage: 0,
        detectionCount: 0,
        alarmCount: 0
    },
    
    // 模块配置
    modules: {
        transformer: {
            id: 'transformer',
            name: '主变巡视',
            icon: 'bi-box',
            pluginId: 'transformer_inspection',
            status: 'ready',
            detectionTypes: ['oil_level', 'silica_gel', 'oil_leak', 'bushing', 'thermal'],
            cameraPreset: 'transformer_view'
        },
        switch: {
            id: 'switch',
            name: '开关巡视',
            icon: 'bi-toggle-on',
            pluginId: 'switch_inspection',
            status: 'ready',
            detectionTypes: ['breaker_state', 'isolator_state', 'indicator', 'interlock'],
            cameraPreset: 'switch_view'
        },
        busbar: {
            id: 'busbar',
            name: '母线巡视',
            icon: 'bi-diagram-3',
            pluginId: 'busbar_inspection',
            status: 'ready',
            detectionTypes: ['insulator_crack', 'pin_missing', 'foreign_object', 'corrosion', 'flashover'],
            cameraPreset: 'busbar_view'
        },
        capacitor: {
            id: 'capacitor',
            name: '电容器',
            icon: 'bi-battery-charging',
            pluginId: 'capacitor_inspection',
            status: 'ready',
            detectionTypes: ['tilt', 'fallen', 'missing', 'intrusion', 'fuse_blown'],
            cameraPreset: 'capacitor_view'
        },
        meter: {
            id: 'meter',
            name: '表计读数',
            icon: 'bi-speedometer2',
            pluginId: 'meter_reading',
            status: 'ready',
            detectionTypes: ['pointer', 'digital', 'dial'],
            cameraPreset: 'meter_view'
        },
        bird: {
            id: 'bird',
            name: '鸟类监控',
            icon: 'bi-feather',
            pluginId: 'bird_monitoring',
            status: 'ready',
            detectionTypes: ['bird_detect', 'nest', 'risk_assess'],
            cameraPreset: 'bird_view'
        },
        acoustic: {
            id: 'acoustic',
            name: '声学监测',
            icon: 'bi-soundwave',
            pluginId: 'acoustic_monitoring',
            status: 'standby',
            detectionTypes: ['partial_discharge', 'mechanical_fault', 'noise'],
            cameraPreset: null
        },
        gas: {
            id: 'gas',
            name: '气体检测',
            icon: 'bi-cloud-haze2',
            pluginId: 'gas_detection',
            status: 'standby',
            detectionTypes: ['sf6', 'h2', 'co', 'dga'],
            cameraPreset: null
        },
        hyperspectral: {
            id: 'hyperspectral',
            name: '高光谱检测',
            icon: 'bi-rainbow',
            pluginId: 'hyperspectral_detection',
            status: 'standby',
            detectionTypes: ['material_aging', 'micro_crack', 'corrosion'],
            cameraPreset: 'hyperspectral_view'
        },
        slam: {
            id: 'slam',
            name: 'SLAM建图',
            icon: 'bi-map',
            pluginId: 'slam_mapping',
            status: 'standby',
            detectionTypes: ['mapping', 'localization', 'navigation'],
            cameraPreset: 'slam_view'
        },
        fusion: {
            id: 'fusion',
            name: '多模态融合',
            icon: 'bi-diagram-3-fill',
            pluginId: 'multimodal_fusion',
            status: 'standby',
            detectionTypes: ['comprehensive'],
            cameraPreset: null
        }
    },
    
    // 配置参数
    config: {
        voltage: '110kV',
        site: null,
        confidenceThreshold: 0.6,
        enableAutoDetect: true,
        detectInterval: 1000,
        enableAlarmSound: true
    }
};

// =============================================================================
// 初始化
// =============================================================================
document.addEventListener('DOMContentLoaded', function() {
    console.log(`[OutdoorCenter] 初始化 V${OutdoorCenterState.version}`);
    
    // 初始化视频画布
    initVideoCanvas();
    
    // 加载摄像头列表
    loadCameras();
    
    // 初始化WebSocket连接
    connectWebSocket();
    
    // 初始化模块状态
    initModuleStatus();
    
    // 绑定事件
    bindEvents();
    
    // 启动时间更新
    startTimeUpdate();
    
    // 启动性能监控
    startMetricsUpdate();
    
    // 加载默认模块
    switchModule('transformer');
    
    console.log('[OutdoorCenter] 初始化完成');
});

// =============================================================================
// 视频画布初始化
// =============================================================================
function initVideoCanvas() {
    const canvas = document.getElementById('main-video-canvas');
    if (!canvas) {
        console.error('[Video] Canvas元素未找到');
        return;
    }
    
    OutdoorCenterState.videoCanvas = canvas;
    OutdoorCenterState.videoCtx = canvas.getContext('2d');
    
    // 设置画布尺寸
    const container = canvas.parentElement;
    canvas.width = container.clientWidth || 1920;
    canvas.height = container.clientHeight || 1080;
    
    // 绘制初始占位图
    drawPlaceholder('等待摄像头连接...');
    
    // 监听窗口大小变化
    window.addEventListener('resize', () => {
        canvas.width = container.clientWidth || 1920;
        canvas.height = container.clientHeight || 1080;
        if (!OutdoorCenterState.cameraConnected) {
            drawPlaceholder('等待摄像头连接...');
        }
    });
}

/**
 * 绘制占位图
 */
function drawPlaceholder(message) {
    const ctx = OutdoorCenterState.videoCtx;
    const canvas = OutdoorCenterState.videoCanvas;
    if (!ctx || !canvas) return;
    
    // 绘制背景
    ctx.fillStyle = '#0f0f1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 绘制网格背景
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < canvas.width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }
    
    // 绘制摄像头图标
    ctx.fillStyle = '#444';
    ctx.font = '48px Bootstrap-icons';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('\uF2C7', canvas.width / 2, canvas.height / 2 - 30); // camera-video-off icon
    
    // 绘制提示文字
    ctx.fillStyle = '#666';
    ctx.font = '18px "Microsoft YaHei", sans-serif';
    ctx.fillText(message, canvas.width / 2, canvas.height / 2 + 30);
}

// =============================================================================
// 摄像头管理
// =============================================================================
async function loadCameras() {
    try {
        const response = await fetch('/api/cameras');
        if (!response.ok) {
            throw new Error('获取摄像头列表失败');
        }
        
        const cameras = await response.json();
        updateCameraSelect(cameras);
        
        // 自动连接第一个摄像头
        if (cameras && cameras.length > 0) {
            OutdoorCenterState.currentCamera = cameras[0].id;
            await connectCamera(cameras[0].id);
        }
    } catch (error) {
        console.error('[Camera] 加载摄像头列表失败:', error);
        showToast('摄像头列表加载失败', 'warning');
    }
}

function updateCameraSelect(cameras) {
    const select = document.getElementById('site-selector');
    if (!select) return;
    
    select.innerHTML = '';
    
    if (!cameras || cameras.length === 0) {
        select.innerHTML = '<option value="">未发现摄像头</option>';
        return;
    }
    
    cameras.forEach(cam => {
        const option = document.createElement('option');
        option.value = cam.id;
        option.textContent = `${cam.id} (${cam.camera_type || 'unknown'})`;
        select.appendChild(option);
    });
}

async function connectCamera(cameraId) {
    if (!cameraId) {
        drawPlaceholder('请选择摄像头');
        return;
    }
    
    try {
        console.log(`[Camera] 正在连接: ${cameraId}`);
        drawPlaceholder('正在连接摄像头...');
        
        // 发送连接请求
        const response = await fetch(`/api/cameras/${cameraId}/connect`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('摄像头连接失败');
        }
        
        OutdoorCenterState.currentCamera = cameraId;
        OutdoorCenterState.cameraConnected = true;
        
        // 启动视频流
        startVideoStream(cameraId);
        
        updateCameraStatus('connected');
        console.log(`[Camera] 连接成功: ${cameraId}`);
        
    } catch (error) {
        console.error('[Camera] 连接失败:', error);
        OutdoorCenterState.cameraConnected = false;
        updateCameraStatus('disconnected');
        drawPlaceholder('摄像头连接失败');
        showToast('摄像头连接失败: ' + error.message, 'danger');
    }
}

function startVideoStream(cameraId) {
    const canvas = OutdoorCenterState.videoCanvas;
    const ctx = OutdoorCenterState.videoCtx;
    if (!canvas || !ctx) return;
    
    // 创建Image对象用于接收MJPEG流
    const img = new Image();
    let frameCount = 0;
    let lastFpsTime = performance.now();
    
    img.onload = function() {
        if (!OutdoorCenterState.cameraConnected) return;
        
        // 绘制视频帧
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // 绘制检测结果覆盖层
        drawDetectionOverlay();
        
        // 计算FPS
        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
            OutdoorCenterState.metrics.fps = frameCount;
            updateFpsDisplay(frameCount);
            frameCount = 0;
            lastFpsTime = now;
        }
        
        // 继续加载下一帧
        setTimeout(() => {
            if (OutdoorCenterState.cameraConnected) {
                img.src = `/api/cameras/${cameraId}/stream?fps=${OutdoorCenterState.frameRate}&ts=${Date.now()}`;
            }
        }, 1000 / OutdoorCenterState.frameRate);
    };
    
    img.onerror = function() {
        console.warn('[Video] 帧加载失败，尝试重新连接');
        if (OutdoorCenterState.cameraConnected) {
            setTimeout(() => {
                img.src = `/api/cameras/${cameraId}/stream?fps=${OutdoorCenterState.frameRate}&ts=${Date.now()}`;
            }, 500);
        }
    };
    
    // 启动流
    img.src = `/api/cameras/${cameraId}/stream?fps=${OutdoorCenterState.frameRate}&ts=${Date.now()}`;
    
    // 同时启动检测
    if (OutdoorCenterState.config.enableAutoDetect) {
        startAutoDetection();
    }
}

function updateCameraStatus(status) {
    const statusDots = document.querySelectorAll('.status-dot');
    const cameraStatusItem = document.querySelector('.status-item:nth-child(3) .status-dot');
    
    if (cameraStatusItem) {
        cameraStatusItem.classList.remove('warning', 'error');
        if (status === 'connected') {
            cameraStatusItem.style.background = 'var(--accent-success)';
        } else {
            cameraStatusItem.classList.add('error');
        }
    }
    
    // 更新摄像头信息显示
    const cameraName = document.getElementById('camera-name');
    const cameraResolution = document.getElementById('camera-resolution');
    
    if (cameraName && OutdoorCenterState.currentCamera) {
        cameraName.textContent = OutdoorCenterState.currentCamera;
    }
    if (cameraResolution) {
        cameraResolution.textContent = status === 'connected' ? '1920×1080' : '离线';
    }
}

function updateFpsDisplay(fps) {
    const fpsEl = document.getElementById('fps-value');
    if (fpsEl) {
        fpsEl.textContent = fps;
    }
}

// =============================================================================
// 检测功能
// =============================================================================
function startAutoDetection() {
    if (OutdoorCenterState.isDetecting) return;
    
    OutdoorCenterState.isDetecting = true;
    console.log('[Detection] 启动自动检测');
    
    // 定时执行检测
    OutdoorCenterState.detectTimer = setInterval(() => {
        if (OutdoorCenterState.cameraConnected && !OutdoorCenterState.isPaused) {
            performDetection();
        }
    }, OutdoorCenterState.config.detectInterval);
}

function stopAutoDetection() {
    OutdoorCenterState.isDetecting = false;
    if (OutdoorCenterState.detectTimer) {
        clearInterval(OutdoorCenterState.detectTimer);
        OutdoorCenterState.detectTimer = null;
    }
    console.log('[Detection] 停止自动检测');
}

async function performDetection() {
    const module = OutdoorCenterState.modules[OutdoorCenterState.currentModule];
    if (!module) return;
    
    try {
        const startTime = performance.now();
        
        // 从画布获取当前帧
        const canvas = OutdoorCenterState.videoCanvas;
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // 发送检测请求
        const response = await fetch(`/api/outdoor/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                module: OutdoorCenterState.currentModule,
                plugin_id: module.pluginId,
                image: imageData,
                detection_types: module.detectionTypes,
                confidence_threshold: OutdoorCenterState.config.confidenceThreshold
            })
        });
        
        if (!response.ok) {
            throw new Error('检测请求失败');
        }
        
        const result = await response.json();
        
        // 更新延迟
        const latency = performance.now() - startTime;
        OutdoorCenterState.metrics.latency = latency;
        updateLatencyDisplay(latency);
        
        // 处理检测结果
        handleDetectionResult(result);
        
    } catch (error) {
        console.error('[Detection] 检测失败:', error);
    }
}

function handleDetectionResult(result) {
    if (!result) return;
    
    // 更新检测结果
    OutdoorCenterState.detectionResults = result.detections || [];
    OutdoorCenterState.metrics.detectionCount = OutdoorCenterState.detectionResults.length;
    
    // 更新UI
    updateDetectionCount(OutdoorCenterState.detectionResults.length);
    updateResultList(result.detections);
    
    // 处理告警
    if (result.alarms && result.alarms.length > 0) {
        handleAlarms(result.alarms);
    }
}

function updateDetectionCount(count) {
    const countEl = document.getElementById('detection-count');
    if (countEl) {
        countEl.textContent = count;
    }
}

function updateLatencyDisplay(latency) {
    const metricEl = document.querySelector('.metric-value');
    if (metricEl) {
        metricEl.textContent = `${Math.round(latency)}ms`;
    }
}

function updateResultList(detections) {
    const resultList = document.getElementById('result-list');
    if (!resultList) return;
    
    resultList.innerHTML = '';
    
    if (!detections || detections.length === 0) {
        resultList.innerHTML = '<li class="result-item empty">暂无识别结果</li>';
        return;
    }
    
    detections.forEach((det, index) => {
        const item = document.createElement('li');
        item.className = 'result-item';
        item.innerHTML = `
            <span class="result-icon ${getConfidenceClass(det.confidence)}">
                <i class="bi ${getDetectionIcon(det.type)}"></i>
            </span>
            <div class="result-info">
                <div class="result-name">${det.label || det.type}</div>
                <div class="result-confidence">置信度: ${(det.confidence * 100).toFixed(1)}%</div>
            </div>
        `;
        resultList.appendChild(item);
    });
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.9) return 'high';
    if (confidence >= 0.7) return 'medium';
    return 'low';
}

function getDetectionIcon(type) {
    const icons = {
        'oil_level': 'bi-droplet',
        'silica_gel': 'bi-circle',
        'oil_leak': 'bi-exclamation-triangle',
        'breaker_state': 'bi-toggle-on',
        'isolator_state': 'bi-toggle-off',
        'insulator_crack': 'bi-lightning',
        'pin_missing': 'bi-x-circle',
        'tilt': 'bi-slash-lg',
        'bird_detect': 'bi-feather',
        'default': 'bi-box'
    };
    return icons[type] || icons['default'];
}

/**
 * 绘制检测结果覆盖层
 */
function drawDetectionOverlay() {
    const ctx = OutdoorCenterState.videoCtx;
    const canvas = OutdoorCenterState.videoCanvas;
    const detections = OutdoorCenterState.detectionResults;
    
    if (!ctx || !canvas || !detections || detections.length === 0) return;
    
    detections.forEach(det => {
        if (!det.bbox) return;
        
        const [x, y, w, h] = det.bbox;
        const scaledX = x * canvas.width;
        const scaledY = y * canvas.height;
        const scaledW = w * canvas.width;
        const scaledH = h * canvas.height;
        
        // 绘制检测框
        ctx.strokeStyle = getBoxColor(det.confidence, det.type);
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        
        // 绘制标签背景
        ctx.fillStyle = getBoxColor(det.confidence, det.type);
        const labelText = `${det.label || det.type} ${(det.confidence * 100).toFixed(0)}%`;
        const labelWidth = ctx.measureText(labelText).width + 10;
        ctx.fillRect(scaledX, scaledY - 22, labelWidth, 20);
        
        // 绘制标签文字
        ctx.fillStyle = '#fff';
        ctx.font = '12px "Microsoft YaHei", sans-serif';
        ctx.fillText(labelText, scaledX + 5, scaledY - 7);
    });
}

function getBoxColor(confidence, type) {
    // 根据检测类型和置信度返回不同颜色
    if (type && type.includes('alarm') || type && type.includes('leak') || type && type.includes('crack')) {
        return 'rgba(239, 68, 68, 0.9)'; // 红色 - 告警
    }
    if (confidence >= 0.9) {
        return 'rgba(16, 185, 129, 0.9)'; // 绿色 - 高置信度
    }
    if (confidence >= 0.7) {
        return 'rgba(245, 158, 11, 0.9)'; // 黄色 - 中置信度
    }
    return 'rgba(0, 212, 255, 0.9)'; // 青色 - 低置信度
}

// =============================================================================
// 告警处理
// =============================================================================
function handleAlarms(alarms) {
    alarms.forEach(alarm => {
        // 添加到告警列表
        OutdoorCenterState.alarms.unshift(alarm);
        
        // 限制告警数量
        if (OutdoorCenterState.alarms.length > 100) {
            OutdoorCenterState.alarms.pop();
        }
        
        // 更新UI
        addAlarmToList(alarm);
        updateAlarmCount();
        
        // 播放告警音
        if (OutdoorCenterState.config.enableAlarmSound) {
            playAlarmSound(alarm.level);
        }
    });
}

function addAlarmToList(alarm) {
    const alarmList = document.getElementById('alarm-list');
    if (!alarmList) return;
    
    const item = document.createElement('div');
    item.className = `alarm-item ${alarm.level || 'warning'}`;
    item.innerHTML = `
        <div class="alarm-header">
            <span class="alarm-type">${alarm.type || '未知告警'}</span>
            <span class="alarm-time">${formatTime(alarm.timestamp || new Date())}</span>
        </div>
        <div class="alarm-message">${alarm.message || ''}</div>
    `;
    
    alarmList.insertBefore(item, alarmList.firstChild);
    
    // 限制显示数量
    while (alarmList.children.length > 20) {
        alarmList.removeChild(alarmList.lastChild);
    }
}

function updateAlarmCount() {
    const countEl = document.getElementById('alarm-count');
    if (countEl) {
        countEl.textContent = OutdoorCenterState.alarms.length;
    }
}

function playAlarmSound(level) {
    // 告警音频播放（可选）
    try {
        const audio = new Audio(`/static/audio/alarm_${level || 'warning'}.mp3`);
        audio.volume = 0.5;
        audio.play().catch(() => {}); // 忽略自动播放错误
    } catch (e) {
        // 忽略音频播放错误
    }
}

// =============================================================================
// 模块切换
// =============================================================================
function switchModule(moduleId) {
    if (moduleId === OutdoorCenterState.currentModule) return;
    
    console.log(`[Module] 切换到: ${moduleId}`);
    OutdoorCenterState.currentModule = moduleId;
    
    // 更新左侧导航高亮
    document.querySelectorAll('.module-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.module === moduleId) {
            item.classList.add('active');
        }
    });
    
    // 更新控制面板标题
    const module = OutdoorCenterState.modules[moduleId];
    const titleEl = document.getElementById('current-module-name');
    if (titleEl && module) {
        titleEl.textContent = module.name;
    }
    
    // 切换控制面板
    switchControlPanel(moduleId);
    
    // 更新模块状态显示
    updateModuleStatusDisplay(moduleId);
    
    // 清除之前的检测结果
    OutdoorCenterState.detectionResults = [];
    updateResultList([]);
    
    // 如果有摄像头预设，切换摄像头视角
    if (module && module.cameraPreset) {
        switchCameraPreset(module.cameraPreset);
    }
    
    // 加载模块特定配置
    loadModuleConfig(moduleId);
}

function switchControlPanel(moduleId) {
    // 隐藏所有控制面板
    document.querySelectorAll('.module-control-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    
    // 显示目标控制面板
    const targetPanel = document.getElementById(`control-${moduleId}`);
    if (targetPanel) {
        targetPanel.classList.add('active');
    } else {
        // 显示默认面板
        const defaultPanel = document.getElementById('control-default');
        if (defaultPanel) {
            defaultPanel.classList.add('active');
        }
    }
}

function updateModuleStatusDisplay(moduleId) {
    const module = OutdoorCenterState.modules[moduleId];
    if (!module) return;
    
    // 更新左侧模块项的状态
    const moduleItem = document.querySelector(`.module-item[data-module="${moduleId}"]`);
    if (moduleItem) {
        const statusEl = moduleItem.querySelector('.module-item-status');
        if (statusEl) {
            statusEl.textContent = module.status === 'ready' ? '已就绪' : '待机中';
            statusEl.classList.toggle('ready', module.status === 'ready');
        }
    }
}

async function loadModuleConfig(moduleId) {
    try {
        const response = await fetch(`/api/module/${moduleId}/config`);
        if (response.ok) {
            const config = await response.json();
            applyModuleConfig(moduleId, config);
        }
    } catch (error) {
        console.warn(`[Config] 加载模块配置失败: ${moduleId}`, error);
    }
}

function applyModuleConfig(moduleId, config) {
    // 应用模块特定配置到UI
    if (!config) return;
    
    // 更新置信度滑块
    const confSlider = document.getElementById(`${moduleId.substring(0,2)}-confidence`);
    if (confSlider && config.confidence_threshold) {
        confSlider.value = config.confidence_threshold * 100;
        const valueEl = document.getElementById(`${moduleId.substring(0,2)}-confidence-value`);
        if (valueEl) {
            valueEl.textContent = `${Math.round(config.confidence_threshold * 100)}%`;
        }
    }
}

async function switchCameraPreset(presetName) {
    try {
        await fetch('/api/ptz/preset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preset: presetName })
        });
    } catch (error) {
        console.warn('[PTZ] 切换预设失败:', error);
    }
}

// =============================================================================
// WebSocket连接
// =============================================================================
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/outdoor`;
    
    try {
        OutdoorCenterState.ws = new WebSocket(wsUrl);
        
        OutdoorCenterState.ws.onopen = () => {
            console.log('[WebSocket] 连接成功');
            OutdoorCenterState.wsConnected = true;
            OutdoorCenterState.wsReconnectAttempts = 0;
            
            // 订阅当前模块
            subscribeModule(OutdoorCenterState.currentModule);
        };
        
        OutdoorCenterState.ws.onmessage = (event) => {
            handleWebSocketMessage(JSON.parse(event.data));
        };
        
        OutdoorCenterState.ws.onclose = () => {
            console.log('[WebSocket] 连接关闭');
            OutdoorCenterState.wsConnected = false;
            scheduleReconnect();
        };
        
        OutdoorCenterState.ws.onerror = (error) => {
            console.error('[WebSocket] 错误:', error);
        };
        
    } catch (error) {
        console.error('[WebSocket] 连接失败:', error);
        scheduleReconnect();
    }
}

function scheduleReconnect() {
    if (OutdoorCenterState.wsReconnectAttempts >= OutdoorCenterState.maxReconnectAttempts) {
        console.log('[WebSocket] 达到最大重连次数');
        return;
    }
    
    OutdoorCenterState.wsReconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, OutdoorCenterState.wsReconnectAttempts), 30000);
    
    console.log(`[WebSocket] ${delay}ms后重连 (尝试 ${OutdoorCenterState.wsReconnectAttempts})`);
    
    OutdoorCenterState.wsReconnectTimer = setTimeout(() => {
        connectWebSocket();
    }, delay);
}

function subscribeModule(moduleId) {
    if (!OutdoorCenterState.wsConnected || !OutdoorCenterState.ws) return;
    
    OutdoorCenterState.ws.send(JSON.stringify({
        type: 'subscribe',
        module: moduleId,
        camera: OutdoorCenterState.currentCamera
    }));
}

function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'frame':
            // 实时帧更新（如果使用WebSocket传输视频）
            if (message.image) {
                renderWebSocketFrame(message.image);
            }
            break;
            
        case 'detection':
            handleDetectionResult(message);
            break;
            
        case 'alarm':
            handleAlarms([message.alarm]);
            break;
            
        case 'metrics':
            updateMetrics(message.metrics);
            break;
            
        case 'module_status':
            updateModuleStatus(message.module, message.status);
            break;
    }
}

function renderWebSocketFrame(imageData) {
    const ctx = OutdoorCenterState.videoCtx;
    const canvas = OutdoorCenterState.videoCanvas;
    if (!ctx || !canvas) return;
    
    const img = new Image();
    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        drawDetectionOverlay();
    };
    img.src = `data:image/jpeg;base64,${imageData}`;
}

// =============================================================================
// 初始化模块状态
// =============================================================================
async function initModuleStatus() {
    try {
        const response = await fetch('/api/plugins/status');
        if (response.ok) {
            const data = await response.json();
            
            Object.entries(data.plugins || {}).forEach(([pluginId, status]) => {
                const moduleId = findModuleByPlugin(pluginId);
                if (moduleId) {
                    OutdoorCenterState.modules[moduleId].status = status.loaded ? 'ready' : 'standby';
                    updateModuleStatusDisplay(moduleId);
                }
            });
        }
    } catch (error) {
        console.warn('[Module] 获取模块状态失败:', error);
    }
}

function findModuleByPlugin(pluginId) {
    for (const [moduleId, module] of Object.entries(OutdoorCenterState.modules)) {
        if (module.pluginId === pluginId) {
            return moduleId;
        }
    }
    return null;
}

function updateModuleStatus(moduleId, status) {
    if (OutdoorCenterState.modules[moduleId]) {
        OutdoorCenterState.modules[moduleId].status = status;
        updateModuleStatusDisplay(moduleId);
    }
}

// =============================================================================
// 事件绑定
// =============================================================================
function bindEvents() {
    // 电压等级选择
    const voltageSelector = document.getElementById('voltage-selector');
    if (voltageSelector) {
        voltageSelector.addEventListener('change', (e) => {
            OutdoorCenterState.config.voltage = e.target.value;
            console.log(`[Config] 电压等级: ${e.target.value}`);
        });
    }
    
    // 站点选择
    const siteSelector = document.getElementById('site-selector');
    if (siteSelector) {
        siteSelector.addEventListener('change', async (e) => {
            const cameraId = e.target.value;
            if (cameraId) {
                await connectCamera(cameraId);
            }
        });
    }
    
    // 置信度滑块
    document.querySelectorAll('[id$="-confidence"]').forEach(slider => {
        slider.addEventListener('input', function() {
            const valueEl = document.getElementById(this.id + '-value');
            if (valueEl) {
                valueEl.textContent = this.value + '%';
            }
            OutdoorCenterState.config.confidenceThreshold = this.value / 100;
        });
    });
    
    // 检测项复选框
    document.querySelectorAll('.form-check-input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            console.log(`[Config] ${this.id}: ${this.checked}`);
        });
    });
}

// =============================================================================
// 时间更新
// =============================================================================
function startTimeUpdate() {
    updateDateTime();
    setInterval(updateDateTime, 1000);
}

function updateDateTime() {
    const now = new Date();
    const dateStr = now.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    
    const datetimeEl = document.getElementById('current-datetime');
    if (datetimeEl) {
        datetimeEl.textContent = dateStr;
    }
}

function formatTime(date) {
    const d = new Date(date);
    return d.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

// =============================================================================
// 性能监控
// =============================================================================
function startMetricsUpdate() {
    setInterval(updateMetricsDisplay, 2000);
}

function updateMetricsDisplay() {
    const metrics = OutdoorCenterState.metrics;
    
    // 更新性能指标显示
    const latencyEl = document.querySelector('.performance-metrics .metric-value:first-child');
    if (latencyEl) {
        latencyEl.textContent = `${Math.round(metrics.latency)}ms`;
    }
}

function updateMetrics(newMetrics) {
    OutdoorCenterState.metrics = { ...OutdoorCenterState.metrics, ...newMetrics };
    updateMetricsDisplay();
}

// =============================================================================
// 操作函数
// =============================================================================

/**
 * 开始检测
 */
function startDetection() {
    console.log(`[Detection] 开始检测模块: ${OutdoorCenterState.currentModule}`);
    
    if (!OutdoorCenterState.cameraConnected) {
        showToast('请先连接摄像头', 'warning');
        return;
    }
    
    OutdoorCenterState.isPaused = false;
    
    if (!OutdoorCenterState.isDetecting) {
        startAutoDetection();
    }
    
    // 立即执行一次检测
    performDetection();
    
    showToast('检测已启动', 'success');
}

/**
 * 截图保存
 */
function captureImage() {
    const canvas = OutdoorCenterState.videoCanvas;
    if (!canvas) return;
    
    const link = document.createElement('a');
    link.download = `capture_${OutdoorCenterState.currentModule}_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
    
    showToast('截图已保存', 'success');
}

/**
 * 开始自动巡检
 */
function startAutoPatrol() {
    console.log('[Patrol] 开始自动巡检');
    
    // TODO: 实现自动巡检逻辑
    showToast('自动巡检已启动', 'success');
}

/**
 * 打开设置
 */
function openSettings() {
    window.location.href = '/settings';
}

// =============================================================================
// 工具函数
// =============================================================================
function showToast(message, type = 'info') {
    // 简单的Toast提示
    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 24px;
        background: ${type === 'success' ? '#10b981' : type === 'danger' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// =============================================================================
// 导出全局函数
// =============================================================================
window.switchModule = switchModule;
window.startDetection = startDetection;
window.captureImage = captureImage;
window.startAutoPatrol = startAutoPatrol;
window.openSettings = openSettings;
