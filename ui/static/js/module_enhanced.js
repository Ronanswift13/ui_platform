/**
 * 增强版模块JavaScript
 * 输变电激光监测平台 - 全自动AI巡检改造
 * 
 * 功能:
 * - 实时视频流和检测结果叠加
 * - 自动ROI可视化
 * - 云台控制
 * - 模型管理
 * - 多证据融合可视化
 */

// ============== 全局状态 ==============
const EnhancedState = {
    moduleId: null,
    ws: null,
    videoStream: null,
    isRunning: false,
    isPatrolling: false,
    currentDetections: [],
    autoROIs: [],
    fusionWeights: {
        dl: 0.6,
        ocr: 0.3,
        color: 0.2,
        angle: 0.2,
    },
    metrics: {
        fps: 0,
        latency: 0,
        gpuUsage: 0,
    },
};

// ============== 初始化 ==============
function initEnhancedModule(moduleId) {
    EnhancedState.moduleId = moduleId;
    
    // 加载站点列表
    loadSites();
    
    // 初始化视频流
    initVideoStream();
    
    // 初始化WebSocket
    initWebSocket();
    
    // 绑定事件
    bindEnhancedEvents();
    
    // 初始化云台控制
    initPTZControls();
    
    // 初始化融合面板
    initFusionPanel();
    
    // 启动性能监控
    startPerformanceMonitor();
    
    console.log(`[Enhanced] 模块初始化完成: ${moduleId}`);
}

// ============== 站点加载 ==============
async function loadSites() {
    try {
        const response = await fetch('/api/sites');
        const sites = await response.json();
        
        const select = document.getElementById('select-site');
        select.innerHTML = '<option value="">请选择站点...</option>';
        
        sites.forEach(site => {
            select.innerHTML += `<option value="${site.id}">${site.name}</option>`;
        });
    } catch (err) {
        console.error('加载站点失败:', err);
    }
}

// ============== 视频流 ==============
function initVideoStream() {
    const canvas = document.getElementById('video-canvas');
    const ctx = canvas.getContext('2d');
    
    // 设置canvas尺寸
    canvas.width = 1280;
    canvas.height = 720;
    
    // 隐藏加载提示
    document.getElementById('video-loading').style.display = 'none';
    
    // 绘制占位图
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#888';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('等待视频流...', canvas.width / 2, canvas.height / 2);
}

// ============== WebSocket ==============
function initWebSocket() {
    const wsUrl = `ws://${window.location.host}/ws/inference`;
    
    EnhancedState.ws = new WebSocket(wsUrl);
    
    EnhancedState.ws.onopen = () => {
        console.log('[WS] 连接成功');
        updateStreamStatus(true);
    };
    
    EnhancedState.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    EnhancedState.ws.onclose = () => {
        console.log('[WS] 连接断开');
        updateStreamStatus(false);
        // 5秒后重连
        setTimeout(initWebSocket, 5000);
    };
    
    EnhancedState.ws.onerror = (err) => {
        console.error('[WS] 错误:', err);
    };
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'frame':
            renderFrame(data.frame, data.detections);
            break;
        case 'detection':
            updateDetections(data.detections);
            break;
        case 'metrics':
            updateMetrics(data.metrics);
            break;
        case 'alarm':
            addAlarm(data.alarm);
            break;
        case 'roi_update':
            updateAutoROIs(data.rois);
            break;
        case 'reshoot_suggestion':
            showReshootSuggestion(data);
            break;
    }
}

function updateStreamStatus(online) {
    const badge = document.getElementById('stream-status');
    if (online) {
        badge.textContent = '在线';
        badge.className = 'badge bg-success';
    } else {
        badge.textContent = '离线';
        badge.className = 'badge bg-danger';
    }
}

// ============== 检测结果渲染 ==============
function renderFrame(frameData, detections) {
    const canvas = document.getElementById('video-canvas');
    const ctx = canvas.getContext('2d');
    
    // 创建图像
    const img = new Image();
    img.onload = () => {
        // 绘制帧
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // 绘制检测框
        if (detections) {
            drawDetections(ctx, detections, canvas.width, canvas.height);
        }
        
        // 绘制自动ROI
        drawAutoROIs();
    };
    img.src = 'data:image/jpeg;base64,' + frameData;
    
    // 更新检测计数
    document.getElementById('detection-count').textContent = detections ? detections.length : 0;
}

function drawDetections(ctx, detections, width, height) {
    detections.forEach(det => {
        const { bbox, label, confidence, severity } = det;
        
        // 颜色映射
        let color = '#00ff00';  // 绿色 - 正常
        if (severity === 'warning') color = '#ffff00';  // 黄色 - 告警
        if (severity === 'error') color = '#ff0000';    // 红色 - 异常
        if (severity === 'critical') color = '#ff00ff'; // 紫色 - 严重
        
        // 转换坐标
        const x = bbox.x * width;
        const y = bbox.y * height;
        const w = bbox.width * width;
        const h = bbox.height * height;
        
        // 绘制框
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        
        // 绘制标签背景
        ctx.fillStyle = color;
        const labelText = `${label} ${(confidence * 100).toFixed(1)}%`;
        const labelWidth = ctx.measureText(labelText).width + 10;
        ctx.fillRect(x, y - 20, labelWidth, 20);
        
        // 绘制标签文字
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        ctx.fillText(labelText, x + 5, y - 6);
    });
}

function drawAutoROIs() {
    const svg = document.getElementById('roi-overlay');
    svg.innerHTML = '';
    
    EnhancedState.autoROIs.forEach(roi => {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', `${roi.bbox.x * 100}%`);
        rect.setAttribute('y', `${roi.bbox.y * 100}%`);
        rect.setAttribute('width', `${roi.bbox.width * 100}%`);
        rect.setAttribute('height', `${roi.bbox.height * 100}%`);
        rect.setAttribute('fill', 'none');
        rect.setAttribute('stroke', '#00ffff');
        rect.setAttribute('stroke-width', '1');
        rect.setAttribute('stroke-dasharray', '5,5');
        svg.appendChild(rect);
        
        // 添加标签
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', `${roi.bbox.x * 100}%`);
        text.setAttribute('y', `${(roi.bbox.y - 0.01) * 100}%`);
        text.setAttribute('fill', '#00ffff');
        text.setAttribute('font-size', '10');
        text.textContent = `[自动] ${roi.class_name}`;
        svg.appendChild(text);
    });
}

function updateDetections(detections) {
    EnhancedState.currentDetections = detections;
    
    // 更新统计
    let normal = 0, warning = 0, error = 0, manual = 0;
    detections.forEach(det => {
        if (det.need_manual_review) manual++;
        else if (det.severity === 'error' || det.severity === 'critical') error++;
        else if (det.severity === 'warning') warning++;
        else normal++;
    });
    
    document.getElementById('result-normal').textContent = normal;
    document.getElementById('result-warning').textContent = warning;
    document.getElementById('result-error').textContent = error;
    document.getElementById('result-manual').textContent = manual;
    
    // 更新表格
    updateResultsTable(detections);
}

function updateResultsTable(detections) {
    const tbody = document.getElementById('results-body');
    
    if (!detections || detections.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">暂无检测结果</td></tr>';
        return;
    }
    
    tbody.innerHTML = detections.map(det => `
        <tr class="${getSeverityClass(det.severity)}">
            <td>${det.target_name || det.roi_id || '--'}</td>
            <td>${det.label}</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar ${getConfidenceClass(det.confidence)}" 
                         style="width: ${det.confidence * 100}%">
                        ${(det.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            </td>
            <td>${det.reason_code || '--'}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="viewDetail('${det.id}')">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-outline-warning" onclick="markReview('${det.id}')">
                    <i class="bi bi-flag"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

function getSeverityClass(severity) {
    switch (severity) {
        case 'error': return 'table-danger';
        case 'critical': return 'table-danger';
        case 'warning': return 'table-warning';
        default: return '';
    }
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'bg-success';
    if (confidence >= 0.5) return 'bg-warning';
    return 'bg-danger';
}

// ============== 云台控制 ==============
function initPTZControls() {
    // 方向控制
    document.getElementById('ptz-up').addEventListener('click', () => sendPTZCommand('tilt_up'));
    document.getElementById('ptz-down').addEventListener('click', () => sendPTZCommand('tilt_down'));
    document.getElementById('ptz-left').addEventListener('click', () => sendPTZCommand('pan_left'));
    document.getElementById('ptz-right').addEventListener('click', () => sendPTZCommand('pan_right'));
    document.getElementById('ptz-home').addEventListener('click', () => sendPTZCommand('goto_preset', { preset_id: 0 }));
    
    // 变焦控制
    document.getElementById('ptz-zoom-in').addEventListener('click', () => sendPTZCommand('zoom_in'));
    document.getElementById('ptz-zoom-out').addEventListener('click', () => sendPTZCommand('zoom_out'));
    
    // 焦点控制
    document.getElementById('ptz-focus-near').addEventListener('click', () => sendPTZCommand('focus_near'));
    document.getElementById('ptz-focus-far').addEventListener('click', () => sendPTZCommand('focus_far'));
    document.getElementById('ptz-auto-focus').addEventListener('click', () => sendPTZCommand('auto_focus'));
    
    // 预置位
    document.getElementById('btn-goto-preset').addEventListener('click', () => {
        const presetId = document.getElementById('preset-select').value;
        sendPTZCommand('goto_preset', { preset_id: parseInt(presetId) });
    });
    
    document.getElementById('btn-set-preset').addEventListener('click', () => {
        const presetId = document.getElementById('preset-select').value;
        sendPTZCommand('set_preset', { preset_id: parseInt(presetId) });
    });
    
    // 巡航
    document.getElementById('btn-start-patrol').addEventListener('click', startPatrol);
    document.getElementById('btn-stop-patrol').addEventListener('click', stopPatrol);
}

async function sendPTZCommand(command, params = {}) {
    try {
        const response = await fetch('/api/ptz/command', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command, ...params }),
        });
        
        if (!response.ok) {
            throw new Error('PTZ命令执行失败');
        }
        
        console.log(`[PTZ] 命令执行: ${command}`);
    } catch (err) {
        console.error('[PTZ] 错误:', err);
        showToast('云台控制失败', 'danger');
    }
}

function startPatrol() {
    EnhancedState.isPatrolling = true;
    document.getElementById('btn-start-patrol').disabled = true;
    document.getElementById('btn-stop-patrol').disabled = false;
    sendPTZCommand('start_patrol', { route_id: 'default' });
}

function stopPatrol() {
    EnhancedState.isPatrolling = false;
    document.getElementById('btn-start-patrol').disabled = false;
    document.getElementById('btn-stop-patrol').disabled = true;
    sendPTZCommand('stop_patrol');
}

// ============== 融合面板 ==============
function initFusionPanel() {
    // 权重滑块事件
    ['dl', 'ocr', 'color'].forEach(type => {
        const slider = document.getElementById(`slider-weight-${type}`);
        const display = document.getElementById(`weight-${type}`);
        
        if (slider && display) {
            slider.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                display.textContent = value.toFixed(1);
                EnhancedState.fusionWeights[type] = value;
            });
        }
    });
}

function updateFusionResult(result) {
    document.getElementById('fusion-state').textContent = result.state || '--';
    document.getElementById('fusion-confidence').textContent = 
        result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : '--';
    
    const conflictBadge = document.getElementById('fusion-conflict');
    if (result.conflict_detected) {
        conflictBadge.textContent = '冲突';
        conflictBadge.className = 'badge bg-warning';
    } else {
        conflictBadge.textContent = '无';
        conflictBadge.className = 'badge bg-secondary';
    }
}

// ============== 告警管理 ==============
function addAlarm(alarm) {
    const list = document.getElementById('alarm-list');
    
    // 移除空状态提示
    if (list.querySelector('.text-muted')) {
        list.innerHTML = '';
    }
    
    const levelClass = {
        'info': 'list-group-item-info',
        'warning': 'list-group-item-warning',
        'error': 'list-group-item-danger',
        'critical': 'list-group-item-danger',
    }[alarm.level] || '';
    
    const item = document.createElement('a');
    item.className = `list-group-item list-group-item-action ${levelClass}`;
    item.innerHTML = `
        <div class="d-flex justify-content-between">
            <strong>${alarm.type}</strong>
            <small>${new Date(alarm.timestamp).toLocaleTimeString()}</small>
        </div>
        <p class="mb-0 small">${alarm.message}</p>
    `;
    
    list.insertBefore(item, list.firstChild);
    
    // 更新计数
    const count = list.querySelectorAll('.list-group-item').length;
    document.getElementById('alarm-count').textContent = count;
    
    // 限制最多显示20条
    while (list.children.length > 20) {
        list.removeChild(list.lastChild);
    }
}

// ============== 复拍建议 ==============
function showReshootSuggestion(data) {
    document.getElementById('reshoot-reason').textContent = data.reason;
    document.getElementById('current-clarity').textContent = data.clarity_score.toFixed(2);
    
    const suggestionsList = document.getElementById('reshoot-suggestions');
    suggestionsList.innerHTML = data.suggestions.map(s => 
        `<li><i class="bi bi-check"></i> ${s}</li>`
    ).join('');
    
    // 显示模态框
    const modal = new bootstrap.Modal(document.getElementById('reshootModal'));
    modal.show();
}

document.getElementById('btn-auto-reshoot')?.addEventListener('click', async () => {
    try {
        await fetch('/api/ptz/smart_reshoot', { method: 'POST' });
        bootstrap.Modal.getInstance(document.getElementById('reshootModal')).hide();
        showToast('正在执行自动复拍...', 'info');
    } catch (err) {
        showToast('复拍失败', 'danger');
    }
});

// ============== 性能监控 ==============
function startPerformanceMonitor() {
    setInterval(async () => {
        try {
            const response = await fetch('/api/metrics');
            const metrics = await response.json();
            updateMetrics(metrics);
        } catch (err) {
            // 忽略错误
        }
    }, 1000);
}

function updateMetrics(metrics) {
    document.getElementById('inference-latency').textContent = 
        metrics.latency ? metrics.latency.toFixed(1) : '--';
    document.getElementById('gpu-usage').textContent = 
        metrics.gpu_usage ? metrics.gpu_usage.toFixed(0) : '--';
    document.getElementById('fps-counter').textContent = 
        metrics.fps ? metrics.fps.toFixed(1) : '--';
}

// ============== 事件绑定 ==============
function bindEnhancedEvents() {
    // 运行任务
    document.getElementById('btn-run').addEventListener('click', runTask);
    document.getElementById('btn-stop').addEventListener('click', stopTask);
    
    // 站点选择
    document.getElementById('select-site').addEventListener('change', loadPositions);
    
    // 导出结果
    document.getElementById('btn-export-results').addEventListener('click', exportResults);
    
    // 显示模式切换
    document.getElementById('btn-visible').addEventListener('click', () => setDisplayMode('visible'));
    document.getElementById('btn-thermal').addEventListener('click', () => setDisplayMode('thermal'));
    document.getElementById('btn-fusion').addEventListener('click', () => setDisplayMode('fusion'));
}

async function runTask() {
    const btn = document.getElementById('btn-run');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 运行中...';
    
    document.getElementById('btn-stop').disabled = false;
    EnhancedState.isRunning = true;
    
    try {
        const response = await fetch('/api/tasks/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                site_id: document.getElementById('select-site').value,
                position_id: document.getElementById('select-position').value,
                task_template: document.getElementById('select-task').value,
                module_id: EnhancedState.moduleId,
            }),
        });
        
        const result = await response.json();
        console.log('任务启动:', result);
        
        // 开始轮询进度
        pollTaskProgress(result.task_id);
        
    } catch (err) {
        console.error('任务启动失败:', err);
        showToast('任务启动失败', 'danger');
        resetRunButton();
    }
}

function stopTask() {
    EnhancedState.isRunning = false;
    document.getElementById('btn-stop').disabled = true;
    resetRunButton();
    
    fetch('/api/tasks/current/stop', { method: 'POST' });
}

function resetRunButton() {
    const btn = document.getElementById('btn-run');
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-play-fill"></i> 运行任务';
}

async function pollTaskProgress(taskId) {
    const progressBar = document.getElementById('task-progress');
    
    while (EnhancedState.isRunning) {
        try {
            const response = await fetch(`/api/tasks/${taskId}`);
            const task = await response.json();
            
            const progress = task.progress || 0;
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${progress}%`;
            
            if (task.status === 'completed' || task.status === 'failed') {
                EnhancedState.isRunning = false;
                resetRunButton();
                document.getElementById('btn-stop').disabled = true;
                
                if (task.status === 'completed') {
                    showToast('任务完成', 'success');
                } else {
                    showToast('任务失败: ' + task.error, 'danger');
                }
                break;
            }
            
            await new Promise(resolve => setTimeout(resolve, 500));
        } catch (err) {
            console.error('获取任务状态失败:', err);
            break;
        }
    }
}

async function loadPositions() {
    const siteId = document.getElementById('select-site').value;
    if (!siteId) return;
    
    try {
        const response = await fetch(`/api/sites/${siteId}/positions`);
        const positions = await response.json();
        
        const select = document.getElementById('select-position');
        select.innerHTML = '<option value="">请选择点位...</option>';
        
        positions.forEach(pos => {
            select.innerHTML += `<option value="${pos.id}">${pos.name}</option>`;
        });
    } catch (err) {
        console.error('加载点位失败:', err);
    }
}

function setDisplayMode(mode) {
    // 更新按钮状态
    ['visible', 'thermal', 'fusion'].forEach(m => {
        const btn = document.getElementById(`btn-${m}`);
        btn.classList.toggle('active', m === mode);
    });
    
    // 发送显示模式切换命令
    if (EnhancedState.ws && EnhancedState.ws.readyState === WebSocket.OPEN) {
        EnhancedState.ws.send(JSON.stringify({ type: 'set_display_mode', mode }));
    }
}

function exportResults() {
    const results = EnhancedState.currentDetections;
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `detection_results_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    
    URL.revokeObjectURL(url);
}

function updateAutoROIs(rois) {
    EnhancedState.autoROIs = rois;
    drawAutoROIs();
}

// ============== 工具函数 ==============
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed top-0 end-0 m-3`;
    toast.style.zIndex = '9999';
    toast.innerHTML = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function viewDetail(detectionId) {
    console.log('查看详情:', detectionId);
    // TODO: 实现详情查看
}

function markReview(detectionId) {
    console.log('标记复核:', detectionId);
    // TODO: 实现标记复核
}
