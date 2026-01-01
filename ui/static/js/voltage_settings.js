/**
 * 电压等级设置页面 JavaScript
 * 破夜绘明激光监测平台
 * 
 * 功能:
 * 1. 电压等级选择和切换
 * 2. 模型库状态显示
 * 3. 配置详情展示
 * 4. API 调用
 */

// 全局变量
let currentVoltageLevel = null;
let selectedVoltageLevel = null;
let confirmModal = null;
let successModal = null;

/**
 * 初始化电压设置页面
 * @param {string} initialLevel - 初始电压等级
 */
function initVoltageSettings(initialLevel) {
    console.log('初始化电压设置页面, 当前等级:', initialLevel);
    
    currentVoltageLevel = initialLevel || null;
    selectedVoltageLevel = null;
    
    // 初始化 Bootstrap 模态框
    confirmModal = new bootstrap.Modal(document.getElementById('confirmVoltageModal'));
    successModal = new bootstrap.Modal(document.getElementById('successModal'));
    
    // 绑定卡片点击事件
    document.querySelectorAll('.voltage-option').forEach(card => {
        card.addEventListener('click', function() {
            selectVoltageCard(this.dataset.voltage);
        });
    });
    
    // 绑定选择按钮事件
    document.querySelectorAll('.select-voltage-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            selectVoltageCard(this.dataset.voltage);
        });
    });
    
    // 如果有当前电压等级，高亮显示
    if (currentVoltageLevel) {
        highlightCurrentVoltage(currentVoltageLevel);
        refreshModelStatus();
    }
    
    // 获取当前电压等级
    fetchCurrentVoltageLevel();
}

/**
 * 从服务器获取当前电压等级
 */
async function fetchCurrentVoltageLevel() {
    try {
        const response = await fetch('/api/voltage/current');
        const data = await response.json();
        
        if (data.success && data.voltage_level) {
            currentVoltageLevel = data.voltage_level;
            document.getElementById('current-voltage-level').textContent = currentVoltageLevel;
            highlightCurrentVoltage(currentVoltageLevel);
            refreshModelStatus();
        }
    } catch (error) {
        console.error('获取当前电压等级失败:', error);
    }
}

/**
 * 高亮当前电压等级卡片
 * @param {string} level - 电压等级
 */
function highlightCurrentVoltage(level) {
    // 移除所有卡片的当前标记
    document.querySelectorAll('.voltage-option').forEach(card => {
        card.classList.remove('selected');
    });
    
    // 添加当前等级的标记
    const cardId = level === '220kV' ? 'card-220kv' : 'card-500kv';
    const card = document.getElementById(cardId);
    if (card) {
        card.classList.add('selected');
    }
}

/**
 * 选择电压等级卡片
 * @param {string} level - 电压等级
 */
function selectVoltageCard(level) {
    selectedVoltageLevel = level;
    
    // 更新卡片样式
    document.querySelectorAll('.voltage-option').forEach(card => {
        card.classList.remove('selected');
    });
    
    const cardId = level === '220kV' ? 'card-220kv' : 'card-500kv';
    document.getElementById(cardId).classList.add('selected');
    
    // 启用应用按钮
    document.getElementById('btn-apply-voltage').disabled = false;
    
    // 显示配置详情
    showConfigDetails(level);
    
    console.log('选择电压等级:', level);
}

/**
 * 显示配置详情
 * @param {string} level - 电压等级
 */
async function showConfigDetails(level) {
    const detailsCard = document.getElementById('config-details-card');
    detailsCard.style.display = 'block';
    
    try {
        // 获取设备配置
        const plugins = ['transformer', 'switch', 'busbar', 'capacitor', 'meter'];
        let allClasses = [];
        let equipmentParams = {};
        
        for (const plugin of plugins) {
            const response = await fetch(`/api/voltage/config/${plugin}?level=${level}`);
            const data = await response.json();
            
            if (data.success && data.config) {
                const classes = data.config.detection_classes || [];
                allClasses = allClasses.concat(classes);
                equipmentParams[plugin] = data.config;
            }
        }
        
        // 显示检测类别
        const classesHtml = [...new Set(allClasses)].map(cls => 
            `<span class="badge bg-secondary me-1 mb-1">${cls}</span>`
        ).join('');
        document.getElementById('detection-classes-list').innerHTML = classesHtml || '<span class="text-muted">暂无数据</span>';
        
        // 显示设备参数
        const paramsHtml = Object.entries(equipmentParams).map(([plugin, config]) => `
            <div class="mb-2">
                <strong>${getPluginName(plugin)}:</strong>
                <small class="text-muted d-block">${getConfigSummary(config)}</small>
            </div>
        `).join('');
        document.getElementById('equipment-params').innerHTML = paramsHtml || '<span class="text-muted">暂无数据</span>';
        
    } catch (error) {
        console.error('获取配置详情失败:', error);
        document.getElementById('detection-classes-list').innerHTML = '<span class="text-danger">加载失败</span>';
        document.getElementById('equipment-params').innerHTML = '<span class="text-danger">加载失败</span>';
    }
}

/**
 * 获取插件中文名称
 */
function getPluginName(plugin) {
    const names = {
        'transformer': '主变巡视',
        'switch': '开关间隔',
        'busbar': '母线巡视',
        'capacitor': '电容器',
        'meter': '表计读数'
    };
    return names[plugin] || plugin;
}

/**
 * 获取配置摘要
 */
function getConfigSummary(config) {
    const items = [];
    if (config.detection_classes) {
        items.push(`${config.detection_classes.length}个检测类别`);
    }
    if (config.thermal_threshold_celsius) {
        const t = config.thermal_threshold_celsius;
        items.push(`热成像阈值: ${t.warning || '-'}°C`);
    }
    return items.join(', ') || '标准配置';
}

/**
 * 应用电压等级（显示确认对话框）
 */
function applyVoltageLevel() {
    if (!selectedVoltageLevel) {
        showToast('请先选择电压等级', 'warning');
        return;
    }
    
    // 如果选择的和当前的相同
    if (selectedVoltageLevel === currentVoltageLevel) {
        showToast('当前已是 ' + selectedVoltageLevel + ' 模式', 'info');
        return;
    }
    
    // 更新确认对话框内容
    document.getElementById('confirm-voltage-text').textContent = selectedVoltageLevel;
    
    // 显示确认对话框
    confirmModal.show();
}

/**
 * 确认切换电压等级
 */
async function confirmVoltageSwitch() {
    const level = selectedVoltageLevel;
    
    // 关闭确认对话框
    confirmModal.hide();
    
    // 显示加载状态
    showLoadingState(true);
    
    try {
        // 调用API切换电压等级
        const response = await fetch('/api/voltage/set', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ level: level })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 更新当前电压等级
            currentVoltageLevel = level;
            document.getElementById('current-voltage-level').textContent = level;
            
            // 刷新模型状态
            await refreshModelStatus();
            
            // 显示成功消息
            document.getElementById('success-message').textContent = 
                `已成功切换到 ${level} 模式，AI模型库已更新`;
            successModal.show();
            
            showToast('电压等级切换成功', 'success');
        } else {
            throw new Error(data.message || '切换失败');
        }
    } catch (error) {
        console.error('切换电压等级失败:', error);
        showToast('切换失败: ' + error.message, 'danger');
    } finally {
        showLoadingState(false);
    }
}

/**
 * 刷新模型状态
 */
async function refreshModelStatus() {
    const tableBody = document.getElementById('model-table-body');
    const spinner = document.getElementById('model-loading-spinner');
    
    spinner.classList.remove('d-none');
    
    try {
        const response = await fetch('/api/voltage/models');
        const data = await response.json();
        
        if (data.success && data.models) {
            let html = '';
            
            for (const [plugin, models] of Object.entries(data.models)) {
                for (const [modelName, modelPath] of Object.entries(models)) {
                    // 检查模型文件是否存在
                    const statusResponse = await fetch(`/api/voltage/model-status?path=${encodeURIComponent(modelPath)}`);
                    const statusData = await statusResponse.json();
                    
                    const exists = statusData.exists || false;
                    const statusBadge = exists 
                        ? '<span class="badge bg-success model-status-badge"><i class="bi bi-check-circle"></i> 已就绪</span>'
                        : '<span class="badge bg-warning model-status-badge"><i class="bi bi-exclamation-triangle"></i> 待训练</span>';
                    
                    html += `
                        <tr>
                            <td><strong>${getPluginName(plugin)}</strong></td>
                            <td><code class="small">${modelPath.split('/').pop()}</code></td>
                            <td>${statusBadge}</td>
                        </tr>
                    `;
                }
            }
            
            tableBody.innerHTML = html || '<tr><td colspan="3" class="text-center text-muted">暂无模型配置</td></tr>';
        }
    } catch (error) {
        console.error('刷新模型状态失败:', error);
        tableBody.innerHTML = '<tr><td colspan="3" class="text-center text-danger">加载失败</td></tr>';
    } finally {
        spinner.classList.add('d-none');
    }
}

/**
 * 显示/隐藏加载状态
 */
function showLoadingState(show) {
    const applyBtn = document.getElementById('btn-apply-voltage');
    const refreshBtn = document.getElementById('btn-refresh-models');
    
    if (show) {
        applyBtn.disabled = true;
        applyBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> 切换中...';
        refreshBtn.disabled = true;
    } else {
        applyBtn.disabled = false;
        applyBtn.innerHTML = '<i class="bi bi-check-lg"></i> 应用设置';
        refreshBtn.disabled = false;
    }
}

/**
 * 显示 Toast 消息
 */
function showToast(message, type = 'info') {
    // 创建 Toast 容器（如果不存在）
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';
        document.body.appendChild(toastContainer);
    }
    
    // 创建 Toast
    const toastId = 'toast-' + Date.now();
    const bgClass = {
        'success': 'bg-success',
        'danger': 'bg-danger',
        'warning': 'bg-warning text-dark',
        'info': 'bg-info text-dark'
    }[type] || 'bg-info text-dark';
    
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center ${bgClass} text-white border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: 3000 });
    toast.show();
    
    // 自动移除
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

// 导出全局函数（供 HTML 内联调用）
window.initVoltageSettings = initVoltageSettings;
window.applyVoltageLevel = applyVoltageLevel;
window.confirmVoltageSwitch = confirmVoltageSwitch;
window.refreshModelStatus = refreshModelStatus;
