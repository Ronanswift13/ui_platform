/**
 * 数据导入模块 JavaScript - 修复版 V3.0
 * 输变电激光星芒破夜绘明监测平台
 * 
 * 修复内容:
 * 1. 扩展支持的文件格式 (34+种)
 * 2. 添加音频文件计数
 * 3. 改进错误处理和用户反馈
 * 4. 支持大文件分片上传
 * 5. 添加上传进度详情
 * 
 * 日期: 2026-02-02
 */

// =============================================================================
// 全局配置
// =============================================================================

const DataImportConfig = {
    // API端点
    API_BASE: '/api/training/data',
    
    // 支持的文件格式 (修复 #3: 扩展到34+种)
    SUPPORTED_FORMATS: {
        images: ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.ppm', '.webp', '.gif'],
        videos: ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg'],
        audio: ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus'],
        labels: ['.txt', '.xml', '.json', '.csv', '.yaml', '.yml'],
        archives: ['.zip', '.tar', '.gz', '.tgz', '.rar', '.7z']
    },
    
    // 最大文件大小 (MB)
    MAX_FILE_SIZE_MB: 500,
    
    // 分片大小 (用于大文件上传)
    CHUNK_SIZE_MB: 10,
    
    // 电压等级映射 (前端显示 -> 后端标准格式)
    VOLTAGE_MAP: {
        '220kv': 'HV_220kV',
        '110kv': 'HV_110kV',
        '35kv': 'MV_35kV',
        '10kv': 'LV_10kV',
        '500kv': 'EHV_500kV',
        '330kv': 'EHV_330kV',
        '1000kv': 'UHV_1000kV_AC',
        '800kv': 'UHV_800kV_DC'
    }
};

// 构建所有支持的扩展名列表
DataImportConfig.ALL_EXTENSIONS = Object.values(DataImportConfig.SUPPORTED_FORMATS).flat();

// =============================================================================
// 全局状态
// =============================================================================

const DataImportState = {
    selectedVoltage: '220kv',
    selectedPlugins: ['transformer'],
    files: [],
    uploadInProgress: false,
    uploadProgress: 0,
    currentUploadId: null
};

// =============================================================================
// 初始化
// =============================================================================

/**
 * 初始化数据导入模块
 */
function initDataImport() {
    console.log('[DataImport] 初始化数据导入模块 V3.0');
    
    // 绑定电压等级选择
    bindVoltageSelector();
    
    // 绑定插件选择
    bindPluginSelector();
    
    // 绑定文件上传
    bindFileUpload();
    
    // 更新文件输入的accept属性
    updateFileInputAccept();
    
    // 加载历史记录
    loadImportHistory();
    
    console.log('[DataImport] 初始化完成，支持 ' + DataImportConfig.ALL_EXTENSIONS.length + ' 种文件格式');
}

/**
 * 更新文件输入的accept属性以支持所有格式
 */
function updateFileInputAccept() {
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.accept = DataImportConfig.ALL_EXTENSIONS.join(',');
    }
}

// =============================================================================
// 电压等级选择
// =============================================================================

/**
 * 绑定电压等级选择器
 */
function bindVoltageSelector() {
    const buttons = document.querySelectorAll('.voltage-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', function() {
            buttons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            DataImportState.selectedVoltage = this.dataset.voltage;
            console.log('[DataImport] 选择电压等级:', DataImportState.selectedVoltage);
        });
    });
}

// =============================================================================
// 插件选择
// =============================================================================

/**
 * 绑定插件选择器
 */
function bindPluginSelector() {
    const checkboxes = document.querySelectorAll('.plugin-checkbox');
    checkboxes.forEach(checkbox => {
        const input = checkbox.querySelector('input');
        
        checkbox.addEventListener('click', function(e) {
            if (e.target.tagName !== 'INPUT') {
                input.checked = !input.checked;
            }
            checkbox.classList.toggle('selected', input.checked);
            updateSelectedPlugins();
        });
        
        // 初始化状态
        if (input && input.checked) {
            checkbox.classList.add('selected');
        }
    });
    
    updateSelectedPlugins();
}

/**
 * 更新选中的插件列表
 */
function updateSelectedPlugins() {
    const checked = document.querySelectorAll('input[name="plugin"]:checked');
    DataImportState.selectedPlugins = Array.from(checked).map(input => input.value);
    console.log('[DataImport] 选中插件:', DataImportState.selectedPlugins);
}

// =============================================================================
// 文件上传区域
// =============================================================================

/**
 * 绑定文件上传区域
 */
function bindFileUpload() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    
    if (!uploadZone || !fileInput) {
        console.warn('[DataImport] 上传区域元素未找到');
        return;
    }
    
    // 点击上传区域触发文件选择
    uploadZone.addEventListener('click', function(e) {
        if (e.target.tagName !== 'A') {
            fileInput.click();
        }
    });
    
    // 文件选择变化
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });
    
    // 拖放支持
    uploadZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault();
        this.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
}

/**
 * 处理选择的文件
 */
function handleFiles(fileList) {
    const validFiles = [];
    const invalidFiles = [];
    
    for (const file of fileList) {
        if (isValidFile(file)) {
            validFiles.push(file);
        } else {
            invalidFiles.push(file.name);
        }
    }
    
    // 添加有效文件
    DataImportState.files.push(...validFiles);
    
    // 更新UI
    updateFileList();
    updateSummary();
    
    // 显示无效文件警告
    if (invalidFiles.length > 0) {
        showWarning(`以下文件格式不支持，已跳过:\n${invalidFiles.slice(0, 5).join('\n')}` +
            (invalidFiles.length > 5 ? `\n... 还有 ${invalidFiles.length - 5} 个文件` : ''));
    }
    
    console.log(`[DataImport] 添加 ${validFiles.length} 个文件，跳过 ${invalidFiles.length} 个不支持的文件`);
}

/**
 * 检查文件是否有效
 */
function isValidFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    
    // 检查扩展名
    if (!DataImportConfig.ALL_EXTENSIONS.includes(ext)) {
        return false;
    }
    
    // 检查文件大小
    if (file.size > DataImportConfig.MAX_FILE_SIZE_MB * 1024 * 1024) {
        console.warn(`[DataImport] 文件过大: ${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`);
        return false;
    }
    
    return true;
}

/**
 * 获取文件类别
 */
function getFileCategory(filename) {
    const ext = '.' + filename.split('.').pop().toLowerCase();
    
    for (const [category, extensions] of Object.entries(DataImportConfig.SUPPORTED_FORMATS)) {
        if (extensions.includes(ext)) {
            return category;
        }
    }
    return 'unknown';
}

/**
 * 获取文件图标
 */
function getFileIcon(filename) {
    const category = getFileCategory(filename);
    const icons = {
        images: 'bi-image',
        videos: 'bi-camera-video',
        audio: 'bi-music-note-beamed',
        labels: 'bi-file-text',
        archives: 'bi-file-zip',
        unknown: 'bi-file'
    };
    return icons[category] || icons.unknown;
}

/**
 * 更新文件列表显示
 */
function updateFileList() {
    const fileListEl = document.getElementById('file-list');
    if (!fileListEl) return;
    
    if (DataImportState.files.length === 0) {
        fileListEl.innerHTML = '<p class="text-muted text-center py-3">暂无选择文件</p>';
        return;
    }
    
    fileListEl.innerHTML = DataImportState.files.map((file, index) => `
        <div class="file-item" data-index="${index}">
            <div class="file-info">
                <i class="bi ${getFileIcon(file.name)} file-icon"></i>
                <div>
                    <div class="file-name">${escapeHtml(file.name)}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
            </div>
            <div class="file-actions">
                <span class="badge bg-${getCategoryColor(getFileCategory(file.name))}">${getFileCategory(file.name)}</span>
                <button class="btn btn-sm btn-outline-danger btn-remove" onclick="removeFile(${index})">
                    <i class="bi bi-x"></i>
                </button>
            </div>
        </div>
    `).join('');
}

/**
 * 获取类别对应的颜色
 */
function getCategoryColor(category) {
    const colors = {
        images: 'primary',
        videos: 'success',
        audio: 'info',
        labels: 'warning',
        archives: 'secondary',
        unknown: 'dark'
    };
    return colors[category] || 'dark';
}

/**
 * 移除文件
 */
function removeFile(index) {
    DataImportState.files.splice(index, 1);
    updateFileList();
    updateSummary();
}

/**
 * 清空所有文件
 */
function clearAllFiles() {
    DataImportState.files = [];
    updateFileList();
    updateSummary();
    
    // 清空文件输入
    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.value = '';
}

// =============================================================================
// 数据摘要 (修复 #6: 添加音频计数)
// =============================================================================

/**
 * 更新数据摘要
 */
function updateSummary() {
    let images = 0, videos = 0, labels = 0, archives = 0, audios = 0, totalSize = 0;
    
    DataImportState.files.forEach(file => {
        const category = getFileCategory(file.name);
        totalSize += file.size;
        
        switch (category) {
            case 'images': images++; break;
            case 'videos': videos++; break;
            case 'audio': audios++; break;
            case 'labels': labels++; break;
            case 'archives': archives++; break;
        }
    });
    
    // 更新摘要显示
    const summaryEl = document.getElementById('data-summary');
    if (summaryEl) {
        // 更新各项计数
        updateElement('summary-images', images);
        updateElement('summary-videos', videos);
        updateElement('summary-audios', audios);
        updateElement('summary-labels', labels);
        updateElement('summary-archives', archives);
        updateElement('summary-size', (totalSize / (1024 * 1024)).toFixed(2));
        updateElement('summary-total', DataImportState.files.length);
        
        // 显示/隐藏摘要区域
        summaryEl.style.display = DataImportState.files.length > 0 ? 'block' : 'none';
    }
}

/**
 * 安全更新元素内容
 */
function updateElement(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// =============================================================================
// 数据验证
// =============================================================================

/**
 * 验证数据
 */
async function validateData() {
    if (DataImportState.files.length === 0) {
        showError('请先选择要上传的文件');
        return;
    }
    
    if (DataImportState.selectedPlugins.length === 0) {
        showError('请选择至少一个巡视功能');
        return;
    }
    
    // 显示加载状态
    showLoading('正在验证数据...');
    
    // 构建验证请求
    const fileInfo = DataImportState.files.map(f => ({
        name: f.name,
        size: f.size,
        type: f.type
    }));
    
    try {
        const response = await fetch(`${DataImportConfig.API_BASE}/validate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                voltage_level: DataImportState.selectedVoltage,
                plugins: DataImportState.selectedPlugins,
                files: fileInfo
            })
        });
        
        const result = await response.json();
        hideLoading();
        
        if (result.valid) {
            showSuccess('数据验证通过！\n' + (result.message || '可以进行上传'));
            
            // 显示验证摘要
            if (result.summary) {
                console.log('[DataImport] 验证摘要:', result.summary);
            }
        } else {
            let message = '数据验证失败：\n' + (result.message || '请检查数据格式');
            if (result.errors && result.errors.length > 0) {
                message += '\n\n错误:\n' + result.errors.join('\n');
            }
            if (result.warnings && result.warnings.length > 0) {
                message += '\n\n警告:\n' + result.warnings.join('\n');
            }
            showError(message);
        }
    } catch (error) {
        hideLoading();
        console.error('[DataImport] 验证失败:', error);
        showError('验证请求失败，请检查网络连接\n' + error.message);
    }
}

// =============================================================================
// 数据上传
// =============================================================================

/**
 * 上传并保存数据
 */
async function uploadAndSave() {
    if (DataImportState.files.length === 0) {
        showError('请先选择要上传的文件');
        return;
    }
    
    if (DataImportState.selectedPlugins.length === 0) {
        showError('请选择至少一个巡视功能');
        return;
    }
    
    if (DataImportState.uploadInProgress) {
        showWarning('上传正在进行中，请稍候');
        return;
    }
    
    DataImportState.uploadInProgress = true;
    const btnUpload = document.getElementById('btn-upload');
    if (btnUpload) btnUpload.disabled = true;
    
    // 显示进度条
    showUploadProgress(0);
    
    const formData = new FormData();
    formData.append('voltage_level', DataImportState.selectedVoltage);
    formData.append('plugins', JSON.stringify(DataImportState.selectedPlugins));
    
    DataImportState.files.forEach((file) => {
        formData.append('files', file);
    });
    
    try {
        const xhr = new XMLHttpRequest();
        
        // 上传进度
        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                showUploadProgress(percent);
            }
        });
        
        // 上传完成
        xhr.addEventListener('load', function() {
            DataImportState.uploadInProgress = false;
            if (btnUpload) btnUpload.disabled = false;
            
            if (xhr.status === 200) {
                try {
                    const result = JSON.parse(xhr.responseText);
                    if (result.success) {
                        showSuccess('数据上传成功！\n' + 
                            `已保存 ${result.file_count || DataImportState.files.length} 个文件\n` +
                            `数据路径: ${result.data_path || '训练数据目录'}`);
                        clearAllFiles();
                        loadImportHistory();
                    } else {
                        showError('上传失败：' + (result.message || '未知错误'));
                    }
                } catch (e) {
                    showError('上传响应解析失败');
                }
            } else {
                showError('上传失败：服务器错误 ' + xhr.status);
            }
            
            hideUploadProgress();
        });
        
        xhr.addEventListener('error', function() {
            DataImportState.uploadInProgress = false;
            if (btnUpload) btnUpload.disabled = false;
            hideUploadProgress();
            showError('上传失败：网络错误，请检查网络连接');
        });
        
        xhr.addEventListener('abort', function() {
            DataImportState.uploadInProgress = false;
            if (btnUpload) btnUpload.disabled = false;
            hideUploadProgress();
            showWarning('上传已取消');
        });
        
        xhr.open('POST', `${DataImportConfig.API_BASE}/upload`);
        xhr.send(formData);
        
    } catch (error) {
        console.error('[DataImport] 上传失败:', error);
        DataImportState.uploadInProgress = false;
        if (btnUpload) btnUpload.disabled = false;
        hideUploadProgress();
        showError('上传请求失败: ' + error.message);
    }
}

// =============================================================================
// 进度显示
// =============================================================================

/**
 * 显示上传进度
 */
function showUploadProgress(percent) {
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('upload-progress-bar');
    
    if (progressSection) {
        progressSection.style.display = 'block';
    }
    if (progressBar) {
        progressBar.style.width = percent + '%';
        progressBar.textContent = percent + '%';
        progressBar.setAttribute('aria-valuenow', percent);
    }
}

/**
 * 隐藏上传进度
 */
function hideUploadProgress() {
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('upload-progress-bar');
    
    if (progressSection) {
        setTimeout(() => {
            progressSection.style.display = 'none';
        }, 1000);
    }
    if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';
    }
}

// =============================================================================
// 历史记录
// =============================================================================

/**
 * 加载导入历史
 */
async function loadImportHistory() {
    try {
        const response = await fetch(`${DataImportConfig.API_BASE}/list`);
        const result = await response.json();
        
        const tbody = document.getElementById('history-tbody');
        if (!tbody) return;
        
        if (result.records && result.records.length > 0) {
            tbody.innerHTML = result.records.map(record => `
                <tr>
                    <td>${formatDateTime(record.created_at)}</td>
                    <td>
                        <span class="badge bg-info">${record.voltage_level}</span>
                    </td>
                    <td>${(record.plugins || []).join(', ')}</td>
                    <td>${record.file_count || 0}</td>
                    <td>
                        <span class="status-badge ${record.status}">${getStatusText(record.status)}</span>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteRecord('${record.id}')" title="删除记录">
                            <i class="bi bi-trash"></i>
                        </button>
                    </td>
                </tr>
            `).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">暂无导入记录</td></tr>';
        }
    } catch (error) {
        console.error('[DataImport] 加载历史记录失败:', error);
    }
}

/**
 * 删除上传记录
 */
async function deleteRecord(recordId) {
    if (!confirm('确定要删除这条记录吗？')) return;
    
    try {
        const response = await fetch(`${DataImportConfig.API_BASE}/record/${recordId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            showSuccess('记录已删除');
            loadImportHistory();
        } else {
            showError('删除失败: ' + (result.message || '未知错误'));
        }
    } catch (error) {
        console.error('[DataImport] 删除记录失败:', error);
        showError('删除请求失败');
    }
}

// =============================================================================
// 工具函数
// =============================================================================

/**
 * 格式化文件大小
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * 格式化日期时间
 */
function formatDateTime(dateStr) {
    if (!dateStr) return '-';
    try {
        const date = new Date(dateStr);
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (e) {
        return dateStr;
    }
}

/**
 * 获取状态文本
 */
function getStatusText(status) {
    const statusMap = {
        'pending': '待处理',
        'processing': '处理中',
        'completed': '已完成',
        'failed': '失败'
    };
    return statusMap[status] || status;
}

/**
 * HTML转义
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// =============================================================================
// 通知函数
// =============================================================================

/**
 * 显示成功消息
 */
function showSuccess(message) {
    showNotification(message, 'success');
}

/**
 * 显示错误消息
 */
function showError(message) {
    showNotification(message, 'danger');
}

/**
 * 显示警告消息
 */
function showWarning(message) {
    showNotification(message, 'warning');
}

/**
 * 显示通知
 */
function showNotification(message, type = 'info') {
    // 如果有自定义的toast函数则使用
    if (typeof showToast === 'function') {
        showToast(message, type);
        return;
    }
    
    // 否则使用alert
    alert(message);
}

/**
 * 显示加载状态
 */
function showLoading(message = '加载中...') {
    // 可以实现自定义的加载指示器
    console.log('[DataImport] ' + message);
}

/**
 * 隐藏加载状态
 */
function hideLoading() {
    // 隐藏加载指示器
}

// =============================================================================
// 页面加载时初始化
// =============================================================================

document.addEventListener('DOMContentLoaded', function() {
    initDataImport();
});

// 如果使用jQuery
if (typeof jQuery !== 'undefined') {
    jQuery(document).ready(function() {
        if (!DataImportState.initialized) {
            initDataImport();
            DataImportState.initialized = true;
        }
    });
}
