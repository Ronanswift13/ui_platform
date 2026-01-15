/**
 * 数据导入模块 JavaScript
 * 输变电激光星芒破夜绘明监测平台 V3.0
 */

// 全局状态
const DataImportState = {
    selectedVoltage: '220kv',
    selectedPlugins: ['transformer'],
    files: [],
    uploadInProgress: false
};

/**
 * 初始化数据导入模块
 */
function initDataImport() {
    console.log('[DataImport] 初始化数据导入模块');

    // 绑定电压等级选择
    bindVoltageSelector();

    // 绑定插件选择
    bindPluginSelector();

    // 绑定文件上传
    bindFileUpload();

    // 加载历史记录
    loadImportHistory();

    console.log('[DataImport] 初始化完成');
}

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
        if (input.checked) {
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

/**
 * 绑定文件上传区域
 */
function bindFileUpload() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');

    // 拖放事件
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

    // 点击上传
    uploadZone.addEventListener('click', function() {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
        this.value = ''; // 清除以便重复选择
    });
}

/**
 * 处理选择的文件
 */
function handleFiles(fileList) {
    const allowedTypes = ['image/jpeg', 'image/png', 'video/mp4', 'video/avi', 'text/plain', 'application/json', 'application/xml'];
    const allowedExtensions = ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.txt', '.xml', '.json'];

    Array.from(fileList).forEach(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (allowedExtensions.includes(ext)) {
            // 检查是否已存在
            if (!DataImportState.files.find(f => f.name === file.name && f.size === file.size)) {
                DataImportState.files.push(file);
            }
        } else {
            console.warn('[DataImport] 不支持的文件类型:', file.name);
        }
    });

    renderFileList();
    updateSummary();
}

/**
 * 渲染文件列表
 */
function renderFileList() {
    const container = document.getElementById('file-list');
    container.innerHTML = '';

    if (DataImportState.files.length === 0) {
        return;
    }

    DataImportState.files.forEach((file, index) => {
        const ext = file.name.split('.').pop().toLowerCase();
        let iconClass = 'bi-file-earmark';
        let iconColor = 'text-secondary';

        if (['jpg', 'jpeg', 'png'].includes(ext)) {
            iconClass = 'bi-file-earmark-image';
            iconColor = 'text-success';
        } else if (['mp4', 'avi'].includes(ext)) {
            iconClass = 'bi-file-earmark-play';
            iconColor = 'text-primary';
        } else if (['txt', 'xml', 'json'].includes(ext)) {
            iconClass = 'bi-file-earmark-text';
            iconColor = 'text-warning';
        }

        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <i class="file-icon bi ${iconClass} ${iconColor}"></i>
                <div>
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
            </div>
            <div>
                <span class="file-status text-muted">待上传</span>
                <i class="btn-remove bi bi-x-circle" onclick="removeFile(${index})"></i>
            </div>
        `;
        container.appendChild(fileItem);
    });
}

/**
 * 移除文件
 */
function removeFile(index) {
    DataImportState.files.splice(index, 1);
    renderFileList();
    updateSummary();
}

/**
 * 清除所有文件
 */
function clearAllFiles() {
    DataImportState.files = [];
    renderFileList();
    updateSummary();
    document.getElementById('data-summary').style.display = 'none';
}

/**
 * 更新数据摘要
 */
function updateSummary() {
    const files = DataImportState.files;

    if (files.length === 0) {
        document.getElementById('data-summary').style.display = 'none';
        return;
    }

    let images = 0, videos = 0, labels = 0, totalSize = 0;

    files.forEach(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        totalSize += file.size;

        if (['jpg', 'jpeg', 'png'].includes(ext)) images++;
        else if (['mp4', 'avi'].includes(ext)) videos++;
        else if (['txt', 'xml', 'json'].includes(ext)) labels++;
    });

    document.getElementById('summary-images').textContent = images;
    document.getElementById('summary-videos').textContent = videos;
    document.getElementById('summary-labels').textContent = labels;
    document.getElementById('summary-size').textContent = (totalSize / (1024 * 1024)).toFixed(2);

    document.getElementById('data-summary').style.display = 'block';
}

/**
 * 验证数据
 */
async function validateData() {
    if (DataImportState.files.length === 0) {
        alert('请先选择要上传的文件');
        return;
    }

    if (DataImportState.selectedPlugins.length === 0) {
        alert('请选择至少一个巡视功能');
        return;
    }

    // 构建验证请求
    const fileInfo = DataImportState.files.map(f => ({
        name: f.name,
        size: f.size,
        type: f.type
    }));

    try {
        const response = await fetch('/api/training/data/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                voltage_level: DataImportState.selectedVoltage,
                plugins: DataImportState.selectedPlugins,
                files: fileInfo
            })
        });

        const result = await response.json();

        if (result.valid) {
            alert('数据验证通过！\n' + (result.message || '可以进行上传'));
        } else {
            alert('数据验证失败：\n' + (result.message || '请检查数据格式'));
        }
    } catch (error) {
        console.error('[DataImport] 验证失败:', error);
        alert('验证请求失败，请检查网络连接');
    }
}

/**
 * 上传并保存数据
 */
async function uploadAndSave() {
    if (DataImportState.files.length === 0) {
        alert('请先选择要上传的文件');
        return;
    }

    if (DataImportState.selectedPlugins.length === 0) {
        alert('请选择至少一个巡视功能');
        return;
    }

    if (DataImportState.uploadInProgress) {
        alert('上传正在进行中，请稍候');
        return;
    }

    DataImportState.uploadInProgress = true;
    document.getElementById('btn-upload').disabled = true;

    // 显示进度条
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('upload-progress-bar');
    progressSection.style.display = 'block';

    const formData = new FormData();
    formData.append('voltage_level', DataImportState.selectedVoltage);
    formData.append('plugins', JSON.stringify(DataImportState.selectedPlugins));

    DataImportState.files.forEach((file, index) => {
        formData.append('files', file);
    });

    try {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', function(e) {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                progressBar.style.width = percent + '%';
                progressBar.textContent = percent + '%';
            }
        });

        xhr.addEventListener('load', function() {
            if (xhr.status === 200) {
                const result = JSON.parse(xhr.responseText);
                if (result.success) {
                    alert('数据上传成功！\n已保存到训练数据目录');
                    clearAllFiles();
                    loadImportHistory();
                } else {
                    alert('上传失败：' + (result.message || '未知错误'));
                }
            } else {
                alert('上传失败：服务器错误 ' + xhr.status);
            }

            DataImportState.uploadInProgress = false;
            document.getElementById('btn-upload').disabled = false;
            progressSection.style.display = 'none';
            progressBar.style.width = '0%';
        });

        xhr.addEventListener('error', function() {
            alert('上传失败：网络错误');
            DataImportState.uploadInProgress = false;
            document.getElementById('btn-upload').disabled = false;
            progressSection.style.display = 'none';
        });

        xhr.open('POST', '/api/training/data/upload');
        xhr.send(formData);

    } catch (error) {
        console.error('[DataImport] 上传失败:', error);
        alert('上传请求失败');
        DataImportState.uploadInProgress = false;
        document.getElementById('btn-upload').disabled = false;
        progressSection.style.display = 'none';
    }
}

/**
 * 加载导入历史
 */
async function loadImportHistory() {
    try {
        const response = await fetch('/api/training/data/list');
        const result = await response.json();

        const tbody = document.getElementById('history-tbody');

        if (result.records && result.records.length > 0) {
            tbody.innerHTML = result.records.map(record => `
                <tr>
                    <td>${formatDateTime(record.created_at)}</td>
                    <td>${record.voltage_level}</td>
                    <td>${record.plugins.join(', ')}</td>
                    <td>${record.file_count}</td>
                    <td>
                        <span class="status-badge ${record.status}">${getStatusText(record.status)}</span>
                    </td>
                    <td>
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteDataset('${record.id}')">
                            <i class="bi bi-trash"></i>
                        </button>
                    </td>
                </tr>
            `).join('');
        } else {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">暂无导入记录</td></tr>';
        }
    } catch (error) {
        console.error('[DataImport] 加载历史失败:', error);
    }
}

/**
 * 删除数据集
 */
async function deleteDataset(datasetId) {
    if (!confirm('确定要删除该数据集吗？此操作不可恢复。')) {
        return;
    }

    try {
        const response = await fetch(`/api/training/data/${datasetId}`, {
            method: 'DELETE'
        });

        const result = await response.json();

        if (result.success) {
            alert('数据集已删除');
            loadImportHistory();
        } else {
            alert('删除失败：' + (result.message || '未知错误'));
        }
    } catch (error) {
        console.error('[DataImport] 删除失败:', error);
        alert('删除请求失败');
    }
}

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
function formatDateTime(isoString) {
    if (!isoString) return '-';
    const date = new Date(isoString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * 获取状态文本
 */
function getStatusText(status) {
    const statusMap = {
        'success': '已完成',
        'pending': '处理中',
        'error': '失败'
    };
    return statusMap[status] || status;
}
