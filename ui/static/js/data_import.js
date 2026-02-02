/**
 * 数据导入模块 V2.0 - 完整前端实现
 * ===================================
 * 
 * 对应后端: apps/data_upload_api.py
 * 
 * 功能清单:
 *   1. 拖拽/选择文件上传 (直接上传, 支持压缩包)
 *   2. 分片断点续传 (大文件 > 100MB)
 *   3. 上传前数据预验证
 *   4. 上传后自动触发后台校验 (图片-标签对齐, 格式探测)
 *   5. 服务器端扫描导入 (超大数据集 > 2GB)
 *   6. 导入历史列表 (带状态徽章)
 *   7. 数据集删除
 *   8. 数据聚合与训练拆分
 * 
 * 修改说明 (相比演示版本):
 *   - 真正调用后端 API, 不再是空壳
 *   - 新增分片上传逻辑
 *   - 新增服务器端导入面板
 *   - 新增校验报告展示
 *   - 上传后自动轮询处理状态
 */

// =============================================================================
// 全局状态
// =============================================================================

const DataImportState = {
    files: [],
    selectedVoltage: '',
    selectedPlugins: [],
    uploadInProgress: false,
    // 分片上传阈值 (100MB 以上走分片)
    CHUNK_THRESHOLD: 100 * 1024 * 1024,
    // 分片大小 (5MB)
    CHUNK_SIZE: 5 * 1024 * 1024,
    // 上传者名 (可从用户系统获取)
    uploader: 'default_user',
};

// =============================================================================
// 初始化
// =============================================================================

document.addEventListener('DOMContentLoaded', function () {
    initVoltageSelector();
    initPluginSelector();
    initUploadZone();
    loadImportHistory();

    // 检查URL参数 (支持外部跳转带参数)
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('voltage')) {
        DataImportState.selectedVoltage = urlParams.get('voltage');
    }

    console.log('[DataImport V2.0] 初始化完成');
});

// =============================================================================
// 电压等级选择
// =============================================================================

function initVoltageSelector() {
    const voltageCards = document.querySelectorAll('.voltage-card');
    voltageCards.forEach(card => {
        card.addEventListener('click', function () {
            voltageCards.forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            DataImportState.selectedVoltage = this.dataset.voltage;
            console.log('[DataImport] 选择电压:', DataImportState.selectedVoltage);
        });
    });
}

// =============================================================================
// 插件选择
// =============================================================================

function initPluginSelector() {
    const pluginCheckboxes = document.querySelectorAll('input[name="plugin"]');
    pluginCheckboxes.forEach(cb => {
        cb.addEventListener('change', function () {
            DataImportState.selectedPlugins = Array.from(
                document.querySelectorAll('input[name="plugin"]:checked')
            ).map(el => el.value);
            console.log('[DataImport] 选择插件:', DataImportState.selectedPlugins);
        });
    });
}

// =============================================================================
// 上传区域 (拖拽 + 点击)
// =============================================================================

function initUploadZone() {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');

    if (!uploadZone || !fileInput) return;

    // 拖放事件
    uploadZone.addEventListener('dragover', function (e) {
        e.preventDefault();
        this.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', function (e) {
        e.preventDefault();
        this.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', function (e) {
        e.preventDefault();
        this.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    // 点击上传
    uploadZone.addEventListener('click', function () {
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
        this.value = '';
    });
}

// =============================================================================
// 文件处理
// =============================================================================

function handleFiles(fileList) {
    const allowedExtensions = [
        // 图片
        '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.ppm', '.webp',
        // 视频
        '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg',
        // 音频
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus',
        // 标注
        '.txt', '.xml', '.json',
        // 压缩包
        '.zip', '.tar', '.gz', '.tgz', '.rar',
    ];

    let addedCount = 0;
    Array.from(fileList).forEach(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (allowedExtensions.includes(ext) || file.name.endsWith('.tar.gz')) {
            // 检查重复
            if (!DataImportState.files.find(f => f.name === file.name && f.size === file.size)) {
                DataImportState.files.push(file);
                addedCount++;
                console.log('[DataImport] 添加文件:', file.name, `(${(file.size / 1024 / 1024).toFixed(2)} MB)`);
            }
        } else {
            console.warn('[DataImport] 不支持的文件类型:', file.name);
            alert(`不支持的文件类型: ${file.name}\n请上传图片、视频、音频、标注文件或压缩包`);
        }
    });

    if (addedCount > 0) {
        renderFileList();
        updateSummary();
    }
}

// =============================================================================
// 文件列表渲染
// =============================================================================

function renderFileList() {
    const container = document.getElementById('file-list');
    if (!container) return;

    container.innerHTML = '';

    if (DataImportState.files.length === 0) return;

    DataImportState.files.forEach((file, index) => {
        const ext = file.name.split('.').pop().toLowerCase();
        let icon = 'bi-file-earmark';
        let iconColor = '#6c757d';

        // 图片
        if (['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'ppm', 'webp'].includes(ext)) {
            icon = 'bi-file-earmark-image';
            iconColor = '#28a745';
        }
        // 视频
        else if (['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'm4v', 'mpeg', 'mpg'].includes(ext)) {
            icon = 'bi-file-earmark-play';
            iconColor = '#007bff';
        }
        // 音频
        else if (['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma', 'opus'].includes(ext)) {
            icon = 'bi-file-earmark-music';
            iconColor = '#17a2b8';
        }
        // 标注
        else if (['txt', 'xml', 'json'].includes(ext)) {
            icon = 'bi-file-earmark-text';
            iconColor = '#fd7e14';
        }
        // 压缩包
        else if (['zip', 'tar', 'gz', 'tgz', 'rar'].includes(ext) || file.name.endsWith('.tar.gz')) {
            icon = 'bi-file-earmark-zip';
            iconColor = '#6f42c1';
        }

        const html = `
            <div class="file-item" data-index="${index}">
                <div class="file-info">
                    <i class="bi ${icon} file-icon" style="color: ${iconColor}"></i>
                    <div>
                        <div class="file-name">${escapeHtml(file.name)}</div>
                        <div class="file-size">${formatFileSize(file.size)}</div>
                    </div>
                </div>
                <span class="btn-remove" onclick="removeFile(${index})" title="移除">
                    <i class="bi bi-x-circle"></i>
                </span>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', html);
    });
}

function removeFile(index) {
    DataImportState.files.splice(index, 1);
    renderFileList();
    updateSummary();
}

function clearAllFiles() {
    DataImportState.files = [];
    renderFileList();
    updateSummary();
    const summaryEl = document.getElementById('data-summary');
    if (summaryEl) summaryEl.style.display = 'none';
}

// =============================================================================
// 数据摘要
// =============================================================================

function updateSummary() {
    let images = 0, videos = 0, labels = 0, archives = 0, audios = 0, totalSize = 0;

    DataImportState.files.forEach(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        totalSize += file.size;

        // 图片
        if (['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'ppm', 'webp'].includes(ext)) {
            images++;
        }
        // 视频
        else if (['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'm4v', 'mpeg', 'mpg'].includes(ext)) {
            videos++;
        }
        // 音频
        else if (['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a', 'wma', 'opus'].includes(ext)) {
            audios++;
        }
        // 标注
        else if (['txt', 'xml', 'json'].includes(ext)) {
            labels++;
        }
        // 压缩包
        else if (['zip', 'tar', 'gz', 'tgz', 'rar'].includes(ext) || file.name.endsWith('.tar.gz')) {
            archives++;
        }
    });

    const el = id => document.getElementById(id);
    if (el('summary-images')) el('summary-images').textContent = images;
    if (el('summary-videos')) el('summary-videos').textContent = videos;
    if (el('summary-labels')) el('summary-labels').textContent = labels;
    if (el('summary-archives')) el('summary-archives').textContent = archives;
    if (el('summary-size')) el('summary-size').textContent = (totalSize / (1024 * 1024)).toFixed(2);

    const summaryEl = document.getElementById('data-summary');
    if (summaryEl) {
        summaryEl.style.display = DataImportState.files.length > 0 ? 'block' : 'none';
    }
}

// =============================================================================
// 数据验证 (上传前预检)
// =============================================================================

async function validateData() {
    if (DataImportState.files.length === 0) {
        alert('请先选择要上传的文件');
        return;
    }

    if (DataImportState.selectedPlugins.length === 0) {
        alert('请选择至少一个巡视功能');
        return;
    }

    const fileInfo = DataImportState.files.map(f => ({
        name: f.name,
        size: f.size,
        type: f.type,
    }));

    try {
        const response = await fetch('/api/training/data/validate-preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
                voltage_level: DataImportState.selectedVoltage,
                plugins: JSON.stringify(DataImportState.selectedPlugins),
                files: JSON.stringify(fileInfo),
            }),
        });

        const result = await response.json();

        if (result.valid) {
            alert('✅ 数据验证通过！\n' +
                  `图片: ${result.summary.images}, 标注: ${result.summary.labels}, ` +
                  `压缩包: ${result.summary.archives}\n\n可以进行上传。`);
        } else {
            alert('⚠️ 数据验证警告：\n' + (result.message || '请检查数据'));
        }
    } catch (error) {
        console.error('[DataImport] 验证失败:', error);
        alert('验证请求失败，请检查网络连接');
    }
}

// =============================================================================
// 上传逻辑 (核心)
// =============================================================================

async function uploadAndSave() {
    if (DataImportState.files.length === 0) {
        alert('请先选择要上传的文件');
        return;
    }

    if (!DataImportState.selectedVoltage) {
        alert('请选择电压等级');
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
    const btnUpload = document.getElementById('btn-upload');
    if (btnUpload) btnUpload.disabled = true;

    // 判断是否需要分片上传 (单文件 > 100MB)
    const totalSize = DataImportState.files.reduce((sum, f) => sum + f.size, 0);
    const hasLargeFile = DataImportState.files.some(f => f.size > DataImportState.CHUNK_THRESHOLD);

    try {
        if (hasLargeFile && DataImportState.files.length === 1) {
            // 单个大文件 -> 分片上传
            await doChunkUpload(DataImportState.files[0]);
        } else {
            // 普通上传
            await doDirectUpload();
        }
    } catch (error) {
        console.error('[DataImport] 上传失败:', error);
        alert('上传失败: ' + (error.message || '未知错误'));
    } finally {
        DataImportState.uploadInProgress = false;
        if (btnUpload) btnUpload.disabled = false;
    }
}

// ---------- 普通直接上传 ----------

async function doDirectUpload() {
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('upload-progress-bar');
    if (progressSection) progressSection.style.display = 'block';

    const formData = new FormData();
    formData.append('voltage_level', DataImportState.selectedVoltage);
    formData.append('plugins', JSON.stringify(DataImportState.selectedPlugins));
    formData.append('uploader', DataImportState.uploader);
    formData.append('dataset_name', generateDatasetName());

    DataImportState.files.forEach(file => {
        formData.append('files', file);
    });

    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', function (e) {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                if (progressBar) {
                    progressBar.style.width = percent + '%';
                    progressBar.textContent = percent + '%';
                }
            }
        });

        xhr.addEventListener('load', function () {
            if (progressSection) progressSection.style.display = 'none';
            if (progressBar) progressBar.style.width = '0%';

            if (xhr.status === 200) {
                try {
                    const result = JSON.parse(xhr.responseText);
                    if (result.success) {
                        alert('✅ 数据上传成功！\n' +
                              `已接收 ${result.file_count} 个文件\n` +
                              '后台正在进行解压与校验...');
                        clearAllFiles();
                        loadImportHistory();

                        // 轮询处理状态
                        if (result.results) {
                            result.results.forEach(r => {
                                if (r.dataset_id) {
                                    pollDatasetStatus(r.dataset_id);
                                }
                            });
                        }
                        resolve(result);
                    } else {
                        alert('上传失败：' + (result.message || '未知错误'));
                        reject(new Error(result.message));
                    }
                } catch (e) {
                    reject(e);
                }
            } else {
                alert('上传失败：服务器错误 ' + xhr.status);
                reject(new Error('Server error: ' + xhr.status));
            }
        });

        xhr.addEventListener('error', function () {
            if (progressSection) progressSection.style.display = 'none';
            alert('上传失败：网络错误');
            reject(new Error('Network error'));
        });

        xhr.open('POST', '/api/training/data/upload');
        xhr.send(formData);
    });
}

// ---------- 分片上传 (大文件) ----------

async function doChunkUpload(file) {
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('upload-progress-bar');
    if (progressSection) progressSection.style.display = 'block';

    const totalChunks = Math.ceil(file.size / DataImportState.CHUNK_SIZE);

    // Step 1: 初始化会话
    const initResp = await fetch('/api/training/data/chunk/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            file_name: file.name,
            file_size: file.size,
            total_chunks: totalChunks,
            file_md5: '',
            voltage_level: DataImportState.selectedVoltage,
            plugin_type: DataImportState.selectedPlugins[0],
            uploader: DataImportState.uploader,
        }),
    });
    const initData = await initResp.json();
    if (!initData.success) {
        throw new Error('初始化分片上传失败: ' + initData.message);
    }
    const sessionId = initData.session_id;

    // Step 2: 逐个上传分片
    for (let i = 0; i < totalChunks; i++) {
        const start = i * DataImportState.CHUNK_SIZE;
        const end = Math.min(start + DataImportState.CHUNK_SIZE, file.size);
        const blob = file.slice(start, end);

        const chunkForm = new FormData();
        chunkForm.append('session_id', sessionId);
        chunkForm.append('chunk_index', i.toString());
        chunkForm.append('chunk_md5', '');
        chunkForm.append('chunk', blob, `chunk_${i}`);

        const chunkResp = await fetch('/api/training/data/chunk/upload', {
            method: 'POST',
            body: chunkForm,
        });
        const chunkData = await chunkResp.json();

        if (!chunkData.success) {
            throw new Error(`分片 ${i} 上传失败`);
        }

        // 更新进度
        const percent = Math.round(((i + 1) / totalChunks) * 100);
        if (progressBar) {
            progressBar.style.width = percent + '%';
            progressBar.textContent = `${percent}% (${i + 1}/${totalChunks})`;
        }
    }

    // Step 3: 合并分片
    if (progressBar) progressBar.textContent = '合并中...';

    const mergeResp = await fetch('/api/training/data/chunk/merge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
    });
    const mergeData = await mergeResp.json();

    if (progressSection) progressSection.style.display = 'none';

    if (mergeData.success) {
        alert('✅ 大文件上传成功！\n后台正在解压与校验...');
        clearAllFiles();
        loadImportHistory();
        if (mergeData.dataset_id) {
            pollDatasetStatus(mergeData.dataset_id);
        }
    } else {
        throw new Error('分片合并失败: ' + (mergeData.message || ''));
    }
}

// =============================================================================
// 轮询数据集处理状态
// =============================================================================

function pollDatasetStatus(datasetId, maxRetries = 30) {
    let retries = 0;

    const interval = setInterval(async () => {
        retries++;
        if (retries > maxRetries) {
            clearInterval(interval);
            return;
        }

        try {
            const resp = await fetch(`/api/training/data/status/${datasetId}`);
            const data = await resp.json();

            const taskStatus = data.task_status || {};
            const status = taskStatus.status;

            if (status === 'completed' || status === 'error') {
                clearInterval(interval);
                loadImportHistory();

                if (status === 'completed') {
                    const report = taskStatus.validation_report || {};
                    console.log('[DataImport] 处理完成:', report);
                    showValidationToast(datasetId, report);
                } else {
                    console.warn('[DataImport] 处理出错:', taskStatus.message);
                }
            }
        } catch (e) {
            console.error('[DataImport] 状态轮询失败:', e);
        }
    }, 3000);  // 每 3 秒轮询一次
}

function showValidationToast(datasetId, report) {
    const valid = report.valid;
    const msg = valid
        ? `✅ 数据集校验通过\n图片: ${report.image_count}, 标签: ${report.label_count}, 格式: ${report.format}`
        : `⚠️ 数据集校验有问题\n${(report.warnings || []).concat(report.errors || []).join('\n')}`;

    // 使用简单的 toast 通知
    if (typeof showToast === 'function') {
        showToast(msg, valid ? 'success' : 'warning');
    } else {
        console.log(msg);
    }
}

// =============================================================================
// 服务器端扫描导入 (策略B)
// =============================================================================

async function scanLocalImport() {
    if (!DataImportState.selectedVoltage) {
        alert('请先选择电压等级');
        return;
    }
    if (DataImportState.selectedPlugins.length === 0) {
        alert('请选择至少一个插件类型');
        return;
    }

    const confirmed = confirm(
        '将扫描服务器 training/data/temp_upload/ 目录下的压缩包并导入。\n\n' +
        '请确认已将数据文件拷贝到该目录。\n\n继续？'
    );
    if (!confirmed) return;

    try {
        const resp = await fetch('/api/training/data/import/local', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                voltage_level: DataImportState.selectedVoltage,
                plugin_type: DataImportState.selectedPlugins[0],
                uploader: DataImportState.uploader,
            }),
        });

        const result = await resp.json();

        if (result.success) {
            alert(`✅ 扫描导入完成\n发现 ${result.imported_count} 个文件，已加入处理队列。`);
            loadImportHistory();

            // 轮询每个导入的数据集状态
            (result.results || []).forEach(r => {
                if (r.dataset_id) pollDatasetStatus(r.dataset_id);
            });
        } else {
            alert('导入失败: ' + (result.message || ''));
        }
    } catch (error) {
        console.error('[DataImport] 扫描导入失败:', error);
        alert('扫描导入请求失败');
    }
}

// =============================================================================
// 数据聚合 (模块四 - 前端触发)
// =============================================================================

async function aggregateDatasets() {
    if (!DataImportState.selectedVoltage || DataImportState.selectedPlugins.length === 0) {
        alert('请先选择电压等级和插件类型');
        return;
    }

    try {
        const resp = await fetch('/api/training/data/aggregate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                voltage_level: DataImportState.selectedVoltage,
                plugin_type: DataImportState.selectedPlugins[0],
            }),
        });

        const result = await resp.json();

        if (result.success) {
            alert(
                `✅ 数据聚合完成\n\n` +
                `数据集数量: ${result.dataset_count}\n` +
                `总图片数: ${result.total_images}\n` +
                `总标签数: ${result.total_labels}\n` +
                `YAML路径: ${result.yaml_path}\n\n` +
                `可用于训练。`
            );
        } else {
            alert('聚合失败: ' + (result.message || ''));
        }
    } catch (error) {
        console.error('[DataImport] 聚合失败:', error);
        alert('聚合请求失败');
    }
}

// =============================================================================
// 导入历史
// =============================================================================

async function loadImportHistory() {
    try {
        const response = await fetch('/api/training/data/list');
        const result = await response.json();

        const tbody = document.getElementById('history-tbody');
        if (!tbody) return;

        if (result.records && result.records.length > 0) {
            tbody.innerHTML = result.records.map(record => {
                const statusClass = getStatusClass(record.status);
                const statusText = getStatusText(record.status);
                const formatBadge = record.format && record.format !== 'UNKNOWN'
                    ? `<span class="badge bg-info">${record.format}</span>` : '';

                return `
                <tr>
                    <td>${formatDateTime(record.created_at)}</td>
                    <td>${record.voltage_level || '-'}</td>
                    <td>${(record.plugins || []).join(', ')}</td>
                    <td>
                        <span title="图片: ${record.image_count || 0}, 标注: ${record.label_count || 0}">
                            ${record.image_count || 0} 图 / ${record.label_count || 0} 标
                        </span>
                    </td>
                    <td>
                        <span class="status-badge ${statusClass}">${statusText}</span>
                        ${formatBadge}
                    </td>
                    <td>${record.uploader || '-'}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-info me-1" 
                                onclick="viewDatasetDetail('${record.id}')" 
                                title="查看详情">
                            <i class="bi bi-eye"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-warning me-1" 
                                onclick="revalidateDataset('${record.id}')" 
                                title="重新校验">
                            <i class="bi bi-arrow-repeat"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-danger" 
                                onclick="deleteDataset('${record.id}')" 
                                title="删除">
                            <i class="bi bi-trash"></i>
                        </button>
                    </td>
                </tr>`;
            }).join('');
        } else {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center text-muted py-4">
                        <i class="bi bi-inbox" style="font-size: 2rem;"></i>
                        <p class="mt-2 mb-0">暂无导入记录</p>
                    </td>
                </tr>`;
        }
    } catch (error) {
        console.error('[DataImport] 加载历史失败:', error);
    }
}

// =============================================================================
// 数据集操作
// =============================================================================

async function deleteDataset(datasetId) {
    if (!confirm('确定要删除该数据集吗？此操作不可恢复。')) return;

    try {
        const response = await fetch(`/api/training/data/${datasetId}`, {
            method: 'DELETE',
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

async function revalidateDataset(datasetId) {
    try {
        const resp = await fetch('/api/training/data/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: datasetId }),
        });
        const result = await resp.json();

        if (result.valid) {
            alert('✅ 重新校验通过');
        } else {
            const report = result.report || {};
            const issues = (report.warnings || []).concat(report.errors || []);
            alert('⚠️ 校验结果:\n' + issues.join('\n'));
        }
        loadImportHistory();
    } catch (error) {
        console.error('[DataImport] 校验失败:', error);
        alert('校验请求失败');
    }
}

async function viewDatasetDetail(datasetId) {
    try {
        const resp = await fetch(`/api/training/data/status/${datasetId}`);
        const data = await resp.json();

        const meta = data.metadata || {};
        const report = meta.validation_report || {};

        let detail = `数据集详情: ${datasetId}\n\n`;
        detail += `上传者: ${meta.uploader || '-'}\n`;
        detail += `上传时间: ${meta.upload_time || '-'}\n`;
        detail += `电压等级: ${meta.voltage_level || '-'}\n`;
        detail += `插件类型: ${meta.plugin_type || '-'}\n`;
        detail += `状态: ${meta.status || '-'}\n`;
        detail += `图片数量: ${meta.image_count || 0}\n`;
        detail += `标注数量: ${meta.label_count || 0}\n`;
        detail += `标注格式: ${meta.format || 'UNKNOWN'}\n`;

        if (report.match_ratio !== undefined) {
            detail += `\n--- 校验报告 ---\n`;
            detail += `匹配率: ${(report.match_ratio * 100).toFixed(1)}%\n`;
            detail += `缺失标注: ${report.missing_labels || 0}\n`;
            detail += `孤立标注: ${report.orphan_labels || 0}\n`;
            detail += `清理垃圾文件: ${report.garbage_removed || 0}\n`;
            if (report.warnings && report.warnings.length > 0) {
                detail += `警告: ${report.warnings.join('; ')}\n`;
            }
            if (report.errors && report.errors.length > 0) {
                detail += `错误: ${report.errors.join('; ')}\n`;
            }
        }

        detail += `\n路径: ${meta.data_path || '-'}`;

        alert(detail);
    } catch (error) {
        console.error('[DataImport] 获取详情失败:', error);
        alert('获取详情失败');
    }
}

// =============================================================================
// 工具函数
// =============================================================================

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDateTime(isoString) {
    if (!isoString) return '-';
    const date = new Date(isoString);
    return date.toLocaleString('zh-CN', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit',
    });
}

function getStatusText(status) {
    const map = {
        'verified': '✅ 已验证',
        'warning': '⚠️ 有警告',
        'error': '❌ 错误',
        'pending': '⏳ 处理中',
        'validating': '🔍 校验中',
        'archived': '📦 已归档',
    };
    return map[status] || status || '-';
}

function getStatusClass(status) {
    const map = {
        'verified': 'success',
        'warning': 'warning',
        'error': 'error',
        'pending': 'pending',
        'validating': 'pending',
        'archived': 'success',
    };
    return map[status] || 'pending';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function generateDatasetName() {
    const now = new Date();
    return `batch_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`;
}
