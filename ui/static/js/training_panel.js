/**
 * 破夜绘明激光监测平台 - 训练控制面板脚本
 * 处理训练任务管理、数据下载、进度显示等功能
 */

// 训练管理器
const TrainingPanel = {
    // 当前配置
    currentPlugin: null,
    currentVoltageLevel: 'HV_220kV',
    currentTaskId: null,
    pollingInterval: null,
    downloadPollingInterval: null,
    
    // 公开数据集映射
    datasetMapping: {
        'transformer': ['transformer_thermal'],
        'switch': ['switch_indicator'],
        'busbar': ['insulator_defect', 'mpid_insulator'],
        'capacitor': ['capacitor_structure'],
        'meter': ['ufpr_amr']
    },

    /**
     * 初始化训练面板
     * @param {string} pluginId - 插件ID
     */
    init: function(pluginId) {
        this.currentPlugin = pluginId;
        console.log('初始化训练面板:', pluginId);
        
        // 绑定事件
        this.bindEvents();
        
        // 加载初始状态
        this.loadStatus();
        
        // 加载可用数据集
        this.loadAvailableDatasets();
    },

    /**
     * 绑定事件处理
     */
    bindEvents: function() {
        const self = this;
        
        // 电压等级选择
        const voltageSelect = document.getElementById('training-voltage-select');
        if (voltageSelect) {
            voltageSelect.addEventListener('change', function() {
                self.currentVoltageLevel = this.value;
                self.loadStatus();
            });
        }
        
        // 开始训练按钮
        const btnStart = document.getElementById('btn-start-training');
        if (btnStart) {
            btnStart.addEventListener('click', function() {
                self.startTraining();
            });
        }
        
        // 取消训练按钮
        const btnCancel = document.getElementById('btn-cancel-training');
        if (btnCancel) {
            btnCancel.addEventListener('click', function() {
                self.cancelTraining();
            });
        }
        
        // 使用模型按钮
        const btnUse = document.getElementById('btn-use-model');
        if (btnUse) {
            btnUse.addEventListener('click', function() {
                self.useModel();
            });
        }
        
        // 重新训练按钮
        const btnRetrain = document.getElementById('btn-retrain');
        if (btnRetrain) {
            btnRetrain.addEventListener('click', function() {
                document.getElementById('trained-model-section').style.display = 'none';
                document.getElementById('btn-start-training').disabled = false;
            });
        }
    },

    /**
     * 加载训练状态
     */
    loadStatus: async function() {
        if (!this.currentPlugin || !this.currentVoltageLevel) {
            return;
        }
        
        try {
            const response = await fetch(`/api/training/status/${this.currentVoltageLevel}/${this.currentPlugin}`);
            const data = await response.json();
            
            if (data.success) {
                this.updateDatasetStatus(data.dataset);
                this.updateTaskStatus(data.task);
                this.updateCompletedModel(data.completed_model);
            }
        } catch (error) {
            console.error('加载训练状态失败:', error);
        }
    },

    /**
     * 更新数据集状态显示
     */
    updateDatasetStatus: function(dataset) {
        const statusCard = document.getElementById('dataset-status-card');
        const statusText = document.getElementById('dataset-status-text');
        const statusBadge = document.getElementById('dataset-status-badge');
        const trainCount = document.getElementById('train-count');
        const valCount = document.getElementById('val-count');
        const testCount = document.getElementById('test-count');
        const btnStart = document.getElementById('btn-start-training');
        const placeholderWarning = document.getElementById('placeholder-warning');
        
        if (!dataset) return;
        
        // 更新计数
        trainCount.textContent = dataset.train_count || 0;
        valCount.textContent = dataset.val_count || 0;
        testCount.textContent = dataset.test_count || 0;
        
        // 更新状态
        statusCard.classList.remove('ready', 'placeholder', 'downloading');
        
        switch (dataset.status) {
            case 'ready':
                statusText.textContent = '数据就绪';
                statusBadge.textContent = '就绪';
                statusBadge.className = 'badge bg-success';
                statusCard.classList.add('ready');
                btnStart.disabled = false;
                placeholderWarning.style.display = 'none';
                break;
                
            case 'downloaded':
                statusText.textContent = '数据已下载';
                statusBadge.textContent = '已下载';
                statusBadge.className = 'badge bg-info';
                btnStart.disabled = dataset.total_count < 10;
                placeholderWarning.style.display = 'none';
                break;
                
            case 'placeholder':
                statusText.textContent = '占位符数据';
                statusBadge.textContent = '待采集';
                statusBadge.className = 'badge bg-warning';
                statusCard.classList.add('placeholder');
                btnStart.disabled = true;
                placeholderWarning.style.display = 'block';
                break;
                
            case 'downloading':
                statusText.textContent = '正在下载...';
                statusBadge.textContent = '下载中';
                statusBadge.className = 'badge bg-primary';
                statusCard.classList.add('downloading');
                btnStart.disabled = true;
                placeholderWarning.style.display = 'none';
                break;
                
            default:
                statusText.textContent = '无数据';
                statusBadge.textContent = '未下载';
                statusBadge.className = 'badge bg-secondary';
                btnStart.disabled = true;
                placeholderWarning.style.display = 'none';
        }
    },

    /**
     * 更新任务状态显示
     */
    updateTaskStatus: function(task) {
        const progressSection = document.getElementById('training-progress-section');
        const statusBadge = document.getElementById('training-status-badge');
        const btnStart = document.getElementById('btn-start-training');
        const btnCancel = document.getElementById('btn-cancel-training');
        
        if (!task) {
            statusBadge.textContent = '未训练';
            statusBadge.className = 'badge bg-secondary';
            progressSection.style.display = 'none';
            btnStart.style.display = 'block';
            btnCancel.style.display = 'none';
            return;
        }
        
        this.currentTaskId = task.task_id;
        
        switch (task.status) {
            case 'training':
            case 'preparing':
            case 'downloading':
                statusBadge.textContent = '训练中';
                statusBadge.className = 'badge bg-primary training-active';
                progressSection.style.display = 'block';
                btnStart.style.display = 'none';
                btnCancel.style.display = 'block';
                this.updateProgress(task);
                this.startPolling();
                break;
                
            case 'completed':
                statusBadge.textContent = '已完成';
                statusBadge.className = 'badge bg-success';
                progressSection.style.display = 'none';
                btnStart.style.display = 'block';
                btnCancel.style.display = 'none';
                this.stopPolling();
                break;
                
            case 'failed':
                statusBadge.textContent = '失败';
                statusBadge.className = 'badge bg-danger';
                progressSection.style.display = 'block';
                btnStart.style.display = 'block';
                btnCancel.style.display = 'none';
                this.updateProgress(task);
                this.stopPolling();
                break;
                
            case 'cancelled':
                statusBadge.textContent = '已取消';
                statusBadge.className = 'badge bg-warning';
                progressSection.style.display = 'none';
                btnStart.style.display = 'block';
                btnCancel.style.display = 'none';
                this.stopPolling();
                break;
                
            default:
                statusBadge.textContent = '空闲';
                statusBadge.className = 'badge bg-secondary';
                progressSection.style.display = 'none';
                btnStart.style.display = 'block';
                btnCancel.style.display = 'none';
        }
    },

    /**
     * 更新训练进度
     */
    updateProgress: function(task) {
        const progressBar = document.getElementById('training-progress-bar');
        const progressText = document.getElementById('training-progress-text');
        const currentEpoch = document.getElementById('current-epoch');
        const totalEpochs = document.getElementById('total-epochs');
        const currentMap = document.getElementById('current-map');
        const message = document.getElementById('training-message');
        
        const progress = task.progress || 0;
        
        progressBar.style.width = `${progress}%`;
        progressText.textContent = `${progress.toFixed(1)}%`;
        currentEpoch.textContent = task.current_epoch || 0;
        totalEpochs.textContent = task.total_epochs || 100;
        currentMap.textContent = task.best_map50 ? task.best_map50.toFixed(4) : '-';
        message.textContent = task.message || '处理中...';
        
        // 根据状态设置进度条颜色
        progressBar.classList.remove('bg-success', 'bg-danger', 'bg-warning');
        if (task.status === 'failed') {
            progressBar.classList.add('bg-danger');
        } else if (task.status === 'completed') {
            progressBar.classList.add('bg-success');
        }
    },

    /**
     * 更新已完成模型信息
     */
    updateCompletedModel: function(model) {
        const section = document.getElementById('trained-model-section');
        
        if (!model) {
            section.style.display = 'none';
            return;
        }
        
        section.style.display = 'block';
        
        document.getElementById('trained-model-name').textContent = 
            `${model.voltage_level} - ${model.plugin}`;
        document.getElementById('trained-model-map').textContent = 
            model.best_map50 ? model.best_map50.toFixed(4) : '-';
        document.getElementById('trained-model-time').textContent = 
            model.created_at ? new Date(model.created_at).toLocaleDateString() : '-';
    },

    /**
     * 加载可用数据集
     */
    loadAvailableDatasets: async function() {
        if (!this.currentPlugin) return;
        
        try {
            const response = await fetch(`/api/training/datasets/${this.currentPlugin}`);
            const data = await response.json();
            
            if (data.success) {
                this.renderDownloadButtons(data.datasets);
            }
        } catch (error) {
            console.error('加载数据集列表失败:', error);
        }
    },

    /**
     * 渲染下载按钮
     */
    renderDownloadButtons: function(datasets) {
        const container = document.getElementById('download-buttons');
        container.innerHTML = '';
        
        for (const [id, dataset] of Object.entries(datasets)) {
            const btn = document.createElement('button');
            btn.className = 'btn btn-outline-primary download-btn';
            btn.dataset.datasetId = id;
            
            if (dataset.requires_manual) {
                btn.className = 'btn btn-outline-warning download-btn';
                btn.innerHTML = `
                    <div class="dataset-info">
                        <div class="dataset-name">${dataset.name}</div>
                        <div class="dataset-count">${dataset.image_count || '?'} 张图像 - 需手动下载</div>
                    </div>
                    <i class="bi bi-exclamation-triangle"></i>
                `;
                btn.onclick = () => this.showManualInstructions(dataset);
            } else {
                btn.innerHTML = `
                    <div class="dataset-info">
                        <div class="dataset-name">${dataset.name}</div>
                        <div class="dataset-count">${dataset.image_count || '?'} 张图像</div>
                    </div>
                    <i class="bi bi-cloud-download"></i>
                `;
                btn.onclick = () => this.downloadDataset(id);
            }
            
            container.appendChild(btn);
        }
        
        if (Object.keys(datasets).length === 0) {
            container.innerHTML = '<small class="text-muted">暂无公开数据集</small>';
        }
    },

    /**
     * 显示手动下载说明
     */
    showManualInstructions: function(dataset) {
        alert(`${dataset.name}\n\n${dataset.manual_instructions || '需要手动下载数据集'}`);
    },

    /**
     * 下载数据集
     */
    downloadDataset: async function(datasetId) {
        const self = this;
        
        try {
            const response = await fetch('/api/training/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: datasetId,
                    voltage_level: this.currentVoltageLevel
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // 显示下载进度
                document.getElementById('download-progress-container').style.display = 'block';
                document.getElementById('download-status-text').textContent = '正在下载...';
                
                // 开始轮询下载进度
                this.startDownloadPolling(datasetId);
            } else {
                if (data.manual_instructions) {
                    alert(data.manual_instructions);
                } else {
                    alert('下载失败: ' + (data.message || '未知错误'));
                }
            }
        } catch (error) {
            console.error('下载请求失败:', error);
            alert('下载请求失败');
        }
    },

    /**
     * 开始下载进度轮询
     */
    startDownloadPolling: function(datasetId) {
        const self = this;
        
        if (this.downloadPollingInterval) {
            clearInterval(this.downloadPollingInterval);
        }
        
        this.downloadPollingInterval = setInterval(async function() {
            try {
                const response = await fetch(`/api/training/download/progress/${datasetId}`);
                const data = await response.json();
                
                if (data.success) {
                    const progress = data.progress;
                    const progressBar = document.getElementById('download-progress-bar');
                    const statusText = document.getElementById('download-status-text');
                    
                    if (progress >= 0) {
                        progressBar.style.width = `${progress}%`;
                        statusText.textContent = `下载进度: ${progress.toFixed(1)}%`;
                    }
                    
                    if (!data.downloading || progress >= 100) {
                        // 下载完成
                        clearInterval(self.downloadPollingInterval);
                        self.downloadPollingInterval = null;
                        
                        document.getElementById('download-progress-container').style.display = 'none';
                        statusText.textContent = '下载完成';
                        
                        // 刷新状态
                        setTimeout(() => self.loadStatus(), 1000);
                    }
                }
            } catch (error) {
                console.error('获取下载进度失败:', error);
            }
        }, 1000);
    },

    /**
     * 开始训练
     */
    startTraining: async function() {
        const epochs = parseInt(document.getElementById('training-epochs').value) || 100;
        const batchSize = parseInt(document.getElementById('training-batch-size').value) || 16;
        
        try {
            const response = await fetch('/api/training/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    voltage_level: this.currentVoltageLevel,
                    plugin: this.currentPlugin,
                    epochs: epochs,
                    batch_size: batchSize
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentTaskId = data.task_id;
                this.updateTaskStatus(data.task);
                this.startPolling();
            } else {
                alert('启动训练失败: ' + (data.message || '未知错误'));
            }
        } catch (error) {
            console.error('启动训练请求失败:', error);
            alert('启动训练请求失败');
        }
    },

    /**
     * 取消训练
     */
    cancelTraining: async function() {
        if (!this.currentTaskId) return;
        
        if (!confirm('确定要取消当前训练吗？')) return;
        
        try {
            const response = await fetch(`/api/training/cancel/${this.currentTaskId}`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.stopPolling();
                this.loadStatus();
            }
        } catch (error) {
            console.error('取消训练失败:', error);
        }
    },

    /**
     * 使用已训练模型
     */
    useModel: function() {
        // 发送事件通知其他组件使用此模型
        const event = new CustomEvent('model-selected', {
            detail: {
                voltage_level: this.currentVoltageLevel,
                plugin: this.currentPlugin
            }
        });
        document.dispatchEvent(event);
        
        alert(`模型已激活: ${this.currentVoltageLevel} - ${this.currentPlugin}`);
    },

    /**
     * 开始状态轮询
     */
    startPolling: function() {
        const self = this;
        
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        this.pollingInterval = setInterval(function() {
            self.loadStatus();
        }, 3000);
    },

    /**
     * 停止状态轮询
     */
    stopPolling: function() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
};

// 页面加载完成后自动初始化
document.addEventListener('DOMContentLoaded', function() {
    // 从页面获取当前模块ID
    const moduleIdMeta = document.querySelector('meta[name="module-id"]');
    const moduleId = moduleIdMeta ? moduleIdMeta.content : null;
    
    // 或者从URL路径获取
    const pathParts = window.location.pathname.split('/');
    const urlModuleId = pathParts[pathParts.length - 1] || pathParts[pathParts.length - 2];
    
    const pluginId = moduleId || urlModuleId;
    
    // 插件ID映射
    const pluginMapping = {
        'transformer': 'transformer',
        'switch': 'switch',
        'busbar': 'busbar',
        'capacitor': 'capacitor',
        'meter': 'meter'
    };
    
    const normalizedPlugin = pluginMapping[pluginId];
    
    if (normalizedPlugin && document.getElementById('training-panel')) {
        TrainingPanel.init(normalizedPlugin);
    }
});

// 导出供外部使用
window.TrainingPanel = TrainingPanel;
