/**
 * DarkBreaker SDK - Standalone Plugin JavaScript
 *
 * Handles WebSocket connection, image upload, result rendering,
 * and auto-refresh functionality.
 */

const DarkBreakerSDK = {
    config: {
        pluginId: '',
        pluginName: '',
        wsUrl: '',
        apiBase: '',
    },
    ws: null,
    autoRefreshInterval: null,
    currentFile: null,

    /**
     * Initialize the SDK dashboard.
     */
    init(config) {
        this.config = { ...this.config, ...config };
        this._bindEvents();
        this._connectWebSocket();
        this._loadConfig();
        this._loadStatus();
    },

    /**
     * Bind UI event handlers.
     */
    _bindEvents() {
        // Image upload
        const uploadInput = document.getElementById('image-upload');
        if (uploadInput) {
            uploadInput.addEventListener('change', (e) => this._onImageSelected(e));
        }

        // Detect button
        const detectBtn = document.getElementById('btn-detect');
        if (detectBtn) {
            detectBtn.addEventListener('click', () => this._runDetection());
        }

        // Auto-refresh toggle
        const autoRefreshBtn = document.getElementById('btn-auto-refresh');
        if (autoRefreshBtn) {
            autoRefreshBtn.addEventListener('click', () => this._toggleAutoRefresh());
        }

        // Save config
        const saveConfigBtn = document.getElementById('btn-save-config');
        if (saveConfigBtn) {
            saveConfigBtn.addEventListener('click', () => this._saveConfig());
        }

        // Reload config
        const reloadConfigBtn = document.getElementById('btn-reload-config');
        if (reloadConfigBtn) {
            reloadConfigBtn.addEventListener('click', () => this._loadConfig());
        }
    },

    /**
     * Connect to the WebSocket stream.
     */
    _connectWebSocket() {
        if (!this.config.wsUrl) return;

        const statusEl = document.getElementById('connection-status');

        try {
            this.ws = new WebSocket(this.config.wsUrl);

            this.ws.onopen = () => {
                if (statusEl) {
                    statusEl.className = 'badge connected';
                    statusEl.innerHTML = '<i class="bi bi-circle-fill me-1"></i>已连接';
                }
            };

            this.ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'detection') {
                        this._handleDetectionResult(msg.data);
                    } else if (msg.type === 'stats') {
                        this._updateStats(msg.data);
                    }
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            this.ws.onclose = () => {
                if (statusEl) {
                    statusEl.className = 'badge disconnected';
                    statusEl.innerHTML = '<i class="bi bi-circle-fill me-1"></i>未连接';
                }
                // Reconnect after 3 seconds
                setTimeout(() => this._connectWebSocket(), 3000);
            };

            this.ws.onerror = () => {
                if (statusEl) {
                    statusEl.className = 'badge error';
                    statusEl.innerHTML = '<i class="bi bi-circle-fill me-1"></i>错误';
                }
            };
        } catch (e) {
            console.error('WebSocket connection failed:', e);
        }
    },

    /**
     * Handle image file selection.
     */
    _onImageSelected(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.currentFile = file;

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.getElementById('detection-image');
            const placeholder = document.getElementById('canvas-placeholder');
            if (img) {
                img.src = e.target.result;
                img.style.display = 'block';
            }
            if (placeholder) {
                placeholder.style.display = 'none';
            }
        };
        reader.readAsDataURL(file);

        // Enable detect button
        const detectBtn = document.getElementById('btn-detect');
        if (detectBtn) detectBtn.disabled = false;

        // Update input status
        const statusEl = document.getElementById('input-status');
        if (statusEl) {
            statusEl.className = 'badge bg-success';
            statusEl.textContent = '就绪';
        }
    },

    /**
     * Run detection on the current image.
     */
    async _runDetection() {
        if (!this.currentFile) return;

        const detectBtn = document.getElementById('btn-detect');
        if (detectBtn) {
            detectBtn.disabled = true;
            detectBtn.innerHTML = '<i class="bi bi-hourglass-split me-1"></i>处理中...';
        }

        const formData = new FormData();
        formData.append('file', this.currentFile);

        try {
            const response = await fetch(this.config.apiBase + '/api/detect', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                this._handleDetectionResult(data);
            } else {
                console.error('Detection failed:', data.error);
            }
        } catch (e) {
            console.error('Detection request failed:', e);
        } finally {
            if (detectBtn) {
                detectBtn.disabled = false;
                detectBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i>运行检测';
            }
        }
    },

    /**
     * Handle detection results.
     */
    _handleDetectionResult(data) {
        this._renderDetections(data.results || []);
        this._renderAlarms(data.alarms || []);
        if (data.stats) {
            this._updateStats(data.stats);
        }
    },

    /**
     * Render detection results in the list.
     */
    _renderDetections(results) {
        const listEl = document.getElementById('detection-list');
        const countEl = document.getElementById('detection-count');

        if (!listEl) return;

        if (results.length === 0) {
            listEl.innerHTML = '<div class="list-group-item bg-dark border-secondary text-secondary text-center small py-3">暂无检测结果</div>';
            if (countEl) countEl.textContent = '0';
            return;
        }

        if (countEl) countEl.textContent = results.length.toString();

        listEl.innerHTML = results.map((r, i) => `
            <div class="list-group-item detection-item">
                <div class="d-flex justify-content-between align-items-center">
                    <span class="label">${r.label || 'Unknown'}</span>
                    <span class="confidence">${(r.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="small text-secondary mt-1">
                    ROI: ${r.roi_id || '-'} | Value: ${r.value || '-'}
                </div>
            </div>
        `).join('');
    },

    /**
     * Render alarms in the alarm panel.
     */
    _renderAlarms(alarms) {
        const listEl = document.getElementById('alarm-list');
        const countEl = document.getElementById('alarm-count');

        if (!listEl) return;

        if (alarms.length === 0) {
            listEl.innerHTML = '<div class="list-group-item bg-dark border-secondary text-secondary text-center small py-3">暂无告警</div>';
            if (countEl) countEl.textContent = '0';
            return;
        }

        if (countEl) countEl.textContent = alarms.length.toString();

        listEl.innerHTML = alarms.map(a => `
            <div class="list-group-item alarm-item alarm-${a.level || 'warning'}">
                <div class="d-flex justify-content-between align-items-center">
                    <span class="fw-bold">${a.title || 'Alarm'}</span>
                    <span class="badge bg-${a.level === 'critical' || a.level === 'error' ? 'danger' : a.level === 'warning' ? 'warning' : 'info'}">${a.level || 'warning'}</span>
                </div>
                <div class="small text-secondary mt-1">${a.message || ''}</div>
            </div>
        `).join('');
    },

    /**
     * Update statistics display.
     */
    _updateStats(stats) {
        const mappings = {
            'stat-fps': stats.fps,
            'stat-frames': stats.total_frames,
            'stat-detections': stats.total_detections,
            'stat-alarms': stats.total_alarms,
            'stat-avg-time': stats.avg_inference_time_ms,
            'stats-fps': stats.fps,
            'stats-frames': stats.total_frames,
            'stats-detections': stats.total_detections,
            'stats-alarms': stats.total_alarms,
            'stats-avg-time': stats.avg_inference_time_ms,
        };

        for (const [id, value] of Object.entries(mappings)) {
            const el = document.getElementById(id);
            if (el && value !== undefined) {
                el.textContent = typeof value === 'number' ?
                    (Number.isInteger(value) ? value : value.toFixed(1)) :
                    value;
            }
        }
    },

    /**
     * Toggle auto-refresh mode.
     */
    _toggleAutoRefresh() {
        const btn = document.getElementById('btn-auto-refresh');
        if (!btn) return;

        const isActive = btn.dataset.active === 'true';

        if (isActive) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
            btn.dataset.active = 'false';
            btn.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>自动刷新: 关闭';
            btn.classList.remove('btn-primary');
            btn.classList.add('btn-outline-secondary');
        } else {
            this.autoRefreshInterval = setInterval(() => {
                if (this.currentFile) {
                    this._runDetection();
                }
            }, 2000);
            btn.dataset.active = 'true';
            btn.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>自动刷新: 开启';
            btn.classList.remove('btn-outline-secondary');
            btn.classList.add('btn-primary');
        }
    },

    /**
     * Load plugin configuration.
     */
    async _loadConfig() {
        try {
            const response = await fetch(this.config.apiBase + '/api/config');
            const data = await response.json();
            this._renderConfigForm(data.config || {}, data.config_schema || {});
        } catch (e) {
            console.error('Failed to load config:', e);
        }
    },

    /**
     * Render the dynamic config form.
     */
    _renderConfigForm(config, schema) {
        const formEl = document.getElementById('config-form');
        if (!formEl) return;

        const saveBtn = document.getElementById('btn-save-config');
        if (saveBtn) saveBtn.disabled = false;

        if (Object.keys(config).length === 0) {
            formEl.innerHTML = '<div class="text-secondary small text-center py-2">无配置参数</div>';
            return;
        }

        formEl.innerHTML = Object.entries(config).map(([key, value]) => {
            const inputType = typeof value === 'boolean' ? 'checkbox' :
                              typeof value === 'number' ? 'number' : 'text';

            if (inputType === 'checkbox') {
                return `
                    <div class="config-field form-check">
                        <input class="form-check-input" type="checkbox" id="config-${key}"
                               data-key="${key}" ${value ? 'checked' : ''}>
                        <label class="form-check-label" for="config-${key}">${key}</label>
                    </div>
                `;
            }

            return `
                <div class="config-field">
                    <label for="config-${key}">${key}</label>
                    <input type="${inputType}" class="form-control form-control-sm bg-dark border-secondary text-light"
                           id="config-${key}" data-key="${key}" value="${value}">
                </div>
            `;
        }).join('');
    },

    /**
     * Save configuration.
     */
    async _saveConfig() {
        const formEl = document.getElementById('config-form');
        if (!formEl) return;

        const config = {};
        const inputs = formEl.querySelectorAll('input[data-key]');
        inputs.forEach(input => {
            const key = input.dataset.key;
            if (input.type === 'checkbox') {
                config[key] = input.checked;
            } else if (input.type === 'number') {
                config[key] = parseFloat(input.value);
            } else {
                config[key] = input.value;
            }
        });

        try {
            const response = await fetch(this.config.apiBase + '/api/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config }),
            });

            const data = await response.json();
            if (data.status === 'updated') {
                // Flash success feedback
                const saveBtn = document.getElementById('btn-save-config');
                if (saveBtn) {
                    const original = saveBtn.innerHTML;
                    saveBtn.innerHTML = '<i class="bi bi-check-lg me-1"></i>已保存';
                    saveBtn.classList.replace('btn-outline-primary', 'btn-success');
                    setTimeout(() => {
                        saveBtn.innerHTML = original;
                        saveBtn.classList.replace('btn-success', 'btn-outline-primary');
                    }, 2000);
                }
            }
        } catch (e) {
            console.error('Failed to save config:', e);
        }
    },

    /**
     * Load plugin status.
     */
    async _loadStatus() {
        try {
            const response = await fetch(this.config.apiBase + '/api/status');
            const data = await response.json();
            if (data.stats) {
                this._updateStats(data.stats);
            }
        } catch (e) {
            console.error('Failed to load status:', e);
        }
    },
};
