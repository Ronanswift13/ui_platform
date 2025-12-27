/**
 * 模块页面通用逻辑（摄像头接入 + 检测）
 */
document.addEventListener('DOMContentLoaded', function() {
    const MODULES = {
        transformer: { pluginId: 'transformer_inspection' },
        switch: { pluginId: 'switch_inspection' },
        busbar: { pluginId: 'busbar_inspection' },
        capacitor: { pluginId: 'capacitor_inspection' },
        meter: { pluginId: 'meter_reading' },
    };

    function getModuleId() {
        const path = window.location.pathname.replace(/\/+$/, '');
        const parts = path.split('/');
        return parts[parts.length - 1] || '';
    }

    const moduleId = getModuleId();
    const moduleInfo = MODULES[moduleId];
    if (!moduleInfo) {
        return;
    }

    const cameraStream = document.getElementById('camera-stream');
    const noSignal = document.getElementById('no-signal');
    const noSignalText = document.getElementById('no-signal-text');
    const detectLoading = document.getElementById('detect-loading');
    const connectionBadge = document.getElementById('camera-connection-badge');
    const btnDetect = document.getElementById('btn-detect');
    const btnClear = document.getElementById('btn-clear');
    const resultsCard = document.getElementById('results-card');
    const resultsList = document.getElementById('results-list');
    const noResults = document.getElementById('no-results');
    const alarmsCard = document.getElementById('alarms-card');
    const alarmsList = document.getElementById('alarms-list');
    const snapshotCard = document.getElementById('snapshot-card');
    const snapshotImage = document.getElementById('snapshot-image');

    const cameraSelect = document.getElementById('camera-select');
    const cameraStatus = document.getElementById('camera-status');
    const cameraStatusText = document.getElementById('camera-status-text');
    const btnRefreshCameras = document.getElementById('btn-refresh-cameras');
    const btnConnectCamera = document.getElementById('btn-connect-camera');
    const btnDisconnectCamera = document.getElementById('btn-disconnect-camera');
    const btnRegisterCamera = document.getElementById('btn-register-camera');

    const cameraType = document.getElementById('camera-type');
    const cameraUrlGroup = document.getElementById('camera-url-group');
    const cameraIndexGroup = document.getElementById('camera-index-group');

    const thermalStatus = document.getElementById('thermal-status');
    const thermalMaxValue = document.getElementById('thermal-max-value');
    const thermalAvgValue = document.getElementById('thermal-avg-value');
    const thermalMinValue = document.getElementById('thermal-min-value');
    const thermalHotspot = document.getElementById('thermal-hotspot');

    if (!cameraStream || !cameraSelect || !btnDetect) {
        return;
    }

    let currentCameraId = '';
    let statusTimer = null;

    function setNoSignal(message) {
        if (noSignal) {
            noSignal.classList.remove('d-none');
        }
        cameraStream.style.display = 'none';
        cameraStream.src = '';
        if (noSignalText) {
            noSignalText.textContent = message || '请接入激光摄像头';
        }
        if (btnDetect) {
            btnDetect.disabled = true;
        }
        if (connectionBadge) {
            connectionBadge.className = 'badge bg-secondary';
            connectionBadge.textContent = '未连接';
        }
    }

    function setSignalReady() {
        if (noSignal) {
            noSignal.classList.add('d-none');
        }
        cameraStream.style.display = 'block';
        if (btnDetect) {
            btnDetect.disabled = false;
        }
        if (connectionBadge) {
            connectionBadge.className = 'badge bg-success';
            connectionBadge.textContent = '已连接';
        }
    }

    function updateCameraBadge(status) {
        if (!connectionBadge) {
            return;
        }
        if (status === 'connected') {
            setSignalReady();
        } else if (status === 'connecting') {
            connectionBadge.className = 'badge bg-warning';
            connectionBadge.textContent = '连接中';
        } else {
            setNoSignal('无信号输入');
        }
    }

    function updateThermalDisplay(data) {
        if (!thermalStatus || !thermalMaxValue || !thermalAvgValue || !thermalMinValue || !thermalHotspot) {
            return;
        }
        if (!data) {
            thermalStatus.className = 'badge bg-secondary';
            thermalStatus.textContent = '未分析';
            thermalMaxValue.textContent = '--';
            thermalAvgValue.textContent = '--';
            thermalMinValue.textContent = '--';
            thermalHotspot.textContent = '--';
            return;
        }
        thermalMaxValue.textContent = data.max_temp.toFixed(1) + '°C';
        thermalAvgValue.textContent = data.avg_temp.toFixed(1) + '°C';
        thermalMinValue.textContent = data.min_temp.toFixed(1) + '°C';
        thermalHotspot.textContent = (data.hotspot_ratio * 100).toFixed(1) + '%';
        if (data.over_threshold) {
            thermalStatus.className = 'badge bg-danger';
            thermalStatus.textContent = '超温';
        } else {
            thermalStatus.className = 'badge bg-success';
            thermalStatus.textContent = '正常';
        }
    }

    function toggleCameraFields() {
        if (!cameraType || !cameraUrlGroup || !cameraIndexGroup) {
            return;
        }
        const type = cameraType.value;
        if (type === 'usb') {
            cameraUrlGroup.classList.add('d-none');
            cameraIndexGroup.classList.remove('d-none');
        } else {
            cameraUrlGroup.classList.remove('d-none');
            cameraIndexGroup.classList.add('d-none');
        }
    }

    function setStream(cameraId) {
        if (!cameraId) {
            setNoSignal('未选择摄像头');
            return;
        }
        cameraStream.src = `/api/cameras/${cameraId}/stream?fps=10&ts=${Date.now()}`;
    }

    async function loadCameras() {
        try {
            const response = await fetch('/api/cameras');
            const cameras = await response.json();
            cameraSelect.innerHTML = '';
            if (!cameras || cameras.length === 0) {
                cameraSelect.innerHTML = '<option value="">未发现摄像头</option>';
                currentCameraId = '';
                setNoSignal('未发现摄像头');
                return;
            }
            cameras.forEach(cam => {
                const option = document.createElement('option');
                option.value = cam.id;
                option.textContent = `${cam.id} (${cam.camera_type})`;
                cameraSelect.appendChild(option);
            });
            if (!currentCameraId) {
                currentCameraId = cameras[0].id;
            }
            cameraSelect.value = currentCameraId;
            await refreshCameraStatus();
        } catch (error) {
            setNoSignal('摄像头列表加载失败');
        }
    }

    async function refreshCameraStatus() {
        if (!currentCameraId) {
            setNoSignal('未选择摄像头');
            return;
        }
        try {
            const response = await fetch(`/api/cameras/${currentCameraId}/status`);
            const data = await response.json();
            const status = data.status || 'disconnected';
            if (cameraStatus) {
                cameraStatus.textContent = status === 'connected' ? '已连接' : '未连接';
                cameraStatus.className = status === 'connected' ? 'badge bg-success' : 'badge bg-secondary';
            }
            if (cameraStatusText) {
                cameraStatusText.textContent = data.last_error || '';
            }
            updateCameraBadge(status);
            if (status === 'connected') {
                setStream(currentCameraId);
            }
        } catch (error) {
            setNoSignal('摄像头状态获取失败');
        }
    }

    async function connectSelectedCamera() {
        if (!currentCameraId) return;
        await fetch(`/api/cameras/${currentCameraId}/connect`, { method: 'POST' });
        await refreshCameraStatus();
    }

    async function disconnectSelectedCamera() {
        if (!currentCameraId) return;
        await fetch(`/api/cameras/${currentCameraId}/disconnect`, { method: 'POST' });
        setNoSignal('摄像头已断开');
    }

    async function registerCamera() {
        const cameraId = document.getElementById('camera-id');
        if (!cameraId || !cameraId.value.trim()) {
            alert('请填写摄像头ID');
            return;
        }
        const payload = {
            camera_id: cameraId.value.trim(),
            camera_type: cameraType ? cameraType.value : 'rtsp',
            url: (document.getElementById('camera-url') || {}).value || '',
            device_index: parseInt((document.getElementById('camera-index') || {}).value || '0', 10),
            width: parseInt((document.getElementById('camera-width') || {}).value || '1920', 10),
            height: parseInt((document.getElementById('camera-height') || {}).value || '1080', 10),
            fps: parseInt((document.getElementById('camera-fps') || {}).value || '25', 10),
            auto_connect: true
        };
        const response = await fetch('/api/cameras/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const data = await response.json();
            alert(data.detail || '接入失败');
            return;
        }
        currentCameraId = payload.camera_id;
        await loadCameras();
    }

    function readNumber(id, fallback) {
        const el = document.getElementById(id);
        if (!el) return fallback;
        const value = parseFloat(el.value);
        return Number.isFinite(value) ? value : fallback;
    }

    function readInt(id, fallback) {
        const el = document.getElementById(id);
        if (!el) return fallback;
        const value = parseInt(el.value, 10);
        return Number.isFinite(value) ? value : fallback;
    }

    function readSelectValue(id, fallback) {
        const el = document.getElementById(id);
        return el ? el.value : fallback;
    }

    function isChecked(id, fallback) {
        const el = document.getElementById(id);
        return el ? !!el.checked : !!fallback;
    }

    function getCheckedValues(selector) {
        return Array.from(document.querySelectorAll(selector))
            .filter(el => el.checked)
            .map(el => el.value);
    }

    function buildAlgorithmsConfig() {
        if (moduleId === 'switch') {
            return {
                state_recognition: {
                    breaker: {
                        enabled: isChecked('check-breaker', true),
                        confidence_threshold: readNumber('breaker-confidence', 0.6),
                        method: readSelectValue('breaker-method', 'indicator'),
                    },
                    isolator: {
                        enabled: isChecked('check-isolator', true),
                        confidence_threshold: readNumber('isolator-confidence', 0.6),
                        method: readSelectValue('isolator-method', 'indicator'),
                    },
                    grounding: {
                        enabled: isChecked('check-grounding', true),
                        confidence_threshold: readNumber('grounding-confidence', 0.6),
                    },
                },
                logic_validation: {
                    enabled: isChecked('check-interlock', true),
                },
                image_quality: {
                    enabled: isChecked('check-clarity', true),
                    min_clarity_score: readNumber('min-clarity-score', 0.7),
                    edge_energy_weight: readNumber('edge-energy-weight', 0.5),
                },
                gauge_reading: {
                    enabled: isChecked('check-sf6-reading', false),
                    pointer: {
                        confidence_threshold: readNumber('sf6-confidence', 0.6),
                        max_rotation: readNumber('sf6-max-rotation', 30),
                    },
                },
            };
        }

        if (moduleId === 'busbar') {
            return {
                defect_types: {
                    pin_missing: {
                        confidence_threshold: readNumber('pin-missing-confidence', 0.5),
                        min_size: readInt('pin-missing-min-size', 20),
                    },
                    crack: {
                        edge_density: readNumber('crack-edge-density', 0.15),
                        min_length: readInt('crack-min-length', 50),
                    },
                    foreign_object: {
                        min_area: readInt('foreign-min-area', 200),
                        max_area: readInt('foreign-max-area', 2000),
                        circularity: readNumber('foreign-circularity', 0.3),
                    },
                },
                quality: {
                    enabled: isChecked('check-quality-gate', true),
                    blur_thr: readNumber('quality-min-clarity', 0.6),
                    y_high: readInt('quality-max-exposure', 240),
                    dr_min: readInt('quality-min-contrast', 20),
                },
                zoom: {
                    enabled: isChecked('check-zoom-suggestion', true),
                    min_obj_px: readInt('zoom-min-size', 15),
                    target_px: readInt('zoom-target-size', 40),
                },
                tiling: {
                    enabled: isChecked('check-tiling', true),
                    tile_size: readInt('tiling-size', 640),
                    overlap: readNumber('tiling-overlap', 0.25),
                    nms_iou: readNumber('tiling-nms', 0.45),
                },
            };
        }

        if (moduleId === 'capacitor') {
            return {
                structural_integrity: {
                    tilt_detection: {
                        enabled: isChecked('check-tilt', true),
                        warning_angle: readNumber('tilt-warning-angle', 5),
                        max_tilt_angle: readNumber('tilt-error-angle', 15),
                    },
                    collapse_detection: {
                        enabled: isChecked('check-collapse', true),
                        min_height_ratio: readNumber('collapse-height-ratio', 0.3),
                        confidence_threshold: readNumber('collapse-confidence', 0.55),
                    },
                    missing_detection: {
                        enabled: isChecked('check-missing', true),
                        expected_count: readInt('missing-expected-count', 12),
                        tolerance: readInt('missing-tolerance', 0),
                    },
                },
                intrusion_detection: {
                    enabled: isChecked('check-intrusion-person', true) ||
                        isChecked('check-intrusion-vehicle', true) ||
                        isChecked('check-intrusion-animal', false),
                    types: {
                        person: {
                            enabled: isChecked('check-intrusion-person', true),
                            confidence_threshold: readNumber('intrusion-person-confidence', 0.6),
                            delay_s: readNumber('intrusion-person-delay', 2.0),
                        },
                        vehicle: {
                            enabled: isChecked('check-intrusion-vehicle', true),
                            confidence_threshold: readNumber('intrusion-vehicle-confidence', 0.6),
                            delay_s: readNumber('intrusion-vehicle-delay', 2.0),
                        },
                        animal: {
                            enabled: isChecked('check-intrusion-animal', false),
                            confidence_threshold: readNumber('intrusion-animal-confidence', 0.5),
                            delay_s: readNumber('intrusion-animal-delay', 5.0),
                        },
                    },
                },
            };
        }

        if (moduleId === 'meter') {
            return {
                reading: {
                    pointer: {
                        enabled: isChecked('check-pointer', true),
                        confidence_threshold: readNumber('pointer-confidence', 0.6),
                        max_rotation: readNumber('pointer-max-rotation', 30),
                        retry_count: readInt('pointer-retry-count', 3),
                    },
                    digital: {
                        enabled: isChecked('check-digital', true),
                        confidence_threshold: readNumber('digital-confidence', 0.7),
                        retry_count: readInt('digital-retry-count', 3),
                    },
                    seven_segment: {
                        enabled: isChecked('check-seven-segment', false),
                        threshold: readNumber('seven-segment-threshold', 0.6),
                        digits: readInt('seven-segment-digits', 4),
                    },
                    led: {
                        enabled: isChecked('check-led', false),
                        brightness_threshold: readInt('led-brightness-threshold', 150),
                        color: readSelectValue('led-color', 'red'),
                    },
                },
                quality: {
                    keypoint: {
                        enabled: isChecked('check-keypoint', true),
                        confidence: readNumber('keypoint-confidence', 0.5),
                        count: readInt('keypoint-count', 4),
                    },
                    perspective: {
                        enabled: isChecked('check-perspective', true),
                        max_angle: readNumber('perspective-max-angle', 45),
                        output_size: readInt('perspective-output-size', 512),
                    },
                    range: {
                        enabled: isChecked('check-range', true),
                        ocr_confidence: readNumber('range-ocr-confidence', 0.6),
                        scale_spacing: readInt('range-scale-spacing', 20),
                    },
                    manual_review: {
                        enabled: isChecked('check-manual-review', true),
                        threshold: readNumber('manual-review-threshold', 0.5),
                        history_window: readInt('manual-review-history-window', 5),
                    },
                },
            };
        }

        return {};
    }

    function buildDetectionPayload() {
        const payload = {
            camera_id: currentCameraId,
            defect_types: [],
            state_types: [],
            thermal_enabled: false,
            thermal_threshold: 80,
            thermal_min_temp: 20,
            thermal_max_temp: 120,
            algorithms: buildAlgorithmsConfig(),
        };

        if (moduleId === 'switch') {
            if (isChecked('check-breaker', true)) payload.state_types.push('breaker');
            if (isChecked('check-isolator', true)) payload.state_types.push('isolator');
            if (isChecked('check-grounding', true)) payload.state_types.push('grounding');
            if (isChecked('check-interlock', true)) payload.state_types.push('interlock');
            if (isChecked('check-clarity', true)) payload.state_types.push('clarity');
            if (isChecked('check-sf6-reading', false)) payload.state_types.push('sf6_reading');
            return payload;
        }

        if (moduleId === 'busbar') {
            payload.defect_types = getCheckedValues('.defect-type:checked');
            return payload;
        }

        if (moduleId === 'capacitor') {
            payload.defect_types = getCheckedValues('.defect-type:checked');
            payload.state_types = getCheckedValues('.intrusion-type:checked')
                .map(type => `intrusion_${type}`);
            return payload;
        }

        if (moduleId === 'meter') {
            payload.state_types = getCheckedValues('.reading-type:checked');
            return payload;
        }

        const selected = getCheckedValues('.detection-type:checked');
        if (selected.includes('defect')) {
            payload.defect_types = ['defect'];
        }
        if (selected.includes('state')) {
            payload.state_types = ['state'];
        }
        if (isChecked('check-thermal', false)) {
            payload.thermal_enabled = true;
            payload.thermal_threshold = readNumber('thermal-threshold', 80);
            payload.thermal_min_temp = readNumber('thermal-min', 20);
            payload.thermal_max_temp = readNumber('thermal-max', 120);
        }
        return payload;
    }

    const LABEL_NAME_MAP = {
        switch_inspection: {
            breaker_open: '断路器分闸',
            breaker_closed: '断路器合闸',
            breaker_intermediate: '断路器中间态',
            isolator_open: '隔离开关分闸',
            isolator_closed: '隔离开关合闸',
            grounding_open: '接地开关分闸',
            grounding_closed: '接地开关合闸',
            clarity_low: '清晰度过低',
            gauge_reading: 'SF6表计读数',
            gauge_reading_failed: '表计读数失败',
            sf6_pressure: 'SF6压力读数',
            sf6_density: 'SF6密度读数',
            logic_error: '五防逻辑异常(严重)',
            logic_warning: '五防逻辑异常(警告)',
        },
        busbar_inspection: {
            pin_missing: '销钉缺失',
            crack: '裂纹',
            foreign_object: '异物',
            quality_failed: '质量门禁未通过',
        },
        capacitor_inspection: {
            tilt_warning: '电容器倾斜(警告)',
            tilt_error: '电容器倾斜(严重)',
            collapse: '电容器倒塌',
            missing_unit: '电容器单元缺失',
            intrusion_person: '人员入侵',
            intrusion_vehicle: '车辆入侵',
            intrusion_animal: '动物入侵',
            intrusion_unknown: '未知入侵',
        },
    };

    const METER_NAME_MAP = {
        pressure_gauge: '压强表',
        temperature_gauge: '温度表',
        oil_level_gauge: '油位表',
        sf6_density_gauge: 'SF6密度表',
        digital_display: '数字显示',
        led_indicator: 'LED指示',
        ammeter: '电流表',
        voltmeter: '电压表',
        seven_segment: '七段码',
    };

    function getLabelName(label, labelCn) {
        if (labelCn && labelCn !== label) {
            return labelCn;
        }
        const map = LABEL_NAME_MAP[moduleInfo.pluginId];
        if (map && map[label]) {
            return map[label];
        }
        if (moduleInfo.pluginId === 'meter_reading' && label) {
            const isFailed = label.endsWith('_failed');
            const isReading = label.endsWith('_reading');
            if (isFailed || isReading) {
                const base = label.replace(/_(failed|reading)$/, '');
                const meterName = METER_NAME_MAP[base] || base;
                return isFailed ? `${meterName}读数失败` : `${meterName}读数`;
            }
        }
        return label;
    }

    function getLabelLevel(label) {
        if (moduleInfo.pluginId === 'switch_inspection') {
            if (label.startsWith('logic_')) {
                return label.includes('error') ? 'error' : 'warning';
            }
            if (label === 'clarity_low' || label === 'gauge_reading_failed') {
                return 'warning';
            }
            return 'normal';
        }
        if (moduleInfo.pluginId === 'busbar_inspection') {
            if (label === 'pin_missing') return 'error';
            if (label === 'crack' || label === 'foreign_object') return 'warning';
            if (label === 'quality_failed') return 'info';
            return 'normal';
        }
        if (moduleInfo.pluginId === 'capacitor_inspection') {
            if (label === 'tilt_error' || label === 'collapse' || label === 'missing_unit') return 'error';
            if (label === 'tilt_warning' || label === 'intrusion_animal') return 'warning';
            if (label === 'intrusion_unknown') return 'info';
            if (label.startsWith('intrusion_')) return 'error';
            return 'normal';
        }
        if (moduleInfo.pluginId === 'meter_reading') {
            return label.endsWith('_failed') ? 'warning' : 'normal';
        }
        if (['oil_leak', 'damage'].includes(label)) return 'error';
        if (['rust', 'foreign_object', 'silica_gel_abnormal'].includes(label)) return 'warning';
        return 'normal';
    }

    function getLevelClass(level) {
        if (level === 'error') {
            return 'bg-danger bg-opacity-10 border-start border-danger border-4';
        }
        if (level === 'warning') {
            return 'bg-warning bg-opacity-10 border-start border-warning border-4';
        }
        if (level === 'info') {
            return 'bg-info bg-opacity-10 border-start border-info border-4';
        }
        return 'bg-success bg-opacity-10 border-start border-success border-4';
    }

    function getBadgeClass(level) {
        if (level === 'error') {
            return 'bg-danger';
        }
        if (level === 'warning') {
            return 'bg-warning text-dark';
        }
        if (level === 'info') {
            return 'bg-info text-dark';
        }
        return 'bg-success';
    }

    function displayResults(results) {
        if (!resultsCard || !resultsList) return;
        resultsCard.style.display = 'block';
        resultsList.innerHTML = '';

        if (!results || results.length === 0) {
            if (noResults) noResults.style.display = 'block';
            return;
        }

        if (noResults) noResults.style.display = 'none';

        results.forEach((result) => {
            const level = getLabelLevel(result.label);
            const levelClass = getLevelClass(level);
            const badgeClass = getBadgeClass(level);
            const labelText = getLabelName(result.label, result.label_cn);
            const valueText = result.value !== undefined && result.value !== null
                ? ` | ${result.value}`
                : '';

            const html = `
                <div class="d-flex justify-content-between align-items-center p-2 mb-2 rounded ${levelClass}">
                    <div>
                        <strong>${labelText}</strong>
                        <small class="text-muted ms-2">置信度: ${(result.confidence * 100).toFixed(1)}%${valueText}</small>
                    </div>
                    <span class="badge ${badgeClass}">${result.label}</span>
                </div>
            `;
            resultsList.innerHTML += html;
        });
    }

    function displayAlarms(alarms) {
        if (!alarmsCard || !alarmsList) return;
        if (!alarms || alarms.length === 0) {
            alarmsCard.style.display = 'none';
            return;
        }

        alarmsCard.style.display = 'block';
        alarmsList.innerHTML = '';

        alarms.forEach(alarm => {
            const levelIcon = alarm.level === 'error' ? 'exclamation-octagon' : 'exclamation-triangle';
            const levelClass = alarm.level === 'error' ? 'list-group-item-danger' : 'list-group-item-warning';
            const html = `
                <div class="list-group-item ${levelClass}">
                    <div class="d-flex align-items-center">
                        <i class="bi bi-${levelIcon} me-2"></i>
                        <div>
                            <strong>${alarm.title}</strong>
                            <p class="mb-0 small">${alarm.message}</p>
                        </div>
                    </div>
                </div>
            `;
            alarmsList.innerHTML += html;
        });
    }

    btnDetect.addEventListener('click', async function() {
        if (!currentCameraId) {
            alert('请先接入摄像头');
            return;
        }

        const payload = buildDetectionPayload();
        if (payload.defect_types.length === 0 && payload.state_types.length === 0 && !payload.thermal_enabled) {
            alert('请至少选择一种检测类型');
            return;
        }

        if (detectLoading) {
            detectLoading.classList.remove('d-none');
        }
        btnDetect.disabled = true;

        try {
            const response = await fetch(`/api/detect/${moduleInfo.pluginId}/camera`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || '检测失败');
            }

            if (snapshotCard && snapshotImage) {
                if (data.annotated_image) {
                    snapshotImage.src = data.annotated_image;
                    snapshotCard.style.display = 'block';
                } else if (data.snapshot_image) {
                    snapshotImage.src = data.snapshot_image;
                    snapshotCard.style.display = 'block';
                }
            }

            displayResults(data.results);
            displayAlarms(data.alarms);
            updateThermalDisplay(data.thermal);
        } catch (error) {
            alert('检测失败: ' + error.message);
        } finally {
            if (detectLoading) {
                detectLoading.classList.add('d-none');
            }
            btnDetect.disabled = false;
        }
    });

    if (btnClear) {
        btnClear.addEventListener('click', function() {
            if (resultsCard) resultsCard.style.display = 'none';
            if (alarmsCard) alarmsCard.style.display = 'none';
            if (snapshotCard) snapshotCard.style.display = 'none';
            if (resultsList) resultsList.innerHTML = '';
            if (alarmsList) alarmsList.innerHTML = '';
            if (noResults) noResults.style.display = 'none';
            updateThermalDisplay(null);
        });
    }

    cameraSelect.addEventListener('change', function() {
        currentCameraId = cameraSelect.value;
        refreshCameraStatus();
    });

    if (btnRefreshCameras) btnRefreshCameras.addEventListener('click', loadCameras);
    if (btnConnectCamera) btnConnectCamera.addEventListener('click', connectSelectedCamera);
    if (btnDisconnectCamera) btnDisconnectCamera.addEventListener('click', disconnectSelectedCamera);
    if (btnRegisterCamera) btnRegisterCamera.addEventListener('click', registerCamera);
    if (cameraType) cameraType.addEventListener('change', toggleCameraFields);

    cameraStream.addEventListener('error', function() {
        setNoSignal('摄像头连接失败');
    });

    toggleCameraFields();
    loadCameras();
    if (statusTimer) clearInterval(statusTimer);
    statusTimer = setInterval(refreshCameraStatus, 5000);
});
