/**
 * 变电站电压等级管理系统 - 前端交互模块
 * ==========================================
 * 
 * 提供完整的电压等级管理前端功能:
 * 1. 电压等级选择和切换
 * 2. 配置信息展示
 * 3. 模型状态监控
 * 4. 插件功能管理
 * 
 * 使用方法:
 *   <script src="/static/js/voltage_manager.js"></script>
 *   <script>
 *     const vm = new VoltageManager();
 *     vm.init();
 *   </script>
 * 
 * 作者: 破夜绘明团队
 * 版本: 2.0.0
 * 日期: 2025
 */

class VoltageManager {
    constructor(options = {}) {
        this.apiBase = options.apiBase || '/api/voltage';
        this.currentLevel = null;
        this.currentCategory = null;
        this.voltageInfo = null;
        this.callbacks = {
            onLevelChanged: options.onLevelChanged || null,
            onError: options.onError || null,
            onLoading: options.onLoading || null
        };
    }

    /**
     * 初始化管理器
     */
    async init() {
        try {
            await this.fetchCurrentLevel();
            if (this.currentLevel) {
                await this.fetchVoltageInfo();
            }
            return true;
        } catch (error) {
            this._handleError('初始化失败', error);
            return false;
        }
    }

    /**
     * 获取当前电压等级
     */
    async fetchCurrentLevel() {
        try {
            const response = await fetch(`${this.apiBase}/current`);
            const data = await response.json();
            
            if (data.success) {
                this.currentLevel = data.voltage_level;
                this.currentCategory = data.category;
                return data;
            }
            return null;
        } catch (error) {
            this._handleError('获取当前电压等级失败', error);
            return null;
        }
    }

    /**
     * 设置电压等级
     * @param {string} level - 电压等级
     */
    async setVoltageLevel(level) {
        this._setLoading(true);
        
        try {
            const response = await fetch(`${this.apiBase}/set`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ level: level })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentLevel = data.voltage_level;
                this.currentCategory = data.category;
                await this.fetchVoltageInfo();
                
                if (this.callbacks.onLevelChanged) {
                    this.callbacks.onLevelChanged(data);
                }
                
                return data;
            } else {
                throw new Error(data.message || '切换失败');
            }
        } catch (error) {
            this._handleError('设置电压等级失败', error);
            return null;
        } finally {
            this._setLoading(false);
        }
    }

    /**
     * 获取完整电压等级信息
     */
    async fetchVoltageInfo() {
        try {
            const response = await fetch(`${this.apiBase}/info`);
            const data = await response.json();
            
            if (data.success) {
                this.voltageInfo = data.info;
                return data.info;
            }
            return null;
        } catch (error) {
            this._handleError('获取电压等级信息失败', error);
            return null;
        }
    }

    /**
     * 获取指定电压等级的信息
     * @param {string} level - 电压等级
     */
    async fetchLevelInfo(level) {
        try {
            const response = await fetch(`${this.apiBase}/info/${encodeURIComponent(level)}`);
            const data = await response.json();
            return data.success ? data.info : null;
        } catch (error) {
            this._handleError('获取电压等级信息失败', error);
            return null;
        }
    }

    /**
     * 获取所有电压分类
     */
    async fetchCategories() {
        try {
            const response = await fetch(`${this.apiBase}/categories`);
            const data = await response.json();
            return data.success ? data.categories : [];
        } catch (error) {
            this._handleError('获取电压分类失败', error);
            return [];
        }
    }

    /**
     * 获取所有可用电压等级
     */
    async fetchAvailableLevels() {
        try {
            const response = await fetch(`${this.apiBase}/available`);
            const data = await response.json();
            return data.success ? data.levels : {};
        } catch (error) {
            this._handleError('获取可用电压等级失败', error);
            return {};
        }
    }

    /**
     * 获取设备配置
     * @param {string} equipmentType - 设备类型
     * @param {string} [level] - 指定电压等级（可选）
     */
    async fetchEquipmentConfig(equipmentType, level = null) {
        try {
            let url = `${this.apiBase}/config/${equipmentType}`;
            if (level) {
                url += `?level=${encodeURIComponent(level)}`;
            }
            
            const response = await fetch(url);
            const data = await response.json();
            return data.success ? data.config : null;
        } catch (error) {
            this._handleError('获取设备配置失败', error);
            return null;
        }
    }

    /**
     * 获取所有设备配置
     */
    async fetchAllConfig() {
        try {
            const response = await fetch(`${this.apiBase}/config`);
            const data = await response.json();
            return data.success ? data.config : null;
        } catch (error) {
            this._handleError('获取所有配置失败', error);
            return null;
        }
    }

    /**
     * 获取模型状态
     */
    async fetchModelStatus() {
        try {
            const response = await fetch(`${this.apiBase}/model-status`);
            const data = await response.json();
            return data.success ? data : null;
        } catch (error) {
            this._handleError('获取模型状态失败', error);
            return null;
        }
    }

    /**
     * 获取所有模型路径
     */
    async fetchModels() {
        try {
            const response = await fetch(`${this.apiBase}/models`);
            const data = await response.json();
            return data.success ? data.models : null;
        } catch (error) {
            this._handleError('获取模型列表失败', error);
            return null;
        }
    }

    /**
     * 获取支持的插件
     */
    async fetchPlugins() {
        try {
            const response = await fetch(`${this.apiBase}/plugins`);
            const data = await response.json();
            return data.success ? data.plugins : [];
        } catch (error) {
            this._handleError('获取插件列表失败', error);
            return [];
        }
    }

    /**
     * 获取所有可用插件
     */
    async fetchAllPlugins() {
        try {
            const response = await fetch(`${this.apiBase}/plugins/all`);
            const data = await response.json();
            return data.success ? data.plugins : [];
        } catch (error) {
            this._handleError('获取所有插件失败', error);
            return [];
        }
    }

    /**
     * 获取插件详情
     * @param {string} pluginId - 插件ID
     */
    async fetchPluginInfo(pluginId) {
        try {
            const response = await fetch(`${this.apiBase}/plugins/${pluginId}`);
            const data = await response.json();
            return data.success ? data.plugin : null;
        } catch (error) {
            this._handleError('获取插件信息失败', error);
            return null;
        }
    }

    /**
     * 获取检测类别
     * @param {string} [equipmentType] - 设备类型（可选）
     */
    async fetchDetectionClasses(equipmentType = null) {
        try {
            const url = equipmentType 
                ? `${this.apiBase}/detection-classes/${equipmentType}`
                : `${this.apiBase}/detection-classes`;
            
            const response = await fetch(url);
            const data = await response.json();
            return data.success ? data.detection_classes : null;
        } catch (error) {
            this._handleError('获取检测类别失败', error);
            return null;
        }
    }

    /**
     * 获取热成像阈值
     */
    async fetchThermalThresholds() {
        try {
            const response = await fetch(`${this.apiBase}/thermal-thresholds`);
            const data = await response.json();
            return data.success ? data.thresholds : null;
        } catch (error) {
            this._handleError('获取热成像阈值失败', error);
            return null;
        }
    }

    /**
     * 获取开关角度参考值
     * @param {string} switchType - 开关类型 (breaker/isolator/grounding)
     */
    async fetchAngleReference(switchType) {
        try {
            const response = await fetch(`${this.apiBase}/angle-reference/${switchType}`);
            const data = await response.json();
            return data.success ? data.angle_reference : null;
        } catch (error) {
            this._handleError('获取角度参考值失败', error);
            return null;
        }
    }

    /**
     * 获取特殊功能
     */
    async fetchSpecialFeatures() {
        try {
            const response = await fetch(`${this.apiBase}/special-features`);
            const data = await response.json();
            return data.success ? data.special_features : [];
        } catch (error) {
            this._handleError('获取特殊功能失败', error);
            return [];
        }
    }

    /**
     * 比较两个电压等级
     * @param {string} level1 - 第一个电压等级
     * @param {string} level2 - 第二个电压等级
     */
    async compareLevels(level1, level2) {
        try {
            const url = `${this.apiBase}/compare?level1=${encodeURIComponent(level1)}&level2=${encodeURIComponent(level2)}`;
            const response = await fetch(url);
            const data = await response.json();
            return data.success ? data.comparison : null;
        } catch (error) {
            this._handleError('比较电压等级失败', error);
            return null;
        }
    }

    // -------------------------------------------------------------------------
    // 辅助方法
    // -------------------------------------------------------------------------

    /**
     * 获取电压分类的颜色
     * @param {string} category - 电压分类
     */
    getCategoryColor(category) {
        const colors = {
            '特高压': '#dc3545',
            '超高压': '#fd7e14',
            '高压': '#28a745',
            '中压': '#17a2b8',
            '低压': '#6c757d'
        };
        return colors[category] || '#6c757d';
    }

    /**
     * 获取电压分类的图标
     * @param {string} category - 电压分类
     */
    getCategoryIcon(category) {
        const icons = {
            '特高压': 'bi-lightning-fill',
            '超高压': 'bi-broadcast',
            '高压': 'bi-building',
            '中压': 'bi-house-door',
            '低压': 'bi-plug'
        };
        return icons[category] || 'bi-circle';
    }

    /**
     * 获取设备类型的中文名
     * @param {string} type - 设备类型
     */
    getEquipmentName(type) {
        const names = {
            'transformer': '变压器',
            'switch': '开关设备',
            'busbar': '母线',
            'capacitor': '电容器',
            'meter': '表计',
            'dc_system': '直流系统',
            'gis': 'GIS组合电器'
        };
        return names[type] || type;
    }

    /**
     * 格式化模型状态
     * @param {boolean} ready - 是否就绪
     */
    formatModelStatus(ready) {
        return ready 
            ? '<span class="badge bg-success">已就绪</span>'
            : '<span class="badge bg-warning">待训练</span>';
    }

    /**
     * 格式化热成像阈值显示
     * @param {object} thresholds - 阈值对象
     */
    formatThermalThresholds(thresholds) {
        if (!thresholds) return '';
        
        return `
            <span class="badge bg-success">正常: ${thresholds.normal || '-'}°C</span>
            <span class="badge bg-warning">警告: ${thresholds.warning || '-'}°C</span>
            <span class="badge bg-danger">告警: ${thresholds.alarm || '-'}°C</span>
        `;
    }

    // -------------------------------------------------------------------------
    // 私有方法
    // -------------------------------------------------------------------------

    _handleError(message, error) {
        console.error(message, error);
        if (this.callbacks.onError) {
            this.callbacks.onError(message, error);
        }
    }

    _setLoading(loading) {
        if (this.callbacks.onLoading) {
            this.callbacks.onLoading(loading);
        }
    }
}

// -------------------------------------------------------------------------
// UI组件类
// -------------------------------------------------------------------------

/**
 * 电压等级选择器UI组件
 */
class VoltageSelector {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.manager = options.manager || new VoltageManager();
        this.onSelect = options.onSelect || null;
        this.categories = [];
    }

    async render() {
        this.categories = await this.manager.fetchCategories();
        
        let html = '<div class="voltage-selector">';
        
        for (const category of this.categories) {
            const color = this.manager.getCategoryColor(category.category);
            const icon = this.manager.getCategoryIcon(category.category);
            
            html += `
                <div class="voltage-category mb-3">
                    <div class="category-header p-2 text-white rounded-top" style="background: ${color}">
                        <i class="bi ${icon}"></i> ${category.category}
                        <small class="d-block">${category.description}</small>
                    </div>
                    <div class="category-levels p-2 border border-top-0 rounded-bottom">
            `;
            
            for (const level of category.levels) {
                const isActive = level === this.manager.currentLevel;
                html += `
                    <button class="btn btn-sm ${isActive ? 'btn-primary' : 'btn-outline-secondary'} voltage-btn m-1"
                            data-level="${level}">
                        ${level}
                    </button>
                `;
            }
            
            html += '</div></div>';
        }
        
        html += '</div>';
        this.container.innerHTML = html;
        
        // 绑定事件
        this.container.querySelectorAll('.voltage-btn').forEach(btn => {
            btn.addEventListener('click', () => this._onLevelClick(btn.dataset.level));
        });
    }

    async _onLevelClick(level) {
        const result = await this.manager.setVoltageLevel(level);
        if (result && this.onSelect) {
            this.onSelect(result);
        }
        this.render(); // 重新渲染更新状态
    }
}

/**
 * 电压等级信息显示组件
 */
class VoltageInfoDisplay {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.manager = options.manager || new VoltageManager();
    }

    async render() {
        const info = this.manager.voltageInfo;
        if (!info) {
            this.container.innerHTML = '<div class="alert alert-info">请先选择电压等级</div>';
            return;
        }

        const html = `
            <div class="voltage-info">
                <div class="row">
                    <div class="col-md-6">
                        <h5>基本信息</h5>
                        <table class="table table-sm">
                            <tr><td>电压等级</td><td><strong>${info.voltage_level}</strong></td></tr>
                            <tr><td>分类</td><td><span class="badge" style="background: ${this.manager.getCategoryColor(info.category)}">${info.category}</span></td></tr>
                            <tr><td>描述</td><td>${info.category_description}</td></tr>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h5>热成像阈值</h5>
                        ${this.manager.formatThermalThresholds(info.thermal_thresholds)}
                    </div>
                </div>
                
                <h5 class="mt-3">支持的插件 (${info.supported_plugins?.length || 0})</h5>
                <div class="row">
                    ${(info.supported_plugins || []).map(p => `
                        <div class="col-md-4 mb-2">
                            <div class="card card-body p-2">
                                <strong>${p.name}</strong>
                                <small class="text-muted d-block">${p.description}</small>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <h5 class="mt-3">特殊功能</h5>
                <div>
                    ${(info.special_features || []).map(f => `
                        <span class="badge bg-warning text-dark me-1 mb-1">${f}</span>
                    `).join('')}
                </div>
            </div>
        `;
        
        this.container.innerHTML = html;
    }
}

// 导出到全局
if (typeof window !== 'undefined') {
    window.VoltageManager = VoltageManager;
    window.VoltageSelector = VoltageSelector;
    window.VoltageInfoDisplay = VoltageInfoDisplay;
}

// ES6模块导出
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { VoltageManager, VoltageSelector, VoltageInfoDisplay };
}
