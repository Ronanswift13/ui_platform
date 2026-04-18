# Web Modeling Prompt — 表计读数

## 用途
当需要将表计读数算法的内部状态可视化到 Standalone Web 界面时使用。

## 提示模板

你正在为表计读数插件 (meter_reading) 的 Standalone Web 界面建模。
当前界面在 `standalone/templates/meter_reading.html`。

### 已有指标
- Reading Status (Ready/Running/Error)
- Success Rate (%)
- Inference Count (总推理次数)
- Manual Review Count (待复核数)

### 建模要求
1. 新增指标时，遵循已有的 `updateDashboard(data)` 函数模式
2. 数据通过 Flask 后端 (standalone/app.py) 的 StandalonePluginRunner 传递
3. 不得在前端直接调用算法，所有数据经后端聚合
4. 界面风格保持一致 (Bootstrap/原生 CSS)

### 输出格式
- HTML 片段 (嵌入 meter_reading.html)
- 对应的 JavaScript 更新逻辑
- 后端需要提供的数据字段
