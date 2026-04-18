# 07_learning_log

用于记录本插件重要故障、根因与预防动作。每次 `/repair` 或重大质量问题修复后必须追加。

## Entry Template
- Date:
- Context:
- Symptom:
- Root cause:
- Fix:
- Prevention:
- Follow-up:

## Entries

- Date: 2026-04-10
- Context: 独立启动页需要提供可切换的变电站温度仿真，并覆盖热成像与传感器阵列两条输入路径。
- Symptom: `StandalonePluginRunner` 启动时提示插件缺少 `get_standalone_routes`；独立页没有仿真闭环；传感器仿真场景返回 `index 6 is out of bounds for axis 1 with size 6`。
- Root cause: 插件未提供 standalone 专用仿真路由，模板也没有场景驱动逻辑；`detector._interpolate_sensors()` 直接用 `[cols, rows]` 创建网格，导致 `grid[row, col]` 在传感器路径下越界。
- Fix: 新增 standalone 仿真引擎与 `/api/simulation/*` 路由，使用独立插件实例执行演示；重写独立页模板以消费仿真接口；修正传感器插值网格为 `(rows, cols)`。
- Prevention: 将 standalone 仿真路由、页面控件、传感器场景成功返回纳入回归测试；后续涉及 `resolution` 的逻辑统一按 `col,row -> grid[row,col]` 校验。
- Follow-up: 若后续需要更强的热点可视化，可继续在 standalone 场景参数层迭代，不直接改动真实检测阈值与热点算法。
