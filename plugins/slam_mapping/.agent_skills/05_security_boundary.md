# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：默认只读或只改当前插件目录。
2. 高风险改动先确认：平台合同、导出路径、外部模型、服务暴露。
3. 不在日志或测试快照中输出完整点云或大地图数据。
4. 禁止破坏性命令和无说明的跨插件改动。

## 2. 当前文件边界

### 允许自动改动

- `plugins/slam_mapping/.agent_skills/**`
- `plugins/slam_mapping/tests/**`
- `plugins/slam_mapping/scripts/**`（若后续新建）

### 需人工确认后改动

- `plugins/slam_mapping/plugin.py`
- `plugins/slam_mapping/semantic_slam_plugin.py`
- `plugins/slam_mapping/manifest.json`
- `plugins/slam_mapping/requirements.txt`
- `plugins/slam_mapping/standalone/**`
- `plugins/slam_mapping/data/results.db`

### 禁止擅自改动

- `darkbreaker_sdk/**`
- 其他插件目录
- 外部模型仓或平台核心模块

## 3. 本项目特殊安全关注

1. **服务暴露边界**
   - standalone 默认监听 `0.0.0.0:8084`
   - 未确认网络边界前只应视为本地/内网调试入口
2. **导出边界**
   - `export_map(filepath)` 可写任意路径
   - 启用或修改时必须先确认写入范围
3. **点云与地图数据最小披露**
   - 点云、地图、设备位置属于敏感站区数据
   - 调试时应使用小型模拟数据
4. **模型依赖边界**
   - 当前 `model_registry` 仅为开关/句柄
   - 不应自动扩展为远程模型访问

## 4. 依赖安全

1. 当前主链路核心依赖：`darkbreaker-sdk`、`numpy`
2. manifest 中的 `scipy` 当前应视为声明依赖，不应假设部署环境已有
3. 不允许引入需要外网访问的运行时依赖作为默认前提
