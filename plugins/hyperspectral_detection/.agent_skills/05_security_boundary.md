# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：默认只读或只改当前插件目录。
2. 高风险改动先确认：平台契约、服务暴露、真实模型依赖、文件 I/O。
3. 不在日志或测试输出中打印完整高光谱数组。
4. 禁止破坏性命令和无说明的跨插件改动。

## 2. 当前文件边界

### 允许自动改动

- `plugins/hyperspectral_detection/.agent_skills/**`
- `plugins/hyperspectral_detection/tests/**`
- `plugins/hyperspectral_detection/scripts/**`

### 需人工确认后改动

- `plugins/hyperspectral_detection/plugin.py`
- `plugins/hyperspectral_detection/manifest.json`
- `plugins/hyperspectral_detection/requirements.txt`
- `plugins/hyperspectral_detection/standalone/**`
- `plugins/hyperspectral_detection/demo/**`

### 禁止擅自改动

- `darkbreaker_sdk/**`
- 其他插件目录
- 平台级服务或模型仓目录

## 3. 本项目特殊安全关注

1. **服务暴露边界**
   - standalone 默认监听 `0.0.0.0:8095`
   - 在未确认网络边界前，只能视为本地/内网调试入口
2. **高维数据最小披露**
   - 高光谱立方体体量大，可能包含敏感设备细节
   - 调试与测试中应优先使用小尺寸模拟数据
3. **模型依赖最小化**
   - 不允许为了“补齐能力”自动引入远端模型下载
4. **文件输入边界**
   - 当前主链路不依赖真实文件路径或相机设备
   - 若未来引入真实数据采集，必须先明确文件大小、格式和权限边界

## 4. 依赖安全

1. 当前主链路核心依赖：`darkbreaker-sdk`、`numpy`
2. 其他声明依赖当前并未在主链路真正使用
3. 不允许引入需要外网访问的运行时依赖作为默认前提
