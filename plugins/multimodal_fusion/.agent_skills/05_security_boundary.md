# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：默认只读或只改当前插件目录。
2. 高风险改动先确认：平台合同、外部模态依赖、决策总线、闭环控制、服务暴露。
3. 不在日志或测试输出中落真实站端多模态数据。
4. 禁止破坏性命令和无说明的跨插件修改。

## 2. 当前文件边界

### 允许自动改动

- `plugins/multimodal_fusion/.agent_skills/**`
- `plugins/multimodal_fusion/tests/**`
- `plugins/multimodal_fusion/scripts/**`

### 需人工确认后改动

- `plugins/multimodal_fusion/plugin.py`
- `plugins/multimodal_fusion/fusion_engine.py`
- `plugins/multimodal_fusion/fusion_engine_enhanced.py`
- `plugins/multimodal_fusion/plugin_v4_bayesian.py`
- `plugins/multimodal_fusion/manifest.json`
- `plugins/multimodal_fusion/requirements.txt`
- `plugins/multimodal_fusion/standalone/**`

### 禁止擅自改动

- `darkbreaker_sdk/**`
- 外部模态插件目录
- 平台核心决策总线或闭环控制模块

## 3. 本项目特殊安全关注

1. **服务暴露边界**
   - standalone 默认监听 `0.0.0.0:8096`
   - 未确认网络边界前只应视为本地/内网调试入口
2. **跨插件依赖边界**
   - 当前 manifest 声明依赖 `acoustic_monitoring` / `gas_detection` / `hyperspectral_detection`
   - 当前代码并不会自动拉起这些插件，修改联动方式需人工确认
3. **闭环/决策边界**
   - 目录文案提到决策总线、闭环控制、证据链
   - 在未看到真实下游接口前，不得擅自扩展自动控制路径
4. **数据最小披露**
   - 多模态原始数据可能包含敏感站端信息
   - 调试与测试应优先使用小型模拟输入

## 4. 依赖安全

1. 当前主链路核心依赖：`darkbreaker-sdk`、`numpy`
2. manifest 里的 `onnxruntime` 当前是声明型依赖，不应假设环境已装
3. 不允许引入需要外网访问的运行时依赖作为默认前提
