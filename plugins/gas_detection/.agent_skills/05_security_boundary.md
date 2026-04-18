# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：默认只读或只改当前插件目录。
2. 高风险改动先确认：平台契约、服务暴露、外部模型依赖、持久化行为。
3. 不把站端运行数据、设备 ID 批量写进日志或测试快照。
4. 禁止破坏性命令和无说明的跨插件批量修改。

## 2. 当前文件边界

### 允许自动改动

- `plugins/gas_detection/.agent_skills/**`
- `plugins/gas_detection/tests/**`
- `plugins/gas_detection/scripts/**`

### 需人工确认后改动

- `plugins/gas_detection/plugin.py`
- `plugins/gas_detection/predictor.py`
- `plugins/gas_detection/analyzer.py`
- `plugins/gas_detection/manifest.json`
- `plugins/gas_detection/requirements.txt`
- `plugins/gas_detection/standalone/**`

### 禁止擅自改动

- `darkbreaker_sdk/**`
- `ai_models/**`
- 其他插件目录
- 已存在的 `data/results.db` 历史数据内容

## 3. 本项目特殊安全关注

1. **服务暴露边界**
   - standalone 默认 `0.0.0.0:8094`
   - 在未确认网络边界前，只能视为本地/内网调试入口
2. **运行数据最小披露**
   - `gas_readings`、环境参数、设备 ID 可能属于运维敏感数据
   - 调试时尽量使用最小示例数据
3. **模型依赖最小化**
   - 不允许为了“修好文档”自动引入远端下载模型的逻辑
4. **持久化边界**
   - 当前不得宣称插件会把每次检测结果写库
   - 若未来接入 `results.db`，应最小化保存内容并避免落全量原始历史

## 4. 依赖安全

1. 当前主链路核心依赖：`darkbreaker_sdk`、`numpy`
2. 可选增强依赖：`ai_models.deep_learning.gl_translstm`
3. 声明型依赖与实际依赖存在漂移，修复时应优先同步文档与实现，而不是新增不必要依赖
4. 不允许引入需要外网访问的运行时依赖作为默认前提
