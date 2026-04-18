# 05_security_boundary

## 1. 固定母版规则（跨项目）

1. 最小权限：只读/只改当前插件目录。
2. 高风险操作先确认：系统配置、跨仓库批量改动、外发动作。
3. 秘密信息最小暴露：日志/测试样本不落敏感信息。
4. 任何破坏性命令默认禁止。

## 2. 本项目差异边界（busbar_inspection）

### 2.1 文件边界

- **允许自动改动**：
  - `plugins/busbar_inspection/.agent_skills/**`
  - `plugins/busbar_inspection/tests/**`
  - `plugins/busbar_inspection/scripts/**`
  - `plugins/busbar_inspection/CLAUDE.md`
  - `plugins/busbar_inspection/PROJECT_CARD.md`
- **需人工确认后改动**：
  - `plugins/busbar_inspection/plugin.py`
  - `plugins/busbar_inspection/detector_enhanced.py`
  - `plugins/busbar_inspection/configs/default.yaml`
- **默认禁止改动**：
  - 插件目录外其他业务插件文件
  - 平台级 SDK 核心文件

### 2.2 行为边界

- 禁止访问外网 API。
- 禁止把原始巡检图写入仓库。
- 禁止把设备 ID/站点 ID 全量写进调试输出。
- 禁止在测试夹具中放入真实生产数据。

## 3. 可执行安全检查

```bash
# 1) 检查潜在外网调用
rg -n "requests\.|http://|https://|urllib|aiohttp" plugin.py detector_enhanced.py standalone

# 2) 检查潜在敏感输出
rg -n "task_id|site_id|device_id|component_id|token|secret|password" plugin.py detector_enhanced.py tests

# 3) 检查破坏性命令残留
rg -n "rm -rf|sudo |os\.remove\(|shutil\.rmtree\(" scripts
```

## 4. AI 自动闭环 / 人工确认

### 可自动闭环

- 安全扫描与报告
- 日志脱敏修复
- 测试数据路径隔离

### 必须人工确认

- 是否允许跨插件批量修复
- 是否允许改动平台级 SDK 代码
- 是否允许引入新外部依赖

## 5. 阻断条件

1. 扫描到外网调用并未在白名单内。
2. 扫描到原始图像持久化到仓库路径。
3. 扫描到敏感信息明文输出。
