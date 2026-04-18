# 01_architecture_rules

## 1. 固定母版规则（跨项目不可变）

1. **层级方向固定**：接口层 -> 算法层 -> 配置层；禁止反向依赖。
2. **算法层纯业务**：算法层不得依赖 SDK schema；SDK 适配必须留在接口层。
3. **standalone 隔离**：`standalone/` 仅做运行与展示，不承载算法决策。
4. **配置单一来源**：运行参数只从 YAML/环境注入，不在算法主链写死阈值。
5. **循环依赖禁止**：任意模块间不得形成循环 import。

## 2. 本项目差异规则（busbar_inspection）

### 2.1 目录改动权限

- **允许直接修改**：`tests/`、`.agent_skills/`、`scripts/`、`CLAUDE.md`
- **允许但需契约同步**：`plugin.py`、`detector_enhanced.py`、`configs/default.yaml`
- **禁止修改**：`manifest.json` 的 `id/entrypoint/plugin_class`（除非人工批准并同步平台）

### 2.2 依赖方向（必须满足）

```
plugin.py ---------------> detector_enhanced.py
plugin.py ---------------> darkbreaker_sdk.*
standalone/* ------------> plugin.py

detector_enhanced.py -X-> darkbreaker_sdk.*
detector_enhanced.py -X-> standalone.*
```

### 2.3 架构不变量

1. `plugin.py` 负责：ROI 提取、结果组装、告警组装、健康检查。
2. `detector_enhanced.py` 负责：质量门禁、检测链路、NMS、变焦建议。
3. `configs/default.yaml` 负责：阈值与运行参数；算法代码仅读取，不新增业务常量。
4. `tests/` 负责：契约验证与回归，不允许只做“能跑”测试。

## 3. 强制反模式拦截

1. 禁止新增 `print()` 到生产模块（`plugin.py`、`detector_enhanced.py`）。
2. 禁止裸 `except:` 或 `except Exception: pass`。
3. 禁止在算法层直接读取 SDK `ROI/RecognitionResult` 对象。
4. 禁止在 `infer()` 中写磁盘原始图像。

## 4. 可执行架构校验

```bash
# A. 检查 detector_enhanced.py 不依赖 SDK/standalone
rg -n "darkbreaker_sdk|standalone" detector_enhanced.py
# 期望：无匹配

# B. 检查生产代码无 print()（技术债清零后应无输出）
rg -n "\bprint\(" plugin.py detector_enhanced.py

# C. 检查裸 except
rg -n "except\s*:|except\s+Exception\s*:\s*pass" plugin.py detector_enhanced.py
# 期望：无匹配

# D. 检查循环依赖（可选）
python -m pip show pydeps >/dev/null 2>&1 && pydeps . --max-bacon=2 --noshow || true
```

## 5. AI 自动闭环 / 人工确认

### 可自动闭环

- import 关系修复
- 生产模块 `print()` -> `logger` 迁移
- 目录职责越界修复
- 架构检查脚本更新

### 必须人工确认

- 是否引入新顶层目录（如 `core/`、`adapters/`）
- 是否调整 `manifest.json` 插件元信息
- 是否扩展缺陷分类体系并同步平台字典

## 6. 违反规则的阻断条件

任一条件命中即阻断合并：

1. `detector_enhanced.py` 出现 SDK 依赖。
2. 生产模块出现裸 `except`。
3. 生产模块新增 `print()`。
4. 架构校验命令返回非零并且不是“工具未安装”导致。
