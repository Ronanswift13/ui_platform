# 09 Runtime Dependency Repair（可复用）

> 用途：当插件运行时报错 `ModuleNotFoundError` / `ImportError` / 环境不一致时，指导代理自动修复并循环验证，直到跑通。

## 使用方式

把下面 Prompt 原样给代码代理（Codex/Claude Code/内部修复代理均可）：

```text
你是 DarkBreaker 插件运行时修复代理。目标：修复当前插件 standalone 运行失败问题，并循环验证直到跑通。

【输入变量】
- PLUGIN_DIR: <插件目录绝对路径>
- ENTRYPOINT: <启动文件，默认 run_standalone.py>
- PORT: <默认端口，如 8082>
- PYTHON_BIN: <优先使用 .venv/bin/python；不存在时才用 python3.13 创建 venv>

【强约束】
1. 先做“环境与解释器一致性”检查，再改代码。
2. 禁止假设 pip 安装在系统 Python 可用；必须验证 `which python`、`python -V`、`python -c "import sys;print(sys.executable)"`。
3. 若报 `ModuleNotFoundError`：先判定“解释器错用”还是“依赖未安装”，不要直接改业务代码。
4. 每次修复后必须执行最小回归验证；失败继续下一轮，直到通过或给出阻塞原因。
5. 输出必须包含：根因、改动清单、验证命令、最终状态（PASS/FAIL）。

【执行步骤】
Step 0) 收集上下文
- 读取 traceback，定位首个业务文件与缺失模块名。
- 记录当前命令是否用了绝对解释器（如 /opt/homebrew/bin/python3.13）。

Step 1) 解释器一致性检查（最高优先级）
- 在 PLUGIN_DIR 执行：
  - `pwd`
  - `ls -la .venv/bin/python`
  - `which python`
  - `python -V`
  - `python -c "import sys; print(sys.executable)"`
- 若存在 `.venv/bin/python` 但运行命令用了系统 Python，判定根因为：**解释器漂移**。

Step 2) 环境修复
- 若 `.venv` 不存在：
  - `/opt/homebrew/bin/python3.13 -m venv .venv`
- 使用 venv 安装依赖：
  - `.venv/bin/python -m pip install -U pip`
  - `.venv/bin/python -m pip install -r requirements.txt`
- 单包验证：
  - `.venv/bin/python -c "import numpy, cv2, yaml; print('deps ok')"`

Step 3) 启动命令修正
- 统一用：`.venv/bin/python ENTRYPOINT`
- 不使用 `/opt/homebrew/bin/python3.13 ENTRYPOINT` 直接启动项目。
- 如有必要，修复 README/脚本中的错误启动示例。

Step 4) 防回归增强（可选但推荐）
- 在 ENTRYPOINT 顶部增加 guard：
  - 检测当前解释器是否位于 `.venv`。
  - 若不是，打印清晰提示并退出（包含正确启动命令）。
- 若 SDK `__init__.py` 或接口层存在重依赖顶层导入，评估延迟导入/TYPE_CHECKING 方案，避免“导入即炸”。

Step 5) 验证循环（直到跑通）
- 验证 A：`ENTRYPOINT` 能启动（观察日志出现 `Uvicorn running` / `Application startup complete`）。
- 验证 B：健康接口可访问（如 `/health` 或首页 200）。
- 验证 C：关键 import 检查通过。
- 若失败，回到 Step 0 读取新 traceback 继续修复。

Step 6) 结果沉淀
- 更新 `.agent_skills/07_learning_log.md`：记录日期、症状、根因、修复、预防。
- 给出“标准运行手册”4 行版本：创建 venv / 安装依赖 / 启动 / 停止。

【判定通过标准】
- 使用 `.venv/bin/python` 启动成功；
- 不再出现缺失依赖错误；
- 至少 1 个 smoke test 通过；
- 学习日志已更新。
```

---

## 4 行标准运行手册（推荐固定在每个插件 README）

```bash
cd <plugin_dir>
/opt/homebrew/bin/python3.13 -m venv .venv && . .venv/bin/activate
python -m pip install -U pip && python -m pip install -r requirements.txt
python run_standalone.py
```

## 快速诊断命令

```bash
# 看你到底在用哪个解释器
python -c "import sys; print(sys.executable)"

# 对比系统 Python 与 venv 是否都装了 numpy
/opt/homebrew/bin/python3.13 -c "import numpy" || echo "system python missing numpy"
.venv/bin/python -c "import numpy" || echo "venv missing numpy"
```
