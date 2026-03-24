# 07 学习日志

## 用途
记录项目开发过程中的经验教训、问题解决方案和最佳实践。

## 日志格式

```markdown
### [日期] 标题

**问题**: 描述遇到的问题
**原因**: 根因分析
**解决**: 解决方案
**教训**: 经验总结
```

---

## 日志记录

### [2026-03-19] 初始化质量保障基础设施

**问题**: animal_detection 插件缺乏系统化的质量保障流程
**原因**: 早期快速开发，未建立完整的 CI/CD 流程
**解决**: 参考 indoor_fence 插件，建立完整的 .agent_skills 和 scripts 体系
**教训**: 质量保障基础设施应在项目初期就建立，而非事后补充

---

### [2026-03-19] Python 3.13 Homebrew 环境依赖问题

**问题**: 使用 `/opt/homebrew/bin/python3.13` 运行 `run_standalone.py` 报错 `ModuleNotFoundError: No module named 'numpy'`

**原因**:
1. Homebrew 管理的 Python 3.13 是 externally-managed-environment，禁止直接 `pip install`
2. 系统 Python 环境没有安装项目依赖

**解决**:
```bash
# 在项目目录创建虚拟环境
/opt/homebrew/bin/python3.13 -m venv .venv

# 激活并安装依赖
source .venv/bin/activate
pip install numpy opencv-python pyyaml flask onnxruntime pydantic fastapi uvicorn python-multipart
```

**教训**:
- Homebrew Python 3.13+ 强制使用虚拟环境 (PEP 668)
- 项目应提供完整的 requirements.txt，包含 SDK 的隐式依赖
- 启动脚本应检测虚拟环境并给出提示

**推荐启动方式**:
```bash
# 首次运行
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 后续运行
source .venv/bin/activate
python run_standalone.py
```

---

### [2026-03-19] 端口占用问题

**问题**: 服务启动报错 `[Errno 48] address already in use`

**原因**: 之前的进程未正常退出，端口 8082 仍被占用

**解决**:
```bash
# 查找并杀死占用端口的进程
lsof -ti:8082 | xargs kill -9

# 或使用 pkill
pkill -f "run_standalone.py"
```

**教训**: 开发时应使用 Ctrl+C 正常退出，避免后台残留进程

---

### [2026-03-25] 多插件重复出现 numpy 缺失的统一根因与修复模板

**问题**: 多个插件 standalone 启动时报 `ModuleNotFoundError: No module named 'numpy'`，常见命令形态为：
`/opt/homebrew/bin/python3.13 <plugin>/run_standalone.py`

**原因**:
1. **解释器漂移**：即使 shell 显示 `(.venv)`，只要显式调用 `/opt/homebrew/bin/python3.13`，实际仍在用系统解释器，不会读取插件 `.venv` 的依赖。
2. Homebrew Python 3.13 属于 externally-managed-environment，系统层依赖管理不应承担项目运行依赖。
3. 启动流程缺少“解释器一致性检查 + 依赖验证 + 启动回归”的标准闭环。

**解决**:
- 新增可复用修复模板：`08_runtime_dependency_repair_prompt.md`
- 统一修复策略：
  - 先判定解释器是否为 `.venv/bin/python`
  - 再执行 `.venv` 内 `pip install -r requirements.txt`
  - 用 `.venv/bin/python run_standalone.py` 验证启动
  - 失败则读取新 traceback 继续循环

**教训**:
- 这类问题优先修复“启动方式”，其次才是“代码依赖”。
- 只看提示符 `(.venv)` 不可靠，必须看 `sys.executable`。
- 应把修复经验沉淀成 prompt，形成可重复执行的自动修复路径。

**本次验证**:
- 系统 Python 启动：复现 `No module named 'numpy'`
- `.venv/bin/python` 启动：服务正常启动（Uvicorn running）

---

*后续问题解决记录将追加在此*
