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
**教训**: 质量保障基础设施应在项目初期就建立，而非事后补充 (QR-2)

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
- Homebrew Python 3.13+ 强制使用虚拟环境 (PEP 668) (QR-6)
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
3. 启动流程缺少"解释器一致性检查 + 依赖验证 + 启动回归"的标准闭环。

**解决**:
- 新增可复用修复模板：`09_runtime_dependency_repair.md`
- 统一修复策略：
  - 先判定解释器是否为 `.venv/bin/python`
  - 再执行 `.venv` 内 `pip install -r requirements.txt`
  - 用 `.venv/bin/python run_standalone.py` 验证启动
  - 失败则读取新 traceback 继续循环

**教训**:
- 这类问题优先修复"启动方式"，其次才是"代码依赖" (QR-6)
- 只看提示符 `(.venv)` 不可靠，必须看 `sys.executable`
- 应把修复经验沉淀成 prompt，形成可重复执行的自动修复路径

---

### [2026-03-25] 自动修复循环验证完整执行记录

**问题**: 执行 08_runtime_dependency_repair_prompt 全流程，验证 animal_detection 插件能否跑通

**检查结果**:
1. `.venv` 已存在，Python 3.13.12，解释器路径正确
2. `requirements.txt` 中 6 个依赖全部已安装（numpy, cv2, yaml, pydantic, onnxruntime, fastapi/uvicorn）
3. 启动成功：Uvicorn running on 0.0.0.0:8082
4. 健康检查通过：`/api/health` → `{"healthy":true}`
5. 首页 `/` → 200, API 文档 `/docs` → 200
6. 核心类导入正常：`YOLOv8Detector`, `AnimalDetectionResult`

**非致命警告**: `'AnimalDetectionPlugin' object has no attribute 'get_standalone_routes'`
- 原因：`AnimalDetectionPlugin` 未继承 SDK 基类 `BasePlugin`，缺少该方法
- 影响：无，SDK runner 已用 try/except 处理
- 后续可选：让插件类继承 BasePlugin 或手动添加 `get_standalone_routes()` 方法

**新增防回归**: 在 `run_standalone.py` 顶部增加 venv guard
- 若使用系统 Python 启动，打印清晰错误信息并退出
- 检测逻辑：判断 `sys.executable` 是否包含 `/.venv/`

**教训**:
- 本插件环境已完善，首次循环即通过，无需多轮修复
- venv guard 是最有效的防回归措施 (QR-6)
- 健康端点路径是 `/api/health` 而非 `/health`
- 核心类名为 `YOLOv8Detector` 和 `AnimalDetectionResult`

**标准运行手册 (4行)**:
```bash
cd plugins/animal_detection
/opt/homebrew/bin/python3.13 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
python run_standalone.py
```

---

### [2026-03-26] 跨插件可复用审查规则同步 (meter_reading → animal_detection)

**背景**: 从 meter_reading V3.0/V3.1 迭代中提炼出 14 条通用审查规则 (QR-1 ~ QR-14)，
同步到 animal_detection 的 `04_quality_audit.md`，并逐条对照本插件的代码和合约进行适配。

**同步的核心经验**:

1. **配置覆盖"最后一公里" (QR-1)**: 推理中间环节的浮点字面量必须有配置来源。animal_detection 中 bbox 越界衰减系数 `0.8` 需要配置化。
2. **枚举状态锁定 (QR-8)**: 已在 test_event_schema_contract.py 中加入 `len(Enum)` 断言。
3. **置信度入口清洗 (QR-11)**: 已实现 `_sanitize_confidence()` 并有独立测试覆盖。
4. **越界值显式处理 (QR-9)**: detector.py 中 bbox 越界已加 warning + 置信度衰减，但仍可改进为 metadata 标记。
5. **降级链路可审计 (QR-3)**: 降级策略表已在 01_architecture_rules.md 和 02_algorithm_contract.md 中定义，但缺少 metadata 中的降级标记。
6. **幻象默认值 (QR-10)**: `class_map.get(cls_id, OTHER)` 和 `ANIMAL_RISK_MAP.get(class, MEDIUM)` 使用默认值——可接受但应记录。
7. **文档锚点 (QR-14)**: 热验证和驱离功能在配置中默认关闭，文档需标注为可选功能。

**行动项**:

| 优先级 | 行动 | 关联规则 | 状态 |
|--------|------|----------|------|
| P1 | 将 bbox 越界衰减系数 0.8 移入配置 | QR-1 | 🔄 待实施 |
| P1 | 为配置传递链路补充端到端测试 | QR-4 | 🔄 待实施 |
| P1 | 降级链路输出 metadata 中增加 fallback_level | QR-3 | 🔄 待实施 |
| P2 | 图像帧尺寸验证提取为独立函数 | QR-13 | 🔄 待实施 |
| P2 | ROI 坐标裁剪逻辑提取为独立函数 | QR-13 | 🔄 待实施 |
| P2 | 引入按动物类型差异化的置信度阈值 | QR-1, QR-5 | 🔄 待实施 |
| P3 | 准备标定图片集激活回归测试 | QR-2 | 🔄 待实施 |

---

### 跨插件可复用经验总结（供后续插件读取）

**适用范围**: DarkBreaker 所有插件的 standalone 启动修复

**通用修复清单**:
1. **检查 `.venv` 是否存在** → 不存在则 `python3.13 -m venv .venv`
2. **检查解释器** → `sys.executable` 必须包含 `/.venv/`，否则为"解释器漂移"
3. **安装依赖** → `.venv/bin/python -m pip install -r requirements.txt`
4. **端口冲突** → `lsof -ti:<PORT> | xargs kill -9`
5. **启动验证** → `.venv/bin/python run_standalone.py`，检查 `/api/health` 返回 200
6. **添加 venv guard** → 在 `run_standalone.py` 顶部检测解释器路径

**常见陷阱**:
- shell 显示 `(.venv)` 不代表脚本在用 venv — 显式调用 `/opt/homebrew/bin/python3.13` 会绕过 venv
- 健康端点统一为 `/api/health`（SDK runner 注册），而非 `/health`
- `get_standalone_routes` 警告为非致命，不影响正常运行
- Homebrew Python 3.13 是 externally-managed，禁止系统级 pip install

---

## 可复用审查规则索引

以下规则已同步写入 `04_quality_audit.md`，此处仅索引便于快速查阅：

| 规则 ID | 一句话摘要 | 来源 |
|---------|-----------|------|
| QR-1 | 推理路径中的浮点字面量必须有配置来源 | meter_reading 2026-03-10 |
| QR-2 | 代码 > 300 行必须有同级合约 + 测试覆盖 | meter_reading 2026-03-10 |
| QR-3 | 降级链路必须有策略表、降级测试、metadata 标记 | meter_reading 2026-03-10 |
| QR-4 | 配置路径 YAML → 映射层 → 算法层必须端到端可测试 | meter_reading 2026-03-10 |
| QR-5 | 多链路共享的质量评分字段必须有统一的计算语义 | meter_reading 2026-03-10 |
| QR-6 | 运行入口必须含 venv 自动激活守卫 | meter_reading 2026-03-10 |
| QR-7 | `__init__.py` 顶层不得引入 C 扩展依赖 | meter_reading 2026-03-10 |
| QR-8 | 枚举状态集严格等于合约定义，测试锁定成员数 | meter_reading 2026-03-19 |
| QR-9 | 越界值显式处理不得静默截断 | meter_reading 2026-03-19 |
| QR-10 | 数据缺失不得用幻象默认值掩盖 | meter_reading 2026-03-19 |
| QR-11 | 浮点质量信号必须在赋值入口清洗 | meter_reading 2026-03-19 |
| QR-12 | 输出 schema 字段必须由合约驱动测试锁定 | meter_reading 2026-03-19 |
| QR-13 | 输入清洗规则必须有独立纯函数 + 独立测试 | meter_reading 2026-03-19 |
| QR-14 | 文档描述的行为必须有代码实现和测试锚点 | meter_reading 2026-03-19 |

---

*后续问题解决记录将追加在此*
