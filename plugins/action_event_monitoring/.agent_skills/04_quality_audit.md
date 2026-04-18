# 04_quality_audit

## 1. 当前审计范围（按真实结构）

当前本地只能审计以下内容：

1. `manifest.json`
2. `plugin.py`
3. `configs/default.yaml`
4. `configs/topology/*.yaml`
5. `requirements.txt`
6. `demo/run_demo.py`
7. `__main__.py` / `run_standalone.py`
8. `standalone/app.py`
9. `tests/`
10. `scripts/run_sanity_checks.sh`

当前**不能**审计不存在的内容：

1. 本地 REST API
2. 本地 CandidateEvent 接口
3. 本地人工复核接口
4. 本地 `tests/` 覆盖率
5. UI/dashboard/cockpit 前端入口
6. 真实协议联调闭环

## 2. 从旧文档保留下来的有效经验

以下经验仍然有效，但必须按“当前实现能力”解释：

1. 证据不足时不能给高置信度结论。
2. 自动分析结论必须带原因和置信度，不能伪装成确定结论。
3. 误动、顺序异常、缺少断路器响应这类场景天然需要人工介入。
4. 历史库未接入或证据缺失时，应给出明确降级说明，而不是 TODO。

说明：

- 这些是对 `analysis_result` 质量的审查方向。
- 它们不等于当前本地已经实现了 CandidateEvent / review API。

## 3. 当前高优先级审计项

1. `start()` 在协议连接失败时仍返回 `True`，是否会误导上层“插件健康”判断。
2. `plugin.py` 是否继续扩大为“万能单文件”，导致后续不可维护。
3. 是否继续直接访问 `correlation_service._signal_points` 这类私有成员。
4. `default.yaml` 中是否出现真实主机、用户名、密码等敏感配置。
5. `process()` 的返回结构是否仍保持 B 类统一事件输出壳。
6. 全局 installer 和 integration 清单是否仍包含 `action_event_monitoring`，端口是否为 `8097`。

## 4. 最小可执行审计命令

```bash
# 1) manifest 和 config 可解析
python3 - <<'PY'
import json, yaml
from pathlib import Path
json.loads(Path('manifest.json').read_text())
yaml.safe_load(Path('configs/default.yaml').read_text())
print('manifest/config OK')
PY

# 2) 插件最小 sanity
./scripts/run_sanity_checks.sh

# 3) pytest 合同
python3 -m pytest tests -q

# 4) 反模式扫描
rg -n "TODO|FIXME|HACK|except Exception: pass|print\\(" plugin.py
```

## 5. 历史伪完整内容的处理规则

旧版文档里提到的以下内容，不得再当作当前本地事实：

- “统一 `{success, data, message}` 输出”
- “CandidateEvent 数据模型已本地落地”
- “18 个测试已存在”
- “人工复核状态机已本地暴露接口”
- “/api/device-events/{device_id} 已在当前目录实现”
- “UI / dashboard / cockpit 已完成统一入口”

这些若仍需推进，应作为升级路线，不是当前合同。
