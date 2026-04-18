# 05_security_boundary

## 1. 固定母版规则（跨插件）

1. 最小权限：只读写当前插件目录。
2. 默认离线：生产链路不得访问外网。
3. 敏感信息最小暴露：日志与诊断输出不得扩散设备上下文。
4. 破坏性命令默认禁止。

## 2. 本项目差异边界（meter_reading）

### 2.1 运行边界

1. `plugin.py` / `detector_enhanced.py` / `standalone/*.py` 不得引入 `requests`、`aiohttp`、`urllib` 等联网代码。
2. 原始图像不得持久化到仓库；诊断脚本只允许写文本日志和配置快照到 `data/root_cause/`。
3. `task_id / site_id / device_id / component_id` 只用于结果追溯，不得整批打印到调试日志。

### 2.2 输入边界

1. 非法 ROI 必须失败退出，不做静默 clamp 后继续成功。
2. 非法图像维度、dtype、通道数必须在输入校验阶段终止。
3. `context` 字段允许空字符串，但字段名不能缺失。

### 2.3 配置与依赖边界

1. 配置仅接受本地 YAML。
2. 动态加载只允许本地 `detector_enhanced.py` / `detector.py`。
3. 不允许在脚本或运行时动态 `pip install` 依赖。
4. 若未来引入真实模型文件校验，必须在本地完成，不得联网校验。

## 3. 可执行安全检查

```bash
# 1) 外网调用扫描
rg -n "requests\.|http://|https://|urllib|aiohttp" plugin.py detector_enhanced.py standalone

# 2) 原始图像持久化扫描
rg -n "cv2\.imwrite|imwrite\(|Image\.save|open\(.*['\"]wb" plugin.py detector_enhanced.py standalone scripts

# 3) 敏感上下文输出扫描
rg -n "task_id|site_id|device_id|component_id|token|secret|password" plugin.py detector_enhanced.py standalone tests
```

## 4. 阻断条件

1. 发现生产链路外网调用。
2. 发现原始图像写盘到非诊断目录。
3. 发现上下文字段被大批量明文输出。
