# 02 算法契约 — {{PLUGIN_DISPLAY_NAME}}

## 输入契约

<!-- BUSINESS: 定义本插件的输入数据格式、类型、取值范围 -->

## 输出契约

<!-- BUSINESS: 定义本插件的输出结果结构、状态枚举、置信度范围 -->

## 配置映射

所有算法阈值必须来自 `configs/default.yaml`，经适配后注入算法层。

```yaml
# configs/default.yaml 中应包含的关键配置项
# <!-- BUSINESS: 列出本插件的关键配置项 -->
```

## 降级链路

<!-- BUSINESS: 定义当依赖不可用时的降级行为（如模型缺失、传感器离线等） -->

## 验证命令

```bash
./scripts/run_targeted_tests.sh all
```
