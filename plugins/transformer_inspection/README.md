# 主变自主巡视插件 (A组)

## 功能范围

主变压器本体及附属设施（套管、散热器、油位计、呼吸器、端子箱等）

## 核心任务

1. **外观缺陷识别**: 识别破损、锈蚀、渗漏油、异物悬挂
2. **状态识别**: 呼吸器硅胶变色识别、阀门开闭状态
3. **热成像集成**: 提供红外图像的温度提取接口，输出热点坐标及温度值

## 接口规范

### 输入

```python
def infer(
    frame: np.ndarray,      # BGR格式图像
    rois: list[ROI],        # 识别区域列表
    context: PluginContext, # 运行上下文
) -> list[RecognitionResult]
```

### 输出

```json
{
    "task_id": "xxx",
    "roi_id": "xxx",
    "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
    "label": "正常|破损|锈蚀|渗漏油|异物",
    "value": null,
    "confidence": 0.95,
    "model_version": "v1.0",
    "code_version": "abc123"
}
```

## ROI类型

| ROI类型 | 说明 | 识别目标 |
|---------|------|----------|
| bushing | 套管 | 破损、污损 |
| radiator | 散热器 | 渗漏油、锈蚀 |
| oil_level | 油位计 | 油位读数 |
| breather | 呼吸器 | 硅胶颜色 |
| terminal_box | 端子箱 | 外观异常 |

## 交付清单

- [ ] 完整代码包
- [ ] requirements.txt
- [ ] 回放测试数据 (至少20段视频或200张图)
- [ ] 模型文件
- [ ] 配置样例 (configs/default.yaml)
- [ ] 单元测试 (tests/test_plugin.py)

## 性能要求

- 单帧处理时间: < 500ms
- 光照鲁棒性: 支持室外光照变化
- 置信度阈值: 可配置,默认0.5

## 已知限制

待A组交付后补充
