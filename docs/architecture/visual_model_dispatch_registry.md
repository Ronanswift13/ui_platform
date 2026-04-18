# 视觉插件统一模型调度映射方案

## 1. 目标

本方案覆盖 `/Users/ronan/Desktop/DarkBreaker/plugins` 下 11 个视觉类插件与 `/Users/ronan/Desktop/DarkBreaker/training` 训练成果之间的统一调度映射，第一阶段只解决三件事：

1. 可查找：插件能按 `plugin_id + task_type + version` 找到训练导出物。
2. 可加载：插件能一次拿到模型、`label_map`、`preprocess`、`postprocess` 的路径描述。
3. 可错误判别：插件能把“找不到 / 不兼容 / 契约不一致”分成稳定错误码。

本阶段不负责：

- 具体的 `onnxruntime` / `tensorrt` session 创建
- 业务推理编排
- 告警和业务后处理逻辑

## 2. 统一产物布局

训练导出后，统一落到如下目录：

```text
training/
  exports/
    {plugin_id}/
      {task_type}/
        {version}/
          model.onnx | model.engine | model.pt | model.pth
          label_map.json
          preprocess.yaml | preprocess.json
          postprocess.yaml | postprocess.json
          bundle.json
```

约束：

- `plugin_id` 使用插件真实 ID，例如 `busbar_inspection`，不再直接写旧别名。
- `task_type` 使用运行时任务类型，例如 `detection`、`ocr`、`thermal_anomaly`。
- `version` 是唯一检索维度。对于一个插件下有多个同类模型的场景，`version` 必须做角色前缀约束。
  例如 `meter-det-v1`、`keypoint-v1`、`thermal-check-v1`。
- `bundle.json` 是单个导出目录的自描述文件；`training/registry/model_registry.json` 是全局索引。

## 3. 全局 Registry 结构

全局文件建议固定为：

```text
training/registry/model_registry.json
```

单条记录的最小字段如下：

```json
{
  "plugin_id": "busbar_inspection",
  "task_type": "detection",
  "modality": "rgb",
  "version": "det-v2026.04.14",
  "metrics": {
    "mAP@0.5": 0.82,
    "recall": 0.79
  },
  "compatible_runtime": [
    {
      "runtime": "onnxruntime",
      "min_version": "1.16.0",
      "providers": ["CPUExecutionProvider", "CUDAExecutionProvider"],
      "file_extensions": [".onnx"]
    }
  ],
  "artifacts": {
    "export_dir": "training/exports/busbar_inspection/detection/det-v2026.04.14",
    "model_path": "training/exports/busbar_inspection/detection/det-v2026.04.14/model.onnx",
    "label_map_path": "training/exports/busbar_inspection/detection/det-v2026.04.14/label_map.json",
    "preprocess_config_path": "training/exports/busbar_inspection/detection/det-v2026.04.14/preprocess.yaml",
    "postprocess_config_path": "training/exports/busbar_inspection/detection/det-v2026.04.14/postprocess.yaml"
  }
}
```

其中用户要求的核心字段一一对应：

- `plugin_id`
- `task_type`
- `modality`
- `version`
- `metrics`
- `export_path`
  这里拆成了 `artifacts.export_dir` 和 `artifacts.model_path`
- `compatible_runtime`

附加建议字段：

- `model_role`
  解决同一 `task_type` 下多模型共存的问题
- `labels`
  直接保存运行时标签序列，避免每次都读 `label_map.json`
- `source`
  标识来自 `registry`、`normalized`、`legacy_export`、`legacy_checkpoint`
- `created_at`

样例文件已放在：

- [model_registry.example.json](/Users/ronan/Desktop/DarkBreaker/training/registry/model_registry.example.json)

## 4. 插件映射表

第一阶段建议的统一映射如下。

| plugin_id | legacy_alias | task_type | modality | 说明 |
| --- | --- | --- | --- | --- |
| `busbar_inspection` | `busbar` | `detection` | `rgb` | 主检测模型 |
| `capacitor_inspection` | `capacitor` | `detection` | `rgb` | 主检测模型 |
| `capacitor_inspection` | `capacitor` | `classification` | `rgb` | 可选分类模型 |
| `switch_inspection` | `switch` | `detection` | `rgb` | 主检测模型 |
| `switch_inspection` | `switch` | `classification` | `rgb` | 可选状态分类模型 |
| `transformer_inspection` | `transformer` | `detection` | `rgb` | 主检测模型 |
| `transformer_inspection` | `transformer` | `classification` | `rgb` | 可选状态分类模型 |
| `animal_detection` | - | `detection` | `rgb+thermal` | 主检测模型 |
| `animal_detection` | - | `classification` | `rgb` | 种类分类 |
| `animal_detection` | - | `thermal_anomaly` | `thermal` | 热像活体校验 |
| `bird_monitoring` | - | `detection` | `rgb` | 主检测模型 |
| `bird_monitoring` | - | `classification` | `rgb` | 风险分类 |
| `fire_detection` | - | `detection` | `rgb+thermal` | 主检测模型 |
| `fire_detection` | - | `classification` | `rgb` | 火灾等级分类 |
| `fire_detection` | - | `thermal_anomaly` | `thermal` | 热像异常检测 |
| `meter_reading` | `meter` | `ocr` | `rgb` | 主读数模型 |
| `meter_reading` | `meter` | `detection` | `rgb` | 表计检测模型；若有关键点模型，使用 role-scoped version |
| `temperature_monitoring` | - | `thermal_anomaly` | `thermal` | 主热像模型 |
| `temperature_monitoring` | - | `classification` | `thermal` | 可选等级分类 |
| `thermal` | - | `thermal_anomaly` | `thermal` | 占位插件 |
| `hyperspectral_detection` | - | `hyperspectral_classification` | `hyperspectral` | 主高光谱分类 |
| `hyperspectral_detection` | - | `hyperspectral_anomaly` | `hyperspectral` | 高光谱异常检测 |

说明：

- `meter_reading` 存在一个现实问题：`meter_detector` 和 `keypoint_detector` 都可能属于检测链路。
  第一阶段不强行扩展主键，统一要求 `version` 做角色前缀，例如：
  `meter-det-v1`、`keypoint-v1`。
- `transformer_inspection` 当前插件内有热像逻辑，但训练配置主线仍是 `detection + classification`。
  如果后续把热像模型纳入导出，只需追加 `task_type=thermal_anomaly`，不需要改主设计。

## 5. 插件侧解析接口

第一阶段插件只拿“路径描述”，不直接在统一层里创建推理 session。

统一接口放在：

- [visual_model_registry.py](/Users/ronan/Desktop/DarkBreaker/training/registry/visual_model_registry.py)

插件侧使用方式：

```python
from pathlib import Path
from training.registry.visual_model_registry import PluginModelResolver

resolver = PluginModelResolver(
    "fire_detection",
    training_root=Path("/Users/ronan/Desktop/DarkBreaker/training"),
    plugin_dir=Path("/Users/ronan/Desktop/DarkBreaker/plugins/fire_detection"),
)

bundle = resolver.resolve_bundle(
    "detection",
    version="latest",
    runtime="onnxruntime",
    runtime_version="1.18.0",
    provider="CPUExecutionProvider",
    expected_modality="rgb+thermal",
    require_preprocess=True,
)

plugin_model_config = bundle.to_plugin_config()
```

接口职责：

1. 先查 `training/registry/model_registry.json`
2. 查不到时，扫描规范化目录 `training/exports/{plugin_id}/{task_type}/{version}`
3. 仍查不到时，按过渡策略扫描旧路径
4. 返回统一配置结构给插件 detector
5. 在返回前做兼容性校验

## 6. 错误判别场景

本阶段稳定错误码：

| 场景 | 错误码 | 触发条件 |
| --- | --- | --- |
| 模型不存在 | `MODEL_NOT_FOUND` | 没有 registry 记录，且规范化目录与旧路径都未发现模型 |
| 版本不兼容 | `VERSION_INCOMPATIBLE` | 运行时版本低于或高于模型声明的兼容区间 |
| 标签不一致 | `LABEL_MISMATCH` | 插件期望标签序列与 `label_map` / `labels` 不一致 |
| 输入模态错误 | `MODALITY_MISMATCH` | 插件请求的输入模态与模型声明模态不一致 |
| 推理前处理缺失 | `PREPROCESS_MISSING` | `require_preprocess=True` 且导出物中找不到前处理配置 |

补充错误码：

- `RUNTIME_INCOMPATIBLE`
  运行时后端或 provider 不在 `compatible_runtime` 中
- `POSTPROCESS_MISSING`
  当插件显式要求 `postprocess` 时使用

## 7. 旧路径兼容策略

### 7.1 兼容顺序

插件解析顺序固定为：

1. 全局 registry
2. 规范化目录
3. 旧导出目录
4. 旧 checkpoint
5. 插件本地 `models/`

### 7.2 旧路径扫描规则

兼容以下历史布局：

```text
training/checkpoints/{legacy_alias}/**/best.pt
training/exports/{legacy_alias}/**/*.onnx
plugins/{plugin_id}/models/**/*.onnx
```

### 7.3 过渡期要求

过渡阶段建议：

1. 训练导出同时写新目录和 `bundle.json`
2. registry 先允许 `source=legacy_export` / `source=legacy_checkpoint`
3. 插件若命中旧路径，记录 warning，不中断业务
4. 待所有插件切换到新导出后，再逐步下线 `best.pt/.onnx` 扫描

### 7.4 下线条件

只有当以下条件同时满足时，才能移除旧路径兼容：

1. 11 个视觉插件全部能从统一 registry 拿到模型
2. 每个导出 bundle 都具备 `label_map + preprocess + postprocess + bundle.json`
3. 线上日志中连续一个发布周期没有 `legacy_*` 命中

## 8. 本次落地内容

本次已经补齐：

- 统一 registry / resolver 代码骨架
- registry example
- 错误码与异常类型
- `best.pt/.onnx` 过渡扫描
- 针对关键场景的单测

后续第二阶段再接：

- 训练导出器自动写 `bundle.json`
- 插件 detector 实际 session 初始化与热更新
- 业务级调用编排
