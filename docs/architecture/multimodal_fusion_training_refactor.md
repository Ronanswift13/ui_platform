# Multimodal Fusion Training Refactor

## 目标

针对：

- `/Users/ronan/Desktop/DarkBreaker/plugins/multimodal_fusion`
- `/Users/ronan/Desktop/DarkBreaker/training`

建立可工程交付的多模态融合训练库，满足：

- 单样本中多模态对齐上传
- 支持特征级融合和决策级融合
- 支持模态缺失
- 保留“规则 + 模型”的混合诊断能力
- 训练输出直接对齐插件 `supported_modalities`、`plugin_dependencies`、`diagnostic_rules`

## 现状对齐

插件 manifest 已定义：

- `supported_modalities`
  - `visual`
  - `thermal`
  - `acoustic`
  - `ultrasonic`
  - `gas`
  - `hyperspectral`
  - `vibration`
- `plugin_dependencies`
  - `acoustic_monitoring`
  - `gas_detection`
  - `hyperspectral_detection`
- `diagnostic_rules`
  - 变压器综合过热诊断
  - 绝缘劣化诊断
  - 接触电阻升高

插件代码已具备：

- 特征级融合 `_early_fusion()`
- 决策级融合 `_late_fusion()`
- 注意力融合 `_attention_fusion()`
- 混合融合 `_hybrid_fusion()`
- 规则生成 `diagnostic_report / recommendations / alarms`

因此训练库重构不需要推翻插件，而是要把训练产物组织成插件天然能消费的形态。

## 数据结构

### 1. 数据集级 manifest

建议 dataset 级 `manifest.json` 负责声明整体约束：

- `plugin_id`
- `task_type`
- `input_modality=multimodal`
- `supported_modalities`
- `required_modalities`
- `fusion_strategy`
- `missing_modality_policy`
- `alignment_tolerance_ms`
- `multimodal_schema`
- `diagnosis_target_schema`
- `diagnostic_rule_pack`

### 2. 单样本对齐 manifest

单样本对齐使用 `samples.jsonl`，每行一个样本，字段固定为：

- `sample_id`
- `device_id`
- `timestamp`
- `available_modalities`
- `modality_paths`
- `label`
- `diagnosis_target`

可选字段：

- `decision_inputs`
  用于决策级融合时直接提供单模态结果
- `metadata`
  用于设备类型、工况、天气、负载等级等上下文

### 3. `modality_paths` 设计

`modality_paths` 为 `modality -> relative_path` 的映射，例如：

```json
{
  "visual": "visual/sample_0001.json",
  "thermal": "thermal/sample_0001.json",
  "gas": "gas/sample_0001.json"
}
```

工程上不强制所有模态都上传原始大文件。允许三类输入：

- 原始模态数据
- 预提取特征
- 预处理单模态检测结果

这点与插件 `pre_processed / modality_results` 的设计一致。

## 训练任务设计

### 1. `multimodal_feature_fusion`

目标：

- 将多模态编码到统一向量空间
- 融合后直接预测 `overall_status / confidence / detections`

适合：

- 模态特征维度可标准化
- 有较多对齐样本
- 需要学习跨模态互补关系

推荐范式：

- `attention_pooling`
- `transformer_encoder`
- `cross_modal_projection`

导出角色：

- `multimodal_fusion`
- `multimodal_feature_fusion`

### 2. `multimodal_decision_fusion`

目标：

- 各模态先独立出结果
- 再用元学习器 / 贝叶斯 / 校准器做决策融合

适合：

- 各模态模型成熟度不一致
- 线上需更强可解释性
- 模态经常缺失

推荐范式：

- `stacking_ensemble`
- `bayesian_calibration`
- `gradient_boosted_meta_learner`

导出角色：

- `multimodal_decision_fusion`

## 模态缺失补偿策略

### 策略 1: `allow_sparse`

- 样本允许仅有任意子集模态
- 训练时按存在模态聚合
- 适合冷启动阶段

### 策略 2: `required_modalities_only`

- 每个样本必须具备 `required_modalities`
- 其他模态可缺失
- 适合把 `visual + thermal` 作为最低诊断闭环

### 策略 3: `mask_and_gate`

- 为每个样本构建 modality presence mask
- 融合层学习忽略缺失模态
- 推荐作为默认策略

### 策略 4: `teacher_distillation_fallback`

- 全模态样本训练 teacher
- 缺模态样本训练 student
- 用于提升稀疏模态组合场景的稳健性

### 插件侧运行降级顺序

1. `rule_model_hybrid`
2. `decision_level`
3. `rules_only`

也就是模型不完整时仍可用已有 `diagnostic_rules` 维持诊断能力。

## 训练输出设计

统一输出字段直接对齐插件 output schema：

- `overall_status`
- `confidence`
- `detections`
- `modality_contributions`
- `diagnostic_report`

### 输出含义

#### `overall_status`

- `normal / attention / warning / alarm / critical`

#### `confidence`

- 融合总置信度
- 对 feature-level 来自分类头或校准头
- 对 decision-level 来自投票/贝叶斯后验

#### `detections`

- 关联后的融合检测结果
- 每条记录建议至少包含：
  - `fault_type`
  - `severity`
  - `confidence`
  - `supporting_modalities`

#### `modality_contributions`

- 各模态贡献度
- feature-level 可来自注意力权重
- decision-level 可来自加权投票权重

#### `diagnostic_report`

- 保持结构化，不直接只输出纯文本
- 建议包含：
  - `summary`
  - `modality_analysis`
  - `rule_matches`
  - `fault_statistics`
  - `model_evidence`

## 规则 + 模型混合诊断

这是必须保留的能力，不作为 fallback，而作为正式 runtime mode。

### 推荐运行模式

- `rules_only`
  无模型或强降级时使用
- `model_only`
  离线评估或纯模型对比时使用
- `rule_model_hybrid`
  生产默认

### 混合逻辑

1. 单模态子模型先输出特征或检测结果
2. 融合模型给出 `overall_status / confidence / contributions`
3. `diagnostic_rule_pack` 对融合结果做规则匹配
4. 规则结果与模型结果合并成 `diagnostic_report`
5. 最终生成 `recommendations / alarms`

## 模型注册方案

统一使用：

- `plugin_id`
- `task_type`
- `version`
- `model_role`

推荐注册记录：

- `multimodal_fusion:multimodal_feature_fusion:mmf-v1`
- `multimodal_fusion:multimodal_decision_fusion:mmd-v1`

`bundle.json` 额外建议包含：

- `supported_modalities`
- `required_modalities`
- `plugin_dependencies`
- `diagnostic_rule_pack`
- `runtime_mode`
- `missing_modality_policy`
- `alignment_tolerance_ms`

## 插件加载方案

推荐在插件侧增加 3 个组件：

- `FusionFeatureAdapter`
  负责把单模态结果或原始数据映射到统一训练输入结构
- `FusionModelOrchestrator`
  负责按角色加载
  - `multimodal_feature_fusion`
  - `multimodal_decision_fusion`
  - legacy `multimodal_fusion`
- `FusionRulePack`
  负责读取训练导出的规则包或插件 manifest 中的默认规则

### 加载优先级

1. `multimodal_feature_fusion`
2. `multimodal_decision_fusion`
3. `multimodal_fusion` 旧统一模型
4. `rules_only`

## 错误判别与降级机制

### 1. `MODALITY_PATH_MISSING`

- 样本声明该模态可用，但 `modality_paths` 找不到文件
- 处理：
  - 训练侧丢弃该模态输入并记录缺失率
  - 插件侧按缺模态策略降级

### 2. `MODALITY_DEPENDENCY_MISSING`

- 插件 manifest 声明该模态依赖外部插件，但运行环境未注册
- 处理：
  - feature-level 禁用该模态分支
  - decision-level 跳过该模态决策
  - 保留规则诊断

### 3. `ALIGNMENT_WINDOW_EXCEEDED`

- 多模态时间对齐超出 `alignment_tolerance_ms`
- 处理：
  - 训练侧将样本标记为不可靠或降权
  - 插件侧不参与同一轮融合

### 4. `INSUFFICIENT_MODALITIES`

- 不满足 `required_modalities`
- 处理：
  - 若策略是 `required_modalities_only`，直接拒绝融合
  - 否则回退 `rules_only`

### 5. `RULE_PACK_MISSING`

- 决策级融合需要规则包但导出或加载失败
- 处理：
  - 退回纯 decision-level 模型
  - 在 `diagnostic_report` 标注规则链缺失

### 6. `MODEL_MODALITY_MISMATCH`

- 模型 bundle 声明支持模态集合与当前样本不兼容
- 处理：
  - 不加载该模型
  - 尝试更通用的 fusion model 或 rules-only

## 训练数据上传标准

推荐目录：

```text
dataset_root/
├── manifest.json
├── samples.jsonl
├── visual/
├── thermal/
├── acoustic/
├── ultrasonic/
├── gas/
├── hyperspectral/
├── vibration/
├── labels/
└── metadata/
```

说明：

- 各模态目录都可选
- `samples.jsonl` 才是单样本对齐的核心
- `labels/` 可存放全局标签或补充说明
- `metadata/diagnostic_rules.json` 推荐与插件 `diagnostic_rules` 保持同构

## 与插件 manifest 的对齐关系

### `supported_modalities`

训练侧 `supported_modalities` 必须是插件 manifest 子集，不可自行扩展新模态名。

### `plugin_dependencies`

训练侧 `plugin_dependency_map` 直接复用插件 manifest 的依赖关系：

- `acoustic/ultrasonic` -> `acoustic_monitoring`
- `gas` -> `gas_detection`
- `hyperspectral` -> `hyperspectral_detection`
- `visual/thermal` -> `transformer_monitoring`
- `vibration` -> `null`

### `diagnostic_rules`

训练导出的 `diagnostic_rule_pack` 应与插件 manifest 规则同构，允许新增字段：

- `version`
- `source`
- `min_confidence`
- `min_match`

## 已落地到仓库的支撑文件

- [training/multimodal_fusion/task_profiles.py](/Users/ronan/Desktop/DarkBreaker/training/multimodal_fusion/task_profiles.py)
- [training/multimodal_fusion/data_contract.py](/Users/ronan/Desktop/DarkBreaker/training/multimodal_fusion/data_contract.py)
- [training/plugin_configs/multimodal_fusion.yaml](/Users/ronan/Desktop/DarkBreaker/training/plugin_configs/multimodal_fusion.yaml)
- [training/registry/plugin_training_mapping.json](/Users/ronan/Desktop/DarkBreaker/training/registry/plugin_training_mapping.json)
- [plugins/multimodal_fusion/manifest.json](/Users/ronan/Desktop/DarkBreaker/plugins/multimodal_fusion/manifest.json)
- [training/examples/multimodal_fusion_upload/README.md](/Users/ronan/Desktop/DarkBreaker/training/examples/multimodal_fusion_upload/README.md)

## 推荐实施步骤

### Phase 1

- 先上线 `multimodal_feature_fusion` 上传、路由、导出骨架
- 插件仍使用现有 hybrid/rules 逻辑

### Phase 2

- 接入 `multimodal_feature_fusion` 模型加载
- 输出 `overall_status / confidence / modality_contributions`

### Phase 3

- 接入 `multimodal_decision_fusion`
- 与 `diagnostic_rule_pack` 联动

### Phase 4

- 建立模态缺失鲁棒性评估
- 把降级路径固化进插件 orchestrator

### Phase 5

- 统一 model registry + plugin resolver
- 形成上传 -> 训练 -> 导出 -> 插件加载 -> 在线监控闭环
