# 04_quality_audit

## 1. 固定母版规则（零容忍项）

1. 禁止 `except: pass`
2. 禁止在生产主链路新增 `print()`
3. 禁止把未接入的模型能力写成已落地能力
4. 禁止修改合同却不补最小回归验证
5. 禁止把 demo 成功当成测试完备

## 2. 本项目高优先级审计项

1. **manifest 配置键名漂移**
   - `manifest.json.default_config` 用的是 `spectral_bands` / `wavelength_start_nm` / `wavelength_end_nm`
   - `_parse_config()` 只读 `wavelength_range` / `num_bands`
2. **依赖声明漂移**
   - manifest 声明 `scipy` / `scikit-learn` / `onnxruntime`
   - `requirements.txt` 未显式列出
3. **入口文案漂移**
   - `__main__.py` usage 提到 `run_standalone.py`
   - 当前文件不存在
4. **光谱维度推断不稳**
   - `shape[0] < shape[2]` 的 band 轴启发式会误判部分 3D 输入
5. **占位能力未标注风险**
   - `defect_detection` 当前固定无缺陷
   - `material_analysis` 当前固定主材为 `copper`
6. **占位参数未接入**
   - `analysis_type`
   - `_model_registry`
   - `confidence_threshold`
   - `pca_components`
   - `defect_types`
   - `spatial_resolution`
7. **元数据漂移**
   - `manifest.version = 1.0.0`
   - `plugin.PLUGIN_VERSION = 1.0.1`
   - `__init__.__version__ = 1.0.0`

## 3. 反模式清单

| 反模式 | 检测方法 | 严重度 |
|---|---|---|
| 默认配置声明不生效 | `rg 'spectral_bands|wavelength_start_nm|wavelength_end_nm|wavelength_range|num_bands'` | 高 |
| 入口文案引用缺失文件 | `test -f run_standalone.py` | 中 |
| analysis_type 只读不用 | `rg 'analysis_type' plugin.py` | 中 |
| model_registry 只存不用 | `rg '_model_registry' plugin.py` | 中 |
| 维度误判 | 复现 `(32,32,224)` / `(224,32,32)` 输入 | 高 |
| 依赖漂移 | 对比 manifest 与 requirements | 中 |

## 4. 审计命令

```bash
# 配置与依赖漂移
rg 'spectral_bands|wavelength_start_nm|wavelength_end_nm|wavelength_range|num_bands|onnxruntime|scikit-learn|scipy' plugins/hyperspectral_detection

# 未接入参数
rg 'analysis_type|_model_registry|confidence_threshold|pca_components|defect_types|spatial_resolution' plugins/hyperspectral_detection/plugin.py

# 缺失入口
test -f plugins/hyperspectral_detection/run_standalone.py && echo EXISTS || echo MISSING

# demo 回放
python3 -m plugins.hyperspectral_detection.demo.run_demo
```

## 5. 当前阻断/高风险问题

1. manifest 默认配置与实现消费键名不一致。
2. 光谱 band 轴推断对部分合法输入会给出错误长度输出。

## 6. 当前建议级问题

1. `analysis_type` 和 `_model_registry` 当前更像预留接口，应文档化为未生效能力。
2. `requirements.txt` 与 manifest 依赖列表应同步，避免误导部署侧。
3. `__main__.py` 的 usage 文案应纠正缺失的 `run_standalone.py`。
