# 02_algorithm_contract

本插件当前更接近“算法优先的 `process()` 合同”，而不是服务编排/数据融合合同，因此沿用 `02_algorithm_contract` 命名。

## 1. 输入契约

### 1.1 `init(config_or_registry=None)`

支持三种真实模式：

1. `None`
   - 使用默认 `HyperspectralConfig()`
2. `dict`
   - 当前 `_parse_config()` 只解析 `wavelength_range` 和 `num_bands`
3. 其他对象
   - 被存到 `_model_registry`
   - 当前不影响后续 `process()` 逻辑

### 1.2 `process(inputs)`

当前真实输入：

```python
{
    "device_id": str,                 # 可缺省，默认 "unknown"
    "image": np.ndarray | None,       # 可缺省；缺省时自动生成模拟高光谱立方体
    "wavelength_range": [float, float],   # 可缺省
    "analysis_type": str,             # 可缺省，默认 "full"
}
```

补充事实：

1. 当前没有强制要求 `image`。
2. 当前没有 input schema 校验。
3. 当前 `analysis_type` 只是读取，并未改变处理分支。

### 1.3 图像形状契约（当前真实行为）

1. `image` 缺失时，自动生成 `shape=(num_bands, 256, 256)` 的随机浮点立方体。
2. 2D 输入也会被接受，并按 `axis=0` 求均值。
3. 3D 输入会通过 `shape[0] < shape[2]` 的启发式猜 band 轴。

注意：

1. 这套启发式并不稳健。
2. 已验证某些合法 3D 形状会得到错误的光谱长度，例如：
   - `(32, 32, 224)` 返回 32 长度光谱
   - `(224, 32, 32)` 也会返回 32 长度光谱

## 2. 输出契约

### 2.1 `process()` 成功返回（当前真实结构）

```python
{
    "success": True,
    "device_id": str,
    "timestamp": float,
    "overall_status": str,            # 当前仅 "normal" 或 "alarm"
    "spectrum_analysis": {
        "wavelengths": list[float],
        "mean_spectrum": list[float],
        "spectral_range": {
            "min": float,
            "max": float,
            "mean": float,
        },
    },
    "defect_detection": {
        "defects_found": bool,        # 当前固定 False
        "defect_count": int,          # 当前固定 0
        "defects": list,
        "confidence": float,          # 当前固定 0.95
    },
    "material_analysis": {
        "primary_material": str,      # 当前固定 "copper"
        "confidence": float,          # 当前固定 0.92
        "secondary_materials": list[str],
    },
    "wavelength_range": list[float],
    "num_bands": int,
    "recommendations": list[str],
    "alarms": list,                   # 当前固定 []
}
```

### 2.2 失败返回

当前真实失败壳：

```python
{"success": False, "error": "..."}
```

当前常见失败只会来自内部异常，而不是输入缺少图像。

### 2.3 `healthcheck()`

当前真实返回：

```python
HealthStatus(
    healthy=self._is_initialized,
    message="OK" if self._is_initialized else "未初始化"
)
```

## 3. 配置与依赖契约

### 3.1 当前真实配置来源

| 配置项 | 当前来源 | 备注 |
|---|---|---|
| 波长范围 | `HyperspectralConfig.wavelength_min/max` | 默认 400~2500 |
| 波段数 | `HyperspectralConfig.num_bands` | 默认 224 |
| 置信度阈值 | dataclass 中存在 | 当前主链路未使用 |
| PCA 维度 | dataclass 中存在 | 当前主链路未使用 |
| 缺陷类型列表 | dataclass 中存在 | 当前主链路未使用 |
| 空间分辨率 | dataclass 中存在 | 当前主链路未使用 |

### 3.2 manifest 与实现的当前错位

`manifest.json.default_config` 当前声明：

- `spectral_bands`
- `wavelength_start_nm`
- `wavelength_end_nm`
- `pca_components`
- `confidence_threshold`

但 `_parse_config()` 当前只读取：

- `wavelength_range`
- `num_bands`

这意味着：

1. manifest 中的默认配置键名当前不会被完整消费。
2. 即便传入 `spectral_bands=128`，当前仍可能回到 `num_bands=224`。

### 3.3 外部依赖 / 模型依赖

1. `requirements.txt` 当前核心依赖是：
   - `darkbreaker-sdk`
   - `numpy`
   - `opencv-python`
   - `pydantic`
   - `pyyaml`
2. `manifest.json` 还声明了：
   - `scipy`
   - `scikit-learn`
   - `onnxruntime`
3. 当前实现里没有真实 PCA、ONNX 推理、模型文件加载逻辑。
4. `_model_registry` 当前仅被保存，不参与推理。

## 4. 降级策略

| 场景 | 当前真实行为 |
|---|---|
| `configs/default.yaml` 缺失 | `create_standalone()` 仍可成功，并回到 dataclass 默认配置 |
| `image` 缺失 | 自动生成随机高光谱立方体 |
| `analysis_type` 变化 | 当前无分支差异，输出结构相同 |
| 传入 registry 对象 | 当前不报错，但不参与检测逻辑 |
| 缺陷检测无真实模型 | 返回固定“无缺陷”占位结果 |
| 材料识别无真实模型 | 返回固定“copper”占位结果 |
| 内部异常 | `process()` 返回 `{"success": False, "error": ...}` |

## 5. 已验证的最小事实链路

1. `HyperspectralDetectionPlugin().init()` 返回 `True`
2. `process({})` 返回 `success=True`
3. `process({"analysis_type": "full"})`、`process({"analysis_type": "materials_only"})` 当前输出字段一致
4. `demo/run_demo.py` 可执行
5. `create_standalone()` 可执行

因此当前合同应表述为：

- 光谱摘要分析：基础可用
- 缺陷检测：当前为占位输出
- 材料识别：当前为占位输出
- 配置驱动：部分可用，存在 manifest 键名漂移
