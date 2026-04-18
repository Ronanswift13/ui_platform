# 02_algorithm_contract

## 1. 输入契约

### plugin.process() 输入
```python
{
    "audio": np.ndarray,       # 可选，缺失时生成 mock 音频
    "sample_rate": int,        # 默认 16000
    "device_id": str,          # 设备标识
    "data_source": str,        # "simulation" | "real"（可选）
}
```

- `audio` 缺失 → 自动走 mock 音频生成路径（`_generate_mock_audio`）。
- `audio` 为多声道 → detector 内部取均值转单声道。

### detector 输入
- `audio`: `np.ndarray`, 1D float32, 已归一化。
- `sample_rate`: `int`, 来自 config。

## 2. 输出契约

### plugin.process() 输出
```python
{
    "success": bool,
    "anomaly_detected": bool,
    "anomaly_type": str,           # AcousticAnomalyType 枚举值
    "anomaly_score": float,        # [0, 1]
    "confidence": float,           # [0, 1]
    "severity": str,               # "info" | "warning" | "error" | "critical"
    "frequency_analysis": dict,    # 频谱分析摘要
    "alarms": list[Alarm],         # 当 anomaly_detected 且累计超阈值时触发
}
```

## 3. 检测链路

```
audio → 特征提取 (mel/MFCC/spectral) → 传统信号处理检测 → 模型推理(可选) → 综合评分 → 告警判定
```

### 传统信号处理检测子类型

| 检测器 | 核心指标 | 阈值来源 |
|--------|----------|----------|
| 局部放电 (PD) | 高频能量比、脉冲密度、包络峰度 | `config.pd_*` |
| 电晕放电 (Corona) | 频谱质心、频谱平坦度 | `config.corona_*` |
| 轴承故障 (Bearing) | 峰度、周期性、Hilbert 包络 | `config.bearing_*` |
| 变压器嗡鸣 (Transformer) | 谐波能量占比 | `config.transformer_*` |
| 机械故障 (Mechanical) | 峰值因子、能量变异系数 | `config.mechanical_*` |

## 4. 降级策略

| 场景 | 降级行为 |
|------|----------|
| 无 audio 输入 | 生成 mock 音频，标记 `data_source: "mock"` |
| ONNX 模型不可用 | 仅走传统信号处理路径 |
| 采样率不足（< 超声要求） | 跳过 PD 超声检测，仅做可听频段分析 |
| 特征提取异常 | 返回 `success=True, anomaly_detected=False`，记录 warning 日志 |

## 5. 阈值来源

**唯一来源**：`configs/default.yaml` → `AcousticConfig` dataclass。

禁止在 `detector.py` / `analyzer.py` 中新增未在 config 中声明的阈值常量。`analyzer.py` 中的 `reference_frequencies` 为参考常量（物理常数），不受此约束。
