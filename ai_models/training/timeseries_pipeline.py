"""
时间序列数据管道 - 用于GL-TransLSTM气体浓度预测模型训练

功能:
- 时间序列数据预处理和清洗
- 滑动窗口数据集生成
- 数据增强（噪声注入、时间扭曲等）
- 训练/验证/测试集划分
- 数据标准化和逆标准化
- 支持多种数据源导入

作者: Claude AI
版本: V1.0
日期: 2026-01-23
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np

logger = logging.getLogger(__name__)


class GasType(Enum):
    """气体类型枚举"""
    SF6 = "sf6"
    H2 = "h2"
    CO = "co"
    O2 = "o2"
    CH4 = "ch4"
    CO2 = "co2"


class DataSplitStrategy(Enum):
    """数据集划分策略"""
    RANDOM = "random"  # 随机划分
    CHRONOLOGICAL = "chronological"  # 按时间顺序划分
    WALK_FORWARD = "walk_forward"  # 滚动验证


@dataclass
class TimeSeriesConfig:
    """时间序列配置"""
    # 窗口参数
    sequence_length: int = 168  # 输入序列长度 (7天 * 24小时)
    prediction_horizon: int = 24  # 预测时间步数
    stride: int = 1  # 滑动窗口步长

    # 数据参数
    gas_types: List[str] = field(default_factory=lambda: ["sf6", "h2", "co", "o2"])
    env_features: List[str] = field(default_factory=lambda: ["temperature", "humidity", "pressure"])
    target_gas: str = "sf6"

    # 标准化参数
    normalize_method: str = "minmax"  # minmax, zscore, robust

    # 数据增强
    augmentation_enabled: bool = True
    noise_std: float = 0.01
    time_warp_ratio: float = 0.1

    # 划分比例
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_strategy: str = "chronological"


@dataclass
class TimeSeriesSample:
    """时间序列样本"""
    timestamp: datetime
    gas_values: Dict[str, float]
    env_values: Dict[str, float]
    quality_score: float = 1.0
    is_anomaly: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataNormalizer:
    """数据标准化器"""

    def __init__(self, method: str = "minmax"):
        self.method = method
        self.params: Dict[str, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, data: np.ndarray, feature_names: List[str]) -> None:
        """拟合标准化参数"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        for i, name in enumerate(feature_names):
            col = data[:, i]
            if self.method == "minmax":
                self.params[name] = {
                    "min": float(np.nanmin(col)),
                    "max": float(np.nanmax(col))
                }
            elif self.method == "zscore":
                self.params[name] = {
                    "mean": float(np.nanmean(col)),
                    "std": float(np.nanstd(col))
                }
            elif self.method == "robust":
                self.params[name] = {
                    "median": float(np.nanmedian(col)),
                    "iqr": float(np.nanpercentile(col, 75) - np.nanpercentile(col, 25))
                }

        self._fitted = True

    def transform(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """标准化数据"""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        result = np.zeros_like(data, dtype=np.float32)

        for i, name in enumerate(feature_names):
            if name not in self.params:
                logger.warning(f"Feature {name} not in fitted params, using identity")
                result[:, i] = data[:, i]
                continue

            col = data[:, i]
            p = self.params[name]

            if self.method == "minmax":
                denom = p["max"] - p["min"]
                if denom == 0:
                    result[:, i] = 0
                else:
                    result[:, i] = (col - p["min"]) / denom
            elif self.method == "zscore":
                if p["std"] == 0:
                    result[:, i] = 0
                else:
                    result[:, i] = (col - p["mean"]) / p["std"]
            elif self.method == "robust":
                if p["iqr"] == 0:
                    result[:, i] = 0
                else:
                    result[:, i] = (col - p["median"]) / p["iqr"]

        return result

    def inverse_transform(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """逆标准化"""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        result = np.zeros_like(data, dtype=np.float32)

        for i, name in enumerate(feature_names):
            if name not in self.params:
                result[:, i] = data[:, i]
                continue

            col = data[:, i]
            p = self.params[name]

            if self.method == "minmax":
                result[:, i] = col * (p["max"] - p["min"]) + p["min"]
            elif self.method == "zscore":
                result[:, i] = col * p["std"] + p["mean"]
            elif self.method == "robust":
                result[:, i] = col * p["iqr"] + p["median"]

        return result

    def fit_transform(self, data: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """拟合并标准化"""
        self.fit(data, feature_names)
        return self.transform(data, feature_names)

    def save(self, path: str) -> None:
        """保存标准化参数"""
        with open(path, 'w') as f:
            json.dump({
                "method": self.method,
                "params": self.params
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DataNormalizer":
        """加载标准化参数"""
        with open(path, 'r') as f:
            data = json.load(f)

        normalizer = cls(method=data["method"])
        normalizer.params = data["params"]
        normalizer._fitted = True
        return normalizer


class DataAugmentor:
    """时间序列数据增强器"""

    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.rng = np.random.default_rng(42)

    def add_gaussian_noise(self, data: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = self.rng.normal(0, self.config.noise_std, data.shape)
        return data + noise

    def time_warp(self, data: np.ndarray) -> np.ndarray:
        """时间扭曲 - 局部加速/减速"""
        seq_len = data.shape[0]
        warp_amount = int(seq_len * self.config.time_warp_ratio)

        if warp_amount < 2:
            return data

        # 随机选择扭曲位置和方向
        start_idx = self.rng.integers(0, seq_len - warp_amount)
        stretch = self.rng.choice([True, False])

        if stretch:
            # 拉伸 - 插值扩展
            segment = data[start_idx:start_idx + warp_amount]
            new_len = int(warp_amount * 1.2)
            indices = np.linspace(0, warp_amount - 1, new_len)

            if data.ndim == 1:
                warped_segment = np.interp(indices, np.arange(warp_amount), segment)
            else:
                warped_segment = np.zeros((new_len, data.shape[1]))
                for j in range(data.shape[1]):
                    warped_segment[:, j] = np.interp(indices, np.arange(warp_amount), segment[:, j])

            # 截断以保持原长度
            result = np.concatenate([
                data[:start_idx],
                warped_segment[:warp_amount],
                data[start_idx + warp_amount:]
            ], axis=0)
        else:
            # 压缩 - 下采样
            segment = data[start_idx:start_idx + warp_amount]
            new_len = int(warp_amount * 0.8)
            indices = np.linspace(0, warp_amount - 1, new_len)

            if data.ndim == 1:
                warped_segment = np.interp(indices, np.arange(warp_amount), segment)
                # 填充以保持原长度
                pad_len = warp_amount - new_len
                warped_segment = np.concatenate([warped_segment, np.full(pad_len, warped_segment[-1])])
            else:
                warped_segment = np.zeros((new_len, data.shape[1]))
                for j in range(data.shape[1]):
                    warped_segment[:, j] = np.interp(indices, np.arange(warp_amount), segment[:, j])
                # 填充
                pad_len = warp_amount - new_len
                warped_segment = np.vstack([warped_segment, np.tile(warped_segment[-1], (pad_len, 1))])

            result = np.concatenate([
                data[:start_idx],
                warped_segment,
                data[start_idx + warp_amount:]
            ], axis=0)

        return result[:len(data)]

    def magnitude_warp(self, data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """幅值扭曲"""
        seq_len = data.shape[0]
        # 生成平滑的扭曲因子
        knot_count = 4
        knots = self.rng.normal(1.0, sigma, knot_count)
        indices = np.linspace(0, knot_count - 1, seq_len)
        warp_factors = np.interp(indices, np.arange(knot_count), knots)

        if data.ndim == 1:
            return data * warp_factors
        else:
            return data * warp_factors.reshape(-1, 1)

    def window_slice(self, data: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """窗口切片 - 随机截取并插值回原长度"""
        seq_len = data.shape[0]
        target_len = max(int(seq_len * reduce_ratio), 10)

        start = self.rng.integers(0, seq_len - target_len + 1)
        sliced = data[start:start + target_len]

        # 插值回原长度
        indices = np.linspace(0, target_len - 1, seq_len)
        if data.ndim == 1:
            return np.interp(indices, np.arange(target_len), sliced)
        else:
            result = np.zeros((seq_len, data.shape[1]))
            for j in range(data.shape[1]):
                result[:, j] = np.interp(indices, np.arange(target_len), sliced[:, j])
            return result

    def augment(self, data: np.ndarray, methods: Optional[List[str]] = None) -> np.ndarray:
        """应用数据增强"""
        if methods is None:
            methods = ["noise", "time_warp"]

        result = data.copy()

        for method in methods:
            if method == "noise":
                result = self.add_gaussian_noise(result)
            elif method == "time_warp":
                result = self.time_warp(result)
            elif method == "magnitude_warp":
                result = self.magnitude_warp(result)
            elif method == "window_slice":
                result = self.window_slice(result)

        return result


class DataCleaner:
    """数据清洗器"""

    def __init__(self):
        self.stats: Dict[str, Any] = {}

    def remove_outliers(self, data: np.ndarray, method: str = "iqr",
                        threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """移除异常值

        Args:
            data: 输入数据
            method: 异常检测方法 (iqr, zscore, mad)
            threshold: 阈值

        Returns:
            (清洗后数据, 异常值掩码)
        """
        if method == "iqr":
            q1 = np.nanpercentile(data, 25, axis=0)
            q3 = np.nanpercentile(data, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (data < lower) | (data > upper)
        elif method == "zscore":
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
            z_scores = np.abs((data - mean) / (std + 1e-8))
            mask = z_scores > threshold
        elif method == "mad":
            median = np.nanmedian(data, axis=0)
            mad = np.nanmedian(np.abs(data - median), axis=0)
            modified_z = 0.6745 * (data - median) / (mad + 1e-8)
            mask = np.abs(modified_z) > threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        if data.ndim == 1:
            outlier_mask = mask
        else:
            outlier_mask = np.any(mask, axis=1)

        cleaned = data.copy()
        if data.ndim == 1:
            cleaned[outlier_mask] = np.nan
        else:
            cleaned[outlier_mask] = np.nan

        self.stats["outliers_removed"] = int(np.sum(outlier_mask))
        return cleaned, outlier_mask

    def interpolate_missing(self, data: np.ndarray, method: str = "linear",
                           max_gap: int = 6) -> np.ndarray:
        """插值填补缺失值

        Args:
            data: 输入数据
            method: 插值方法 (linear, spline, forward, backward)
            max_gap: 最大允许连续缺失长度

        Returns:
            填补后数据
        """
        result = data.copy()

        if data.ndim == 1:
            result = self._interpolate_1d(result, method, max_gap)
        else:
            for j in range(data.shape[1]):
                result[:, j] = self._interpolate_1d(result[:, j], method, max_gap)

        return result

    def _interpolate_1d(self, arr: np.ndarray, method: str, max_gap: int) -> np.ndarray:
        """一维插值"""
        result = arr.copy()
        nan_mask = np.isnan(result)

        if not np.any(nan_mask):
            return result

        # 检查连续缺失长度
        nan_groups = []
        start = None
        for i, is_nan in enumerate(nan_mask):
            if is_nan:
                if start is None:
                    start = i
            else:
                if start is not None:
                    nan_groups.append((start, i))
                    start = None
        if start is not None:
            nan_groups.append((start, len(nan_mask)))

        # 标记超过max_gap的组
        long_gaps = set()
        for start, end in nan_groups:
            if end - start > max_gap:
                for i in range(start, end):
                    long_gaps.add(i)

        valid_mask = ~nan_mask
        valid_indices = np.where(valid_mask)[0]
        valid_values = result[valid_mask]

        if len(valid_indices) < 2:
            return result

        if method == "linear":
            interp_values = np.interp(np.arange(len(result)), valid_indices, valid_values)
        elif method == "forward":
            interp_values = result.copy()
            last_valid = valid_values[0]
            for i in range(len(result)):
                if nan_mask[i]:
                    interp_values[i] = last_valid
                else:
                    last_valid = result[i]
        elif method == "backward":
            interp_values = result.copy()
            last_valid = valid_values[-1]
            for i in range(len(result) - 1, -1, -1):
                if nan_mask[i]:
                    interp_values[i] = last_valid
                else:
                    last_valid = result[i]
        else:
            interp_values = np.interp(np.arange(len(result)), valid_indices, valid_values)

        # 应用插值，但保留长gap为nan
        for i in range(len(result)):
            if nan_mask[i] and i not in long_gaps:
                result[i] = interp_values[i]

        self.stats["interpolated_count"] = int(np.sum(nan_mask) - len(long_gaps))
        return result

    def resample(self, timestamps: np.ndarray, data: np.ndarray,
                target_freq: str = "1H") -> Tuple[np.ndarray, np.ndarray]:
        """重采样到目标频率

        Args:
            timestamps: 时间戳数组
            data: 数据数组
            target_freq: 目标频率 (1H, 30T, 1D等)

        Returns:
            (新时间戳, 重采样数据)
        """
        # 解析频率
        freq_map = {
            "1H": timedelta(hours=1),
            "30T": timedelta(minutes=30),
            "15T": timedelta(minutes=15),
            "1D": timedelta(days=1),
            "6H": timedelta(hours=6),
        }

        if target_freq not in freq_map:
            raise ValueError(f"Unknown frequency: {target_freq}")

        delta = freq_map[target_freq]

        # 创建新时间轴
        start_time = timestamps[0]
        end_time = timestamps[-1]
        new_timestamps = []
        current = start_time
        while current <= end_time:
            new_timestamps.append(current)
            current += delta

        new_timestamps = np.array(new_timestamps)

        # 对每个新时间点，找最近的原始数据点
        if data.ndim == 1:
            new_data = np.zeros(len(new_timestamps))
        else:
            new_data = np.zeros((len(new_timestamps), data.shape[1]))

        for i, t in enumerate(new_timestamps):
            # 找最近的原始时间点
            time_diffs = np.abs(timestamps - t)
            nearest_idx = np.argmin(time_diffs)

            if data.ndim == 1:
                new_data[i] = data[nearest_idx]
            else:
                new_data[i] = data[nearest_idx]

        return new_timestamps, new_data


class TimeSeriesDataset:
    """时间序列数据集"""

    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.samples: List[TimeSeriesSample] = []
        self.normalizer = DataNormalizer(config.normalize_method)
        self.augmentor = DataAugmentor(config)
        self.cleaner = DataCleaner()

        self._gas_data: Optional[np.ndarray] = None
        self._env_data: Optional[np.ndarray] = None
        self._timestamps: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None

    def load_from_csv(self, path: str,
                      timestamp_col: str = "timestamp",
                      timestamp_format: str = "%Y-%m-%d %H:%M:%S") -> None:
        """从CSV加载数据"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV loading")

        df = pd.read_csv(path)

        # 解析时间戳
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=timestamp_format)
        df = df.sort_values(timestamp_col)

        # 提取数据
        for _, row in df.iterrows():
            gas_values = {}
            for gas in self.config.gas_types:
                if gas in df.columns:
                    gas_values[gas] = float(row[gas])

            env_values = {}
            for env in self.config.env_features:
                if env in df.columns:
                    env_values[env] = float(row[env])

            sample = TimeSeriesSample(
                timestamp=row[timestamp_col].to_pydatetime(),
                gas_values=gas_values,
                env_values=env_values
            )
            self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} samples from {path}")

    def load_from_json(self, path: str) -> None:
        """从JSON加载数据"""
        with open(path, 'r') as f:
            data = json.load(f)

        for item in data:
            sample = TimeSeriesSample(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                gas_values=item.get("gas_values", {}),
                env_values=item.get("env_values", {}),
                quality_score=item.get("quality_score", 1.0),
                is_anomaly=item.get("is_anomaly", False)
            )
            self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} samples from {path}")

    def load_from_database(self, connection_string: str, table_name: str,
                          timestamp_col: str = "timestamp") -> None:
        """从数据库加载数据"""
        try:
            import pandas as pd
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("pandas and sqlalchemy are required for database loading")

        engine = create_engine(connection_string)
        query = f"SELECT * FROM {table_name} ORDER BY {timestamp_col}"
        df = pd.read_sql(query, engine)

        # 转换为samples
        for _, row in df.iterrows():
            gas_values = {gas: float(row[gas]) for gas in self.config.gas_types if gas in df.columns}
            env_values = {env: float(row[env]) for env in self.config.env_features if env in df.columns}

            sample = TimeSeriesSample(
                timestamp=row[timestamp_col],
                gas_values=gas_values,
                env_values=env_values
            )
            self.samples.append(sample)

    def prepare_arrays(self, clean: bool = True, normalize: bool = True) -> None:
        """准备numpy数组"""
        if not self.samples:
            raise ValueError("No samples loaded")

        # 提取数据
        n_samples = len(self.samples)
        n_gas = len(self.config.gas_types)
        n_env = len(self.config.env_features)

        self._gas_data = np.zeros((n_samples, n_gas))
        self._env_data = np.zeros((n_samples, n_env))
        self._timestamps = np.array([s.timestamp for s in self.samples])

        for i, sample in enumerate(self.samples):
            for j, gas in enumerate(self.config.gas_types):
                self._gas_data[i, j] = sample.gas_values.get(gas, np.nan)
            for j, env in enumerate(self.config.env_features):
                self._env_data[i, j] = sample.env_values.get(env, np.nan)

        # 清洗数据
        if clean:
            self._gas_data, _ = self.cleaner.remove_outliers(self._gas_data)
            self._gas_data = self.cleaner.interpolate_missing(self._gas_data)

            self._env_data, _ = self.cleaner.remove_outliers(self._env_data)
            self._env_data = self.cleaner.interpolate_missing(self._env_data)

        # 标准化
        if normalize:
            self._gas_data = self.normalizer.fit_transform(
                self._gas_data, self.config.gas_types
            )
            # 环境特征单独标准化
            env_normalizer = DataNormalizer(self.config.normalize_method)
            self._env_data = env_normalizer.fit_transform(
                self._env_data, self.config.env_features
            )

        # 提取目标变量
        target_idx = self.config.gas_types.index(self.config.target_gas)
        self._targets = self._gas_data[:, target_idx]

        logger.info(f"Prepared arrays: gas {self._gas_data.shape}, env {self._env_data.shape}")

    def create_sequences(self, augment: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """创建序列数据集

        Returns:
            (X_gas, X_env, y) - 输入气体序列, 输入环境序列, 目标序列
        """
        if self._gas_data is None:
            self.prepare_arrays()

        seq_len = self.config.sequence_length
        pred_len = self.config.prediction_horizon
        stride = self.config.stride

        n_samples = len(self._gas_data)
        n_sequences = (n_samples - seq_len - pred_len) // stride + 1

        if n_sequences <= 0:
            raise ValueError(f"Not enough data: {n_samples} samples for {seq_len} + {pred_len} window")

        X_gas = []
        X_env = []
        y = []

        for i in range(0, n_samples - seq_len - pred_len + 1, stride):
            gas_seq = self._gas_data[i:i + seq_len]
            env_seq = self._env_data[i:i + seq_len]
            target_seq = self._targets[i + seq_len:i + seq_len + pred_len]

            # 检查是否有nan
            if np.any(np.isnan(gas_seq)) or np.any(np.isnan(target_seq)):
                continue

            if augment and self.config.augmentation_enabled:
                gas_seq = self.augmentor.augment(gas_seq, ["noise"])

            X_gas.append(gas_seq)
            X_env.append(env_seq)
            y.append(target_seq)

        return np.array(X_gas), np.array(X_env), np.array(y)

    def split(self, X_gas: np.ndarray, X_env: np.ndarray, y: np.ndarray,
             seed: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """划分训练/验证/测试集"""
        n_samples = len(X_gas)

        if self.config.split_strategy == "chronological":
            # 按时间顺序划分
            train_end = int(n_samples * self.config.train_ratio)
            val_end = train_end + int(n_samples * self.config.val_ratio)

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(train_end, val_end)
            test_indices = np.arange(val_end, n_samples)

        elif self.config.split_strategy == "random":
            # 随机划分
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n_samples)

            train_end = int(n_samples * self.config.train_ratio)
            val_end = train_end + int(n_samples * self.config.val_ratio)

            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

        elif self.config.split_strategy == "walk_forward":
            # 滚动验证 - 返回多个fold
            # 这里简化为单fold
            train_end = int(n_samples * self.config.train_ratio)
            val_end = train_end + int(n_samples * self.config.val_ratio)

            train_indices = np.arange(0, train_end)
            val_indices = np.arange(train_end, val_end)
            test_indices = np.arange(val_end, n_samples)
        else:
            raise ValueError(f"Unknown split strategy: {self.config.split_strategy}")

        return {
            "train": (X_gas[train_indices], X_env[train_indices], y[train_indices]),
            "val": (X_gas[val_indices], X_env[val_indices], y[val_indices]),
            "test": (X_gas[test_indices], X_env[test_indices], y[test_indices])
        }

    def save_processed(self, output_dir: str) -> None:
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存标准化参数
        self.normalizer.save(os.path.join(output_dir, "normalizer.json"))

        # 保存配置
        config_dict = {
            "sequence_length": self.config.sequence_length,
            "prediction_horizon": self.config.prediction_horizon,
            "stride": self.config.stride,
            "gas_types": self.config.gas_types,
            "env_features": self.config.env_features,
            "target_gas": self.config.target_gas,
            "normalize_method": self.config.normalize_method,
        }
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)

        # 保存数据
        if self._gas_data is not None:
            np.save(os.path.join(output_dir, "gas_data.npy"), self._gas_data)
            np.save(os.path.join(output_dir, "env_data.npy"), self._env_data)

        logger.info(f"Saved processed data to {output_dir}")


class TimeSeriesPipelineManager:
    """时间序列数据管道管理器 - 端到端训练流程"""

    def __init__(self, config: TimeSeriesConfig):
        self.config = config
        self.dataset = TimeSeriesDataset(config)

    def run_pipeline(self,
                     data_source: str,
                     output_dir: str,
                     source_type: str = "csv",
                     augment_train: bool = True) -> Dict[str, Any]:
        """运行完整数据管道

        Args:
            data_source: 数据源路径或连接字符串
            output_dir: 输出目录
            source_type: 数据源类型 (csv, json, database)
            augment_train: 是否对训练集进行数据增强

        Returns:
            包含数据集路径和统计信息的字典
        """
        logger.info("Starting time series data pipeline...")

        # 1. 加载数据
        if source_type == "csv":
            self.dataset.load_from_csv(data_source)
        elif source_type == "json":
            self.dataset.load_from_json(data_source)
        elif source_type == "database":
            self.dataset.load_from_database(data_source, "gas_readings")
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        # 2. 准备数组
        self.dataset.prepare_arrays(clean=True, normalize=True)

        # 3. 创建序列
        X_gas, X_env, y = self.dataset.create_sequences(augment=False)
        logger.info(f"Created {len(X_gas)} sequences")

        # 4. 划分数据集
        splits = self.dataset.split(X_gas, X_env, y)

        # 5. 对训练集进行数据增强
        if augment_train and self.config.augmentation_enabled:
            train_X_gas, train_X_env, train_y = splits["train"]
            augmented_X_gas = []
            augmented_X_env = []
            augmented_y = []

            for i in range(len(train_X_gas)):
                # 原始数据
                augmented_X_gas.append(train_X_gas[i])
                augmented_X_env.append(train_X_env[i])
                augmented_y.append(train_y[i])

                # 增强数据
                aug_gas = self.dataset.augmentor.augment(train_X_gas[i], ["noise", "time_warp"])
                augmented_X_gas.append(aug_gas)
                augmented_X_env.append(train_X_env[i])
                augmented_y.append(train_y[i])

            splits["train"] = (
                np.array(augmented_X_gas),
                np.array(augmented_X_env),
                np.array(augmented_y)
            )
            logger.info(f"Augmented training set: {len(augmented_X_gas)} samples")

        # 6. 保存数据集
        os.makedirs(output_dir, exist_ok=True)

        for split_name, (X_g, X_e, y_split) in splits.items():
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            np.save(os.path.join(split_dir, "X_gas.npy"), X_g)
            np.save(os.path.join(split_dir, "X_env.npy"), X_e)
            np.save(os.path.join(split_dir, "y.npy"), y_split)

        # 保存标准化参数和配置
        self.dataset.save_processed(output_dir)

        # 7. 生成统计报告
        stats = {
            "total_samples": len(self.dataset.samples),
            "sequence_count": len(X_gas),
            "train_samples": len(splits["train"][0]),
            "val_samples": len(splits["val"][0]),
            "test_samples": len(splits["test"][0]),
            "sequence_length": self.config.sequence_length,
            "prediction_horizon": self.config.prediction_horizon,
            "gas_types": self.config.gas_types,
            "target_gas": self.config.target_gas,
            "output_dir": output_dir
        }

        with open(os.path.join(output_dir, "stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Pipeline complete. Output saved to {output_dir}")
        return stats

    def generate_synthetic_data(self, n_samples: int = 10000,
                                output_path: str = "synthetic_gas_data.csv") -> str:
        """生成合成训练数据（用于测试和演示）

        Args:
            n_samples: 样本数量
            output_path: 输出路径

        Returns:
            输出文件路径
        """
        rng = np.random.default_rng(42)

        # 时间范围
        start_time = datetime(2025, 1, 1)
        timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]

        # 生成气体浓度数据 (带有周期性和趋势)
        t = np.arange(n_samples)

        # SF6 - 基础值 + 日周期 + 随机噪声
        sf6_base = 5.0
        sf6_daily = 0.5 * np.sin(2 * np.pi * t / 24)
        sf6_weekly = 0.3 * np.sin(2 * np.pi * t / 168)
        sf6_noise = rng.normal(0, 0.1, n_samples)
        sf6 = sf6_base + sf6_daily + sf6_weekly + sf6_noise

        # H2 - 类似模式但不同参数
        h2_base = 2.0
        h2_daily = 0.3 * np.sin(2 * np.pi * t / 24 + np.pi/4)
        h2_noise = rng.normal(0, 0.05, n_samples)
        h2 = h2_base + h2_daily + h2_noise

        # CO
        co_base = 0.5
        co_daily = 0.1 * np.sin(2 * np.pi * t / 24 + np.pi/2)
        co_noise = rng.normal(0, 0.02, n_samples)
        co = co_base + co_daily + co_noise

        # O2 - 较稳定
        o2_base = 20.9
        o2_noise = rng.normal(0, 0.05, n_samples)
        o2 = o2_base + o2_noise

        # 环境因素
        temp_base = 25
        temp_daily = 5 * np.sin(2 * np.pi * t / 24 - np.pi/2)
        temp_seasonal = 10 * np.sin(2 * np.pi * t / (24 * 365))
        temperature = temp_base + temp_daily + temp_seasonal + rng.normal(0, 1, n_samples)

        humidity = 60 + 15 * np.sin(2 * np.pi * t / 24 + np.pi) + rng.normal(0, 3, n_samples)
        pressure = 1013 + 5 * np.sin(2 * np.pi * t / (24 * 7)) + rng.normal(0, 2, n_samples)

        # 添加一些异常事件
        anomaly_indices = rng.choice(n_samples, size=int(n_samples * 0.01), replace=False)
        sf6[anomaly_indices] += rng.uniform(1, 3, len(anomaly_indices))

        # 保存为CSV
        try:
            import pandas as pd
            df = pd.DataFrame({
                "timestamp": timestamps,
                "sf6": sf6,
                "h2": h2,
                "co": co,
                "o2": o2,
                "temperature": temperature,
                "humidity": humidity,
                "pressure": pressure
            })
            df.to_csv(output_path, index=False)
        except ImportError:
            # 不使用pandas的备选方案
            with open(output_path, 'w') as f:
                f.write("timestamp,sf6,h2,co,o2,temperature,humidity,pressure\n")
                for i in range(n_samples):
                    f.write(f"{timestamps[i].strftime('%Y-%m-%d %H:%M:%S')},"
                           f"{sf6[i]:.4f},{h2[i]:.4f},{co[i]:.4f},{o2[i]:.4f},"
                           f"{temperature[i]:.2f},{humidity[i]:.2f},{pressure[i]:.2f}\n")

        logger.info(f"Generated {n_samples} synthetic samples to {output_path}")
        return output_path


def create_training_script_template(output_path: str) -> None:
    """创建训练脚本模板"""
    script = '''#!/usr/bin/env python
"""
GL-TransLSTM 训练脚本

使用方法:
    python train_gl_translstm.py --data_dir ./data --output_dir ./models

"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train GL-TransLSTM model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='./models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    return parser.parse_args()


def load_data(data_dir: str):
    """加载预处理的数据"""
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'
    test_dir = Path(data_dir) / 'test'

    train_data = {
        'X_gas': np.load(train_dir / 'X_gas.npy'),
        'X_env': np.load(train_dir / 'X_env.npy'),
        'y': np.load(train_dir / 'y.npy')
    }

    val_data = {
        'X_gas': np.load(val_dir / 'X_gas.npy'),
        'X_env': np.load(val_dir / 'X_env.npy'),
        'y': np.load(val_dir / 'y.npy')
    }

    test_data = {
        'X_gas': np.load(test_dir / 'X_gas.npy'),
        'X_env': np.load(test_dir / 'X_env.npy'),
        'y': np.load(test_dir / 'y.npy')
    }

    return train_data, val_data, test_data


def main():
    args = parse_args()

    logger.info("Loading data...")
    train_data, val_data, test_data = load_data(args.data_dir)

    logger.info(f"Train samples: {len(train_data['y'])}")
    logger.info(f"Val samples: {len(val_data['y'])}")
    logger.info(f"Test samples: {len(test_data['y'])}")

    try:
        import torch
        from ai_models.deep_learning.gl_translstm_torch import (
            GLTransLSTMModel, GLTransLSTMTrainer, ModelConfig
        )

        # 配置模型
        config = ModelConfig(
            input_dim=train_data['X_gas'].shape[2],
            env_dim=train_data['X_env'].shape[2],
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            output_dim=train_data['y'].shape[1],
            dropout=0.1
        )

        model = GLTransLSTMModel(config)

        # 训练
        trainer = GLTransLSTMTrainer(
            model=model,
            learning_rate=args.learning_rate,
            device=args.device
        )

        trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )

        # 评估
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_data)
        logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")

    except ImportError as e:
        logger.error(f"PyTorch not available: {e}")
        logger.info("Please install PyTorch: pip install torch")
        return 1

    logger.info("Training complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''

    with open(output_path, 'w') as f:
        f.write(script)

    logger.info(f"Created training script template at {output_path}")


# 便捷函数
def prepare_gas_training_data(
    csv_path: str,
    output_dir: str,
    target_gas: str = "sf6",
    sequence_length: int = 168,
    prediction_horizon: int = 24
) -> Dict[str, Any]:
    """便捷函数 - 准备气体浓度预测训练数据

    Args:
        csv_path: CSV数据文件路径
        output_dir: 输出目录
        target_gas: 目标气体类型
        sequence_length: 输入序列长度
        prediction_horizon: 预测时间步数

    Returns:
        数据集统计信息
    """
    config = TimeSeriesConfig(
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        target_gas=target_gas
    )

    pipeline = TimeSeriesPipelineManager(config)
    return pipeline.run_pipeline(csv_path, output_dir, source_type="csv")


if __name__ == "__main__":
    # 演示用法
    logging.basicConfig(level=logging.INFO)

    config = TimeSeriesConfig(
        sequence_length=168,
        prediction_horizon=24,
        target_gas="sf6"
    )

    pipeline = TimeSeriesPipelineManager(config)

    # 生成合成数据
    synthetic_path = pipeline.generate_synthetic_data(
        n_samples=5000,
        output_path="/tmp/synthetic_gas_data.csv"
    )

    # 运行管道
    stats = pipeline.run_pipeline(
        data_source=synthetic_path,
        output_dir="/tmp/gas_training_data",
        source_type="csv"
    )

    print("Pipeline stats:", stats)

    # 创建训练脚本模板
    create_training_script_template("/tmp/train_gl_translstm.py")
