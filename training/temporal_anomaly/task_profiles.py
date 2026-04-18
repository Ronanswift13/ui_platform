"""
时序异常训练任务画像

为 training/temporal_anomaly 提供统一的任务定义、默认训练范式、
输出头和工程化落盘约定。
"""

from __future__ import annotations

from dataclasses import dataclass, field


TEMPORAL_DATASET_FAMILY = "temporal_anomaly"

TEMPORAL_TASK_TYPES = {
    "acoustic_time_frequency_anomaly",
    "multivariate_sensor_anomaly",
    "equipment_health_prediction",
    "health_index_calibration",
    "action_event_sequence_recognition",
}

TEMPORAL_PARADIGMS = {
    "autoencoder",
    "lstm",
    "gru",
    "temporal_cnn",
    "transformer_encoder",
    "hybrid_statistical_ml",
}

TEMPORAL_MODALITIES = {
    "audio_waveform",
    "audio_spectrogram",
    "audio_waveform+spectrogram",
    "multivariate_timeseries",
    "event_sequence",
}

TEMPORAL_SOURCE_KINDS = {
    "waveform",
    "spectrogram",
    "timeseries",
    "event_sequence",
    "labels",
    "metadata",
}


@dataclass(frozen=True)
class OutputHeadSpec:
    """模型输出头定义"""

    name: str
    head_type: str
    description: str
    range_hint: str = ""
    required: bool = True


@dataclass(frozen=True)
class TemporalTaskProfile:
    """时序任务画像"""

    task_type: str
    description: str
    modality: str
    storage_family: str = TEMPORAL_DATASET_FAMILY
    supported_formats: tuple[str, ...] = ()
    required_source_kinds: tuple[str, ...] = ()
    optional_source_kinds: tuple[str, ...] = ()
    supported_paradigms: tuple[str, ...] = ()
    default_paradigm: str = ""
    default_model_family: str = ""
    required_manifest_fields: tuple[str, ...] = ()
    evaluation_metrics: tuple[str, ...] = ()
    export_runtimes: tuple[str, ...] = ("onnxruntime",)
    output_heads: tuple[OutputHeadSpec, ...] = ()


DEVICE_MONITORING_OUTPUT_HEADS = (
    OutputHeadSpec(
        name="health_index",
        head_type="regression",
        description="0-100 设备健康指数，用于平台健康看板和分层阈值告警。",
        range_hint="[0, 100]",
    ),
    OutputHeadSpec(
        name="anomaly_score",
        head_type="anomaly_score",
        description="0-1 连续异常分数，用于异常检测与排序。",
        range_hint="[0, 1]",
    ),
    OutputHeadSpec(
        name="predicted_failure",
        head_type="classification",
        description="指定预测时窗内的故障概率、剩余时间或风险等级。",
        range_hint="probability or risk level",
    ),
)


TEMPORAL_TASK_PROFILES: dict[str, TemporalTaskProfile] = {
    "acoustic_time_frequency_anomaly": TemporalTaskProfile(
        task_type="acoustic_time_frequency_anomaly",
        description="声学时频异常检测，支持原始波形与时频图双路径输入。",
        modality="audio_waveform+spectrogram",
        supported_formats=("wav_dir", "flac_dir", "npy_waveform", "mel_npz", "dual_audio_views"),
        required_source_kinds=("waveform",),
        optional_source_kinds=("spectrogram", "labels", "metadata"),
        supported_paradigms=(
            "autoencoder",
            "temporal_cnn",
            "transformer_encoder",
            "hybrid_statistical_ml",
        ),
        default_paradigm="transformer_encoder",
        default_model_family="dual_path_audio_transformer",
        required_manifest_fields=("sample_rate_hz", "window_size", "feature_views"),
        evaluation_metrics=("auc_roc", "auc_pr", "f1", "false_alarm_rate", "latency_ms"),
        output_heads=(
            OutputHeadSpec(
                name="anomaly_score",
                head_type="anomaly_score",
                description="综合声学异常分数。",
                range_hint="[0, 1]",
            ),
            OutputHeadSpec(
                name="anomaly_type",
                head_type="classification",
                description="异常类型分类，例如局放、电晕、摩擦异响。",
            ),
        ),
    ),
    "multivariate_sensor_anomaly": TemporalTaskProfile(
        task_type="multivariate_sensor_anomaly",
        description="多变量数值时序异常检测，适用于气体监测和设备运行监测。",
        modality="multivariate_timeseries",
        supported_formats=("csv_series", "parquet_series", "jsonl_series", "npz_series"),
        required_source_kinds=("timeseries",),
        optional_source_kinds=("labels", "metadata"),
        supported_paradigms=(
            "autoencoder",
            "lstm",
            "gru",
            "temporal_cnn",
            "transformer_encoder",
            "hybrid_statistical_ml",
        ),
        default_paradigm="autoencoder",
        default_model_family="multivariate_autoencoder",
        required_manifest_fields=("sensor_columns", "window_size"),
        evaluation_metrics=("auc_pr", "auc_roc", "f1", "false_positive_rate", "lead_time_minutes"),
        output_heads=(
            OutputHeadSpec(
                name="anomaly_score",
                head_type="anomaly_score",
                description="多变量异常分数。",
                range_hint="[0, 1]",
            ),
        ),
    ),
    "equipment_health_prediction": TemporalTaskProfile(
        task_type="equipment_health_prediction",
        description="设备健康预测，面向健康指数、异常分数和故障预测联合输出。",
        modality="multivariate_timeseries",
        supported_formats=("csv_series", "parquet_series", "jsonl_series", "npz_series"),
        required_source_kinds=("timeseries", "labels"),
        optional_source_kinds=("metadata",),
        supported_paradigms=(
            "lstm",
            "gru",
            "temporal_cnn",
            "transformer_encoder",
            "hybrid_statistical_ml",
        ),
        default_paradigm="lstm",
        default_model_family="health_forecaster",
        required_manifest_fields=("sensor_columns", "target_schema", "prediction_horizon"),
        evaluation_metrics=(
            "health_mae",
            "health_rmse",
            "anomaly_auc",
            "failure_f1",
            "failure_recall",
            "lead_time_hours",
        ),
        output_heads=DEVICE_MONITORING_OUTPUT_HEADS,
    ),
    "health_index_calibration": TemporalTaskProfile(
        task_type="health_index_calibration",
        description="健康指数校准，将设备原始指标映射为稳定可解释的 health_index。",
        modality="multivariate_timeseries",
        supported_formats=("csv_series", "parquet_series", "jsonl_series", "npz_series"),
        required_source_kinds=("timeseries", "labels"),
        optional_source_kinds=("metadata",),
        supported_paradigms=(
            "lstm",
            "gru",
            "temporal_cnn",
            "transformer_encoder",
            "hybrid_statistical_ml",
        ),
        default_paradigm="hybrid_statistical_ml",
        default_model_family="health_calibrator",
        required_manifest_fields=("sensor_columns", "target_schema"),
        evaluation_metrics=("health_mae", "health_rmse", "health_rank_corr", "calibration_mae"),
        output_heads=(
            OutputHeadSpec(
                name="health_index",
                head_type="regression",
                description="0-100 设备健康指数。",
                range_hint="[0, 100]",
            ),
        ),
    ),
    "action_event_sequence_recognition": TemporalTaskProfile(
        task_type="action_event_sequence_recognition",
        description="带时间戳的动作事件序列识别与异常序列分析。",
        modality="event_sequence",
        supported_formats=("jsonl_events", "csv_events", "parquet_events"),
        required_source_kinds=("event_sequence", "labels"),
        optional_source_kinds=("metadata",),
        supported_paradigms=(
            "lstm",
            "gru",
            "temporal_cnn",
            "transformer_encoder",
            "hybrid_statistical_ml",
        ),
        default_paradigm="transformer_encoder",
        default_model_family="event_sequence_encoder",
        required_manifest_fields=("event_types", "timestamp_field"),
        evaluation_metrics=("sequence_accuracy", "macro_f1", "event_recall", "edit_distance"),
        output_heads=(
            OutputHeadSpec(
                name="event_label",
                head_type="classification",
                description="动作事件序列类别或状态转移类别。",
            ),
            OutputHeadSpec(
                name="sequence_risk",
                head_type="classification",
                description="序列级风险等级或异常类型。",
            ),
        ),
    ),
}


PLUGIN_TEMPORAL_MODEL_ROLES: dict[str, tuple[str, ...]] = {
    "acoustic_monitoring": ("audio_anomaly_transformer", "ultrasonic_pd_detector"),
    "gas_detection": ("sf6_forecast", "multi_gas_forecast", "equipment_health_trend"),
    "device_monitoring": (
        "device_anomaly_autoencoder",
        "failure_predictor",
        "health_calibrator",
    ),
    "action_event_monitoring": ("action_sequence_encoder",),
}


PLUGIN_TEMPORAL_DEFAULT_TASKS: dict[str, tuple[str, ...]] = {
    "acoustic_monitoring": ("acoustic_time_frequency_anomaly",),
    "gas_detection": ("multivariate_sensor_anomaly", "equipment_health_prediction"),
    "device_monitoring": (
        "multivariate_sensor_anomaly",
        "equipment_health_prediction",
        "health_index_calibration",
    ),
    "action_event_monitoring": ("action_event_sequence_recognition",),
}


@dataclass(frozen=True)
class TemporalUploadLayout:
    """标准上传目录布局"""

    task_type: str
    required_dirs: tuple[str, ...]
    optional_dirs: tuple[str, ...] = ()
    samples_index_file: str = "samples.jsonl"
    manifest_name: str = "manifest.json"
    notes: tuple[str, ...] = field(default_factory=tuple)


TEMPORAL_UPLOAD_LAYOUTS: dict[str, TemporalUploadLayout] = {
    "acoustic_time_frequency_anomaly": TemporalUploadLayout(
        task_type="acoustic_time_frequency_anomaly",
        required_dirs=("waveforms",),
        optional_dirs=("spectrograms", "labels", "metadata"),
        notes=(
            "推荐同时上传 waveforms/ 与 spectrograms/，支持原始音频与时频图双路径训练。",
            "labels/ 可为异常片段标注、样本级分类或仅 normal/anomaly 二分类。",
        ),
    ),
    "multivariate_sensor_anomaly": TemporalUploadLayout(
        task_type="multivariate_sensor_anomaly",
        required_dirs=("timeseries",),
        optional_dirs=("labels", "metadata"),
        notes=(
            "timeseries/ 下推荐使用按设备分片的 csv/parquet/jsonl 文件。",
            "metadata/ 存放量纲、采样周期、设备类型、区域和缺测策略。",
        ),
    ),
    "equipment_health_prediction": TemporalUploadLayout(
        task_type="equipment_health_prediction",
        required_dirs=("timeseries", "labels"),
        optional_dirs=("metadata",),
        notes=(
            "labels/ 必须包含健康指数、异常分数或故障标签中的至少一种监督信号。",
            "device_monitoring 建议三个输出层同时标注: health_index、anomaly_score、predicted_failure。",
        ),
    ),
    "health_index_calibration": TemporalUploadLayout(
        task_type="health_index_calibration",
        required_dirs=("timeseries", "labels"),
        optional_dirs=("metadata",),
        notes=(
            "labels/ 需提供 health_index 目标值，可由人工巡检评分、SLA 评分或专家规则生成。",
            "建议 metadata/ 提供设备类型、厂家、部署位置、维护等级，用于分层校准。",
        ),
    ),
    "action_event_sequence_recognition": TemporalUploadLayout(
        task_type="action_event_sequence_recognition",
        required_dirs=("events", "labels"),
        optional_dirs=("metadata",),
        notes=(
            "events/ 推荐 jsonl 或 csv，每行一条带时间戳事件。",
            "labels/ 可为序列级类别、风险等级或关键事件区间。",
        ),
    ),
}


def is_temporal_task(task_type: str) -> bool:
    """判断 task_type 是否属于时序任务"""

    return task_type in TEMPORAL_TASK_TYPES


def get_temporal_task_profile(task_type: str) -> TemporalTaskProfile | None:
    """获取时序任务画像"""

    return TEMPORAL_TASK_PROFILES.get(task_type)


def get_temporal_upload_layout(task_type: str) -> TemporalUploadLayout | None:
    """获取标准上传布局"""

    return TEMPORAL_UPLOAD_LAYOUTS.get(task_type)
