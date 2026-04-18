"""Busbar 模型资产训练源阻塞事实测试。"""

from __future__ import annotations

from pathlib import Path
import zipfile

import yaml

from plugins.busbar_inspection.label_contract import RUNTIME_SUPPORTED_DEFECT_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_busbar_detector_checkpoints_are_placeholders_not_real_export_sources():
    """当前 busbar 检测 best.pt 只是占位文本，不能作为真实导出源。"""
    checkpoint_paths = [
        PROJECT_ROOT / "training" / "checkpoints" / "busbar" / "EHV_500kV" / "best.pt",
        PROJECT_ROOT / "training" / "checkpoints" / "busbar" / "HV_220kV" / "best.pt",
    ]

    for path in checkpoint_paths:
        assert path.exists() is True
        assert path.read_text(encoding="utf-8").startswith("# Placeholder model")


def test_busbar_processed_training_classes_do_not_define_a_single_runtime_safe_export_set():
    """各电压等级 busbar 训练类别集合不一致，且包含当前 runtime contract 外类别。"""
    data_yaml_paths = sorted(
        (PROJECT_ROOT / "training" / "data" / "processed").glob("*/busbar/data.yaml")
    )

    observed_class_sets: set[tuple[str, ...]] = set()
    unsupported_labels: set[str] = set()

    for path in data_yaml_paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = data.get("names", {})
        if isinstance(names, dict):
            labels = tuple(str(names[idx]) for idx in sorted(names))
        else:
            labels = tuple(str(label) for label in names)
        observed_class_sets.add(labels)
        for label in labels:
            if label not in RUNTIME_SUPPORTED_DEFECT_LABELS:
                unsupported_labels.add(label)

    assert len(observed_class_sets) > 1
    assert "fitting_loose" in unsupported_labels
    assert "bird" in unsupported_labels


def test_busbar_noise_classifier_checkpoint_is_not_the_required_detector_train_source():
    """现有 .pth 是噪声分类器检查点，不是当前 defect detector 的导出源。"""
    path = PROJECT_ROOT / "training" / "checkpoints" / "busbar" / "busbar_noise_classifier_best.pth"

    assert path.exists() is True
    assert path.name == "busbar_noise_classifier_best.pth"
    assert zipfile.is_zipfile(path) is True


def test_busbar_export_directory_has_no_generated_triplet_ready_for_preflight():
    """当前训练导出目录为空，说明尚无可追溯的新三件套产物。"""
    export_dir = PROJECT_ROOT / "training" / "exports" / "busbar"

    assert export_dir.exists() is True
    assert list(export_dir.rglob("*.*")) == []
