#!/usr/bin/env python3
"""真实 ONNX 模型 preflight 检查 CLI

用途
----
用于真实 YOLOv8 ONNX 模型交付时的 gating 检查。直接调用
`BirdDetector._preflight_onnx()` 的契约逻辑，输出结构化报告。

不做：
  - 不下载模型
  - 不训练模型
  - 不伪造推理结果
  - 不修改 configs/default.yaml

只做：
  - 临时把传入的模型路径塞进 BirdDetector 的初始化路径
  - 让 detector 真实跑 _load_model + _preflight_onnx
  - 打印 healthcheck.preflight 报告
  - 退出码反映 preflight passed/failed/blocked

退出码
------
  0  preflight passed (model_loaded == True, runtime_mode 可切 real_dl)
  1  preflight performed but failed（issues 非空）
  2  preflight 未执行（onnxruntime 缺失 / 模型文件不存在 / session 加载失败）
  3  CLI 参数错误

用法
----
  scripts/check_real_model.py path/to/bird_yolov8n.onnx
  scripts/check_real_model.py path/to/bird_yolov8n.onnx --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_repo_root_to_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bird_monitoring real_dl preflight")
    parser.add_argument("model_path", help="ONNX 模型绝对或相对路径")
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出完整 preflight 报告（机器可读）",
    )
    args = parser.parse_args(argv)

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        print(f"[FAIL] 模型文件不存在: {model_path}", file=sys.stderr)
        return 2

    _add_repo_root_to_syspath()
    from plugins.bird_monitoring.detector import BirdDetector

    detector = BirdDetector({"model": {"path": str(model_path)}})
    report = detector._preflight_report
    runtime_status = detector.get_runtime_status() if hasattr(
        detector, "get_runtime_status"
    ) else {}

    payload = {
        "model_path": str(model_path),
        "real_model_loaded": bool(detector.real_model_loaded),
        "onnx_session_ready": bool(detector.onnx_session_ready),
        "model_load_error": detector.model_load_error,
        "preflight": report,
        "runtime_status": runtime_status,
    }

    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(f"model_path             : {model_path}")
        print(f"real_model_loaded      : {detector.real_model_loaded}")
        print(f"onnx_session_ready     : {detector.onnx_session_ready}")
        print(f"model_load_error       : {detector.model_load_error}")
        print(f"preflight.performed    : {report.get('performed')}")
        print(f"preflight.passed       : {report.get('passed')}")
        checks = report.get("checks", {})
        for key in (
            "input_count",
            "input_shape",
            "output_count",
            "output_shape",
            "expected_channels",
            "channels_match",
            "class_map_size",
        ):
            if key in checks:
                print(f"checks.{key:<20s}: {checks[key]}")
        issues = report.get("issues", [])
        if issues:
            print("issues:")
            for issue in issues:
                print(f"  - {issue}")

    if not report.get("performed"):
        return 2
    if not report.get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
