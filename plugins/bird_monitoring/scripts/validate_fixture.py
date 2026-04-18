#!/usr/bin/env python3
"""真实 bird fixture intake 校验 CLI

用途
----
当有人按 `prompts/fixture_collection_prompt.md` 采集真实 bird 图片并落入
`tests/fixtures/{normal,anomaly,boundary,quality_fail}/*.jpg` 时，用本脚本做
intake 校验：
  - 文件确实存在且非空
  - cv2 可读、HxW > min_dimension
  - sample_id 与 `tests/replay/expected_results.json` 槽位一致
  - 槽位 collection_status 仍为 planned/planned_blocked_by_model 时，提示
    采集者手动改为 collected（本脚本不写入 JSON）

不做：
  - 不写 expected_results.json
  - 不删除/移动文件
  - 不做模型推理（preflight 走 check_real_model.py）

退出码
------
  0  全部 fixture 通过 intake 校验
  1  存在校验失败的 fixture
  2  CLI 参数错误 / 槽位 JSON 缺失
  3  cv2 不可用

用法
----
  scripts/validate_fixture.py                    # 校验 expected_results.json 中所有声明的样本
  scripts/validate_fixture.py --sample-id bird_quality_dark_001
  scripts/validate_fixture.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PLUGIN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_JSON = PLUGIN_DIR / "tests" / "replay" / "expected_results.json"
MIN_DIMENSION_DEFAULT = 32


def _load_expected() -> dict[str, Any]:
    if not EXPECTED_JSON.exists():
        raise FileNotFoundError(EXPECTED_JSON)
    return json.loads(EXPECTED_JSON.read_text(encoding="utf-8"))


def _resolve_asset_path(asset_relpath: str) -> Path:
    p = Path(asset_relpath)
    if p.is_absolute():
        return p
    return REPO_ROOT / asset_relpath


def _check_one(sample: dict[str, Any], min_dim: int) -> dict[str, Any]:
    sample_id = sample.get("sample_id", "<unknown>")
    asset_path = _resolve_asset_path(sample.get("asset_relpath", ""))
    status = sample.get("collection_status", "unknown")
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "asset_path": str(asset_path),
        "collection_status": status,
        "exists": False,
        "readable": False,
        "size_ok": False,
        "shape": None,
        "issues": [],
        "passed": False,
    }

    if not asset_path.exists():
        if status in ("planned", "planned_blocked_by_model"):
            result["issues"].append(f"slot_pending:{status}")
        else:
            result["issues"].append("file_missing")
        return result
    result["exists"] = True

    if asset_path.stat().st_size == 0:
        result["issues"].append("empty_file")
        return result

    import cv2  # local import — only required when fixtures actually exist

    img = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        result["issues"].append("cv2_unreadable")
        return result
    result["readable"] = True
    result["shape"] = list(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    if h < min_dim or w < min_dim:
        result["issues"].append(f"dim_below_min:{h}x{w}<{min_dim}")
    else:
        result["size_ok"] = True

    if status in ("planned", "planned_blocked_by_model"):
        result["issues"].append(
            "collection_status_not_updated:still_marked_" + status
        )

    result["passed"] = not result["issues"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bird_monitoring fixture intake validator")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="只校验指定 sample_id（可重复）",
    )
    parser.add_argument(
        "--min-dimension",
        type=int,
        default=MIN_DIMENSION_DEFAULT,
        help="最小宽高像素（默认 %(default)s，与 quality 门 min_dimension 对齐）",
    )
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args(argv)

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[FAIL] cv2 (opencv-python) 不可用", file=sys.stderr)
        return 3

    try:
        expected = _load_expected()
    except FileNotFoundError:
        print(f"[FAIL] expected_results.json 缺失: {EXPECTED_JSON}", file=sys.stderr)
        return 2

    samples = expected.get("samples", [])
    if args.sample_id:
        wanted = set(args.sample_id)
        samples = [s for s in samples if s.get("sample_id") in wanted]
        if not samples:
            print(f"[FAIL] 找不到 sample_id: {args.sample_id}", file=sys.stderr)
            return 2

    results = [_check_one(s, args.min_dimension) for s in samples]
    failed = [r for r in results if not r["passed"]]

    if args.json:
        json.dump(
            {"checked": len(results), "failed": len(failed), "results": results},
            sys.stdout,
            ensure_ascii=False,
            indent=2,
        )
        sys.stdout.write("\n")
    else:
        for r in results:
            tag = "OK  " if r["passed"] else "FAIL"
            extra = ""
            if r["shape"]:
                extra = f" shape={r['shape']}"
            print(f"[{tag}] {r['sample_id']:<32s} status={r['collection_status']}{extra}")
            for issue in r["issues"]:
                print(f"        - {issue}")
        print()
        print(f"checked={len(results)} failed={len(failed)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
