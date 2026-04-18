"""Synthetic fixture builder for bird_monitoring replay baseline.

只生成 `simulation` 主链能稳定复现的两类样本：
- `no_bird_clear`：正常亮度的无鸟图（plugin 应输出 `no_bird`）
- `quality_dark`：亮度低到触发 hard_fail 的图（plugin 应输出 `quality_failed`）

其它 label（`bird_safe` / `unknown_bird` / `review_required` / …）**不得**由合成数据
伪造，必须依赖真实图片 + real_dl 模型。该约束由 `expected_results.json` 的
`collection_status=planned_blocked_by_model` 字段表达。

运行:
    python3 plugins/bird_monitoring/tests/regression/build_synthetic_fixtures.py
    → 写入 plugins/bird_monitoring/tests/regression/fixtures/{no_bird_clear,quality_dark}.npy

被 `tests/test_replay_baseline.py` 调用以构建确定性回归样本；不会覆盖任何
真实采集的 .jpg 样本（真实样本以 .jpg 落入 `tests/fixtures/` 的对应子目录）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def build_no_bird_clear(seed: int = 2026) -> np.ndarray:
    """640x480 亮度合理的噪声图；不含鸟。"""
    rng = np.random.default_rng(seed)
    # 均值约 128，模拟自然背景亮度
    frame = rng.integers(80, 180, size=(480, 640, 3), dtype=np.uint8)
    return np.ascontiguousarray(frame)


def build_quality_dark(seed: int = 2026) -> np.ndarray:
    """640x480 全黑+极微噪声，触发 detector 亮度不足分支。"""
    rng = np.random.default_rng(seed)
    base = np.zeros((480, 640, 3), dtype=np.uint8)
    noise = rng.integers(0, 5, size=(480, 640, 3), dtype=np.uint8)
    return np.ascontiguousarray(base + noise)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURE_DIR / "no_bird_clear.npy", build_no_bird_clear())
    np.save(FIXTURE_DIR / "quality_dark.npy", build_quality_dark())
    print(f"[build_synthetic_fixtures] wrote to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
