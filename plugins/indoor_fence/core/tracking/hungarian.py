"""
Indoor Fence V3.0 - Hungarian Algorithm
=========================================

Simplified implementation of the Hungarian (Munkres) algorithm
for optimal assignment in multi-target tracking.
Uses scipy if available, falls back to greedy assignment.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def linear_assignment(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Solve the linear assignment problem.

    Uses scipy.optimize.linear_sum_assignment if available,
    otherwise falls back to a greedy approximation.

    Args:
        cost_matrix: (N, M) cost matrix where N = tracks, M = detections

    Returns:
        List of (row, col) assignment pairs.
    """
    if cost_matrix.size == 0:
        return []

    try:
        from scipy.optimize import linear_sum_assignment as scipy_lsa
        row_ind, col_ind = scipy_lsa(cost_matrix)
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    except ImportError:
        return _greedy_assignment(cost_matrix)


def _greedy_assignment(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Greedy approximation of the assignment problem."""
    n_rows, n_cols = cost_matrix.shape
    assignments: List[Tuple[int, int]] = []
    used_rows: set = set()
    used_cols: set = set()

    # Flatten and sort by cost
    flat = []
    for i in range(n_rows):
        for j in range(n_cols):
            flat.append((cost_matrix[i, j], i, j))
    flat.sort(key=lambda x: x[0])

    for cost, r, c in flat:
        if r not in used_rows and c not in used_cols:
            assignments.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
            if len(assignments) == min(n_rows, n_cols):
                break

    return assignments
