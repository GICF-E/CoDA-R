"""欧氏投影到概率单纯形。"""
from __future__ import annotations

import numpy as np


def simplex_proj(v: np.ndarray) -> np.ndarray:
    """
    Duchi / Michelot 水位法：找唯一的水位 tau 使 sum_i (v_i - tau)^+ = 1，返回 (v - tau)^+。
    够不到水位的分量精确为 0（论文附录 A.1）。
    """
    v = np.asarray(v, dtype = np.float64).reshape(-1)
    n = int(v.size)
    if n == 0:
        return v.copy()
    if n == 1:
        return np.ones(1, dtype = np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    j = np.arange(1, n + 1, dtype = np.float64)
    # 最大的 j（1-based）使 u_j > (css_j - 1) / j
    k = int(np.max(np.flatnonzero(u > (css - 1.0) / j)) + 1)
    tau = (css[k - 1] - 1.0) / float(k)
    return np.maximum(v - tau, 0.0)
