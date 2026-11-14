"""零和矩阵博弈的 LP：行玩家最小化损失，列玩家最大化损失。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from experiment.core.simplex import simplex_proj


def _solve(c: np.ndarray, a_ub: np.ndarray, b_ub: np.ndarray, n: int) -> np.ndarray:
    a_eq = np.zeros((1, n + 1))
    a_eq[0, :n] = 1.0
    res = linprog(c, A_ub = a_ub, b_ub = b_ub, A_eq = a_eq, b_eq = np.array([1.0]),
                  bounds = [(0.0, None)] * n + [(None, None)], method = "highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return np.asarray(res.x, dtype = np.float64)


def row_minimax(loss: np.ndarray) -> np.ndarray:
    """行玩家的极小极大策略：min_x max_j (L^T x)_j。"""
    loss = np.asarray(loss, dtype = np.float64)
    m, n = loss.shape
    c = np.zeros(m + 1)
    c[-1] = 1.0
    sol = _solve(c, np.hstack([loss.T, -np.ones((n, 1))]), np.zeros(n), m)
    return simplex_proj(sol[:m])
