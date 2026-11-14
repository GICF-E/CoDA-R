"""有限维实验的博弈族与初始池。

所有博弈写成 l(x; y) = 1/2 + x^T S y，S 反对称且元素在 [-1/2, 1/2] 内，所以败率在 [0, 1] 内。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from experiment.theory.lp import row_minimax

FAMILIES = ("random", "cyclic", "spinning", "blotto")


@dataclass
class MatrixGame:
    s: np.ndarray
    # 初始池成员，每列一个混合策略
    pool: np.ndarray
    nash: np.ndarray
    rng: np.random.Generator

    @property
    def m(self) -> int:
        return int(self.s.shape[0])


def _skew(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a - a.T)


def _random_skew(m: int, rng: np.random.Generator) -> np.ndarray:
    return _skew(rng.uniform(-0.5, 0.5, (m, m)))


def _cyclic(m: int) -> np.ndarray:
    """广义石头剪刀布：i 输给 i + 1、赢 i - 1（循环）。"""
    c = np.zeros((m, m))
    for i in range(m):
        c[i, (i + 1) % m] += 0.5
        c[i, (i - 1) % m] -= 0.5
    return c


def _spinning(m: int, rng: np.random.Generator) -> np.ndarray:
    """传递加循环（spinning top 型）：评分高的策略赢面大，同时叠加循环结构。"""
    rating = rng.normal(size = m)
    transitive = -0.5 * np.tanh(rating[:, None] - rating[None, :])
    return 0.5 * transitive + 0.5 * _cyclic(m)


def _blotto(soldiers: int, fields: int) -> np.ndarray:
    """Colonel Blotto：每个战场兵多者胜，败率为 1/2 减去净胜战场比例的一半。"""
    allocations = [a for a in product(range(soldiers + 1), repeat = fields) if sum(a) == soldiers]
    m = len(allocations)
    s = np.zeros((m, m))
    for i, a in enumerate(allocations):
        for j, b in enumerate(allocations):
            net = sum(np.sign(x - y) for x, y in zip(a, b))
            s[i, j] = -0.5 * net / fields
    return s


def make_game(family: str, size: int, n_pool: int, seed: int, blotto_fields: int = 3) -> MatrixGame:
    """
    size 对 random / cyclic / spinning 是纯策略数 m，对 blotto 是兵力数（战场数为 blotto_fields）。
    初始池成员取均匀 Dirichlet 的随机混合策略。
    """
    rng = np.random.default_rng(seed)
    if family == "random":
        s = _random_skew(size, rng)
    elif family == "cyclic":
        s = 0.8 * _cyclic(size) + 0.2 * _random_skew(size, rng)
    elif family == "spinning":
        s = _spinning(size, rng)
    elif family == "blotto":
        s = _blotto(size, blotto_fields)
    else:
        raise ValueError(f"unknown family {family!r}")
    m = int(s.shape[0])
    pool = rng.dirichlet(np.ones(m), size = n_pool).T if n_pool else np.zeros((m, 0))
    # 对称博弈中行玩家的极小极大策略就是纳什策略
    nash = row_minimax(0.5 + s)
    return MatrixGame(s = s, pool = pool, nash = nash, rng = rng)


def rps() -> np.ndarray:
    return 0.5 * np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])


def exploitability(s: np.ndarray, x: np.ndarray) -> float:
    return float(max(np.max(x @ s), 0.0))


def best_response(s: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.eye(s.shape[0])[:, int(np.argmax(x @ s))]
