"""有限矩阵博弈上的精确 CoDA-R 一步（隐式 VI）与同一输出点上的全部证书量。

正确算法：对偶锚固定在 e_m、自博弈份额为常数、镜像对手是本步的解、池系数按解重新投影。
其余开关只用于必要性对照（论文 §3 与附录），由调用方标为诊断。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiment.core.simplex import simplex_proj


@dataclass
class Solution:
    x: np.ndarray
    # 在解处按（含噪声的）败率投影得到的对偶权重 (w_P, w_m)
    w: np.ndarray
    # 原侧训练实际使用的池系数；冻结对照中与 w 不同
    used_w: np.ndarray
    delta: float
    e: float
    iterations: int


def dual_weights(s: np.ndarray, pool: np.ndarray, x: np.ndarray, c: float, anchor: np.ndarray,
                 noise: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """w = Pi(anchor + c * l~)；误差只注入池成员，镜像坐标恒为 1/2。返回 (w, 实际误差的无穷范数)。"""
    losses = np.append(0.5 + (s @ pool).T @ x, 0.5)
    perturbation = np.zeros_like(losses) if noise is None else np.array(noise, dtype = np.float64)
    perturbation[-1] = 0.0
    estimate = np.clip(losses + perturbation, 0.0, 1.0)
    return simplex_proj(anchor + c * estimate), float(np.max(np.abs(estimate - losses)))


def fixed_anchor(n: int) -> np.ndarray:
    return np.append(np.zeros(n), 1.0)


def solve_step(
    s: np.ndarray,
    pool: np.ndarray,
    x0: np.ndarray,
    eta: float,
    c: float,
    share: float = 1.0,
    anchor: np.ndarray | None = None,
    noise: np.ndarray | None = None,
    target_delta: float = 0.0,
    explicit_mirror: bool = False,
    live_share: bool = False,
    frozen_weights: np.ndarray | None = None,
    tol: float = 1e-11,
    max_iter: int = 200_000,
) -> Solution:
    """
    外梯度法求解 (P)；每个试探点都重新投影对偶权重，保证停机点的原、对偶量互相配对。
    停机条件是含近端项的 Frank-Wolfe 间隙不超过 max(target_delta, tol)，它就是 (P) 的松弛量 delta。
    anchor 缺省为固定锚 e_m；explicit_mirror / live_share / frozen_weights 只用于必要性对照。
    """
    n = pool.shape[1]
    a = s @ pool
    center = fixed_anchor(n) if anchor is None else np.asarray(anchor, dtype = np.float64)

    def field(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        w, e = dual_weights(s, pool, x, c, center, noise)
        used = w if frozen_weights is None else frozen_weights
        lam = float(w[-1]) if live_share else share
        g = lam * (s @ (x0 if explicit_mirror else x)) + a @ used[:n] + eta * (x - x0)
        return g, w, used, e

    lip = eta + max(abs(share), float(live_share)) * np.linalg.norm(s, 2)
    if n:
        lip += c * np.linalg.norm(a, 2) ** 2 + (c * np.linalg.norm(s, 2) * np.linalg.norm(a, 2) if live_share else 0.0)
    step = 0.8 / lip
    x = np.array(x0, dtype = np.float64)
    gap = float("inf")
    for it in range(max_iter):
        g, w, used, e = field(x)
        # 单纯形上线性函数的最小值在顶点取到，所以 FW 间隙可以精确计算
        gap = max(0.0, float(g @ x - g.min()))
        if gap <= max(target_delta, tol):
            return Solution(x = x, w = w, used_w = np.array(used, dtype = np.float64), delta = gap, e = e, iterations = it)
        y = simplex_proj(x - step * g)
        gy, _, _, _ = field(y)
        x = simplex_proj(x - step * gy)
    raise RuntimeError(f"VI did not reach tolerance: gap={gap:.3g} after {max_iter} iterations")


def certificate(s: np.ndarray, pool: np.ndarray, x0: np.ndarray, sol: Solution, nash: np.ndarray,
                eta: float, c: float, share: float) -> dict[str, Any]:
    """所有量都在同一个输出点上计算；error = delta + a * e 是定理 7/8 的误差项。"""
    x, w = sol.x, sol.w
    n = pool.shape[1]
    dx = float(np.linalg.norm(x - x0))
    mass = max(0.0, 1.0 - float(w[-1]))
    dual_sq = float(w[:-1] @ w[:-1] + (w[-1] - 1.0) ** 2)
    primal = eta * np.sqrt(2.0) * dx
    dual = np.sqrt(2.0 * dual_sq) / c
    pool_expl = max(0.0, float(np.max(x @ s @ pool, initial = 0.0)))
    full_expl = max(0.0, float(np.max(x @ s)))
    before = float(np.sum((x0 - nash) ** 2))
    after = float(np.sum((x - nash) ** 2))
    movement = dx ** 2 + 2.0 * dual_sq / (c * eta)
    error = sol.delta + mass * sol.e
    # 当前混合 q = v / t，v = share * x + sum_i w_i y_i
    v = share * x + pool @ w[:n]
    t = float(share + mass)
    q_expl = max(0.0, float(np.max((v / t) @ s))) if t > 0 else float("nan")
    full_rhs = 0.5 * mass + primal + error
    pool_rhs = dual + primal + 2.0 * sol.e + sol.delta
    mixture_rhs = primal + error
    pairing = float(np.sqrt(2.0) * np.linalg.norm((s @ pool) @ (w[:n] - sol.used_w[:n]))) if n else 0.0
    return dict(
        dx = dx, pool_mass = mass, dual_distance_squared = dual_sq, mirror_weight = float(w[-1]), share = share,
        full_expl = full_expl, pool_expl = pool_expl, q_expl = q_expl, t = t, v = v,
        primal_term = primal, dual_term = dual, full_rhs = full_rhs, pool_rhs = pool_rhs, mixture_rhs = mixture_rhs,
        distance_before = before, distance_after = after, movement = movement, error = error,
        e = sol.e, delta = sol.delta, iterations = sol.iterations, pairing_error = pairing,
        fejer_violation = after > before - movement + 2.0 * error / eta + 1e-9,
        full_violation = share * full_expl > full_rhs + 1e-9,
        pool_violation = pool_expl > pool_rhs + 1e-9,
        mixture_violation = t * q_expl > mixture_rhs + 1e-9 if t > 0 else False,
    )
