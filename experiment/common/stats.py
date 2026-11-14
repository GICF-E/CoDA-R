"""跨 seed 的统计：IQM 与 bootstrap 区间、配对差、Wilson 区间，以及预算轴上的末段指标。"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

METRIC_KEYS = ("at_budget", "tail_mean", "tail_worst", "auc")


def _iqm_weights(n: int) -> np.ndarray:
    # 对经验分位函数在 [0.25, 0.75] 上积分，边界样本按落在区间内的比例计权，小样本也不必整条丢弃
    left = np.arange(n, dtype = np.float64)
    return np.maximum(0.0, np.minimum(left + 1, 0.75 * n) - np.maximum(left, 0.25 * n)) / (0.5 * n)


def finite(values: Any) -> np.ndarray:
    xs = np.asarray(values, dtype = np.float64).reshape(-1)
    return xs[np.isfinite(xs)]


def iqm(values: Any) -> float:
    xs = np.sort(finite(values))
    return float(_iqm_weights(xs.size) @ xs) if xs.size else float("nan")


def bootstrap_iqm(values: Any, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """(IQM, 95% 下界, 95% 上界)，按 seed 重采样。"""
    xs = finite(values)
    if xs.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.sort(np.random.default_rng(seed).choice(xs, size = (n_boot, xs.size), replace = True), axis = 1)
    stats = draws @ _iqm_weights(xs.size)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return iqm(xs), float(lo), float(hi)


def bootstrap_mean(values: Any, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    xs = finite(values)
    if xs.size == 0:
        return float("nan"), float("nan"), float("nan")
    draws = np.random.default_rng(seed).choice(xs, (n_boot, xs.size), replace = True).mean(axis = 1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(xs.mean()), float(lo), float(hi)


def paired(treatment: Any, reference: Any) -> dict[str, Any]:
    """逐 seed 配对差 treatment - reference（越小越好）的 IQM 与区间，以及胜负场数。"""
    a = np.asarray(treatment, dtype = np.float64)
    b = np.asarray(reference, dtype = np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    diffs = a[keep] - b[keep]
    return dict(n = int(diffs.size), ci = bootstrap_iqm(diffs), wins = int(np.sum(diffs < 0)),
                losses = int(np.sum(diffs > 0)), diffs = diffs.tolist())


def history_forgetting(losses: Sequence[Sequence[float]], margin: float) -> dict[str, float]:
    """
    losses[t][j]：第 t 次考试时 latest 对第 j 个历史版本的败率，只含考试时已经存在的版本，按加入顺序排列。
    worst_past：每次考试对历史版本的最坏败率减 1/2，再对考试取平均；final_worst_past：最后一次考试的这个量。
    forgotten：曾被 latest 打败（败率 <= 1/2 - margin）、之后又打败 latest（败率 >= 1/2 + margin）的历史版本占比。
    """
    rows = [np.asarray(row, dtype = np.float64) for row in losses if len(row)]
    if not rows:
        return dict(worst_past = float("nan"), final_worst_past = float("nan"), forgotten = float("nan"))
    worst = [float(row.max()) - 0.5 for row in rows]
    n_versions = max(row.size for row in rows)
    forgotten = 0
    for j in range(n_versions):
        beaten = False
        for row in rows:
            if row.size <= j:
                continue
            if row[j] <= 0.5 - margin:
                beaten = True
            elif beaten and row[j] >= 0.5 + margin:
                forgotten += 1
                break
    return dict(worst_past = float(np.mean(worst)), final_worst_past = worst[-1], forgotten = forgotten / n_versions)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """二项比例的 Wilson 区间：(点估计, 下界, 上界)。"""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, float(center - half), float(center + half)


# -------- 预算轴上的末段指标 --------
def _held(rows: list[tuple[float, float]], x: float) -> float:
    """前值保持：games <= x 的最近一次评估；不借用未来信息。"""
    value = float("nan")
    for games, v in rows:
        if games > x:
            break
        value = v
    return value


def _hold_mean(rows: list[tuple[float, float]], lo: float, hi: float) -> float:
    area, x, value = 0.0, lo, _held(rows, lo)
    for games, v in rows:
        if games <= lo:
            continue
        if games >= hi:
            break
        area += value * (games - x)
        x, value = games, v
    return (area + value * (hi - x)) / (hi - lo)


def tail_metrics(evals: Sequence[tuple[float, float]], budget: float, tail_start: float) -> dict[str, float]:
    """
    评估点按前值保持展开成预算轴上的阶梯函数 E(g)：
    at_budget = E(B)；tail_mean / tail_worst = E 在 [tail_start, B] 上按宽度加权的平均与最大值；auc = [0, B] 上的平均。
    阶梯函数没有覆盖相应区间时记 NaN，不外推。
    """
    rows = sorted((float(g), float(v)) for g, v in evals if np.isfinite(g) and np.isfinite(v))
    nan = float("nan")
    out = {key: nan for key in METRIC_KEYS}
    if not rows or rows[-1][0] < budget:
        return out
    out["at_budget"] = _held(rows, budget)
    if rows[0][0] <= tail_start:
        out["tail_mean"] = _hold_mean(rows, tail_start, budget)
        inside = [v for g, v in rows if tail_start < g <= budget]
        out["tail_worst"] = max([_held(rows, tail_start), *inside])
    if rows[0][0] <= 0.0:
        out["auc"] = _hold_mean(rows, 0.0, budget)
    return out
