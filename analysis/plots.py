"""各图共用的绘制工具：预算轴上的前值保持曲线与 bootstrap 带、末段阴影、共享图例、点估计加区间。"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from analysis.style import Style, apply_rc
from experiment.common.plotting import get_pyplot
from experiment.common.stats import bootstrap_iqm
from experiment.experiments.report import CellResult

CURVE_POINTS = 120
N_BOOT = 2000


def figure(nrows: int, ncols: int, width: float, height: float, **kwargs: Any) -> tuple[Any, np.ndarray] | None:
    """
    返回 (fig, 二维 axes 数组)；matplotlib 不可用时返回 None。
    统一用 constrained layout：标题、轴标签、图外图例都参与排版，彼此不会重叠，各图不必再手动调间距。
    """
    plt = apply_rc()
    if plt is None:
        return None
    fig, axes = plt.subplots(nrows, ncols, figsize = (width, height), squeeze = False, layout = "constrained", **kwargs)
    return fig, axes


def held_matrix(cells: Sequence[CellResult], key: str, xs: np.ndarray) -> np.ndarray:
    """(seed, x) 矩阵：每个 x 取 games <= x 的最近一次评估（前值保持），之前没有评估为 NaN。"""
    out = np.full((len(cells), xs.size), np.nan)
    for i, cell in enumerate(cells):
        rows = cell.series(key)
        if not rows:
            continue
        games, values = np.asarray(sorted(rows), dtype = np.float64).T
        index = np.searchsorted(games, xs, side = "right") - 1
        valid = index >= 0
        out[i, valid] = values[index[valid]]
    return out


def band(matrix: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐列的 seed 均值与 bootstrap 95% 区间；按 seed 整行重采样，列间共用同一组重采样。"""
    n = matrix.shape[0]
    draws = np.random.default_rng(seed).integers(0, n, size = (N_BOOT, n))
    # 早于首个评估点的列全为 NaN，nanmean 会对它们告警，结果仍是 NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(matrix[draws], axis = 1)
        mean = np.nanmean(matrix, axis = 0)
        lo, hi = np.nanquantile(means, [0.025, 0.975], axis = 0)
    return mean, lo, hi


def curve(ax: Any, cells: Sequence[CellResult], key: str, style: Style, floor: float | None = None) -> bool:
    """在 ax 上画一组 seed 的均值阶梯线与区间带；没有任何评估点时不画，返回 False。"""
    budget = max(c.cfg.budget for c in cells)
    xs = np.geomspace(max(budget / 1000.0, 1.0), budget, CURVE_POINTS)
    matrix = held_matrix(cells, key, xs)
    if not np.isfinite(matrix).any():
        return False
    mean, lo, hi = band(matrix)
    if floor is not None:
        mean, lo, hi = (np.maximum(v, floor) for v in (mean, lo, hi))
    ax.step(xs, mean, where = "post", **style.line_kwargs())
    ax.fill_between(xs, lo, hi, step = "post", color = style.color, alpha = 0.12, linewidth = 0)
    return True


def shade_tail(ax: Any, budget: float, start: float = 0.8) -> None:
    ax.axvspan(start * budget, budget, color = "0.9", zorder = 0, linewidth = 0)


def _handles(axes: Any) -> dict[str, Any]:
    handles: dict[str, Any] = {}
    for ax in np.asarray(axes).flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    return handles


def shared_legend(fig: Any, axes: np.ndarray, ncol: int = 5) -> None:
    """所有子图的图例去重后放在图外下方。"""
    handles = _handles(axes)
    if handles:
        fig.legend(list(handles.values()), list(handles), loc = "outside lower center", ncol = min(ncol, len(handles)))


def row_legend(axes_row: np.ndarray, title: str | None = None) -> None:
    """一行子图的图例去重后放在这一行最右侧子图的右边（多行曲线图每行一个图例）。"""
    handles = _handles(axes_row)
    if handles:
        axes_row[-1].legend(list(handles.values()), list(handles), loc = "center left", bbox_to_anchor = (1.02, 0.5),
                            borderaxespad = 0.0, title = title, title_fontsize = 7, alignment = "left")


def tidy_log(ax: Any) -> None:
    """对数轴只标主刻度；跨度不到两个数量级时改在 1、2、5 处标普通数字，避免次刻度标签挤在一起。须在画完数据后调用。"""
    plt = get_pyplot()
    if plt is None:
        return
    for axis, scale, (lo, hi) in ((ax.xaxis, ax.get_xscale(), ax.get_xlim()), (ax.yaxis, ax.get_yscale(), ax.get_ylim())):
        if scale != "log":
            continue
        if hi < 100 * max(lo, 1e-300):
            axis.set_major_locator(plt.LogLocator(base = 10, subs = (1.0, 2.0, 5.0)))
            axis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: _compact(value)))
        axis.set_minor_formatter(plt.NullFormatter())


def _compact(value: float) -> str:
    """刻度标签：1e4 以上写成 2e5 这样的短形式，其余用 %g。"""
    if value < 1e4:
        return f"{value:g}"
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def dot_ci(ax: Any, y: float, values: Sequence[float], style: Style, show_points: bool = True) -> tuple[float, float, float]:
    """一行 IQM 与 95% 区间（横向误差棒），可选地叠加逐 seed 的点；返回 (IQM, 下界, 上界)。"""
    point, lo, hi = bootstrap_iqm(values)
    if not np.isfinite(point):
        return point, lo, hi
    ax.errorbar(point, y, xerr = [[point - lo], [hi - point]], fmt = style.marker, color = style.color, markersize = 4,
                capsize = 2, linewidth = 1.0)
    if show_points:
        jitter = np.linspace(-0.15, 0.15, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, y + jitter, s = 4, color = style.color, alpha = 0.35, linewidths = 0)
    return point, lo, hi


def label_rows(ax: Any, labels: Sequence[str]) -> None:
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)


FOREST_NOTE = ("paired per-seed difference (IQM, 95% CI; negative = first method better); "
               "green / red: CI below / above 0; right column: seeds won / lost")


def forest(ax: Any, rows: Sequence[tuple[str, dict[str, Any] | None]]) -> None:
    """
    逐 seed 配对差（前者 - 后者，越负越好）的 IQM 与 95% 区间，一行一对。
    区间整体小于 0 为绿色，整体大于 0 为红色，跨过 0 为灰色；没有数据的行（None）留空。
    胜/负 seed 数写在子图内部右侧单独的一列（右边留出空白），不会伸到相邻子图里。
    """
    for y, (_label, stats) in enumerate(rows):
        if stats is None or not np.isfinite(stats["ci"][0]):
            continue
        point, lo, hi = stats["ci"]
        color = "#2ca02c" if hi < 0 else "#d62728" if lo > 0 else "0.45"
        ax.errorbar(point, y, xerr = [[point - lo], [hi - point]], fmt = "o", color = color, markersize = 3.5,
                    capsize = 2, linewidth = 1.0)
        ax.annotate(f"{stats['wins']}/{stats['losses']}", xy = (0.985, y), xycoords = ax.get_yaxis_transform(),
                    fontsize = 6.5, va = "center", ha = "right", color = color)
    ax.axvline(0.0, color = "0.6", linewidth = 0.6, zorder = 0)
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + 0.22 * (right - left))
    label_rows(ax, [label for label, _stats in rows])


DUMBBELL_NOTE = "open marker: best checkpoint over all evaluations (optimistic); filled marker: tail exploitability (seed IQM)"


def dumbbell(ax: Any, y: float, best: Sequence[float], tail: Sequence[float], style: Style, floor: float) -> None:
    """同一组 seed 的全程最低点（空心）与末段均值（实心），两者的 IQM 用线段相连。"""
    lo_point = max(bootstrap_iqm(best)[0], floor)
    hi_point = max(bootstrap_iqm(tail)[0], floor)
    if not (np.isfinite(lo_point) and np.isfinite(hi_point)):
        return
    ax.plot([lo_point, hi_point], [y, y], color = style.color, linewidth = 1.0)
    ax.scatter([lo_point], [y], marker = style.marker, s = 20, facecolors = "white", edgecolors = style.color, zorder = 3)
    ax.scatter([hi_point], [y], marker = style.marker, s = 20, color = style.color, zorder = 3)
