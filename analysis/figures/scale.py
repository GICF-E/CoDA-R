"""规模实验（abrupt Dark Hex、abrupt Phantom TTT）的全树精确 exploitability 与历史遗忘。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from analysis.data import Outputs
from analysis.plots import figure, shade_tail, shared_legend, tidy_log
from analysis.style import SCALE_ENVS, env_title, method_style
from experiment.common.plotting import save_figure
from experiment.common.runner import done_records, latest_records
from experiment.common.stats import bootstrap_iqm
from experiment.experiments.protocol import MAIN_BUDGET
from experiment.experiments.scale import SCALE_METHODS, TAIL_FRACTIONS


def seed_tail(rows: list[dict[str, Any]]) -> dict[int, float]:
    """每个 seed 在末段三个 checkpoint 上的精确 exploitability 均值（与 scale-eval 的 tail_mean 相同）。"""
    out: dict[int, float] = {}
    for seed in sorted({int(r["seed"]) for r in rows}):
        tail = [float(r["exploitability"]) for r in rows if int(r["seed"]) == seed and float(r["fraction"]) in TAIL_FRACTIONS]
        out[seed] = float(np.mean(tail)) if tail else float("nan")
    return out


def scale_curves(out: Outputs, dest: Path) -> list[str]:
    """各 checkpoint 处的 seed IQM 与 95% 区间；横轴为该比例处 checkpoint 的局数（seed 中位数）。"""
    by_cell = out.scale_rows()
    made = figure(1, len(SCALE_ENVS), 7.2, 2.6, sharey = True) if by_cell else None
    if made is None:
        return []
    fig, axes = made
    for ax, env in zip(axes[0], SCALE_ENVS):
        for method in SCALE_METHODS:
            rows = by_cell.get((env, method), [])
            if not rows:
                continue
            fractions = sorted({float(r["fraction"]) for r in rows})
            games = [float(np.median([r["games"] for r in rows if float(r["fraction"]) == f])) for f in fractions]
            stats = np.array([bootstrap_iqm([r["exploitability"] for r in rows if float(r["fraction"]) == f])
                              for f in fractions])
            style = method_style(method)
            ax.plot(games, stats[:, 0], marker = style.marker, markersize = 3.5, **style.line_kwargs())
            ax.fill_between(games, stats[:, 1], stats[:, 2], color = style.color, alpha = 0.12, linewidth = 0)
        shade_tail(ax, MAIN_BUDGET)
        ax.set(xscale = "log", title = env_title(env), xlabel = "algorithmic games")
        tidy_log(ax)
    axes[0, 0].set_ylabel("exact exploitability")
    shared_legend(fig, axes, ncol = 6)
    save_figure(fig, dest, "scale_curves")
    return ["scale_curves"]


def scale_history(out: Outputs, dest: Path) -> list[str]:
    """上：对更早 checkpoint 的最坏败率减 1/2；下：曾被打败后又反超更晚 checkpoint 的占比。"""
    path = out.root / "scale-history"
    if not out.has("scale-history"):
        return []
    records = done_records(latest_records(path))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        key = (r["payload"]["config"]["env"], r["payload"]["method"])
        groups.setdefault(key, []).append(r["summary"])
    made = figure(2, len(SCALE_ENVS), 8.0, 4.8, sharex = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for col, env in enumerate(SCALE_ENVS):
        for i, method in enumerate(SCALE_METHODS):
            rows = groups.get((env, method), [])
            if not rows:
                continue
            style = method_style(method)
            for ax, key in ((axes[0, col], "worst_past"), (axes[1, col], "forgotten")):
                mid, lo, hi = bootstrap_iqm([float(row[key]) for row in rows])
                if np.isfinite(mid):
                    ax.errorbar([i], [mid], yerr = [[mid - lo], [hi - mid]], marker = style.marker, capsize = 2,
                                markersize = 4, **style.line_kwargs())
        axes[0, col].axhline(0.0, color = "0.7", linewidth = 0.6)
        axes[0, col].set(title = env_title(env), xlim = (-0.5, len(SCALE_METHODS) - 0.5))
        axes[1, col].set(xlabel = "method", xlim = (-0.5, len(SCALE_METHODS) - 0.5),
                         xticks = list(range(len(SCALE_METHODS))),
                         xticklabels = [method_style(m).label for m in SCALE_METHODS], ylim = (0.0, 1.05))
        axes[1, col].tick_params(axis = "x", labelrotation = 25)
    axes[0, 0].set_ylabel("worst past lead")
    axes[1, 0].set_ylabel("forgotten past checkpoints")
    shared_legend(fig, axes, ncol = 6)
    save_figure(fig, dest, "scale_history")
    return ["scale_history"]
