"""调参的敏感性：阶段 1 每个配置的得分热图，框出最终选定的配置，标题注明选定值是否落在网格边界。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from analysis.data import Outputs
from analysis.plots import figure
from analysis.style import MAIN_ENVS, env_title
from experiment.common.plotting import get_pyplot, save_figure
from experiment.experiments.methods import CODA_C, ENT_FINALS, METHODS, NFSP_LRS, TUNE_STAGE_A, TUNE_STAGE_B, title

Axes = tuple[str, list[float], str, list[float]]


def grid_axes(method: str, env: str) -> Axes:
    """(行参数, 行取值, 列参数, 列取值)，取值从小到大。"""
    spec = METHODS[method]
    if method == "nfsp":
        return "nfsp_lr", sorted(NFSP_LRS), "nfsp_anticipatory", sorted(spec.values)
    assert spec.param is not None
    values = CODA_C[env] if method == "coda_r" else spec.values
    return "ent_final", sorted(ENT_FINALS), spec.param, sorted(float(v) for v in values)


def edge_flags(method: str, env: str, params: dict[str, Any]) -> dict[str, str]:
    """选定配置的每个调参维度：'min' / 'max' 表示落在网格边界，'inner' 表示在内部。"""
    row, rows, col, cols = grid_axes(method, env)
    flags = {}
    for name, values in ((row, rows), (col, cols)):
        value = float(params[name])
        flags[name] = "min" if value == values[0] else "max" if value == values[-1] else "inner"
    return flags


def stage1_scores(out: Outputs, stage: str) -> dict[tuple[str, str], dict[tuple[float, float], float]]:
    """{(游戏, 方法): {(行取值, 列取值): tail_mean 的 seed 均值}}，只取阶段 1。"""
    acc: dict[tuple[str, str], dict[tuple[float, float], list[float]]] = {}
    for r in out.neural(f"tune-{stage}"):
        if int(r.meta["tune_stage"]) != 1:
            continue
        method = str(r.meta["method"])
        row, _rows, col, _cols = grid_axes(method, r.cfg.env)
        key = (float(r.meta["params"][row]), float(r.meta["params"][col]))
        acc.setdefault((r.cfg.env, method), {}).setdefault(key, []).append(r.metrics()["tail_mean"])
    return {k: {cell: float(np.mean(v)) for cell, v in by_cell.items()} for k, by_cell in acc.items()}


def _heatmaps(out: Outputs, dest: Path, stage: str, methods: tuple[str, ...]) -> list[str]:
    scores = stage1_scores(out, stage)
    selected = out.selected(stage)
    made = figure(len(methods), len(MAIN_ENVS), 8.0, 2.1 * len(methods) + 0.4) if scores else None
    if made is None:
        return []
    fig, axes = made
    plt = get_pyplot()
    for i, method in enumerate(methods):
        for j, env in enumerate(MAIN_ENVS):
            ax = axes[i, j]
            row, rows, col, cols = grid_axes(method, env)
            by_cell = scores.get((env, method), {})
            grid = np.array([[by_cell.get((r, c), np.nan) for c in cols] for r in rows])
            ax.imshow(grid, cmap = "RdYlGn_r", aspect = "auto")
            lo, hi = np.nanmin(grid), np.nanmax(grid)
            for a in range(len(rows)):
                for b in range(len(cols)):
                    # 色板两端颜色很深，上面用白字
                    extreme = hi > lo and abs((grid[a, b] - lo) / (hi - lo) - 0.5) > 0.32
                    ax.text(b, a, f"{grid[a, b]:.3g}", ha = "center", va = "center", fontsize = 6.5,
                            color = "white" if extreme else "black")
            chosen = selected.get(env, {}).get(method)
            status = "not selected"
            if chosen is not None and plt is not None:
                a, b = rows.index(float(chosen[row])), cols.index(float(chosen[col]))
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill = False, edgecolor = "black", linewidth = 1.8))
                edges = [f"{k}={v}" for k, v in edge_flags(method, env, chosen).items() if v != "inner"]
                status = "edge: " + ", ".join(edges) if edges else "interior"
            header = f"{env_title(env)}\n" if i == 0 else ""
            ax.set_title(header + status, fontsize = 7)
            ax.set(xticks = range(len(cols)), xticklabels = [f"{v:g}" for v in cols], yticks = range(len(rows)),
                   yticklabels = [f"{v:g}" for v in rows], xlabel = col)
            ax.set_ylabel(f"{title(method)}\n{row}" if j == 0 else row)
    fig.supxlabel("stage-1 tail exploitability (seed mean, lower is better); black frame: configuration selected after stage 2",
                  fontsize = 7)
    stem = f"tuning_{stage}"
    save_figure(fig, dest, stem)
    return [stem]


def tuning_a(out: Outputs, dest: Path) -> list[str]:
    return _heatmaps(out, dest, "a", TUNE_STAGE_A)


def tuning_b(out: Outputs, dest: Path) -> list[str]:
    return _heatmaps(out, dest, "b", TUNE_STAGE_B)
