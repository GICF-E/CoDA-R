"""主实验的图：exploitability 曲线、末段指标、逐 seed 配对差、全程最低点与末段的退化、策略熵。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from analysis.data import Outputs
from analysis.plots import (
    DUMBBELL_NOTE,
    FOREST_NOTE,
    curve,
    dot_ci,
    dumbbell,
    figure,
    forest,
    label_rows,
    row_legend,
    shade_tail,
    tidy_log,
)
from analysis.style import (
    LOG_FLOOR,
    MAIN_ENVS,
    MIXTURE_METHODS,
    SINGLE_METHODS,
    UPPER_BOUND_METHODS,
    env_title,
    method_style,
)
from experiment.common.plotting import save_figure
from experiment.common.stats import paired
from experiment.experiments.report import CellResult, degradation, diagnostics

Groups = dict[tuple[str, str], list[CellResult]]
# CoDA-R 的时间平均 Q 不是单独的运行，而是 coda_r 格子里的 mix_average_expl 序列
DERIVED_SERIES = {"coda_r_q": ("coda_r", "mix_average_expl")}
# 前者 - 后者；与 experiment/experiments/main.py 的预先确定的配对一致，另加 Q 对 NFSP
MAIN_PAIRS = (
    ("coda_r", "pfsp"), ("coda_r", "ufsp"), ("coda_r", "rnad"), ("coda_r", "mmd"), ("coda_r", "coda_r_no_oracle"),
    ("coda_r_exact_br", "pfsp_exact_br"), ("coda_r_q", "psro"), ("coda_r_q", "nfsp"),
)
DEGRADATION_METHODS = (*SINGLE_METHODS, "coda_r_exact_br", "pfsp_exact_br")


def cells_of(groups: Groups, env: str, method: str) -> tuple[list[CellResult], str]:
    """一种方法在一个游戏上的格子与要读的序列名。"""
    source, key = DERIVED_SERIES.get(method, (method, "exploitability"))
    return groups.get((env, source), []), key


def tail_by_seed(cells: list[CellResult], key: str) -> dict[int, float]:
    return {c.cfg.seed: c.metrics(key)["tail_mean"] for c in cells}


def paired_tail(groups: Groups, env: str, treatment: str, reference: str) -> dict[str, Any] | None:
    a, key_a = cells_of(groups, env, treatment)
    b, key_b = cells_of(groups, env, reference)
    if not a or not b:
        return None
    ma, mb = tail_by_seed(a, key_a), tail_by_seed(b, key_b)
    seeds = sorted(set(ma) & set(mb))
    return paired([ma[s] for s in seeds], [mb[s] for s in seeds])


def pair_label(treatment: str, reference: str) -> str:
    return f"{method_style(treatment).label} - {method_style(reference).label}"


CurveRow = tuple[str, tuple[str, ...]]


def _curves(out: Outputs, dest: Path, stem: str, rows: tuple[CurveRow, ...], key_override: str | None = None,
            log_y: bool = True, ylabel: str = "exploitability") -> list[str]:
    """每行一组方法、每列一个游戏；图例放在每行右侧，标题为该行的分组名。"""
    groups = out.by_env_method("main")
    made = figure(len(rows), len(MAIN_ENVS), 8.4, 2.5 * len(rows), sharex = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for i, (row_title, methods) in enumerate(rows):
        for ax, env in zip(axes[i], MAIN_ENVS):
            budget = 0.0
            for method in methods:
                cells, key = cells_of(groups, env, method)
                if cells and curve(ax, cells, key_override or key, method_style(method), LOG_FLOOR if log_y else None):
                    budget = max(budget, *(c.cfg.budget for c in cells))
            if budget:
                shade_tail(ax, budget)
            ax.set(xscale = "log", yscale = "log" if log_y else "linear")
            if i == 0:
                ax.set_title(env_title(env))
            tidy_log(ax)
        axes[i, 0].set_ylabel(ylabel)
        row_legend(axes[i], row_title)
    fig.supxlabel("algorithmic games (grey band: tail [0.8B, B]; seed mean with 95% CI)", fontsize = 8)
    save_figure(fig, dest, stem)
    return [stem]


def main_curves(out: Outputs, dest: Path) -> list[str]:
    """上行：单网络部署；下行：混合部署，CoDA-R learner 作为参照。seed 均值与 95% 区间。"""
    return _curves(out, dest, "main_curves", (("single network", SINGLE_METHODS),
                                              ("mixtures (vs CoDA-R learner)", ("coda_r", *MIXTURE_METHODS))))


def main_curves_upper(out: Outputs, dest: Path) -> list[str]:
    """精确 BR oracle 的上限对照（虚线/点线）及其 PPO oracle 版本（实线）。"""
    return _curves(out, dest, "main_curves_upper", (("PPO oracle vs exact-BR oracle", ("coda_r", "pfsp", "psro",
                                                                                           *UPPER_BOUND_METHODS)),))


def main_entropy(out: Outputs, dest: Path) -> list[str]:
    return _curves(out, dest, "main_entropy", (("single network", SINGLE_METHODS),), key_override = "entropy",
                   log_y = False, ylabel = "policy entropy")


def main_tail(out: Outputs, dest: Path) -> list[str]:
    """tail_mean 的 seed IQM 与 95% 区间及逐 seed 点；三条分隔线依次分开单网络、混合部署与精确 BR 上限。"""
    groups = out.by_env_method("main")
    methods = (*SINGLE_METHODS, *MIXTURE_METHODS, *UPPER_BOUND_METHODS)
    made = figure(1, len(MAIN_ENVS), 8.4, 3.4, sharey = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for ax, env in zip(axes[0], MAIN_ENVS):
        for y, method in enumerate(methods):
            cells, key = cells_of(groups, env, method)
            if cells:
                values = [max(v, LOG_FLOOR) for v in tail_by_seed(cells, key).values()]
                dot_ci(ax, y, values, method_style(method))
        for split in (len(SINGLE_METHODS), len(SINGLE_METHODS) + len(MIXTURE_METHODS)):
            ax.axhline(split - 0.5, color = "0.75", linewidth = 0.6)
        ax.set(xscale = "log", title = env_title(env), xlabel = "tail exploitability")
        tidy_log(ax)
        label_rows(ax, [method_style(m).label for m in methods])
    fig.supxlabel("seed IQM with 95% CI, dots = seeds; rows: single network / mixtures / exact-BR upper bounds",
                  fontsize = 7)
    save_figure(fig, dest, "main_tail")
    return ["main_tail"]


def main_paired(out: Outputs, dest: Path) -> list[str]:
    groups = out.by_env_method("main")
    made = figure(1, len(MAIN_ENVS), 8.4, 3.0, sharey = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for ax, env in zip(axes[0], MAIN_ENVS):
        forest(ax, [(pair_label(a, b), paired_tail(groups, env, a, b)) for a, b in MAIN_PAIRS])
        ax.set(title = env_title(env), xlabel = "tail exploitability difference")
    fig.supxlabel(FOREST_NOTE, fontsize = 7)
    save_figure(fig, dest, "main_paired")
    return ["main_paired"]


def main_degradation(out: Outputs, dest: Path) -> list[str]:
    """全程最低 exploitability（空心）与末段均值（实心）的 seed IQM；最低点取自全部评估点，偏乐观。"""
    groups = out.by_env_method("main")
    made = figure(1, len(MAIN_ENVS), 8.4, 2.8, sharey = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for ax, env in zip(axes[0], MAIN_ENVS):
        for y, method in enumerate(DEGRADATION_METHODS):
            rows = [degradation(c) for c in groups.get((env, method), [])]
            if rows:
                dumbbell(ax, y, [r["best"] for r in rows], [r["tail_mean"] for r in rows], method_style(method), LOG_FLOOR)
        ax.set(xscale = "log", title = env_title(env), xlabel = "exploitability")
        tidy_log(ax)
        label_rows(ax, [method_style(m).label for m in DEGRADATION_METHODS])
    fig.supxlabel(DUMBBELL_NOTE, fontsize = 7)
    save_figure(fig, dest, "main_degradation")
    return ["main_degradation"]


def degradation_rows(out: Outputs, exp_id: str = "main") -> list[dict[str, Any]]:
    """分析表用：每个 (游戏, 方法) 的最低点、出现位置、末段均值、差距与末段熵，均为 seed 中位数。"""
    rows = []
    for (env, method), cells in sorted(out.by_env_method(exp_id).items()):
        values = [degradation(c) | dict(tail_entropy = diagnostics(c)["tail_entropy"]) for c in cells]
        if not any(np.isfinite(v["best"]) for v in values):
            continue
        row: dict[str, Any] = dict(env = env, method = method, n = len(values))
        row.update({k: float(np.nanmedian([v[k] for v in values]))
                    for k in ("best", "best_at", "tail_mean", "gap", "tail_entropy")})
        rows.append(row)
    return rows
