"""消融、尾部回升诊断与机制实验（M1–M4）的图。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from analysis.data import Outputs, group_by, method_key
from analysis.plots import (
    DUMBBELL_NOTE,
    FOREST_NOTE,
    curve,
    dumbbell,
    figure,
    forest,
    label_rows,
    shade_tail,
    shared_legend,
    tidy_log,
)
from analysis.style import ABLATION_ENVS, LOG_FLOOR, env_title, method_style, variant_styles
from experiment.common.io_util import load_json
from experiment.common.plotting import save_figure
from experiment.common.stats import bootstrap_iqm, paired
from experiment.experiments.ablations import BASE_LABEL, VARIANTS as ABLATION_VARIANTS
from experiment.experiments.diagnostics import BASE_VARIANT, DIAG_METHODS, VARIANTS as DIAG_VARIANTS
from experiment.experiments.mechanisms import (
    ATTACKER,
    DEFAULT_SUPPRESSION,
    HOLE_ROLLOUTS,
    HOLE_RULES,
    M1_RULES,
    M2_RULES,
    POOL_SIZES,
    RETENTION_BANDS,
    RETENTION_METHODS,
    SUPPRESSION_ROLLOUTS,
    censored_median,
    hole_metrics,
    open_pool_metrics,
    retention_metrics,
    vuln_metrics,
)
from experiment.experiments.methods import title
from experiment.experiments.report import CellResult, degradation

# 消融变体的名称是 experiment 里的配置键，已经记录在实验输出里，不能改；图上换成论文的术语
ABLATION_DISPLAY = {
    "A2 sampling base = field": "A2 sampling: coefficients",
    "A2 sampling base = half": "A2 sampling: half weights, half coefficients",
    "A6 mirror refreshed per outer step": "A6 self-play opponent refreshed per outer step",
    "A7 pool coefficients frozen per outer step": "A7 pool weights frozen per outer step",
}


def _paired_vs(cells: list[CellResult], base: list[CellResult]) -> dict[str, Any] | None:
    if not cells or not base:
        return None
    mine = {c.cfg.seed: c.metrics()["tail_mean"] for c in cells}
    ref = {c.cfg.seed: c.metrics()["tail_mean"] for c in base}
    seeds = sorted(set(mine) & set(ref))
    return paired([mine[s] for s in seeds], [ref[s] for s in seeds])


# -------- 消融 --------
def ablations_forest(out: Outputs, dest: Path) -> list[str]:
    """每个消融变体相对 CoDA-R 基准的逐 seed 配对差（tail_mean）。"""
    groups = group_by(out.neural("ablations"), lambda r: (r.cfg.env, r.cfg.label))
    made = figure(1, len(ABLATION_ENVS), 8.0, 4.2, sharey = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for ax, env in zip(axes[0], ABLATION_ENVS):
        base = groups.get((env, BASE_LABEL), [])
        forest(ax, [(ABLATION_DISPLAY.get(label, label), _paired_vs(groups.get((env, label), []), base))
                    for label in ABLATION_VARIANTS])
        ax.set(title = env_title(env), xlabel = "tail exploitability: variant - CoDA-R base")
    fig.supxlabel(FOREST_NOTE, fontsize = 7)
    save_figure(fig, dest, "ablations_forest")
    return ["ablations_forest"]


# -------- 尾部回升诊断 --------
def _diag_groups(out: Outputs) -> dict[tuple[str, str, str], list[CellResult]]:
    return group_by(out.neural("diag-tail"), lambda r: (r.cfg.env, str(r.meta["method"]), str(r.meta["variant"])))


def diag_tail_curves(out: Outputs, dest: Path) -> list[str]:
    groups = _diag_groups(out)
    made = figure(len(ABLATION_ENVS), len(DIAG_METHODS), 8.0, 5.2, sharey = "row") if groups else None
    if made is None:
        return []
    fig, axes = made
    styles = variant_styles(DIAG_VARIANTS, highlight = BASE_VARIANT)
    for i, env in enumerate(ABLATION_ENVS):
        for j, method in enumerate(DIAG_METHODS):
            ax = axes[i, j]
            budget = 0.0
            for variant in DIAG_VARIANTS:
                cells = groups.get((env, method, variant), [])
                if cells and curve(ax, cells, "exploitability", styles[variant], LOG_FLOOR):
                    budget = max(budget, *(c.cfg.budget for c in cells))
            if budget:
                shade_tail(ax, budget)
            ax.set(xscale = "log", yscale = "log", title = f"{env_title(env)} / {title(method)}")
            tidy_log(ax)
        axes[i, 0].set_ylabel("exploitability")
    for ax in axes[-1]:
        ax.set_xlabel("algorithmic games")
    shared_legend(fig, axes, ncol = 3)
    save_figure(fig, dest, "diag_tail_curves")
    return ["diag_tail_curves"]


def diag_tail_summary(out: Outputs, dest: Path) -> list[str]:
    """上：每个变体的全程最低点与末段均值；下：变体相对 base 的逐 seed 配对差。"""
    groups = _diag_groups(out)
    made = figure(2, len(ABLATION_ENVS), 8.0, 7.0) if groups else None
    if made is None:
        return []
    fig, axes = made
    rows = [(method, variant) for method in DIAG_METHODS for variant in DIAG_VARIANTS]
    labels = [f"{title(m)} | {v}" for m, v in rows]
    for col, env in enumerate(ABLATION_ENVS):
        ax = axes[0, col]
        for y, (method, variant) in enumerate(rows):
            values = [degradation(c) for c in groups.get((env, method, variant), [])]
            if values:
                dumbbell(ax, y, [v["best"] for v in values], [v["tail_mean"] for v in values], method_style(method),
                         LOG_FLOOR)
        label_rows(ax, labels)
        ax.set(xscale = "log", title = env_title(env), xlabel = "exploitability")
        tidy_log(ax)
        forest(axes[1, col], [(f"{title(m)} | {v}", _paired_vs(groups.get((env, m, v), []),
                                                                groups.get((env, m, BASE_VARIANT), [])))
                              for m in DIAG_METHODS for v in DIAG_VARIANTS if v != BASE_VARIANT])
        axes[1, col].set_xlabel("tail exploitability: variant - base")
    fig.supxlabel(f"top: {DUMBBELL_NOTE}\nbottom: {FOREST_NOTE}", fontsize = 7)
    save_figure(fig, dest, "diag_tail_summary")
    return ["diag_tail_summary"]


# -------- M1 漏洞注入 --------
def attacker_traces(cells: list[CellResult]) -> tuple[np.ndarray, np.ndarray]:
    """攻击者 A 的真实败率与训练系数，逐 rollout、对 seed 取均值。"""
    losses, fields = [], []
    for c in cells:
        rows = [h for h in c.history if h.get("roll", -1) >= 0 and h.get("pool_losses_true") and ATTACKER in h["pool_ids"]]
        losses.append([h["pool_losses_true"][h["pool_ids"].index(ATTACKER)] for h in rows])
        fields.append([h["pool_field"][h["pool_ids"].index(ATTACKER)] for h in rows])
    length = min(len(v) for v in losses)
    return (np.mean([v[:length] for v in losses], axis = 0), np.mean([v[:length] for v in fields], axis = 0))


def vuln_repair(out: Outputs, dest: Path) -> list[str]:
    """上：learner 对攻击者 A 的真实败率；下：A 持有的训练权重；虚线为 learner 重置回底座的时刻（默认压制时长）。"""
    cells = [r for r in out.neural("vuln-repair") if int(r.meta.get("suppression", DEFAULT_SUPPRESSION)) == DEFAULT_SUPPRESSION]
    groups = group_by(cells, lambda r: (r.cfg.env, method_key(r)))
    made = figure(2, len(ABLATION_ENVS), 8.0, 4.8, sharex = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for col, env in enumerate(ABLATION_ENVS):
        for rule in M1_RULES:
            cells = groups.get((env, rule), [])
            if not cells:
                continue
            loss, coef = attacker_traces(cells)
            style = method_style(rule)
            axes[0, col].plot(np.arange(loss.size), loss, **style.line_kwargs())
            axes[1, col].plot(np.arange(coef.size), coef, **style.line_kwargs())
            for ax in axes[:, col]:
                ax.axvline(int(cells[0].meta["reset_at"]), color = "0.5", linestyle = ":", linewidth = 0.8)
        axes[0, col].axhline(0.5, color = "0.7", linewidth = 0.6)
        axes[0, col].set(title = env_title(env))
        axes[1, col].set_xlabel("rollout")
    axes[0, 0].set_ylabel("true loss rate vs attacker A")
    axes[1, 0].set_ylabel("A's training weight")
    shared_legend(fig, axes, ncol = 7)
    save_figure(fig, dest, "vuln_repair")
    return ["vuln_repair"]


def _iqm_line(ax: Any, xs: list[float], samples: list[list[float]], style: Any) -> None:
    stats = np.array([bootstrap_iqm(values) for values in samples])
    keep = np.isfinite(stats[:, 0])
    if keep.any():
        x, s = np.asarray(xs)[keep], stats[keep]
        ax.errorbar(x, s[:, 0], yerr = [s[:, 0] - s[:, 1], s[:, 2] - s[:, 0]], marker = style.marker, capsize = 2,
                    markersize = 3.5, **style.line_kwargs())


def vuln_sleeper(out: Outputs, dest: Path) -> list[str]:
    """上：重置后攻击者 A 的系数回到重置前峰值一半所需的 rollout 数；下：没有在赢 learner 的成员占训练权重的比例。横轴为压制时长。"""
    groups = group_by(out.neural("vuln-repair"), lambda r: (r.cfg.env, method_key(r), int(r.meta.get("suppression", 0))))
    made = figure(2, len(ABLATION_ENVS), 8.0, 4.8, sharex = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for col, env in enumerate(ABLATION_ENVS):
        for rule in M1_RULES:
            if rule == "none":
                continue
            metrics = [[vuln_metrics(c) for c in groups.get((env, rule, t), [])] for t in SUPPRESSION_ROLLOUTS]
            style = method_style(rule)
            _iqm_line(axes[0, col], list(SUPPRESSION_ROLLOUTS), [[m["reweight_delay"] for m in ms] for ms in metrics], style)
            _iqm_line(axes[1, col], list(SUPPRESSION_ROLLOUTS), [[m["idle_field"] for m in ms] for ms in metrics], style)
        axes[0, col].set(title = env_title(env))
        axes[1, col].set(xscale = "log", xlabel = "suppression length T (rollouts)", xticks = list(SUPPRESSION_ROLLOUTS),
                         xticklabels = [str(t) for t in SUPPRESSION_ROLLOUTS])
        axes[1, col].minorticks_off()
    axes[0, 0].set_ylabel("re-weighting delay (rollouts)")
    axes[1, 0].set_ylabel("training weight of members\nnot beating the learner")
    shared_legend(fig, axes, ncol = 6)
    save_figure(fig, dest, "vuln_sleeper")
    return ["vuln_sleeper"]


# -------- M2 开放池 --------
def open_pool(out: Outputs, dest: Path) -> list[str]:
    """上：已被打败的成员占训练权重的比例；下：末段 exploitability，虚线为不用池的 R-NaD-PPO。IQM 与 95% 区间。"""
    results = out.neural("open-pool")
    groups = group_by(results, lambda r: (r.cfg.env, method_key(r), int(r.meta["pool_size"])))
    made = figure(2, len(ABLATION_ENVS), 8.0, 4.8, sharex = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for col, env in enumerate(ABLATION_ENVS):
        for rule in M2_RULES:
            style = method_style(rule)
            for row, values_of in ((0, lambda c: open_pool_metrics(c)["beaten_share"]),
                                   (1, lambda c: c.metrics()["tail_mean"])):
                stats = np.array([bootstrap_iqm([values_of(c) for c in groups.get((env, rule, n), [])])
                                  for n in POOL_SIZES])
                ax = axes[row, col]
                ax.errorbar(POOL_SIZES, stats[:, 0], yerr = [stats[:, 0] - stats[:, 1], stats[:, 2] - stats[:, 0]],
                            marker = style.marker, capsize = 2, **style.line_kwargs())
        reference = groups.get((env, "none", 0), [])
        if reference:
            level = bootstrap_iqm([c.metrics()["tail_mean"] for c in reference])[0]
            axes[1, col].axhline(level, label = method_style("none").label, color = method_style("none").color,
                                 linestyle = "--", linewidth = 1.0)
        axes[0, col].set(title = env_title(env))
        axes[1, col].set(xscale = "log", xlabel = "pool size N")
        axes[1, col].minorticks_off()
        axes[1, col].set(xticks = list(POOL_SIZES), xticklabels = [str(n) for n in POOL_SIZES])
    axes[0, 0].set_ylabel("training weight of beaten members")
    axes[1, 0].set_ylabel("tail exploitability")
    shared_legend(fig, axes, ncol = 4)
    save_figure(fig, dest, "open_pool")
    return ["open_pool"]


# -------- M3 隐藏漏洞 --------
def _band_reach(out: Outputs, cells: list[CellResult]) -> float:
    """一档漏洞的对手到达概率：各 seed 的 prep 里该档漏洞信息集的均值。"""
    reaches = []
    for c in cells:
        band = load_json(out.root / "hidden-hole" / str(c.meta["prep"]) / "holes.json")["bands"][int(c.meta["band"])]
        reaches += [float(h["opponent_reach"]) for h in band["holes"]]
    return float(np.mean(reaches)) if reaches else float("nan")


def hidden_hole(out: Outputs, dest: Path) -> list[str]:
    """
    上：修复时间（漏洞动作的平均概率降到一半以下所需的 rollout 数，seed 中位数；空心标记表示超过一半的 seed 没有修复）
    随漏洞在自博弈下的对手到达概率的变化；下：到达概率最低一档的漏洞动作平均概率。
    """
    groups = group_by(out.neural("hidden-hole"), lambda r: (r.cfg.env, method_key(r), int(r.meta["band"])))
    made = figure(2, len(ABLATION_ENVS), 8.0, 4.8) if groups else None
    if made is None:
        return []
    fig, axes = made
    floor = 1e-4
    for col, env in enumerate(ABLATION_ENVS):
        bands = sorted({b for e, _rule, b in groups if e == env})
        if not bands:
            continue
        for rule in HOLE_RULES:
            style = method_style(rule)
            points = [(max(_band_reach(out, groups[(env, rule, b)]), floor),
                       censored_median([hole_metrics(c)["t_repair"] for c in groups[(env, rule, b)]], HOLE_ROLLOUTS))
                      for b in bands if (env, rule, b) in groups]
            xs = np.array([x for x, _t in points])
            ts = np.array([t for _x, t in points])
            shown = np.where(np.isfinite(ts), ts, 1.15 * HOLE_ROLLOUTS)
            axes[0, col].plot(xs, shown, **style.line_kwargs())
            axes[0, col].scatter(xs[np.isfinite(ts)], shown[np.isfinite(ts)], marker = style.marker, color = style.color, s = 18)
            axes[0, col].scatter(xs[~np.isfinite(ts)], shown[~np.isfinite(ts)], marker = style.marker, s = 18,
                                 facecolors = "white", edgecolors = style.color)
            lowest = groups.get((env, rule, bands[-1]), [])
            traces = [[float(h["hole_mass"]) for h in c.history if h.get("roll", -1) >= 0 and "hole_mass" in h] for c in lowest]
            if traces:
                length = min(len(t) for t in traces)
                axes[1, col].plot(np.arange(length), np.mean([t[:length] for t in traces], axis = 0), **style.line_kwargs())
        axes[0, col].axhline(HOLE_ROLLOUTS, color = "0.7", linewidth = 0.6, linestyle = ":")
        axes[0, col].set(xscale = "log", title = env_title(env), xlabel = "opponent reach of the vulnerability under self-play")
        tidy_log(axes[0, col])
        axes[1, col].axhline(0.5, color = "0.7", linewidth = 0.6)
        axes[1, col].set(xlabel = "rollout", title = "lowest-reach band")
    axes[0, 0].set_ylabel("rollouts to repair (seed median)")
    axes[1, 0].set_ylabel("prob. of the planted action")
    shared_legend(fig, axes, ncol = 3)
    save_figure(fig, dest, "hidden_hole")
    return ["hidden_hole"]


# -------- M4 遗忘 --------
def _retention_main(out: Outputs) -> list[CellResult]:
    # c 敏感性分析只在 summary 里；图上只比调参选出的配置
    return [r for r in out.neural("forgetting") if float(r.meta.get("c_factor", 1.0)) == 1.0]


def forgetting(out: Outputs, dest: Path) -> list[str]:
    """上：末段 exploitability 相对起点的上升；下：曾被打败后又反超 learner 的历史版本占比。"""
    cells = _retention_main(out)
    groups = group_by(cells, lambda r: (r.cfg.env, method_key(r)))
    made = figure(2, len(ABLATION_ENVS), 8.0, 4.8, sharex = True) if groups else None
    if made is None:
        return []
    fig, axes = made
    for col, env in enumerate(ABLATION_ENVS):
        for i, method in enumerate(RETENTION_METHODS):
            rows = [retention_metrics(c) for c in groups.get((env, method), [])]
            if not rows:
                continue
            style = method_style(method)
            for ax, key in ((axes[0, col], "expl_rise"), (axes[1, col], "forgotten")):
                mid, lo, hi = bootstrap_iqm([r[key] for r in rows])
                if np.isfinite(mid):
                    ax.errorbar([i], [mid], yerr = [[mid - lo], [hi - mid]], marker = style.marker, capsize = 2,
                                markersize = 4, **style.line_kwargs())
        axes[0, col].axhline(0.0, color = "0.7", linewidth = 0.6)
        axes[0, col].set(title = env_title(env), xlim = (-0.5, len(RETENTION_METHODS) - 0.5))
        axes[1, col].set(xlabel = "method", xlim = (-0.5, len(RETENTION_METHODS) - 0.5),
                         xticks = list(range(len(RETENTION_METHODS))),
                         xticklabels = [method_style(m).label for m in RETENTION_METHODS], ylim = (0.0, 1.05))
        axes[1, col].tick_params(axis = "x", labelrotation = 25)
    axes[0, 0].set_ylabel("exploitability rise from start")
    axes[1, 0].set_ylabel("forgotten past versions")
    shared_legend(fig, axes, ncol = 5)
    save_figure(fig, dest, "forgetting")
    return ["forgetting"]


def forgetting_drift(out: Outputs, dest: Path) -> list[str]:
    """按自博弈对手到达概率分档的策略漂移（末段每次到达比 CFR+ 底座少得的归一化得分）。"""
    cells = _retention_main(out)
    groups = group_by(cells, lambda r: (r.cfg.env, method_key(r)))
    made = figure(1, len(ABLATION_ENVS), 8.0, 2.8) if groups else None
    if made is None:
        return []
    fig, axes = made
    xs = [0.5 * (lo + min(hi, 1.0)) for lo, hi in RETENTION_BANDS]
    labels = [f"drift_loss opp reach [{lo:g}, {min(hi, 1.0):g})" for lo, hi in RETENTION_BANDS]
    for col, env in enumerate(ABLATION_ENVS):
        ax = axes[0, col]
        for method in RETENTION_METHODS:
            rows = [retention_metrics(c) for c in groups.get((env, method), [])]
            if not rows:
                continue
            samples = [[r[label] for r in rows] for label in labels]
            _iqm_line(ax, xs, samples, method_style(method))
        ax.axhline(0.0, color = "0.7", linewidth = 0.6)
        ax.set(xscale = "log", title = env_title(env), xlabel = "opponent reach under self-play")
        tidy_log(ax)
    axes[0, 0].set_ylabel("drift loss vs CFR+ base")
    shared_legend(fig, axes, ncol = 5)
    save_figure(fig, dest, "forgetting_drift")
    return ["forgetting_drift"]
