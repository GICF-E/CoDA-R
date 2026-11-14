"""理论实验（有限矩阵博弈）的图：必要性对照、移动锚的盆地、带误差的下限、池规则对照、oracle 的速率与对抗池。"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from analysis.data import Outputs
from analysis.plots import band, figure, row_legend
from analysis.style import Style, method_style, variant_styles
from experiment.common.plotting import save_figure
from experiment.common.stats import wilson
from experiment.theory.experiments import ORACLE_ETAS
from experiment.theory.rules import ADVERSARIES, ORACLE_ARMS, RULES, SERIES_EVERY

EXPL_FLOOR = 1e-12
# 图上的记号与论文一致：自博弈权重 w_sp、估计误差 xi、策略更新误差 delta、纯策略数 M、池大小 N
WSP = r"$w_{\mathrm{sp}}$"
EXPL_LABEL = "Expl($x_k$), seed mean + 95% CI"
T6_STYLES = {
    "default": Style("CoDA-R", "#000000", "-", "o", 2.0),
    "explicit": Style("explicit self-play opponent", "#1f77b4", "-", "s"),
    "share0": Style("self-play coefficient 0", "#ff7f0e", "--", "^"),
    "share_live": Style(f"self-play coefficient = current {WSP}", "#2ca02c", "-.", "v"),
    "share_rule_b": Style(f"self-play coefficient = previous {WSP}", "#9467bd", ":", "D"),
    "frozen": Style("pool weights frozen per step", "#8c564b", "--", "P"),
}
# T7 的标签按名称排序取色和标记；系数 1 写成 $1$，排序才与 $w_sp$ 的标签保持原来的先后
T7_ONE = "$1$"
T7_BASE = f"fixed anchor, coefficient {T7_ONE}"
# T10–T12 的池规则；矩阵博弈里没有 PPO，无池的对照是纯近端自博弈
RULE_STYLES = {
    "coda_r": method_style("coda_r"),
    "moving_anchor": method_style("moving_anchor"),
    "pfsp": method_style("pfsp"),
    "uniform": method_style("ufsp"),
    "hedge": method_style("hedge"),
    "none": Style("proximal self-play (no pool)", "#4d4d4d", ":", "v"),
}
# T11 的臂；coda_r_eps 是同一规则配 eps-BR oracle
ARM_STYLES = {**{arm: RULE_STYLES[rule] for arm, rule in ORACLE_ARMS.items() if arm != "coda_r_eps"},
              "coda_r_eps": Style(r"CoDA-R, $\varepsilon$-BR oracle", "#ff9896", "--", "o")}
# T8 的条件：(锚, oracle, 估计误差 e, 名称)；None 表示该维度全部合并
T8_CONDITIONS: tuple[tuple[str, bool | None, float | None, str], ...] = (
    ("moving", False, 0.0, "moving anchor, exact"),
    ("moving", False, 0.05, r"moving anchor, $\xi$ = 0.05"),
    ("moving", True, None, "moving anchor + oracle"),
    ("fixed", None, None, "fixed anchor (all)"),
)


def _series(ax: Any, seqs: Sequence[Sequence[float]], style: Style) -> None:
    length = min(len(s) for s in seqs)
    mean, lo, hi = band(np.array([s[:length] for s in seqs], dtype = np.float64))
    xs = np.arange(1, length + 1) * SERIES_EVERY
    # 多条曲线常常重合在同一水平（同时收敛或同时卡住），稀疏的标记让被盖住的那条仍然可辨
    ax.plot(xs, np.maximum(mean, EXPL_FLOOR), marker = style.marker, markevery = max(length // 8, 1), markersize = 3.5,
            **style.line_kwargs())
    ax.fill_between(xs, np.maximum(lo, EXPL_FLOOR), np.maximum(hi, EXPL_FLOOR), color = style.color, alpha = 0.12,
                    linewidth = 0)


def theory_necessity(out: Outputs, dest: Path) -> list[str]:
    """左：T6 显式自博弈对手、改变自博弈系数或冻结池权重；右：T7 对偶锚 x 自博弈系数，从构造的不动点出发。"""
    t6, t7 = out.theory("theory-necessity"), out.theory("theory-anchor-witness")
    made = figure(1, 2, 8.0, 3.4) if t6 or t7 else None
    if made is None:
        return []
    fig, axes = made
    for payload, runs in t6:
        _series(axes[0, 0], [run["series"]["full_expl"] for run in runs], T6_STYLES[payload["mode"]])
    labels7 = {(p["anchor"], p["witness"]): f"{p['anchor']} anchor, coefficient {T7_ONE if p['witness'] == 'one' else WSP}"
               for p, _runs in t7}
    styles7 = {label: replace(style, marker = marker) for (label, style), marker
               in zip(sorted(variant_styles(labels7.values(), highlight = T7_BASE).items()), ("s", "^", "v", "D"))}
    for payload, runs in t7:
        _series(axes[0, 1], [run["series"]["full_expl"] for run in runs], styles7[labels7[(payload["anchor"], payload["witness"])]])
    for ax, name in zip(axes[0], ("necessity", "dual anchor x self-play coefficient")):
        ax.set(yscale = "log", xlabel = "outer step", title = name)
        ax.legend(loc = "upper center", bbox_to_anchor = (0.5, -0.2), ncol = 2, fontsize = 6.5)
    axes[0, 0].set_ylabel(EXPL_LABEL)
    save_figure(fig, dest, "theory_necessity")
    return ["theory_necessity"]


def anchor_basin_counts(out: Outputs) -> dict[tuple[str, str], tuple[int, int]]:
    """{(博弈族, 条件名): (卡住数, 运行数)}。"""
    counts: dict[tuple[str, str], list[int]] = {}
    for payload, runs in out.theory("theory-anchor-basin"):
        for anchor, oracle, e, name in T8_CONDITIONS:
            if payload["anchor"] != anchor or (oracle is not None and bool(payload["oracle"]) != oracle) \
                    or (e is not None and float(payload["e"]) != e):
                continue
            acc = counts.setdefault((str(payload["family"]), name), [0, 0])
            acc[0] += sum(bool(run["stuck"]) for run in runs)
            acc[1] += len(runs)
    return {k: (v[0], v[1]) for k, v in counts.items()}


def theory_anchor_basin(out: Outputs, dest: Path) -> list[str]:
    """T8：从随机初始点出发卡在非纳什点的比例，Wilson 95% 区间；合并了博弈规模、池大小与 eta。"""
    counts = anchor_basin_counts(out)
    made = figure(1, 1, 7.0, 2.6) if counts else None
    if made is None:
        return []
    fig, axes = made
    ax = axes[0, 0]
    families = sorted({family for family, _name in counts})
    width = 0.8 / len(T8_CONDITIONS)
    colors = ("#ff7f0e", "#ffbb78", "#9467bd", "#d62728")
    for k, (_anchor, _oracle, _e, name) in enumerate(T8_CONDITIONS):
        stats = np.array([wilson(*counts.get((family, name), (0, 0))) for family in families])
        xs = np.arange(len(families)) + (k - (len(T8_CONDITIONS) - 1) / 2) * width
        ax.bar(xs, stats[:, 0], width, color = colors[k], label = name,
               yerr = [stats[:, 0] - stats[:, 1], stats[:, 2] - stats[:, 0]], capsize = 2, error_kw = dict(linewidth = 0.8))
    ax.set(xticks = np.arange(len(families)), xticklabels = families, ylabel = "fraction stuck at non-Nash point",
           xlabel = r"game family (Wilson 95% CI; sizes, pool sizes and $\eta$ pooled)")
    row_legend(axes[0])
    save_figure(fig, dest, "theory_anchor_basin")
    return ["theory_anchor_basin"]


def theory_inexact(out: Outputs, dest: Path) -> list[str]:
    """T5：最终 exploitability 的中位数随策略更新误差 delta 与估计误差 xi（数据里的键为 e）的变化，每个池大小一张热图。"""
    cells = out.theory("theory-inexact")
    if not cells:
        return []
    finals: dict[tuple[int, float, float], list[float]] = {}
    for payload, runs in cells:
        key = (int(payload["n"]), float(payload["delta"]), float(payload["e"]))
        finals.setdefault(key, []).extend(float(run["final_expl"]) for run in runs)
    pools = sorted({k[0] for k in finals})
    deltas = sorted({k[1] for k in finals})
    errors = sorted({k[2] for k in finals})
    made = figure(1, len(pools), 8.0, 2.8)
    if made is None:
        return []
    fig, axes = made
    images = []
    for ax, n in zip(axes[0], pools):
        grid = np.array([[np.median(finals.get((n, d, e), [np.nan])) for e in errors] for d in deltas])
        logs = np.log10(np.maximum(grid, EXPL_FLOOR))
        images.append(ax.imshow(logs, cmap = "viridis", vmin = -12, vmax = -1, aspect = "auto"))
        for i in range(len(deltas)):
            for j in range(len(errors)):
                # viridis 的高端是亮黄色，上面用深色字
                ax.text(j, i, f"{grid[i, j]:.0e}", ha = "center", va = "center", fontsize = 6.5,
                        color = "black" if logs[i, j] > -4 else "white")
        ax.set(xticks = range(len(errors)), xticklabels = [f"{e:g}" for e in errors], yticks = range(len(deltas)),
               yticklabels = [f"{d:g}" for d in deltas], xlabel = r"estimation error $\xi$", title = f"pool size N = {n}")
    axes[0, 0].set_ylabel(r"policy-update error $\delta$")
    fig.colorbar(images[-1], ax = axes[0, -1], label = "log10 final Expl (median)")
    save_figure(fig, dest, "theory_inexact")
    return ["theory_inexact"]


def theory_pool_rules(out: Outputs, dest: Path) -> list[str]:
    """T10：各池规则的 Expl(x_k)；虚线为 learner 被重置回初始点的时刻。"""
    cells = out.theory("theory-pool-rules")
    sizes = sorted({int(p["m"]) for p, _runs in cells})
    made = figure(1, len(sizes), 8.4, 2.8, sharey = True) if cells else None
    if made is None:
        return []
    fig, axes = made
    for ax, m in zip(axes[0], sizes):
        for payload, runs in cells:
            if int(payload["m"]) == m:
                _series(ax, [run["series"]["full_expl"] for run in runs], RULE_STYLES[payload["rule"]])
                ax.axvline(int(payload["reset_at"]), color = "0.5", linestyle = ":", linewidth = 0.8)
        ax.set(yscale = "log", xlabel = "outer step (dotted: learner reset to $x_0$)", title = f"pool rules, M = {m}")
    axes[0, 0].set_ylabel(EXPL_LABEL)
    row_legend(axes[0])
    save_figure(fig, dest, "theory_pool_rules")
    return ["theory_pool_rules"]


def theory_oracle_rate(out: Outputs, dest: Path) -> list[str]:
    """T11：上行为最终 Expl 随 eta 的变化（中位数与四分位距），下行为 eta = 16 时的 Expl(x_k)；每列一个博弈族。"""
    cells = out.theory("theory-oracle-rate")
    games = sorted({(str(p["family"]), int(p["size"])) for p, _runs in cells})
    made = figure(2, len(games), 8.4, 5.0) if cells else None
    if made is None:
        return []
    fig, axes = made
    curve_eta = ORACLE_ETAS[-2]
    for col, (family, size) in enumerate(games):
        mine = [(p, runs) for p, runs in cells if (p["family"], p["size"]) == (family, size)]
        for arm, style in ARM_STYLES.items():
            points = sorted((float(p["eta"]), [float(run["final_expl"]) for run in runs]) for p, runs in mine if p["arm"] == arm)
            if not points:
                continue
            etas = np.array([eta for eta, _v in points])
            q = np.maximum(np.array([np.quantile(v, [0.25, 0.5, 0.75]) for _eta, v in points]), EXPL_FLOOR)
            axes[0, col].errorbar(etas, q[:, 1], yerr = [q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], marker = style.marker,
                                  capsize = 2, markersize = 3.5, **style.line_kwargs())
            for p, runs in mine:
                if p["arm"] == arm and float(p["eta"]) == curve_eta:
                    _series(axes[1, col], [run["series"]["full_expl"] for run in runs], style)
        axes[0, col].set(xscale = "log", yscale = "log", xlabel = r"$\eta$", title = f"{family}, M = {size}")
        axes[1, col].set(yscale = "log", xlabel = rf"outer step ($\eta$ = {curve_eta:g})")
    axes[0, 0].set_ylabel("final Expl (median, IQR)")
    axes[1, 0].set_ylabel(EXPL_LABEL)
    row_legend(axes[0])
    save_figure(fig, dest, "theory_oracle_rate")
    return ["theory_oracle_rate"]


def theory_adversarial_pool(out: Outputs, dest: Path) -> list[str]:
    """T12：对手 x 池规则的最终 Expl 中位数（log10 热图），每个策略数 m 一张，行按博弈族分组。"""
    cells = out.theory("theory-adversarial-pool")
    if not cells:
        return []
    finals: dict[tuple[int, str, str, str], list[float]] = {}
    for p, runs in cells:
        finals.setdefault((int(p["m"]), str(p["family"]), str(p["adversary"]), str(p["rule"])), []).extend(
            float(run["final_expl"]) for run in runs)
    sizes = sorted({k[0] for k in finals})
    families = sorted({k[1] for k in finals})
    rows = [(f, a) for f in families for a in ADVERSARIES]
    made = figure(1, len(sizes), 8.4, 0.28 * len(rows) + 1.4)
    if made is None:
        return []
    fig, axes = made
    images = []
    for ax, m in zip(axes[0], sizes):
        grid = np.array([[np.median(finals.get((m, f, a, rule), [np.nan])) for rule in RULES] for f, a in rows])
        logs = np.log10(np.maximum(grid, EXPL_FLOOR))
        images.append(ax.imshow(logs, cmap = "viridis", vmin = -12, vmax = -1, aspect = "auto"))
        for i in range(len(rows)):
            for j in range(len(RULES)):
                ax.text(j, i, f"{grid[i, j]:.0e}", ha = "center", va = "center", fontsize = 6,
                        color = "black" if logs[i, j] > -4 else "white")
        ax.set(xticks = range(len(RULES)), xticklabels = [RULE_STYLES[r].label for r in RULES], yticks = range(len(rows)),
               yticklabels = [f"{f} / {a}" for f, a in rows], title = f"adversarial pools, M = {m}")
        ax.tick_params(axis = "x", labelrotation = 40)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
    fig.colorbar(images[-1], ax = axes[0, -1], label = "log10 final Expl (median)")
    save_figure(fig, dest, "theory_adversarial_pool")
    return ["theory_adversarial_pool"]
