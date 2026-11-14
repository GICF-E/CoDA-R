"""分析表（analysis.md）：末段退化、调参选择是否落在网格边界、规模实验与主实验的逐 seed 配对差。"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from analysis.data import Outputs
from analysis.figures.neural import MAIN_PAIRS, degradation_rows, pair_label, paired_tail
from analysis.figures.scale import seed_tail
from analysis.figures.tuning import edge_flags
from analysis.style import MAIN_ENVS, SCALE_ENVS, env_title, method_style
from experiment.common.stats import paired
from experiment.experiments.methods import params_label, title
from experiment.experiments.scale import SCALE_METHODS


def _fmt_ci(ci: tuple[float, float, float] | list[float]) -> str:
    return f"{ci[0]:.4g} [{ci[1]:.4g}, {ci[2]:.4g}]" if np.isfinite(ci[0]) else "nan"


def degradation_section(out: Outputs) -> list[str]:
    rows = degradation_rows(out)
    if not rows:
        return []
    lines = ["## 主实验：末段相对全程最低点的退化", "",
             "seed 中位数。best 取自全部评估点（偏乐观），best_at 为它出现的预算比例；gap = tail_mean - best。", "",
             "| Environment | Method | n | best | best_at (x B) | tail_mean | gap | 末段熵 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['env']} | {method_style(r['method']).label} | {r['n']} | {r['best']:.4g} | {r['best_at']:.3f} | "
              f"{r['tail_mean']:.4g} | {r['gap']:.4g} | {r['tail_entropy']:.3g} |" for r in rows]
    return lines + [""]


def grid_edge_section(out: Outputs) -> list[str]:
    lines = ["## 调参：选定配置是否落在网格边界", "",
             "min / max 表示该维度的选定值是网格的最小 / 最大值，真实最优可能在网格之外。", "",
             "| Stage | Environment | Method | Selected | 边界维度 |", "|---|---|---|---|---|"]
    found = False
    for stage in ("a", "b"):
        for env, by_method in sorted(out.selected(stage).items()):
            for method, params in sorted(by_method.items()):
                found = True
                edges = [f"{k}={v}" for k, v in edge_flags(method, env, params).items() if v != "inner"]
                lines.append(f"| {stage} | {env} | {title(method)} | {params_label(params)} | {', '.join(edges) or '-'} |")
    return lines + [""] if found else []


def scale_pairs_section(out: Outputs) -> list[str]:
    by_cell = out.scale_rows()
    if not by_cell:
        return []
    lines = ["## 规模实验：CoDA-R 与各方法的逐 seed 配对差（精确 tail_mean，越负越好）", "",
             "| Environment | Pair | n | IQM [95% CI] | wins / losses |", "|---|---|---:|---|---|"]
    for env in SCALE_ENVS:
        mine = seed_tail(by_cell.get((env, "coda_r"), []))
        for method in SCALE_METHODS:
            other = seed_tail(by_cell.get((env, method), []))
            seeds = sorted(set(mine) & set(other))
            if method == "coda_r" or not seeds:
                continue
            stats = paired([mine[s] for s in seeds], [other[s] for s in seeds])
            lines.append(f"| {env_title(env)} | {pair_label('coda_r', method)} | {stats['n']} | {_fmt_ci(stats['ci'])} | "
                         f"{stats['wins']} / {stats['losses']} |")
    return lines + [""]


def main_pairs_section(out: Outputs) -> list[str]:
    groups = out.by_env_method("main")
    if not groups:
        return []
    lines = ["## 主实验：预先确定的配对（含 Q 对 NFSP），tail_mean，越负越好", "",
             "| Environment | Pair | n | IQM [95% CI] | wins / losses |", "|---|---|---:|---|---|"]
    for env in MAIN_ENVS:
        for a, b in MAIN_PAIRS:
            stats = paired_tail(groups, env, a, b)
            if stats is not None:
                lines.append(f"| {env_title(env)} | {pair_label(a, b)} | {stats['n']} | {_fmt_ci(stats['ci'])} | "
                             f"{stats['wins']} / {stats['losses']} |")
    return lines + [""]


def write_tables(out: Outputs, dest: Path) -> Path:
    dest.mkdir(parents = True, exist_ok = True)
    lines = ["# CoDA-R 实验分析表", "", f"数据根目录：`{out.root}`", ""]
    # 只有一个 seed 的组合会让中位数与区间对空切片告警，结果按 NaN 记入表中
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for section in (degradation_section, main_pairs_section, scale_pairs_section, grid_edge_section):
            lines += section(out)
    path = dest / "analysis.md"
    path.write_text("\n".join(lines), encoding = "utf-8")
    return path
