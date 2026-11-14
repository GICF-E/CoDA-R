"""尾部回升诊断：熵水平与学习率退火能否解释单网络 latest 在训练后期的退化。

R-NaD-PPO（无池）与 CoDA-R 沿用 tune-a / tune-b 的选择，每行只改熵日程或学习率日程；预算与 seed 同消融。
熵变体用恒定熵（ent_start = ent_final），取值覆盖文献中 PG 方法的较优区间 0.05-0.2。
CoDA-R 的 exploiter 熵终值固定为调参选出的值，oracle 强度因此不随 learner 的熵变化。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiment.common.runner import CellSpec, RunOptions, prepare_output, run_cells
from experiment.common.stats import bootstrap_iqm
from experiment.experiments.cells import run_train_cell, train_spec
from experiment.experiments.methods import build_config, deployment, title
from experiment.experiments.protocol import ABLATION_BUDGET, ABLATION_GAMES, SMALL_SEEDS, SMOKE_SEEDS
from experiment.experiments.report import CellResult, Pair, degradation, diagnostics, group, load_results, write_grid_report
from experiment.experiments.tuning import load_selected

DIAG_METHODS = ("rnad", "coda_r")
BASE_VARIANT = "base"
VARIANTS: dict[str, dict[str, Any]] = {
    BASE_VARIANT: {},
    "ent 0.05": dict(ent_start = 0.05, ent_final = 0.05),
    "ent 0.1": dict(ent_start = 0.1, ent_final = 0.1),
    "lr anneal": dict(lr_final_frac = 0.0),
    "ent 0.05 + lr anneal": dict(ent_start = 0.05, ent_final = 0.05, lr_final_frac = 0.0),
    "ent 0.1 + lr anneal": dict(ent_start = 0.1, ent_final = 0.1, lr_final_frac = 0.0),
}
NOTES = [
    f"预算 {ABLATION_BUDGET}，seed {list(SMALL_SEEDS)}；配置沿用调参选择，每行只改熵日程或学习率日程。",
    "ent x：熵系数全程恒定为 x；lr anneal：学习率在整个预算上线性降到 0。",
    "CoDA-R 的 exploiter 熵终值固定为调参选出的值。退化表中的最低点取自全部评估点，偏乐观。",
]


def variant_label(method: str, variant: str) -> str:
    return f"{title(method)} | {variant}"


def specs(out_root: Path, smoke: bool) -> list[CellSpec]:
    selected = load_selected(out_root, smoke)
    seeds = SMOKE_SEEDS if smoke else SMALL_SEEDS
    out = []
    for env in ABLATION_GAMES:
        for method in DIAG_METHODS:
            params = selected[env][method]
            fixed = dict(oracle_ent_final = params["ent_final"]) if method == "coda_r" else {}
            for variant, overrides in VARIANTS.items():
                for seed in seeds:
                    cfg = build_config(method, env, seed, ABLATION_BUDGET, params, label = variant_label(method, variant),
                                       **{**fixed, **overrides})
                    out.append(train_spec(cfg, smoke, method = method, variant = variant, deployment = deployment(method)))
    return out


def _pairs() -> list[Pair]:
    pairs = [Pair(variant_label(m, v), variant_label(m, BASE_VARIANT)) for m in DIAG_METHODS for v in VARIANTS
             if v != BASE_VARIANT]
    return pairs + [Pair(variant_label("coda_r", v), variant_label("rnad", v)) for v in VARIANTS]


def _degradation_lines(results: list[CellResult]) -> list[str]:
    lines = ["## 末段相对全程最低点的退化（seed IQM）", "",
             "| Environment | Method | n | best | best 出现位置 (x B) | tail_mean | tail - best | 末段熵 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (env, label), cells in sorted(group(results).items()):
        rows = [degradation(c) for c in cells]
        entropy = [diagnostics(c)["tail_entropy"] for c in cells]
        stats = [bootstrap_iqm([row[k] for row in rows])[0] for k in ("best", "best_at", "tail_mean", "gap")]
        lines.append(f"| {env} | {label} | {len(cells)} | " + " | ".join(f"{v:.4g}" for v in stats)
                     + f" | {np.nanmean(entropy):.3g} |")
    return lines + [""]


def run_diag_tail(options: RunOptions) -> Path:
    out = prepare_output(options)
    cells = specs(out.parent, options.smoke)
    records = run_cells(cells, run_train_cell, options)
    results = load_results(out, records)
    write_grid_report(out, "diag-tail：尾部回升诊断（熵与学习率日程）", NOTES, results, len(cells), _pairs(),
                      _degradation_lines(results))
    return out


def planned_cells() -> int:
    return len(ABLATION_GAMES) * len(DIAG_METHODS) * len(VARIANTS) * len(SMALL_SEEDS)
