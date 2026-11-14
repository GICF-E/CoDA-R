"""CoDA-R 消融 A1–A9：基准为 tune-b 选出的 CoDA-R，每次只改一个设计，预算 1e6、6 个 seed、熵按自身预算退火。

所有格子每 4 个 rollout 记录一次成员的精确败率，用来统计漏洞的发现延迟与漏检（A3、A4 的主要指标）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiment.common.runner import CellSpec, RunOptions, prepare_output, run_cells
from experiment.experiments.cells import run_train_cell, train_spec
from experiment.experiments.methods import build_config
from experiment.experiments.protocol import ABLATION_BUDGET, ABLATION_GAMES, SMALL_SEEDS, SMOKE_SEEDS
from experiment.experiments.report import CellResult, Pair, detection_latencies, group, load_results, write_grid_report
from experiment.experiments.tuning import load_selected

BASE_LABEL = "CoDA-R (base)"
VARIANTS: dict[str, dict[str, Any]] = {
    "A1 moving dual anchor": dict(rule = "moving_anchor"),
    "A2 sampling base = field": dict(sampling_base = "field"),
    "A2 sampling base = half": dict(sampling_base = "half"),
    "A3 detection = none": dict(detection = "none"),
    "A3 detection = outer sweep": dict(detection = "sweep"),
    "A4 exact member losses": dict(loss_estimator = "exact"),
    "A5 no oracle": dict(oracle = "none"),
    "A5 exact-BR oracle": dict(oracle = "exact_br"),
    "A5 oracle share 6%": dict(oracle_share = 0.06),
    "A5 oracle share 24%": dict(oracle_share = 0.24),
    "A6 mirror refreshed per outer step": dict(mirror_refresh = "outer"),
    "A7 pool coefficients frozen per outer step": dict(freeze_field = True),
    "A8 proximal form = loss KL": dict(proximal = "kl"),
    "A9 pool capacity 4": dict(pool_capacity = 4),
    "A9 pool capacity 16": dict(pool_capacity = 16),
    "A9 snapshots kept 0": dict(snapshots_keep = 0),
    "A9 snapshots kept 4": dict(snapshots_keep = 4),
}
NOTES = [
    f"基准为调参选出的 CoDA-R；预算 {ABLATION_BUDGET}，seed {list(SMALL_SEEDS)}；每行只改一个设计。",
    "A4 的精确败率只替换对偶侧估计（e_k = 0），是上限对照，不是可部署方案。",
    "A5 的 exact-BR oracle 不消耗环境对局，是 oracle 强度上限。",
]


def specs(out_root: Path, smoke: bool) -> list[CellSpec]:
    selected = load_selected(out_root, smoke)
    seeds = SMOKE_SEEDS if smoke else SMALL_SEEDS
    out = []
    for env in ABLATION_GAMES:
        params = selected[env]["coda_r"]
        for label, overrides in {BASE_LABEL: {}, **VARIANTS}.items():
            for seed in seeds:
                cfg = build_config("coda_r", env, seed, ABLATION_BUDGET, params, label = label,
                                   **{"exact_loss_every": 4, **overrides})
                out.append(train_spec(cfg, smoke, method = "coda_r", ablation = label, deployment = "single"))
    return out


def _latency_lines(results: list[CellResult]) -> list[str]:
    lines = ["## 漏洞发现延迟（每 4 个 rollout 记录一次真实败率）", "",
             "真实败率越过 0.52 到训练系数达到 0.01 之间的 rollout 数。", "",
             "| Environment | Variant | 事件数 | 中位数 | p90 |", "|---|---|---:|---:|---:|"]
    for (env, label), cells in sorted(group(results).items()):
        values = [v for c in cells for v in detection_latencies(c)]
        if values:
            lines.append(f"| {env} | {label} | {len(values)} | {np.median(values):.3g} | {np.quantile(values, 0.9):.3g} |")
    return lines + [""]


def run_ablations(options: RunOptions) -> Path:
    out = prepare_output(options)
    cells = specs(out.parent, options.smoke)
    records = run_cells(cells, run_train_cell, options)
    results = load_results(out, records)
    pairs = [Pair(label, BASE_LABEL) for label in VARIANTS]
    write_grid_report(out, "ablations：CoDA-R 消融", NOTES, results, len(cells), pairs, _latency_lines(results))
    return out


def planned_cells() -> int:
    return len(ABLATION_GAMES) * (len(VARIANTS) + 1) * len(SMALL_SEEDS)
