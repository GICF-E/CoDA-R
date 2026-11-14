"""神经实验的统一报告：末段指标、配对差、诊断量、oracle 下界的校验与曲线。

指标都按预算轴上的前值保持阶梯函数计算（见 common/stats.tail_metrics），末段为 [anneal_frac * B, B]。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiment.common.io_util import load_jsonl, write_json, write_summary
from experiment.common.plotting import save_figure, subplots
from experiment.common.runner import done_records
from experiment.common.stats import METRIC_KEYS, bootstrap_iqm, bootstrap_mean, paired, tail_metrics
from experiment.neural.config import RunConfig, config_from_dict

HOLE_LOSS = 0.52
HOLE_WEIGHT = 0.01


@dataclass
class CellResult:
    cell_id: str
    cfg: RunConfig
    meta: dict[str, Any]
    summary: dict[str, Any]
    history: list[dict[str, Any]]
    oracles: list[dict[str, Any]]

    @property
    def key(self) -> tuple[str, str]:
        return self.cfg.env, self.cfg.label

    def series(self, key: str) -> list[tuple[float, float]]:
        return [(float(h["games"]), float(h[key])) for h in self.history
                if key in h and h[key] is not None and np.isfinite(float(h[key]))]

    def metrics(self, key: str = "exploitability") -> dict[str, float]:
        cfg = self.cfg
        return tail_metrics(self.series(key), cfg.budget, cfg.ent_anneal_frac * cfg.budget)


def load_results(out: Path, records: dict[str, dict[str, Any]]) -> list[CellResult]:
    results = []
    for r in done_records(records):
        payload = r["payload"]
        meta = {k: v for k, v in payload.items() if k != "config"}
        cell = out / r["cell_id"]
        results.append(CellResult(r["cell_id"], config_from_dict(payload["config"]), meta, r.get("summary") or {},
                                  load_jsonl(cell / "rollouts.jsonl"), load_jsonl(cell / "oracles.jsonl")))
    return results


def group(results: list[CellResult], key: Callable[[CellResult], Any] = lambda r: r.key) -> dict[Any, list[CellResult]]:
    out: dict[Any, list[CellResult]] = {}
    for r in results:
        out.setdefault(key(r), []).append(r)
    return out


# -------- 诊断 --------
def _mean_of(history: list[dict[str, Any]], key: str) -> float:
    values = [float(h[key]) for h in history if key in h and h.get("roll", -1) >= 0]
    return float(np.mean(values)) if values else float("nan")


def diagnostics(r: CellResult) -> dict[str, float]:
    breakdown = r.summary.get("breakdown", {})
    games = float(r.summary.get("games", 0) or 0)
    tail = [h for h in r.history if h.get("games", 0) >= r.cfg.ent_anneal_frac * r.cfg.budget and "entropy" in h]
    holes = missed = 0
    for h in r.history:
        truth, field = h.get("pool_losses_true"), h.get("pool_field")
        if truth and field and len(truth) == len(field):
            for loss, weight in zip(truth, field):
                if loss > HOLE_LOSS:
                    holes += 1
                    missed += int(weight < HOLE_WEIGHT)
    oracle_games = float(breakdown.get("oracle_train", 0) + breakdown.get("oracle_match", 0))
    return dict(
        pool_mass = _mean_of(r.history, "pool_mass"),
        pool_sample_share = _mean_of(r.history, "pool_sample_share"),
        no_gradient_share = _mean_of(r.history, "no_gradient_share"),
        ess_fraction = _mean_of(r.history, "ess_fraction"),
        tail_entropy = float(np.mean([h["entropy"] for h in tail])) if tail else float("nan"),
        oracle_share = oracle_games / games if games > 0 else float("nan"),
        learner_share = float(r.summary.get("learner_games", 0)) / games if games > 0 else float("nan"),
        holes = float(holes), holes_missed = float(missed),
    )


def degradation(r: CellResult) -> dict[str, float]:
    """全程最低 exploitability、它出现的预算比例，以及末段均值高出最低点多少；最低点取自全部评估点，因而偏乐观。"""
    rows = r.series("exploitability")
    if not rows:
        nan = float("nan")
        return dict(best = nan, best_at = nan, tail_mean = nan, gap = nan)
    games, values = np.asarray(rows, dtype = np.float64).T
    i = int(np.argmin(values))
    tail = r.metrics()["tail_mean"]
    return dict(best = float(values[i]), best_at = float(games[i]) / r.cfg.budget, tail_mean = tail,
                gap = tail - float(values[i]))


def detection_latencies(r: CellResult) -> list[float]:
    """成员真实败率越过 HOLE_LOSS 后，训练系数要过多少个记录点才到 HOLE_WEIGHT；中途漏洞自行消失的不计。"""
    rows = [h for h in r.history if h.get("pool_losses_true") and h.get("pool_ids")]
    latencies: list[float] = []
    open_since: dict[str, int] = {}
    for h in rows:
        for member, loss, weight in zip(h["pool_ids"], h["pool_losses_true"], h["pool_field"]):
            if loss > HOLE_LOSS and member not in open_since:
                open_since[member] = h["roll"]
            if member in open_since and weight >= HOLE_WEIGHT:
                latencies.append(float(h["roll"] - open_since.pop(member)))
            elif member in open_since and loss <= 0.5:
                open_since.pop(member)
    return latencies


def oracle_bound_check(results: list[CellResult]) -> dict[str, Any]:
    """入池时 exploiter 领先给出的 95% 下界不应超过同一时刻的真实 exploitability。"""
    rows = [o for r in results for o in r.oracles if "exploitability" in o]
    above = sum(o["lower_bound"] > o["exploitability"] + 1e-12 for o in rows)
    informative = sum(o["lower_bound"] > 0 for o in rows)
    return dict(insertions = len(rows), bound_above_truth = int(above), informative = int(informative))


# -------- 表格 --------
def _fmt(ci: tuple[float, float, float] | list[float]) -> str:
    return f"{ci[0]:.4g} [{ci[1]:.4g}, {ci[2]:.4g}]" if np.isfinite(ci[0]) else "nan"


def metric_table(groups: dict[Any, list[CellResult]], key: str = "exploitability") -> list[dict[str, Any]]:
    rows = []
    for (env, label), cells in sorted(groups.items()):
        per_seed = {c.cfg.seed: c.metrics(key) for c in cells}
        row: dict[str, Any] = dict(environment = env, method = label, n = len(cells), seeds = sorted(per_seed),
                                   deployment = cells[0].meta.get("deployment", ""))
        for metric in METRIC_KEYS:
            values = [per_seed[s][metric] for s in sorted(per_seed)]
            row[metric] = dict(values = values, ci = bootstrap_iqm(values))
        diag = [diagnostics(c) for c in cells]
        row["diagnostics"] = {k: float(np.nanmean([d[k] for d in diag])) if diag else float("nan") for k in diag[0]}
        rows.append(row)
    return rows


@dataclass(frozen = True)
class Pair:
    """预先确定的配对比较；两侧可以取不同的指标（例如 CoDA-R 的时间平均 Q 对 PSRO 的混合）。"""
    treatment: str
    reference: str
    treatment_key: str = "exploitability"
    reference_key: str = "exploitability"

    def describe(self) -> tuple[str, str]:
        def side(label: str, key: str) -> str:
            return label if key == "exploitability" else f"{label} [{key}]"
        return side(self.treatment, self.treatment_key), side(self.reference, self.reference_key)


def paired_table(groups: dict[Any, list[CellResult]], pairs: list[Pair]) -> list[dict[str, Any]]:
    rows = []
    for env in sorted({env for env, _label in groups}):
        for pair in pairs:
            a, b = groups.get((env, pair.treatment)), groups.get((env, pair.reference))
            if not a or not b:
                continue
            ma = {c.cfg.seed: c.metrics(pair.treatment_key) for c in a}
            mb = {c.cfg.seed: c.metrics(pair.reference_key) for c in b}
            seeds = sorted(set(ma) & set(mb))
            treat, ref = pair.describe()
            for metric in ("tail_mean", "tail_worst", "at_budget"):
                stats = paired([ma[s][metric] for s in seeds], [mb[s][metric] for s in seeds])
                rows.append(dict(environment = env, treatment = treat, reference = ref, metric = metric, **stats))
    return rows


def _table_lines(rows: list[dict[str, Any]], title: str) -> list[str]:
    lines = [f"## {title}", "", "| Environment | Method | Deploy | n | tail_mean | tail_worst | at_budget | auc |",
             "|---|---|---|---:|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['environment']} | {row['method']} | {row['deployment']} | {row['n']} | "
                     + " | ".join(_fmt(row[m]["ci"]) for m in ("tail_mean", "tail_worst", "at_budget", "auc")) + " |")
    lines += ["", "诊断（逐格均值再平均）：", "", "| Environment | Method | pool_mass | 池采样份额 | 无梯度份额 | ESS | 末段熵 | oracle 占比 | learner 占比 | 漏洞 (漏检) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        d = row["diagnostics"]
        lines.append(f"| {row['environment']} | {row['method']} | {d['pool_mass']:.3g} | {d['pool_sample_share']:.3g} | "
                     f"{d['no_gradient_share']:.3g} | {d['ess_fraction']:.3g} | {d['tail_entropy']:.3g} | {d['oracle_share']:.3g} | "
                     f"{d['learner_share']:.3g} | {d['holes']:.0f} ({d['holes_missed']:.0f}) |")
    return lines + [""]


def _paired_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = ["## 逐 seed 配对差（前者 - 后者，越负越好）", "", "| Environment | Pair | Metric | n | IQM [95% CI] | wins / losses |",
             "|---|---|---|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['environment']} | {row['treatment']} - {row['reference']} | {row['metric']} | {row['n']} | "
                     f"{_fmt(row['ci'])} | {row['wins']} / {row['losses']} |")
    return lines + [""]


# -------- 曲线 --------
def plot_curves(out: Path, groups: dict[Any, list[CellResult]], key: str, ylabel: str, log_y: bool = False) -> None:
    for env in sorted({env for env, _label in groups}):
        fig, ax = subplots(figsize = (8, 4.5))
        if ax is None:
            return
        budget = 0.0
        for (e, label), cells in sorted(groups.items()):
            series = [c.series(key) for c in cells if c.series(key)]
            if e != env or not series:
                continue
            budget = max(budget, float(cells[0].cfg.budget))
            hi = min(s[-1][0] for s in series)
            xs = np.geomspace(max(hi / 1000.0, 1.0), hi, 120)
            ys = np.array([[next((v for g, v in reversed(s) if g <= x), np.nan) for x in xs] for s in series])
            stats = np.array([bootstrap_mean(ys[:, i], seed = i) for i in range(xs.size)])
            ax.step(xs, stats[:, 0], where = "post", label = label)
            ax.fill_between(xs, stats[:, 1], stats[:, 2], step = "post", alpha = 0.12)
        if not ax.lines:
            continue
        ax.axvspan(0.8 * budget, budget, color = "0.85", alpha = 0.5)
        ax.set(xscale = "log", yscale = "log" if log_y else "linear", xlabel = "algorithmic games", ylabel = ylabel, title = env)
        ax.legend(fontsize = 6)
        save_figure(fig, out, f"{env}_{key}")


def write_grid_report(out: Path, title: str, notes: list[str], results: list[CellResult], expected: int,
                      pairs: list[Pair] | None = None, extra_lines: list[str] | None = None) -> dict[str, Any]:
    groups = group(results)
    table = metric_table(groups)
    single = [r for r in table if r["deployment"] != "mixture"]
    mixture = [r for r in table if r["deployment"] == "mixture"]
    mix_groups = {k: v for k, v in groups.items() if any(c.series("mix_average_expl") for c in v)}
    mix_tables = {name: metric_table(mix_groups, name) for name in ("mix_current_expl", "mix_average_expl")} if mix_groups else {}
    pair_rows = paired_table(groups, pairs or [])
    bound = oracle_bound_check(results)
    payload = dict(title = title, expected_cells = expected, completed_cells = len(results), table = table,
                   mixture_outputs = mix_tables, paired = pair_rows, oracle_lower_bound = bound)
    write_json(out / "records.json", payload)
    lines = [f"# {title}", "", *notes, "", f"完成 {len(results)}/{expected} 格。", ""]
    if single:
        lines += _table_lines(single, "单网络部署")
    if mixture:
        lines += _table_lines(mixture, "混合部署")
    for name, rows in mix_tables.items():
        label = "CoDA-R 当前混合 q" if name == "mix_current_expl" else "CoDA-R 时间平均 Q"
        lines += _table_lines(rows, label)
    lines += _paired_lines(pair_rows)
    lines += ["## oracle 下界校验", "", f"入池 {bound['insertions']} 次；95% 下界高于真实 exploitability 的次数 "
              f"{bound['bound_above_truth']}（期望约 5% 以内）；下界为正（给出非平凡证据）的次数 {bound['informative']}。", ""]
    lines += extra_lines or []
    write_summary(out, "\n".join(lines))
    plot_curves(out, groups, "exploitability", "exploitability (seed mean + 95% CI)", log_y = True)
    plot_curves(out, groups, "entropy", "policy entropy on the training field")
    return payload
