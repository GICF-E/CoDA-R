"""有限维理论实验 T1–T12：网格、单格运行与报告。

正确算法（固定锚、份额 1、隐式镜像、配对池系数）上的逐步检查计入 violations，应当全部为 0；
故意破坏某个设计的对照计入 diagnostic_violations，用来确认检查能分辨错误算法。
T10–T12 比较池规则，轨迹由 theory/rules.py 的共用驱动给出。
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from experiment.common.io_util import load_json, write_json, write_summary
from experiment.common.plotting import save_figure, subplots
from experiment.common.runner import CellSpec, RunOptions, done_records, prepare_output, run_cells
from experiment.common.stats import bootstrap_mean, wilson
from experiment.core.simplex import simplex_proj
from experiment.theory.games import best_response, exploitability, make_game, rps
from experiment.theory.rules import ADVERSARIES, ORACLE_ARMS, RULES, SERIES_EVERY, adversarial_run, oracle_rate_run, sleeper_run
from experiment.theory.solver import certificate, dual_weights, fixed_anchor, solve_step
from experiment.theory.stochastic import StochasticSetup, run_stochastic

TITLES = {
    "theory-energy": "T1 能量式与累计预算",
    "theory-pool-changes": "T2 入池与淘汰下的能量式",
    "theory-certificates": "T3 latest / 池 / q / Q 证书",
    "theory-convergence": "T4 收敛、best-iterate 与 last-iterate 速率、残差单调",
    "theory-inexact": "T5 带误差的能量式与证书",
    "theory-necessity": "T6 必要性对照：镜像、自博弈份额、池系数配对",
    "theory-anchor-witness": "T7 对偶锚 x 自博弈份额 2x2 的构造不动点",
    "theory-anchor-basin": "T8 移动锚从随机初始点出发卡在非纳什点的比例",
    "theory-sampling": "T9 带采样的有限维实验：采样基比较",
    "theory-pool-rules": "T10 池规则对照：打平成员的权重与 sleeper 延迟",
    "theory-oracle-rate": "T11 oracle 的定量收益：每步加入锚的 BR 时的收敛速率（定理 9）",
    "theory-adversarial-pool": "T12 对抗池：对手挑选池成员时各规则的最终 exploitability",
}
STUCK_EXPL = 1e-3
STUCK_DX = 1e-8
# T11 的 eta 网格：大 eta 对应 PPO 每个外环只走一小步
ORACLE_ETAS = (1.0, 4.0, 16.0, 64.0)
ORACLE_EPS = 0.01


# -------- 网格 --------
def _base_grid(smoke: bool) -> list[tuple[int, int, float, float]]:
    if smoke:
        return [(6, 4, 6.0, 0.1)]
    return [(m, n, c, eta) for (m, n), (c, eta) in product(product((6, 16, 64), (0, 4, 16)), product((2.0, 6.0), (0.1, 1.0)))]


def _payloads(exp_id: str, smoke: bool) -> list[dict[str, Any]]:
    steps = 8 if smoke else 1024
    seeds = list(range(2 if smoke else 5))
    if exp_id in ("theory-energy", "theory-pool-changes", "theory-certificates", "theory-convergence", "theory-inexact"):
        errors = list(product((0.0, 0.01, 0.05, 0.1), repeat = 2)) if exp_id == "theory-inexact" else [(0.0, 0.0)]
        if smoke and exp_id == "theory-inexact":
            errors = [(0.05, 0.05)]
        return [dict(exp = exp_id, family = "random", size = m, n = n, c = c, eta = eta, e = e, delta = d, seeds = seeds,
                     steps = steps, mode = "default", anchor = "fixed")
                for (m, n, c, eta), (e, d) in product(_base_grid(smoke), errors)]
    if exp_id == "theory-necessity":
        modes = ("default", "explicit", "share0", "share_live", "share_rule_b", "frozen")
        return [dict(exp = exp_id, family = "random", size = 6, n = 4, c = 6.0, eta = 0.1, e = 0.0, delta = 0.0,
                     seeds = list(range(2 if smoke else 10)), steps = steps, mode = mode, anchor = "fixed") for mode in modes]
    if exp_id == "theory-anchor-witness":
        return [dict(exp = exp_id, family = "rps", size = 3, n = 1, c = 6.0, eta = 1.0, e = 0.0, delta = 0.0,
                     seeds = list(range(2 if smoke else 10)), steps = steps, anchor = anchor,
                     mode = "default" if share == "one" else "share_live", witness = share)
                for anchor, share in product(("fixed", "moving"), ("one", "wm"))]
    if exp_id == "theory-anchor-basin":
        sizes = [(f, m) for f in ("random", "cyclic", "spinning") for m in (3, 4, 5, 6)] + [("blotto", 2), ("blotto", 3)]
        if smoke:
            sizes = sizes[:1]
        return [dict(exp = exp_id, family = f, size = m, n = n, c = 6.0, eta = eta, e = e, delta = 0.0, oracle = oracle,
                     anchor = anchor, mode = "default", seeds = list(range(2 if smoke else 200)), steps = 8 if smoke else 2048)
                for (f, m), n, eta, oracle, anchor, e in product(sizes, (1, 4), (0.1, 1.0), (False, True),
                                                                ("fixed", "moving"), (0.0, 0.05))]
    if exp_id == "theory-sampling":
        return [dict(exp = exp_id, m = m, c = c, games = g, base = base, seeds = list(range(2 if smoke else 20)), smoke = smoke)
                for m, c, g, base in product((6,) if smoke else (6, 16), (6.0,) if smoke else (2.0, 6.0, 20.0),
                                             (64,) if smoke else (64, 512), ("field", "dual", "half"))]
    if exp_id == "theory-pool-rules":
        return [dict(exp = exp_id, rule = rule, m = m, n = 4, c = 6.0, eta = 0.1, seeds = list(range(2 if smoke else 20)),
                     steps = 16 if smoke else 1024, reset_at = 8 if smoke else 512, insert_every = 4 if smoke else 64)
                for rule, m in product(RULES, (6,) if smoke else (6, 16))]
    if exp_id == "theory-oracle-rate":
        # cyclic(15) 的纳什在内点（纯旋转），random 与 spinning 的纳什在单纯形边界上
        games = [("random", 6)] if smoke else [("random", 16), ("spinning", 15), ("cyclic", 15)]
        return [dict(exp = exp_id, family = f, size = m, eta = eta, arm = arm, eps = ORACLE_EPS,
                     seeds = list(range(2 if smoke else 10)), steps = 16 if smoke else 4096)
                for (f, m), eta, arm in product(games, ORACLE_ETAS[:2] if smoke else ORACLE_ETAS, ORACLE_ARMS)]
    if exp_id == "theory-adversarial-pool":
        return [dict(exp = exp_id, family = f, m = m, rule = rule, adversary = adv, c = 6.0, eta = 0.1,
                     seeds = list(range(2 if smoke else 10)), steps = 16 if smoke else 1024)
                for f, m, rule, adv in product(("random",) if smoke else ("random", "cyclic"), (6,) if smoke else (6, 16),
                                               RULES, ADVERSARIES)]
    raise ValueError(exp_id)


def _cell_name(p: dict[str, Any]) -> str:
    keys = ("family", "rule", "arm", "adversary", "base", "size", "m", "n", "c", "eta", "e", "delta", "mode", "anchor",
            "oracle", "games", "witness")
    return "_".join(f"{k}{p[k]}" for k in keys if k in p).replace(".", "p")


def cells(exp_id: str, smoke: bool) -> list[CellSpec]:
    return [CellSpec(_cell_name(p), p) for p in _payloads(exp_id, smoke)]


# -------- 精确算法轨迹（T1–T8）--------
def _correct(p: dict[str, Any]) -> bool:
    return p["mode"] == "default" and p["anchor"] == "fixed"


def _initial_state(p: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.random.Generator]:
    """返回 (S, 初始池, 纳什, x0, rng)；T7 用构造的见证点，T6 把当前 BR 放进池以制造非零池场。"""
    if p["family"] == "rps":
        rng = np.random.default_rng(seed)
        pool = np.array([[0.0], [0.5], [0.5]])
        x0 = np.array([2 / 3, 1 / 6, 1 / 6]) if p["witness"] == "one" else np.array([1.0, 0.0, 0.0])
        if seed > 0:
            x0 = simplex_proj(x0 + 1e-3 * rng.normal(size = 3))
        return rps(), pool, np.ones(3) / 3, x0, rng
    game = make_game(p["family"], p["size"], p["n"], seed)
    x0 = game.rng.dirichlet(np.ones(game.m))
    pool = game.pool.copy()
    if p["exp"] == "theory-necessity":
        pool[:, 0] = best_response(game.s, x0)
    if p["mode"] == "share0":
        # 份额 0 的停滞见证：从一个已知非纳什的纯策略出发
        x0 = np.eye(game.m)[int(np.argmax(np.max(game.s, axis = 1)))]
    return game.s, pool, game.nash, x0, game.rng


def _initial_anchor(p: dict[str, Any], n: int) -> np.ndarray:
    if p["family"] == "rps" and p["anchor"] == "moving":
        return np.array([1.0, 0.0]) if p["witness"] == "one" else np.array([2 / 3, 1 / 3])
    return fixed_anchor(n)


def exact_run(p: dict[str, Any], seed: int) -> dict[str, Any]:
    s, pool, nash, x, rng = _initial_state(p, seed)
    c, eta, e, delta = p["c"], p["eta"], p["e"], p["delta"]
    mode, moving = p["mode"], p["anchor"] == "moving"
    exact = e == 0.0 and delta == 0.0
    pool_changes = p["exp"] == "theory-pool-changes" or bool(p.get("oracle"))
    anchor = _initial_anchor(p, pool.shape[1])
    share = 0.0 if mode == "share0" else 1.0
    previous_wm = 1.0
    initial_distance = float(np.sum((x - nash) ** 2))
    sums = dict(dx = 0.0, dual = 0.0, error = 0.0, mass = 0.0)
    sum_v = np.zeros_like(x)
    best = previous_dx = float("inf")
    # 当前迭代点相对本步精确解的误差界；x0 是给定的起点，没有求解误差
    previous_error = 0.0
    violations: dict[str, int] = {}
    series: dict[str, list[float]] = dict(full_expl = [], pool_mass = [], Q_expl = [])
    tail: list[float] = []
    rec: dict[str, Any] = {}
    steps = p["steps"]
    for k in range(1, steps + 1):
        if pool_changes and k > 1 and (k - 1) % 64 == 0:
            pool = np.column_stack((pool, best_response(s, x)))
            anchor = np.insert(anchor, -1, 0.0) if moving else fixed_anchor(pool.shape[1])
        if p["exp"] == "theory-pool-changes" and k > 1 and (k - 1) % 128 == 0:
            zeros = np.flatnonzero(dual_weights(s, pool, x, c, fixed_anchor(pool.shape[1]))[0][:-1] == 0)
            if zeros.size:
                pool = np.delete(pool, int(zeros[0]), axis = 1)
                anchor = fixed_anchor(pool.shape[1])
        n = pool.shape[1]
        noise = rng.uniform(-e, e, n + 1) if e > 0 else None
        lam = previous_wm if mode == "share_rule_b" else share
        frozen = dual_weights(s, pool, x, c, anchor)[0] if mode == "frozen" else None
        sol = solve_step(s, pool, x, eta, c, share = lam, anchor = anchor if moving else None, noise = noise,
                         target_delta = delta, explicit_mirror = mode == "explicit", live_share = mode == "share_live",
                         frozen_weights = frozen)
        used_share = float(sol.w[-1]) if mode == "share_live" else lam
        # 本步的 VI 是 eta 强单调的，停机时的 FW 间隙 delta 给出 ||x_{k+1} - 精确解|| <= sqrt(delta / eta)
        solve_error = float(np.sqrt(sol.delta / eta))
        rec = certificate(s, pool, x, sol, nash, eta, c, used_share)
        sums["dx"] += rec["dx"] ** 2
        sums["dual"] += rec["dual_distance_squared"]
        sums["error"] += rec["error"]
        sums["mass"] += rec["t"]
        sum_v += rec["v"]
        q_average = sum_v / sums["mass"] if sums["mass"] > 0 else None
        average_expl = exploitability(s, q_average) if q_average is not None else float("nan")
        best = min(best, rec["full_expl"])
        checks = {key: bool(rec[key]) for key in ("fejer_violation", "full_violation", "pool_violation", "mixture_violation")}
        checks["energy_budget_violation"] = (sums["dx"] + 2 * sums["dual"] / (c * eta)
                                             > initial_distance + 2 * sums["error"] / eta + 1e-9)
        if sums["mass"] > 0:
            checks["average_violation"] = average_expl > (eta + sums["error"]) / sums["mass"] + 1e-9
        if exact and share == 1.0:
            checks["best_rate_violation"] = best > (np.sqrt(c * eta) / 2 + 2 * eta) / np.sqrt(k) + 1e-9
            checks["mass_telescope_violation"] = sums["mass"] < k - 1e-9
            if not pool_changes:
                bound = np.sqrt(c * eta / 2) * k ** -0.25 + 2 * eta * k ** -0.5
                checks["last_rate_violation"] = rec["full_expl"] > bound + 1e-9
                # 精确步是 resolvent，P 与 I - P 都非扩张；迭代点的误差为 r 时只能保证 dx_k <= dx_{k-1} + r_{k+1} + 2 r_k
                checks["residual_monotonicity_violation"] = (rec["dx"] > previous_dx + solve_error + 2.0 * previous_error
                                                             + 1e-9)
        for key, bad in checks.items():
            violations[key] = violations.get(key, 0) + int(bad)
        if k % SERIES_EVERY == 0 or k == steps:
            series["full_expl"].append(rec["full_expl"])
            series["pool_mass"].append(rec["pool_mass"])
            series["Q_expl"].append(average_expl)
        if k > steps * 3 // 4:
            tail.append(rec["full_expl"])
        x = sol.x
        previous_dx = rec["dx"]
        previous_error = solve_error
        previous_wm = float(sol.w[-1])
        if moving:
            anchor = sol.w
    return dict(seed = seed, final_expl = rec["full_expl"], final_pool_mass = rec["pool_mass"], final_dx = rec["dx"],
                tail_mean_expl = float(np.mean(tail)), final_Q_expl = series["Q_expl"][-1],
                stuck = rec["full_expl"] > STUCK_EXPL and rec["dx"] < STUCK_DX, violations = violations, series = series)


# -------- 单格入口 --------
RULE_RUNS = {"theory-pool-rules": sleeper_run, "theory-oracle-rate": oracle_rate_run,
             "theory-adversarial-pool": adversarial_run}


def run_cell(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    exp = payload["exp"]
    runs: list[dict[str, Any]] = []
    for seed in payload["seeds"]:
        if exp == "theory-sampling":
            setup = StochasticSetup(m = payload["m"], c = payload["c"], games = payload["games"], base = payload["base"],
                                    seed = seed, outer_steps = 4 if payload["smoke"] else 128)
            runs.append(dict(seed = seed, **run_stochastic(setup)))
        elif exp in RULE_RUNS:
            runs.append(RULE_RUNS[exp](payload, seed))
        else:
            runs.append(exact_run(payload, seed))
    keep_series = exp not in ("theory-anchor-basin", "theory-inexact")
    if not keep_series:
        for run in runs:
            run.pop("series", None)
    diagnostic = exp in ("theory-necessity", "theory-anchor-witness", "theory-anchor-basin") and not _correct(payload)
    violations: dict[str, int] = {}
    for run in runs:
        for key, value in run.get("violations", {}).items():
            violations[key] = violations.get(key, 0) + value
    summary = dict(runs = runs, violations = violations, diagnostic = diagnostic)
    write_json(cell_dir / "summary.json", summary)
    # cells.jsonl 里只留汇总，逐 seed 记录在格子目录里
    return dict(violations = violations, diagnostic = diagnostic, n_runs = len(runs))


# -------- 报告 --------
def _load_runs(out: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    return load_json(out / record["cell_id"] / "summary.json")["runs"]


def _plot_series(out: Path, groups: dict[str, list[list[float]]], title: str, stem: str) -> None:
    fig, ax = subplots(figsize = (7, 4))
    if ax is None:
        return
    for label, seqs in groups.items():
        length = min(len(seq) for seq in seqs)
        values = np.array([seq[:length] for seq in seqs])
        stats = np.array([bootstrap_mean(values[:, i], seed = i) for i in range(length)])
        xs = np.arange(1, length + 1) * SERIES_EVERY
        ax.plot(xs, np.maximum(stats[:, 0], 1e-14), label = label)
        ax.fill_between(xs, np.maximum(stats[:, 1], 1e-14), np.maximum(stats[:, 2], 1e-14), alpha = 0.15)
    ax.set(yscale = "log", xlabel = "outer step", ylabel = "Expl(x_k), seed mean + 95% CI", title = title)
    ax.legend(fontsize = 7)
    save_figure(fig, out, stem)


def _quantiles(values: list[float]) -> str:
    q = np.quantile(values, [0.5, 0.9, 0.99, 1.0])
    return " / ".join(f"{v:.2e}" for v in q)


def report(exp_id: str, out: Path, records: dict[str, dict[str, Any]]) -> None:
    done = done_records(records)
    totals: dict[str, int] = {}
    diagnostic_totals: dict[str, int] = {}
    for r in done:
        target = diagnostic_totals if r["summary"]["diagnostic"] else totals
        for key, value in r["summary"]["violations"].items():
            target[key] = target.get(key, 0) + value
    lines = [f"# {exp_id}: {TITLES[exp_id]}", "", f"完成 {len(done)}/{len(records)} 格。", "",
             f"正确算法的逐步检查违反次数（应全为 0）：{totals or '无检查项'}", "",
             f"诊断对照的违反次数（故意破坏的算法，非零说明检查有分辨力）：{diagnostic_totals or '无'}", ""]
    rows: list[str] = []
    curves: dict[str, list[list[float]]] = {}
    # T12：每个规则在所有格子里最差的最终 Expl 中位数、全部运行的最大值与最差格子
    worst_case: dict[str, list[Any]] = {}
    for r in done:
        p = r["payload"]
        runs = _load_runs(out, r)
        if exp_id == "theory-anchor-basin":
            stuck = sum(bool(run["stuck"]) for run in runs)
            point, lo, hi = wilson(stuck, len(runs))
            rows.append(f"| {p['family']} | {p['size']} | {p['n']} | {p['eta']} | {p['oracle']} | {p['anchor']} | {p['e']} | "
                        f"{stuck}/{len(runs)} | {point:.3f} [{lo:.3f}, {hi:.3f}] | {_quantiles([run['tail_mean_expl'] for run in runs])} |")
        elif exp_id == "theory-sampling":
            means = {k: np.nanmean([run[k] for run in runs]) for k in
                     ("mean_weighted_error", "mean_ess", "mean_pool_share", "mean_repair_rollouts", "tail_mean_expl")}
            rows.append(f"| {p['m']} | {p['c']} | {p['games']} | {p['base']} | " + " | ".join(f"{v:.4g}" for v in means.values()) + " |")
        elif exp_id == "theory-pool-rules":
            delays = [run["sleeper_delay"] for run in runs]
            censored = sum(bool(run["sleeper_censored"]) for run in runs)
            rows.append(f"| {p['rule']} | {p['m']} | {np.median([run['expl_before_reset'] for run in runs]):.2e} | "
                        f"{np.median([run['final_expl'] for run in runs]):.2e} | "
                        f"{np.mean([run['tied_weight_before_reset'] for run in runs]):.3g} | "
                        f"{np.nanmedian(delays) if np.isfinite(delays).any() else float('nan'):.3g} | {censored} |")
            curves.setdefault(f"{p['rule']} m={p['m']}", []).extend(run["series"]["full_expl"] for run in runs)
        elif exp_id == "theory-oracle-rate":
            reached = [run["steps_to_1e6"] for run in runs if np.isfinite(run["steps_to_1e6"])]
            rows.append(f"| {p['family']} | {p['size']} | {p['eta']:g} | {p['arm']} | {np.median([run['c'] for run in runs]):.3g} | "
                        f"{np.median([run['rate'] for run in runs]):.3e} | {np.median(reached) if reached else float('nan'):.4g} | "
                        f"{len(runs) - len(reached)} | {np.median([run['final_expl'] for run in runs]):.2e} |")
            if p["eta"] == ORACLE_ETAS[-2]:
                curves.setdefault(f"{p['family']} {p['arm']}", []).extend(run["series"]["full_expl"] for run in runs)
        elif exp_id == "theory-adversarial-pool":
            finals = [run["final_expl"] for run in runs]
            rows.append(f"| {p['family']} | {p['m']} | {p['adversary']} | {p['rule']} | {np.median(finals):.2e} | {max(finals):.2e} | "
                        f"{np.median([run['tail_mean_expl'] for run in runs]):.2e} | "
                        f"{np.mean([run['idle_coef'] for run in runs]):.3g} | {max(run['largest_pool'] for run in runs)} |")
            worst = worst_case.setdefault(p["rule"], [0.0, 0.0, ""])
            worst[1] = max(worst[1], max(finals))
            if np.median(finals) > worst[0]:
                worst[0], worst[2] = float(np.median(finals)), f"{p['family']} m={p['m']} {p['adversary']}"
        else:
            stuck = sum(bool(run["stuck"]) for run in runs)
            rows.append(f"| {_cell_name(p)} | {np.median([run['final_expl'] for run in runs]):.2e} | "
                        f"{max(run['final_expl'] for run in runs):.2e} | {np.median([run['final_pool_mass'] for run in runs]):.3g} | "
                        f"{stuck}/{len(runs)} |")
            if "series" in runs[0]:
                label = p["mode"] if exp_id == "theory-necessity" else (
                    f"{p['anchor']} anchor, share {p['witness']}" if exp_id == "theory-anchor-witness" else f"m={p['size']} n={p['n']}")
                curves.setdefault(label, []).extend(run["series"]["full_expl"] for run in runs)
    headers = {
        "theory-anchor-basin": ("| family | size | n | eta | oracle | anchor | e | 卡住 | 比例 [Wilson 95%] | 末段 Expl p50 / p90 / p99 / max |",
                                "|---|---:|---:|---:|---|---|---:|---:|---|---|"),
        "theory-sampling": ("| m | c | G | base | 持权成员估计误差 | ESS 比例 | 池采样份额 | 修复 rollout 数 | 末段 Expl |",
                            "|---:|---:|---:|---|---:|---:|---:|---:|---:|"),
        "theory-pool-rules": ("| rule | m | 重置前 Expl 中位数 | 最终 Expl 中位数 | 打平成员持有的系数 | sleeper 延迟中位数 | 未恢复 |",
                              "|---|---:|---:|---:|---:|---:|---:|"),
        "theory-oracle-rate": ("| family | size | eta | arm | c | ln Expl 斜率中位数 | 到 1e-6 的步数中位数 | 未到 1e-6 | 最终 Expl 中位数 |",
                               "|---|---:|---:|---|---:|---:|---:|---:|---:|"),
        "theory-adversarial-pool": ("| family | m | adversary | rule | 最终 Expl 中位数 | 最终 Expl 最大 | 末段 Expl 中位数 | 不在赢的成员持有的系数 | 最大池规模 |",
                                    "|---|---:|---|---|---:|---:|---:|---:|---:|"),
    }
    header = headers.get(exp_id, ("| cell | 最终 Expl 中位数 | 最终 Expl 最大 | 最终池权重中位数 | 卡住 |", "|---|---:|---:|---:|---:|"))
    lines += [*header, *sorted(rows)]
    if exp_id == "theory-anchor-basin":
        lines += ["", f"卡住 = 末步 Expl > {STUCK_EXPL:g} 且策略移动量 < {STUCK_DX:g}；有误差（e > 0）时看末段 Expl 的分位数。"]
    if exp_id == "theory-oracle-rate":
        lines += ["", f"c 取定理 9 允许的最大值 min(4, 2 eta / ||S||^2)；oracle 每步加入锚的 BR，池里只保留最近 8 个；"
                      f"coda_r_eps 的 oracle 误差 eps = {ORACLE_EPS:g}。斜率是 -ln Expl 对步数的最小二乘斜率（取 1e-10 到 1e-1 之间的点），"
                      "只作描述；主指标是最终 Expl 与到 1e-6 的步数。"]
    if worst_case:
        lines += ["", "## 每个规则的最坏情况", "", "| rule | 最差格子的最终 Expl 中位数 | 全部运行的最大值 | 最差格子 |",
                  "|---|---:|---:|---|"]
        lines += [f"| {rule} | {w[0]:.2e} | {w[1]:.2e} | {w[2]} |" for rule, w in sorted(worst_case.items())]
    write_summary(out, "\n".join(lines) + "\n")
    if curves and exp_id not in ("theory-energy", "theory-pool-changes", "theory-certificates"):
        _plot_series(out, curves, exp_id, "exploitability")


def run(exp_id: str, options: RunOptions) -> Path:
    out = prepare_output(options)
    specs = cells(exp_id, options.smoke)
    records = run_cells(specs, run_cell, options)
    report(exp_id, out, records)
    return out


def planned_cells(exp_id: str, smoke: bool) -> int:
    return len(cells(exp_id, smoke))
