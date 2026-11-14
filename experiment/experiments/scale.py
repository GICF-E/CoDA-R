"""规模实验：3x3 abrupt Dark Hex 与 abrupt Phantom Tic-Tac-Toe（各约 2.7e10 个节点、2e7 量级信息集），全树精确评估。

scale-train：两个游戏上训练各方法（配置沿用 Goofspiel(4) 的调参结果）；OpenSpiel 无法展开全树，训练中只存 checkpoint。
scale-eval：训练结束后用 exp-a-spiel 对选定 checkpoint 做全树精确 exploitability。一个遍历器约 30 GB，
  所以一次只跑一个评估格子、用满全部线程；每个格子是一个 (游戏, 方法)，遍历器只构造一次，依次评估全部 seed。
scale-crossplay：每个游戏内，所有方法、所有 seed 的最终策略循环赛。
scale-history（M4 的规模臂）：每次训练内，均匀取出的 checkpoint 两两对打（采样对局），看 latest 会不会输给自己的历史版本。
"""
from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiment.common.io_util import append_jsonl, load_json, load_jsonl, write_json, write_summary
from experiment.common.plotting import save_figure, subplots
from experiment.common.runner import CellSpec, RunOptions, done_records, prepare_output, require_complete, run_cells
from experiment.common.stats import bootstrap_iqm, history_forgetting
from experiment.experiments.cells import run_train_cell, train_spec
from experiment.experiments.methods import build_config, deployment, title
from experiment.experiments.protocol import MAIN_BUDGET, SCALE_GAMES, SCALE_SEEDS, SCALE_SOURCE_GAME, SMOKE_SEEDS
from experiment.experiments.tuning import load_selected
from experiment.neural.config import RunConfig, config_from_dict
from experiment.neural.envs import load_game, make_scorer, obs_and_action_size
from experiment.neural.exact_large import ExactEvaluator
from experiment.neural.mlp import SmallMLP
from experiment.neural.play import play_match
from experiment.neural.policies import MixtureActor, frozen_actor

SCALE_METHODS = ("selfplay", "rnad", "mmd", "pfsp", "coda_r", "psro")
# 精确评估的 checkpoint：预算这些比例处最近的一个。末段三个点 (0.8B, 0.9B, B) 的平均与最大值给出 tail_mean / tail_worst
EVAL_FRACTIONS = (1 / 64, 1 / 16, 1 / 4, 0.8, 0.9, 1.0)
TAIL_FRACTIONS = (0.8, 0.9, 1.0)
CROSSPLAY_GAMES = 1000
# scale-history：预算上均匀取的 checkpoint 数与每对的局数；1000 局时单对败率的标准误约 0.016
HISTORY_POINTS = 16
HISTORY_GAMES = 1000
HISTORY_MARGIN = 0.05


# -------- checkpoint --------
def _checkpoints(cell_dir: Path) -> list[tuple[int, Path]]:
    paths = sorted((cell_dir / "ckpts").glob("games_*.pt"))
    return [(int(p.stem.split("_")[1]), p) for p in paths if int(p.stem.split("_")[1]) > 0]


def _nearest(checkpoints: list[tuple[int, Path]], target: float) -> tuple[int, Path]:
    return min(checkpoints, key = lambda item: abs(np.log(item[0]) - np.log(target)))


def load_nets(cfg: RunConfig, path: Path) -> tuple[list[SmallMLP], np.ndarray]:
    """checkpoint -> (网络列表, 混合权重)；PSRO 存的是全部成员与元策略，其余方法是单个网络。"""
    obs_dim, n_actions = obs_and_action_size(load_game(cfg.env))

    def net(state: dict[str, torch.Tensor]) -> SmallMLP:
        model = SmallMLP(obs_dim, n_actions, cfg.hidden)
        model.load_state_dict(state)
        return model

    payload = torch.load(path, map_location = "cpu")
    if cfg.algorithm == "psro":
        return [net(s) for s in payload["states"]], np.asarray(payload["weights"], dtype = np.float64)
    return [net(payload)], np.ones(1, dtype = np.float64)


def _actor(nets: list[SmallMLP], weights: np.ndarray) -> Any:
    actors = [frozen_actor(n, "cpu") for n in nets]
    return actors[0] if len(actors) == 1 else MixtureActor(actors, weights)


# -------- scale-train --------
def _train_specs(out_root: Path, smoke: bool) -> list[CellSpec]:
    selected = load_selected(out_root, smoke)[SCALE_SOURCE_GAME]
    seeds = SMOKE_SEEDS if smoke else SCALE_SEEDS
    return [train_spec(build_config(method, env, seed, MAIN_BUDGET, selected[method]), smoke, method = method,
                       deployment = deployment(method))
            for env in SCALE_GAMES for method in SCALE_METHODS for seed in seeds]


def run_scale_train(options: RunOptions) -> Path:
    out = prepare_output(options)
    specs = _train_specs(out.parent, options.smoke)
    records = run_cells(specs, run_train_cell, options)
    lines = ["# scale-train：3x3 abrupt Dark Hex 与 abrupt Phantom Tic-Tac-Toe", "",
             f"配置沿用 {SCALE_SOURCE_GAME} 的调参结果；训练中只存 checkpoint，精确评估见 scale-eval。", "",
             "| Environment | Method | 完成 | 平均耗时 h |", "|---|---|---:|---:|"]
    for env in SCALE_GAMES:
        for method in SCALE_METHODS:
            done = [records[s.cell_id] for s in specs if s.payload["method"] == method
                    and s.payload["config"]["env"] == env and records.get(s.cell_id, {}).get("status") == "done"]
            hours = np.mean([r["elapsed_s"] for r in done]) / 3600 if done else float("nan")
            lines.append(f"| {env} | {title(method)} | {len(done)} | {hours:.2f} |")
    write_summary(out, "\n".join(lines) + "\n")
    return out


def planned_train_cells() -> int:
    return len(SCALE_GAMES) * len(SCALE_METHODS) * len(SCALE_SEEDS)


def _train_records(out_root: Path) -> dict[str, dict[str, Any]]:
    """scale-train 的最新记录（同一 ID 只认最后一行）。"""
    return {r["cell_id"]: r for r in load_jsonl(out_root / "scale-train" / "cells.jsonl")}


# -------- scale-eval --------
def run_exact_eval_cell(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    """一个 (游戏, 方法)：构造一次遍历器，对每个 seed 的选定 checkpoint 做全树精确评估。"""
    source = cell_dir.parent.parent / "scale-train"
    evaluator = ExactEvaluator(payload["env"])
    path_out = cell_dir / "exact.jsonl"
    path_out.unlink(missing_ok = True)
    n_evals = 0
    for train in payload["train"]:
        cfg = config_from_dict(train["config"])
        checkpoints = _checkpoints(source / train["cell_id"])
        targets = {fraction: _nearest(checkpoints, fraction * cfg.budget) for fraction in payload["fractions"]}
        results: dict[Path, dict[str, float]] = {}
        for fraction, (games, path) in targets.items():
            if path not in results:
                t0 = time.time()
                nets, weights = load_nets(cfg, path)
                results[path] = dict(**evaluator.evaluate(evaluator.strategies(nets, weights)), n_members = len(nets),
                                     eval_s = time.time() - t0)
                n_evals += 1
            append_jsonl(path_out, dict(seed = cfg.seed, fraction = fraction, games = games, **results[path]))
    return dict(n_evals = n_evals, n_runs = len(payload["train"]))


def _eval_specs(train_specs: list[CellSpec], smoke: bool) -> list[CellSpec]:
    fractions = (1.0,) if smoke else EVAL_FRACTIONS
    # 冒烟只覆盖单网络与整局混合两条评估路径，每个格子都要构造一次 30 GB 的遍历器
    methods = ("coda_r", "psro") if smoke else SCALE_METHODS
    specs = []
    for env in SCALE_GAMES:
        for method in methods:
            train = [dict(cell_id = s.cell_id, config = s.payload["config"]) for s in train_specs
                     if s.payload["method"] == method and s.payload["config"]["env"] == env]
            specs.append(CellSpec(f"{env}_{method}_exact", dict(env = env, method = method, fractions = list(fractions),
                                                                train = train)))
    return specs


def _eval_rows(out: Path, records: dict[str, dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    return {(r["payload"]["env"], r["payload"]["method"]): load_jsonl(out / r["cell_id"] / "exact.jsonl")
            for r in done_records(records)}


def _seed_metrics(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for seed in sorted({int(r["seed"]) for r in rows}):
        mine = {float(r["fraction"]): float(r["exploitability"]) for r in rows if int(r["seed"]) == seed}
        tail = [mine[f] for f in TAIL_FRACTIONS if f in mine]
        out[seed] = dict(tail_mean = float(np.mean(tail)) if tail else float("nan"),
                         tail_worst = float(np.max(tail)) if tail else float("nan"), at_budget = mine.get(1.0, float("nan")))
    return out


def _plot(out: Path, by_cell: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
    for env in SCALE_GAMES:
        fig, ax = subplots(figsize = (7, 4.2))
        if ax is None:
            return
        for (cell_env, method), rows in sorted(by_cell.items()):
            if cell_env != env or not rows:
                continue
            fractions = sorted({float(r["fraction"]) for r in rows})
            games = [float(np.median([r["games"] for r in rows if float(r["fraction"]) == f])) for f in fractions]
            values = [bootstrap_iqm([r["exploitability"] for r in rows if float(r["fraction"]) == f]) for f in fractions]
            ax.plot(games, [v[0] for v in values], marker = "o", label = title(method))
            ax.fill_between(games, [v[1] for v in values], [v[2] for v in values], alpha = 0.15)
        ax.set(xscale = "log", xlabel = "algorithmic games", ylabel = "exact exploitability (seed IQM + 95% CI)", title = env)
        ax.legend(fontsize = 7)
        save_figure(fig, out, f"{env}_exploitability")


def run_scale_eval(options: RunOptions) -> Path:
    out = prepare_output(options)
    train_specs = _train_specs(out.parent, options.smoke)
    require_complete(train_specs, _train_records(out.parent))
    specs = _eval_specs(train_specs, options.smoke)
    # 遍历器约 30 GB：无论命令行给多少 worker，都只跑一个评估进程，并让它用满全部线程
    records = run_cells(specs, run_exact_eval_cell, replace(options, workers = 1, cell_threads = os.cpu_count() or 1))
    by_cell = _eval_rows(out, records)
    lines = ["# scale-eval：全树精确 exploitability（exp-a-spiel）", "",
             f"checkpoint 取预算 {list(EVAL_FRACTIONS)} 处最近的一个；tail_mean / tail_worst 为 {list(TAIL_FRACTIONS)} "
             "三个点的平均与最大值。收益为 +-1，单位与 OpenSpiel 的 exploitability 相同。", "",
             "| Environment | Method | Deploy | n | tail_mean | tail_worst | at_budget |", "|---|---|---|---:|---|---|---|"]
    table: dict[str, Any] = {}
    for (env, method), rows in sorted(by_cell.items()):
        per_seed = _seed_metrics(rows)
        stats = {k: bootstrap_iqm([m[k] for m in per_seed.values()]) for k in ("tail_mean", "tail_worst", "at_budget")}
        table[f"{env}/{method}"] = dict(per_seed = per_seed, iqm = stats)
        cells = " | ".join(f"{s[0]:.4g} [{s[1]:.4g}, {s[2]:.4g}]" for s in stats.values())
        lines.append(f"| {env} | {title(method)} | {deployment(method)} | {len(per_seed)} | {cells} |")
    write_json(out / "records.json", dict(table = table))
    write_summary(out, "\n".join(lines) + "\n")
    _plot(out, by_cell)
    return out


def planned_eval_cells() -> int:
    return len(SCALE_GAMES) * len(SCALE_METHODS)


# -------- scale-crossplay --------
def run_crossplay(options: RunOptions) -> Path:
    """每个游戏内，对 scale-train 每个格子的最终策略做循环赛，报告每种方法对同游戏其它所有策略的平均得分。"""
    out = prepare_output(options)
    source = out.parent / "scale-train"
    train_specs = _train_specs(out.parent, options.smoke)
    records = _train_records(out.parent)
    require_complete(train_specs, records)
    n_games = 8 if options.smoke else CROSSPLAY_GAMES
    rng = np.random.default_rng(0)
    scores: dict[str, dict[str, list[float]]] = {}
    for env in SCALE_GAMES:
        game = load_game(env)
        scorer = make_scorer(game)
        policies = []
        for spec in train_specs:
            cfg = config_from_dict(spec.payload["config"])
            if cfg.env == env:
                _games, path = _checkpoints(source / spec.cell_id)[-1]
                policies.append((spec.payload["method"], _actor(*load_nets(cfg, path))))
        by_method: dict[str, list[float]] = {}
        for i, (method_i, pol_i) in enumerate(policies):
            for j, (method_j, pol_j) in enumerate(policies):
                if j <= i:
                    continue
                s = play_match(game, pol_i, pol_j, rng, n_games, scorer)
                by_method.setdefault(method_i, []).append(s)
                by_method.setdefault(method_j, []).append(1.0 - s)
        scores[env] = by_method
    write_json(out / "records.json", dict(scores = scores, games_per_pair = n_games))
    lines = ["# scale-crossplay：最终策略循环赛", "", f"每对 {n_games} 局（座位轮换）；得分为对同游戏其它所有策略的平均。", "",
             "| Environment | Method | 平均得分 |", "|---|---|---:|"]
    for env, by_method in scores.items():
        lines += [f"| {env} | {title(method)} | {np.mean(values):.4g} |" for method, values in sorted(by_method.items())]
    write_summary(out, "\n".join(lines) + "\n")
    return out


# -------- scale-history --------
HISTORY_KEYS = ("worst_past", "final_worst_past", "forgotten")


def run_history_cell(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    """一次训练：按预算均匀取出 checkpoint，后者对前者两两对打；losses[i][j] 是第 i 个对第 j 个（j < i）的败率。"""
    cfg = config_from_dict(payload["config"])
    checkpoints = _checkpoints(cell_dir.parent.parent / "scale-train" / payload["train"])
    picked: list[tuple[int, Path]] = []
    for fraction in payload["fractions"]:
        item = _nearest(checkpoints, fraction * cfg.budget)
        if item not in picked:
            picked.append(item)
    game = load_game(cfg.env)
    scorer = make_scorer(game)
    rng = np.random.default_rng(cfg.seed)
    actors = [_actor(*load_nets(cfg, path)) for _games, path in picked]
    losses = [[1.0 - play_match(game, actors[i], actors[j], rng, payload["games"], scorer) for j in range(i)]
              for i in range(len(actors))]
    write_json(cell_dir / "history.json", dict(games = [g for g, _path in picked], losses = losses,
                                               games_per_pair = payload["games"]))
    return history_forgetting(losses, payload["margin"])


def _history_specs(train_specs: list[CellSpec], smoke: bool) -> list[CellSpec]:
    points = 3 if smoke else HISTORY_POINTS
    fractions = [k / points for k in range(1, points + 1)]
    return [CellSpec(f"{s.payload['config']['env']}_{s.payload['method']}_s{s.payload['config']['seed']}_history",
                     dict(train = s.cell_id, config = s.payload["config"], method = s.payload["method"], fractions = fractions,
                          games = 8 if smoke else HISTORY_GAMES, margin = HISTORY_MARGIN))
            for s in train_specs]


def run_scale_history(options: RunOptions) -> Path:
    out = prepare_output(options)
    train_specs = _train_specs(out.parent, options.smoke)
    require_complete(train_specs, _train_records(out.parent))
    specs = _history_specs(train_specs, options.smoke)
    records = run_cells(specs, run_history_cell, options)
    done = done_records(records)
    lines = ["# scale-history：对自己历史版本的采样对局", "",
             f"每次训练按预算均匀取 {HISTORY_POINTS} 个 checkpoint，后者对前者每对 {HISTORY_GAMES} 局（座位轮换）。"
             "worst_past：每个 checkpoint 对更早 checkpoint 的最坏败率减 1/2，按 checkpoint 平均；final_worst_past：最终策略的这个量；"
             f"forgotten：曾被打败（败率 <= {0.5 - HISTORY_MARGIN:g}）后又打败更晚 checkpoint（>= {0.5 + HISTORY_MARGIN:g}）的"
             "checkpoint 占比。seed IQM 与 95% 区间。", "",
             "| Environment | Method | n | " + " | ".join(HISTORY_KEYS) + " |", "|---|---|---:|" + "|".join("---" for _ in HISTORY_KEYS) + "|"]
    table: dict[str, Any] = {}
    for env in SCALE_GAMES:
        for method in SCALE_METHODS:
            rows = [r["summary"] for r in done if r["payload"]["config"]["env"] == env and r["payload"]["method"] == method]
            if not rows:
                continue
            stats = {k: bootstrap_iqm([row[k] for row in rows]) for k in HISTORY_KEYS}
            table[f"{env}/{method}"] = dict(per_seed = rows, iqm = stats)
            lines.append(f"| {env} | {title(method)} | {len(rows)} | "
                         + " | ".join(f"{s[0]:.3g} [{s[1]:.3g}, {s[2]:.3g}]" for s in stats.values()) + " |")
    write_json(out / "records.json", dict(table = table))
    write_summary(out, "\n".join(lines) + "\n")
    return out


def load_history(cell_dir: Path) -> dict[str, Any]:
    return load_json(cell_dir / "history.json")


def planned_history_cells() -> int:
    return planned_train_cells()
