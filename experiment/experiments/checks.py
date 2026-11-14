"""独立检查与参考线：preflight（环境与评估器）、feasibility（单格耗时估计）、references（CFR+ 参考线）。"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from open_spiel.python import policy as sp_policy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability as expl_mod

from experiment.common.io_util import load_jsonl, log, write_json, write_summary
from experiment.common.resources import peak_memory_bytes
from experiment.common.runner import CellSpec, RunOptions, done_records, prepare_output, run_cells
from experiment.experiments.cells import run_train_cell
from experiment.experiments.methods import ENT_FINALS, build_config
from experiment.experiments.protocol import MAIN_BUDGET, MAIN_GAMES, SCALE_GAMES
from experiment.neural.envs import GAME_SPECS, information_tensor, load_game, obs_and_action_size
from experiment.neural.evals import full_exploitability, policy_values
from experiment.neural.exact_large import coherence_mismatches
from experiment.neural.mlp import SmallMLP
from experiment.neural.policies import UniformActor, frozen_actor
from experiment.neural.references import run_cfr_plus

CFR_ITERATIONS = {"leduc_poker": 2000, "goofspiel4": 2000, "liars_dice": 500}


class CheckFailed(RuntimeError):
    """至少一项检查失败；报告已写入输出目录。"""


# -------- preflight --------
def _play_random(game: Any, n: int, rng: np.random.Generator) -> int:
    done = 0
    for _ in range(n):
        state = game.new_initial_state()
        for _step in range(1000):
            if state.is_terminal():
                break
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                state.apply_action(int(actions[int(rng.choice(len(actions), p = np.asarray(probs)))]))
            else:
                legal = state.legal_actions()
                state.apply_action(int(legal[int(rng.integers(len(legal)))]))
        done += int(state.is_terminal())
    return done


def _check(name: str, fn: Any) -> dict[str, Any]:
    log(f"preflight {name}")
    try:
        detail = fn()
        return dict(ok = bool(detail.pop("ok")), **detail)
    except Exception as exc:
        return dict(ok = False, error = f"{type(exc).__name__}: {exc}")


def _games_ok() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    detail: dict[str, Any] = {}
    for env in GAME_SPECS:
        game = load_game(env)
        state = game.new_initial_state()
        while state.is_chance_node():
            state.apply_action(int(state.chance_outcomes()[0][0]))
        obs_dim, n_actions = obs_and_action_size(game)
        detail[env] = dict(terminal = _play_random(game, 8, rng), obs_dim = obs_dim, n_actions = n_actions,
                           obs_ok = len(information_tensor(state, int(state.current_player()))) == obs_dim)
    ok = all(d["terminal"] == 8 and d["obs_ok"] for d in detail.values())
    return dict(ok = ok, games = detail)


def _cfr_kuhn() -> dict[str, Any]:
    game = load_game("kuhn_poker")
    solver = cfr.CFRPlusSolver(game)
    for _ in range(1000):
        solver.evaluate_and_update_policy()
    pol = solver.average_policy()
    expl = float(expl_mod.exploitability(game, pol))
    value = float(policy_values(game, [pol, pol])[0])
    # Kuhn 扑克先手的博弈值是 -1/18
    return dict(ok = expl < 1e-3 and abs(value + 1.0 / 18.0) < 0.05, exploitability = expl, value = value)


def _tabular_expander() -> dict[str, Any]:
    game = load_game("kuhn_poker")
    direct = float(expl_mod.exploitability(game, sp_policy.UniformRandomPolicy(game)))
    expanded = full_exploitability(game, UniformActor())
    return dict(ok = abs(direct - expanded) < 1e-10, direct = direct, expanded = expanded)


def _short_runs() -> dict[str, Any]:
    """四种训练入口各在 Kuhn 上跑一个极小预算，检查能跑完且评估有限。"""
    detail: dict[str, Any] = {}
    for method in ("selfplay", "coda_r", "psro", "nfsp"):
        params = dict(ent_final = ENT_FINALS[0]) if method != "nfsp" else {}
        cfg = build_config(method, "kuhn_poker", 0, MAIN_BUDGET, params).shrunk()
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_train_cell(dict(config = cfg.as_dict()), Path(tmp))
        detail[method] = dict(games = summary.get("games"), budget = cfg.budget)
    ok = all(d["games"] == d["budget"] for d in detail.values())
    return dict(ok = ok, runs = detail)


def _eas_coherence() -> dict[str, Any]:
    """exp-a-spiel 与 OpenSpiel 在规模游戏上逐步一致（当前玩家、合法动作、信息状态张量、胜负），否则精确评估的不是同一个博弈。"""
    rng = np.random.default_rng(0)
    mismatches = {env: coherence_mismatches(env, 2000, rng) for env in SCALE_GAMES}
    return dict(ok = all(n == 0 for n in mismatches.values()), mismatches = mismatches)


def run_preflight(options: RunOptions) -> Path:
    out = prepare_output(options)
    results = {name: _check(name, fn) for name, fn in (
        ("games", _games_ok), ("cfr_plus_kuhn", _cfr_kuhn), ("tabular_expander", _tabular_expander), ("short_runs", _short_runs),
        ("eas_coherence", _eas_coherence))}
    overall = all(r["ok"] for r in results.values())
    write_json(out / "report.json", dict(overall = overall, results = results))
    write_summary(out, "# preflight\n\n" + "\n".join(f"- {k}: **{'PASS' if v['ok'] else 'FAIL'}**" for k, v in results.items())
                  + f"\n\nOverall: **{'PASS' if overall else 'FAIL'}**\n")
    if not overall:
        raise CheckFailed(f"preflight failed: {[k for k, v in results.items() if not v['ok']]}")
    return out


# -------- feasibility --------
def _seconds_per_rollout(env: str, n_rollouts: int) -> float:
    """CoDA-R 单线程的每 rollout 训练耗时（采样加 PPO 更新），取日志里的 rollout_s，不含评估。"""
    params = dict(ent_final = ENT_FINALS[0], eta = 0.1)
    cfg = build_config("coda_r", env, 0, n_rollouts * 512, params, eval_points = 2, tail_eval_rollouts = 10 ** 6,
                       oracle = "none", evaluate_mixtures = False, exact_loss_every = -1)
    with tempfile.TemporaryDirectory() as tmp:
        run_train_cell(dict(config = cfg.as_dict()), Path(tmp))
        rows = [h["rollout_s"] for h in load_jsonl(Path(tmp) / "rollouts.jsonl") if "rollout_s" in h]
    # 第一个 rollout 含首次前向的初始化开销，不计入
    return float(np.mean(rows[1:] or rows))


def run_feasibility(options: RunOptions) -> Path:
    """
    每个游戏：每 rollout 秒数、峰值内存与单格外推。主实验游戏另测单次 OpenSpiel 全树评估；
    规模游戏训练中不评估，它们的精确评估（约 30 GB、一次约 95 秒）在 scale-eval 里单独进行，这里不测。
    """
    out = prepare_output(options)
    # 与并行 worker 相同的单线程条件
    torch.set_num_threads(1)
    n_rollouts = 3 if options.smoke else 8
    rows: dict[str, Any] = {}
    for env in (*MAIN_GAMES, *SCALE_GAMES):
        log(f"feasibility {env}")
        game = load_game(env)
        obs_dim, n_actions = obs_and_action_size(game)
        per_rollout = _seconds_per_rollout(env, n_rollouts)
        row: dict[str, Any] = dict(seconds_per_rollout = per_rollout, exact_eval_seconds = float("nan"))
        if env not in SCALE_GAMES:
            t0 = time.time()
            full_exploitability(game, frozen_actor(SmallMLP(obs_dim, n_actions, 128), "cpu"))
            row["exact_eval_seconds"] = time.time() - t0
        row["peak_memory_gb"] = peak_memory_bytes() / 1024 ** 3
        n_main = MAIN_BUDGET / 512
        evals = 12 + 0.2 * n_main / 8 if env not in SCALE_GAMES else 0.0
        row["estimated_cell_hours"] = (n_main * per_rollout + evals * np.nan_to_num(row["exact_eval_seconds"])) / 3600
        rows[env] = row
    write_json(out / "report.json", dict(overall = True, rows = rows))
    lines = ["# feasibility", "", "| env | s / rollout | 单次评估 s | 峰值内存 GB | 单格估计 h（预算 B） |", "|---|---:|---:|---:|---:|"]
    for env, row in rows.items():
        lines.append(f"| {env} | {row['seconds_per_rollout']:.2f} | {row['exact_eval_seconds']:.1f} | "
                     f"{row['peak_memory_gb']:.2f} | {row['estimated_cell_hours']:.2f} |")
    lines += ["", "单格估计不含池诊断与 q/Q 聚合，需以首格实测为准；规模游戏只含训练，精确评估见 scale-eval。"]
    write_summary(out, "\n".join(lines) + "\n")
    return out


# -------- references --------
def run_reference_cell(payload: dict[str, Any], cell_dir: Path) -> dict[str, Any]:
    return run_cfr_plus(payload["env"], payload["iterations"], cell_dir)


def run_references(options: RunOptions) -> Path:
    out = prepare_output(options)
    specs = [CellSpec(f"{env}_cfr_plus", dict(env = env, iterations = 4 if options.smoke else it))
             for env, it in CFR_ITERATIONS.items()]
    records = run_cells(specs, run_reference_cell, options)
    lines = ["# references：CFR+ 参考线", "", "| env | 迭代 | 最终 exploitability |", "|---|---:|---:|"]
    for r in done_records(records):
        lines.append(f"| {r['payload']['env']} | {r['payload']['iterations']} | {r['summary']['final_exploitability']:.4g} |")
    write_summary(out, "\n".join(lines) + "\n")
    return out
